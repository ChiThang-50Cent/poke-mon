"""BS1.1: live belief search layered on `NeuralPlayer` (C proposes, search may override).

Inputs of the search are only the learner's own team, its pending request, its poke-env battle
snapshot and protocol history, and public data (the BS0 reconstruction path). The real opponent
identity and any server-side state never enter. The live recurrent state is never touched: the
workers restore a deep copy of the pre-frame snapshot and forward their own players.

Time model (all `time.time()`): D is the end-to-end deadline measured from the moment the
player received the request (before C inference) until the choice is handed to the websocket;
M is reserved for order construction and the send. The search budget of one decision is
D - (t_search_start - t_request) - M, and search is skipped (C's action) below `min_budget_s`.
`choose_move` blocks the event loop while searching (one battle at a time, `--concurrent 1`).
"""

from __future__ import annotations

import copy
import hashlib
import json
import pickle
import random
import time
import traceback
import zlib
from dataclasses import dataclass
from typing import Any

MIN_PAIRS = 4

CATEGORIES = (
    "search_override",
    "search_fallback_no_signal",
    "search_fallback_unsupported",
    "search_skipped_budget",
    "search_skipped_workers",
    "search_skipped_no_snapshot",
    "search_error",
    "search_error_new_class",
)

# BS1.2b exception allowlist: the only search exceptions that are known and safely handled (the
# decision falls back to C's legal action). Both classes were FIXED in BS1.2b, so any
# occurrence is reported by category (target 0); anything else is a NEW class and aborts the run.
KNOWN_ERRORS: tuple[tuple[str, str], ...] = (
    ("forced_action_outside_mask", r"RuntimeError: forced action \d+ outside mask"),
    ("parser_forward_exception",
     r"replay_parser\.exceptions\.\w+: Exception on Replay Forward-Fill"),
)


def classify_error(text: str) -> str | None:
    """Name of the known allowlisted class of a traceback, else None (= new class)."""
    import re

    for name, pattern in KNOWN_ERRORS:
        if re.search(pattern, text):
            return name
    return None


@dataclass(frozen=True)
class SearchSettings:
    deadline_s: float = 5.0  # D, request received -> choice handed to the websocket
    margin_s: float = 0.3  # M, reserved for order construction and the send
    min_budget_s: float = 0.5  # skip search below this budget
    opp_model: str = "mixture"
    n_worlds: int = 8
    n_seeds: int = 8
    workers: int = 4
    seed: int = 20261101
    fail_fast: bool = True  # a NEW exception class aborts the run (after C's action is returned)
    fail_dir: str | None = None  # where the new-class evidence (traceback + root payload) goes

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> SearchSettings:
        return cls(**(raw or {}))


_POOL: Any = None


def start_pool(settings: SearchSettings, pool_factory: Any = None) -> Any:
    """Process-wide worker pool (startup blocks, before any request is served)."""
    global _POOL
    if _POOL is None:
        if pool_factory is None:
            from pokeai.cf.recon_search import SearchPool

            pool_factory = SearchPool
        _POOL = pool_factory(settings.workers)
    return _POOL


def stop_pool() -> None:
    global _POOL
    if _POOL is not None:
        _POOL.close()
        _POOL = None


def hidden_digest(hidden: Any) -> str | None:
    if hidden is None:
        return None
    import torch

    def flat(x: Any) -> Any:
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy().tobytes()
        if isinstance(x, (tuple, list)):
            return [flat(i) for i in x]
        return x

    return hashlib.sha256(pickle.dumps(flat(hidden))).hexdigest()[:16]


class LiveSearch:
    """Per-player search layer. `pool` is injectable (tests); default is the shared pool."""

    def __init__(self, settings: SearchSettings, pool: Any = None) -> None:
        self.settings = settings
        self._pool = pool
        self.counts: dict[str, int] = {}
        self.abort_reason: str | None = None
        self._payload: Any = None

    @property
    def pool(self) -> Any:
        return self._pool if self._pool is not None else start_pool(self.settings)

    def _count(self, category: str) -> None:
        self.counts[category] = self.counts.get(category, 0) + 1

    def run(
        self, player: Any, battle: Any, tag: str, c_action: int, mask: Any, probs: Any,
        t_request: float, snapshot: dict[str, Any] | None, pending_start: int,
        decision_no: int,
    ) -> tuple[int, dict[str, Any]]:
        """Return (final action, log row). Never raises: any failure keeps C's legal action."""
        cfg = self.settings
        t_start = time.time()
        self._payload = None
        row: dict[str, Any] = {
            "t_request": t_request, "t_search_start": t_start, "category": None,
            "c_action": c_action, "action": c_action, "budget_s": None,
        }
        try:
            budget = cfg.deadline_s - (t_start - t_request) - cfg.margin_s
            row["budget_s"] = budget
            if snapshot is None:
                row["category"] = "search_skipped_no_snapshot"
            elif budget < cfg.min_budget_s:
                row["category"] = "search_skipped_budget"
            else:
                pool = self.pool
                pool.maintain()  # non-blocking upkeep (async respawn of a dead/stuck worker)
                row["restarts"] = pool.restarts
                if not pool.usable():
                    row["category"] = "search_skipped_workers"
                else:
                    return self._search(
                        player, battle, tag, c_action, mask, probs, t_request, snapshot,
                        pending_start, decision_no, budget, t_start, row)
        except Exception as exc:  # noqa: BLE001 - fail-safe: C's action
            full = traceback.format_exc()
            known = classify_error(full)
            row["category"] = "search_error" if known else "search_error_new_class"
            row["error_class"] = known or "NEW"
            row["error_cause"] = f"{type(exc).__name__}: {exc}"[:300]
            row["error"] = full[-1500:]
            if known is None:
                self._new_class(row, full, tag)
        row["t_search_end"] = time.time()
        self._count(row["category"])
        return c_action, row

    def _new_class(self, row: dict[str, Any], full: str, tag: str) -> None:
        """Log the evidence of an unknown exception class and flag the run to abort."""
        cfg = self.settings
        if cfg.fail_dir:
            from pathlib import Path

            out = Path(cfg.fail_dir)
            out.mkdir(parents=True, exist_ok=True)
            stem = f"new_class_{zlib.crc32(tag.encode()):08x}_{int(time.time())}"
            (out / f"{stem}.txt").write_text(full)
            if self._payload is not None:
                (out / f"{stem}.payload.pkl").write_bytes(pickle.dumps(self._payload))
            row["evidence"] = str(out / stem)
        if cfg.fail_fast:
            self.abort_reason = f"new search exception class: {row['error_cause']}"

    MAX_FORENSICS = 40

    def _keep_forensics(self, payload: Any, tag: str, decision_no: int, d: dict[str, Any]) -> None:
        """Persist the root payload of a decision with rejected worlds / dropped pairs: the world
        construction is a deterministic function of the payload, so it is exactly reproducible."""
        from pathlib import Path

        out = Path(self.settings.fail_dir) / "forensics"
        out.mkdir(parents=True, exist_ok=True)
        if len(list(out.glob("*.payload.pkl"))) >= self.MAX_FORENSICS:
            return
        stem = f"{zlib.crc32(tag.encode()):08x}_{decision_no}"
        (out / f"{stem}.payload.pkl").write_bytes(pickle.dumps(payload))
        (out / f"{stem}.json").write_text(json.dumps({
            "tag": tag, "decision_no": decision_no, "mask_rejects": d["mask_rejects"],
            "dropped": d["dropped"], "world_rejected_mask": d["world_rejected_mask"],
            "dropped_pairs": d["dropped_pairs"]}))

    def _search(
        self, player: Any, battle: Any, tag: str, c_action: int, mask: Any, probs: Any,
        t_request: float, snapshot: dict[str, Any], pending_start: int, decision_no: int,
        budget: float, t_start: float, row: dict[str, Any],
    ) -> tuple[int, dict[str, Any]]:
        from pokeai.cf.cf1 import choose_candidates
        from pokeai.cf.offline import PLAYER_NAMES, swap_roles
        from pokeai.cf.recon_search import Payload
        from pokeai.cf.recon_world import LearnerInfo

        cfg = self.settings
        mask_list = [int(bool(x)) for x in mask]
        probs_list = [float(x) for x in probs]
        rng = random.Random(f"{cfg.seed}:{tag}:{decision_no}")
        slots = [c["slot"] for c in choose_candidates(
            {"mask": mask_list, "probs": probs_list}, rng)]
        candidates = [c_action] + [s for s in slots if s != c_action]
        candidates = candidates[:4]
        row["candidates"] = candidates
        if len(candidates) < 2:
            row["category"] = "search_fallback_no_signal"
            row["reason"] = "fallback:single_candidate"
            row["t_search_end"] = time.time()
            self._count(row["category"])
            return c_action, row
        history = player.history[tag]
        swapped = battle.player_role == "p2"
        names = PLAYER_NAMES

        def to_canonical(text: str) -> str:
            # canonical seat + the harness account names, so the opponent's fresh python
            # battle learns its role from the `|player|` lines as in the offline roots
            text = swap_roles(text) if swapped else text
            if text.startswith("|player|"):
                parts = text.split("|")
                if len(parts) > 3 and parts[2] in names:
                    parts[3] = names[parts[2]]
                    text = "|".join(parts)
            return text

        request = copy.deepcopy(battle._last_request)
        if swapped:  # the sim seats the learner as p1; see offline.swap_roles
            request = json.loads(swap_roles(json.dumps(request)))
        info = LearnerInfo(
            lines=[to_canonical(x) for x in history], request=request,
            own_team=player._team.yield_team(),
            pending_p1_lines=[to_canonical(x) for x in history[pending_start:]],
            swapped=swapped)
        live_names = {}
        for line in history:
            if line.startswith("|player|"):
                parts = line.split("|")
                if len(parts) > 3 and parts[2] in ("p1", "p2") and parts[3]:
                    live_names[parts[2]] = parts[3]
        mine, theirs = battle.player_role, "p2" if battle.player_role == "p1" else "p1"
        if mine in live_names and theirs in live_names:
            info.names = {PLAYER_NAMES["p1"]: live_names[mine],
                          PLAYER_NAMES["p2"]: live_names[theirs]}
        row["learner_role"] = battle.player_role
        root_id = (zlib.crc32(tag.encode()) % 100_000) * 1000 + decision_no
        payload = Payload(
            root_id=root_id, opp_model=cfg.opp_model, seed=cfg.seed, info=info,
            learner_snapshot=snapshot, old_tag=tag, mask=mask_list, candidates=candidates,
            n_worlds=cfg.n_worlds, n_seeds=cfg.n_seeds)
        self._payload = payload
        row["hidden_before"] = hidden_digest(player.hidden.get(tag))
        d = self.pool.decision(payload, budget, t_request=t_start)
        row["hidden_after"] = hidden_digest(player.hidden.get(tag))
        if (d["world_rejected_mask"] or d["dropped_pairs"]) and cfg.fail_dir:
            self._keep_forensics(payload, tag, decision_no, d)
        choice = d["choice"]
        row.update(
            n_pairs=choice["n_pairs"], n_rollouts=d["n_rollouts"], reason=choice["reason"],
            stats=choice["stats"], worlds_built=d["worlds_built"],
            unsupported=sorted({u.split(" (")[0] for u in d["unsupported"]}),
            illegal=d["illegal"], retries=d["retries"], rounds=d["rounds"], turns=d["turns"], rollout_errors=d["errors"][:3],
            hard_deadline_hit=d["hard_deadline_hit"], workers_done=d["workers_done"],
            pool_wall_s=d["wall_s"], timing=d["timing"], restarts=self.pool.restarts,
            opps=sorted({p["opp"] for p in d["pairs"]}),
            world_rejected_mask=d["world_rejected_mask"], mask_rejects=d["mask_rejects"][:3],
            dropped_pairs=d["dropped_pairs"], dropped=d["dropped"][:3])
        action = choice["action"]
        if not mask_list[action]:  # cannot happen (decide() asserts the C fallback is legal)
            action = c_action
        if choice["override"]:
            row["category"] = "search_override"
        elif d["unsupported"] and choice["n_pairs"] == 0:
            row["category"] = "search_fallback_unsupported"
        else:
            row["category"] = "search_fallback_no_signal"
        row["action"] = action
        row["t_search_end"] = time.time()
        self._count(row["category"])
        return action, row
