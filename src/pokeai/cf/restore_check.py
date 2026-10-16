"""CF0.1: snapshot/restore determinism gate over offline battles.

Pre-registered checkpoint rule (``CheckpointRule``): per battle a seeded RNG draws an
early (turn 2-5), mid (6-14) and late (15-25) target; the first decision round whose
upcoming turn >= target is snapshotted. Additionally the first forced-switch round, up
to two Revival-Blessing (``reviving``) rounds and up to two rounds where a side's active
Pokemon is Cramorant. A global cap (default 50) bounds the total.
"""

from __future__ import annotations

import json
import random
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pokeai.cf.offline import (
    SIDES,
    Checkpoint,
    OfflineMatch,
    PlayerSpec,
    make_player,
    new_tag,
    normalise_frame,
    normalise_line,
)
from pokeai.cf.sim import Sim

WINDOW = 3  # rounds replayed from each checkpoint
C_POLICY = "neural:models/pokedecision-v1-value"
PREVIEW = (
    "third_party/metamon/metamon/backend/team_preview/gen9ou_high_elo_v4/best_model.pt"
)
TRAP_PREFIXES = (
    "|error|[Unavailable choice] Can't switch: The active Pokémon is trapped",
    "|error|[Invalid choice] Can't switch: The active Pokémon is trapped",
)


def load_pool(root: str) -> dict[str, tuple[str, str]]:
    """{file name: (packed team, showdown text)} for a validated team directory."""
    from pokeai.env.teams import load_teamset

    teamset = load_teamset(root)
    pool = {}
    for file in sorted(teamset.team_files):
        text = Path(file).read_text()
        pool[Path(file).name] = (
            teamset.join_team(teamset.parse_showdown_team(text)), text
        )
    return pool


def pending_request(pending: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Parsed `|request|` payloads waiting in a pending event list, by side."""
    out = {}
    for ev in pending:
        if ev["t"] == "sideupdate" and ev["data"].startswith("|request|"):
            out[ev["side"]] = json.loads(ev["data"][len("|request|"):])
    return out


def upcoming_turn(match: OfflineMatch) -> int:
    turn = match.turn
    for ev in match.pending:
        if ev["t"] == "update":
            for line in ev["omni"]:
                if line.startswith("|turn|"):
                    turn = max(turn, int(line[6:]))
    return turn


def is_decision_round(requests: dict[str, dict[str, Any]]) -> bool:
    return any(
        not r.get("wait") and not r.get("teamPreview") for r in requests.values()
    )


@dataclass
class CheckpointRule:
    """Pre-registered selection rule (see module docstring).

    Per-battle budget (default 5): the early/mid/late/forced core (4) always counts;
    one extra slot goes to the first `cramorant` (even battle index; Cramorant active
    with Surf/Dive available) or `tera` (odd battle index; the original run is forced
    to Terastallize if legal) opportunity. Revival-Blessing `reviving` rounds are
    exempt from the budget (<= 2 per battle). `cap_remaining` is a hard global cap.
    """

    seed: int
    cap_remaining: list[int]  # shared mutable global cap
    battle_index: int = 0
    budget: int = 5
    targets: dict[str, int] = field(default_factory=dict)
    taken: set[str] = field(default_factory=set)
    counts: Counter = field(default_factory=Counter)

    def __post_init__(self) -> None:
        rng = random.Random(self.seed)
        self.targets = {
            "early": rng.randint(2, 5), "mid": rng.randint(6, 14),
            "late": rng.randint(15, 25),
        }

    def reasons(self, match: OfflineMatch) -> list[str]:
        if self.cap_remaining[0] <= 0 or match.ended:
            return []
        requests = pending_request(match.pending)
        if not is_decision_round(requests):
            return []
        out = []
        turn = upcoming_turn(match)
        for name, target in self.targets.items():
            if name not in self.taken and turn >= target:
                self.taken.add(name)
                out.append(name)
        if "forced" not in self.taken and any(
            "forceSwitch" in r for r in requests.values()
        ):
            self.taken.add("forced")
            out.append("forced")
        if self.counts["reviving"] < 2 and any(
            any(p.get("reviving") for p in r.get("side", {}).get("pokemon", []))
            for r in requests.values()
        ):
            self.counts["reviving"] += 1
            out.append("reviving")
        extra_used = len(self.taken & {"cramorant", "tera"})
        core_used = len(self.taken & {"early", "mid", "late", "forced"})
        if extra_used == 0 and core_used < self.budget:
            learner = requests.get("p1", {})
            if self.battle_index % 2 == 0:
                active = learner.get("active") or [{}]
                moves = {m.get("id") for m in active[0].get("moves", [])}
                details = (learner.get("side", {}).get("pokemon") or [{}])[0].get(
                    "details", ""
                )
                if (
                    details.startswith("Cramorant") and not learner.get("wait")
                    and moves & {"surf", "dive"}
                ):
                    self.taken.add("cramorant")
                    out.append("cramorant")
            elif (
                upcoming_turn(match) >= 2 and not learner.get("wait")
                and (learner.get("active") or [{}])[0].get("canTerastallize")
            ):
                self.taken.add("tera")
                out.append("tera_forced")
        return out


@dataclass
class TakenCheckpoint:
    battle: str
    reasons: list[str]
    checkpoint: Checkpoint
    state_at: dict[str, Any]
    turn: int
    round_index: int


def mechanics(lines: list[str], requests: list[dict[str, Any]]) -> Counter:
    counts: Counter = Counter()
    for line in lines:
        if line.startswith("|-terastallize|"):
            counts["terastallize"] += 1
        if "Cramorant" in line:
            counts["cramorant_lines"] += 1
        if line.startswith("|-formechange|") and "Cramorant" in line:
            counts["cramorant_formechange"] += 1
        if line.startswith("|move|") and "Revival Blessing" in line:
            counts["revival_blessing_move"] += 1
    for req in requests:
        if "forceSwitch" in req:
            counts["force_switch_requests"] += 1
        if any(p.get("reviving") for p in req.get("side", {}).get("pokemon", [])):
            counts["reviving_requests"] += 1
        if req.get("active") and req["active"][0].get("canTerastallize"):
            counts["tera_available_requests"] += 1
    return counts


def match_mechanics(rounds: list[Any]) -> Counter:
    lines = [line for r in rounds for chunk in r.omni for line in chunk]
    requests = []
    for r in rounds:
        for side in SIDES:
            for raw in r.delivered[side]:
                body = raw.split("\n", 1)[1] if "\n" in raw else ""
                for line in body.split("\n"):
                    if line.startswith("|request|"):
                        requests.append(json.loads(line[len("|request|"):]))
    return mechanics(lines, requests)


def signature_mechanics(sig: list[dict[str, Any]]) -> Counter:
    """Mechanics counters from a (normalised) signature."""
    lines = [x for rr in sig for chunk in rr["omni"] for x in chunk]
    requests = [
        json.loads(line[len("|request|"):])
        for rr in sig for s in SIDES for raw in rr["delivered"][s]
        for line in raw.split("\n") if line.startswith("|request|")
    ]
    return mechanics(lines, requests)


def play_with_checkpoints(
    sim: Sim, players: dict[str, Any], tag: str, teams: dict[str, str],
    sim_seed: int, side_seeds: dict[str, int], rule: CheckpointRule,
) -> tuple[OfflineMatch, list[TakenCheckpoint]]:
    match = OfflineMatch(sim, players, tag)
    match.start(sim_seed, teams, side_seeds)
    taken: list[TakenCheckpoint] = []

    def hook(m: OfflineMatch) -> None:
        reasons = rule.reasons(m)
        if not reasons:
            return
        rule.cap_remaining[0] -= 1
        cp = m.checkpoint()
        if "tera_forced" in reasons:
            m.forced["p1"].append("tera")
            m.forced_events.append((m.round_index, "p1", "tera"))
        taken.append(TakenCheckpoint(
            battle=tag, reasons=reasons, checkpoint=cp,
            state_at=sim.state(tag), turn=upcoming_turn(m), round_index=m.round_index,
        ))
        m.state_rounds.update(range(m.round_index, m.round_index + WINDOW))

    match.play(hook=hook)
    return match, taken


def signature(rounds: list[Any], tag: str) -> list[dict[str, Any]]:
    """Normalised comparison view of replayed rounds (only tag and |t:| normalised)."""
    out = []
    for r in rounds:
        out.append({
            "delivered": {
                s: [normalise_frame(x, tag) for x in r.delivered[s]] for s in SIDES
            },
            "omni": [[normalise_line(x) for x in chunk] for chunk in r.omni],
            "events_out": [_norm_event(e) for e in r.events_out],
            "submitted": [list(x) for x in r.submitted],
            "decisions": {
                s: [
                    {
                        "message": d["message"],
                        "model": (
                            None if d["model"] is None else
                            {k: v for k, v in d["model"].items() if k != "probs"}
                        ),
                    }
                    for d in r.decisions[s]
                ]
                for s in SIDES
            },
            "state_after": r.state_after,
        })
    return out


def _norm_event(ev: dict[str, Any]) -> dict[str, Any]:
    ev = dict(ev)
    for key in ("omni", "p1", "p2"):
        if key in ev:
            ev[key] = [normalise_line(x) for x in ev[key]]
    if ev["t"] == "end":
        ev = {"t": "end", "data": {k: v for k, v in ev["data"].items() if k != "seed"}}
    return ev


CATEGORIES = (
    "channel_p1", "channel_p2", "channel_omni", "events_out", "sim_state",
    "observation_tokens", "observation_numbers", "legality_mask", "hidden",
    "choices", "model_action",
)


def diff_signatures(a: list[dict], b: list[dict]) -> Counter:
    """Count mismatching items per category between two signatures."""
    bad: Counter = Counter()
    if len(a) != len(b):
        bad["round_count"] += 1
    for ra, rb in zip(a, b, strict=False):
        if ra["delivered"]["p1"] != rb["delivered"]["p1"]:
            bad["channel_p1"] += 1
        if ra["delivered"]["p2"] != rb["delivered"]["p2"]:
            bad["channel_p2"] += 1
        if ra["omni"] != rb["omni"]:
            bad["channel_omni"] += 1
        if ra["events_out"] != rb["events_out"]:
            bad["events_out"] += 1
        if ra["state_after"] != rb["state_after"]:
            bad["sim_state"] += 1
        if ra["submitted"] != rb["submitted"]:
            bad["choices"] += 1
        for side in SIDES:
            da, db = ra["decisions"][side], rb["decisions"][side]
            if len(da) != len(db):
                bad["choices"] += 1
                continue
            for xa, xb in zip(da, db, strict=True):
                ma, mb = xa["model"], xb["model"]
                if (ma is None) != (mb is None):
                    bad["model_action"] += 1
                elif ma is not None:
                    if ma["tokens_sha"] != mb["tokens_sha"]:
                        bad["observation_tokens"] += 1
                    if ma["numbers_sha"] != mb["numbers_sha"]:
                        bad["observation_numbers"] += 1
                    if ma["mask"] != mb["mask"]:
                        bad["legality_mask"] += 1
                    if ma["hidden_sha"] != mb["hidden_sha"] or ma["value"] != mb["value"]:
                        bad["hidden"] += 1
                    if ma["action"] != mb["action"]:
                        bad["model_action"] += 1
    return bad


def replay_window(
    sim: Sim, players: dict[str, Any], cp: TakenCheckpoint, new_tag: str,
    original_rounds: list[Any], sabotage: str | None = None,
    forced_events: list[tuple[int, str, int | str]] = (),
) -> dict[str, Any]:
    """Restore `cp`, replay the originally submitted orders, return signature + stats.

    `sabotage` ("hidden" | "prng") deliberately breaks the restore; it exists only to
    prove the comparison can see a bad restore (negative control).
    """
    match = OfflineMatch.restore(sim, players, cp.checkpoint, new_tag)
    if sabotage == "hidden":
        for side in SIDES:
            getattr(players[side], "hidden", {}).pop(new_tag, None)
    elif sabotage == "prng":
        sim.set_prng(new_tag, "gen5,0123456789abcdef")
    immediate = sim.state(new_tag)
    for r in original_rounds:
        for side, message in r.submitted:
            match.script[side].append(message)
    match.state_rounds.update(
        range(cp.round_index, cp.round_index + len(original_rounds))
    )
    for _ in original_rounds:
        if match.finished:
            break
        for round_index, side, wanted in forced_events:  # same stress overrides
            if round_index == match.round_index:
                match.forced[side].append(wanted)
        match.step()
    result = {
        "signature": signature(match.rounds, new_tag),
        "immediate_state_equal": immediate == cp.state_at,
        "script_mismatches": len(match.script_mismatches),
        "mismatch_detail": match.script_mismatches[:3],
        "errors": [e for r in match.rounds for e in r.errors],
    }
    match.close()
    return result


def verify_checkpoints(
    sim: Sim, players: dict[str, Any], match: OfflineMatch,
    taken: list[TakenCheckpoint],
) -> list[dict[str, Any]]:
    rows = []
    for cp in taken:
        window = match.rounds[cp.round_index:cp.round_index + WINDOW]
        original = signature(window, match.tag)
        runs = [
            replay_window(
                sim, players, cp, new_tag(), window, forced_events=match.forced_events
            )
            for _ in range(2)
        ]
        row: dict[str, Any] = {
            "battle": cp.battle, "round": cp.round_index, "turn": cp.turn,
            "reasons": cp.reasons, "window_rounds": len(window),
            "restore_immediate_state_equal": [r["immediate_state_equal"] for r in runs],
            "script_mismatches": [r["script_mismatches"] for r in runs],
            "mismatch_detail": [r["mismatch_detail"] for r in runs],
            "mismatch_vs_original": [
                dict(diff_signatures(original, r["signature"])) for r in runs
            ],
            "mismatch_restore1_vs_restore2": dict(
                diff_signatures(runs[0]["signature"], runs[1]["signature"])
            ),
            "model_decisions_compared": sum(
                1 for rr in original for s in SIDES for d in rr["decisions"][s]
                if d["model"] is not None
            ),
            "error_lines": sum(len(r["errors"]) for r in runs),
            "error_lines_detail": [e for r in runs for e in r["errors"]],
            "mechanics": dict(mechanics(
                [x for rr in original for chunk in rr["omni"] for x in chunk],
                [
                    json.loads(line[len("|request|"):])
                    for rr in original for s in SIDES for raw in rr["delivered"][s]
                    for line in raw.split("\n") if line.startswith("|request|")
                ],
            )),
        }
        rows.append(row)
    return rows


def negative_control(
    sim: Sim, players: dict[str, Any], match: OfflineMatch, cp: TakenCheckpoint,
    sabotage: str,
) -> dict[str, int]:
    window = match.rounds[cp.round_index:cp.round_index + WINDOW]
    original = signature(window, match.tag)
    run = replay_window(
        sim, players, cp, new_tag(), window, sabotage, forced_events=match.forced_events
    )
    return dict(diff_signatures(original, run["signature"]))


def timed(fn: Any, *args: Any, **kwargs: Any) -> tuple[Any, float]:
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, time.perf_counter() - start


def write_json(path: str | Path, payload: Any) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, default=str) + "\n")


__all__ = [
    "CheckpointRule",
    "PlayerSpec",
    "make_player",
    "match_mechanics",
    "play_with_checkpoints",
    "timed",
    "verify_checkpoints",
    "write_json",
]
