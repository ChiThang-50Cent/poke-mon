"""BS0.2 / gate S2: belief-world rollouts with the opponent's python state built from the
reconstructed sim from its own point of view; paired (world, seed, candidate) rollouts."""

from __future__ import annotations

import argparse
import contextlib
import json
import multiprocessing as mp
import random
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))
import bs0_1_reconstruct as s1

S2_SEED = 20261200


def outcome(winner: str | None) -> float:
    return 0.5 if winner is None or winner == "tie" else 1.0 if winner == "p1" else 0.0


def revealed_by_lines(lines: list[str]) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    moves: dict[str, set[str]] = {}
    items: dict[str, set[str]] = {}
    for line in lines:
        parts = line.split("|")
        if len(parts) > 3 and parts[1] == "move" and parts[2].startswith("p1"):
            name = parts[2].split(": ", 1)[1]
            if not any(p.startswith("[from]") for p in parts[4:]):
                moves.setdefault(name, set()).add(
                    "".join(c for c in parts[3].lower() if c.isalnum()))
        if len(parts) > 3 and parts[1] in ("-item", "-enditem") and parts[2].startswith("p1"):
            name = parts[2].split(": ", 1)[1]
            items.setdefault(name, set()).add(
                "".join(c for c in parts[3].lower() if c.isalnum()))
        holder = next((t[5:].strip() for t in parts if t.startswith("[of]")), None)
        for tag in parts[3:]:
            if tag.startswith("[from] item:"):
                who = holder if holder and holder.startswith("p1") else (
                    parts[2] if len(parts) > 2 and parts[2].startswith("p1") else None)
                if who:
                    items.setdefault(who.split(": ", 1)[1], set()).add(
                        "".join(c for c in tag[12:].lower() if c.isalnum()))
    return moves, items


def pov_check(facts: dict[str, Any], info: Any, own_items: dict[str, str],
              names_species: dict[str, str]) -> list[str]:
    """Violations of the opponent's point of view (empty = correct)."""
    bad = []
    moves, items = revealed_by_lines(info.lines)
    def sid(x: str) -> str:
        return "".join(c for c in x.lower() if c.isalnum())

    by_species = {sid(names_species[n]): n for n in names_species}
    for species, m in facts["opp"].items():
        name = by_species.get(sid(species))
        known = {x.replace("-", "") for x in m["moves"]}
        allowed = moves.get(name, set()) if name else set()
        if known - allowed - {"struggle"}:
            bad.append(f"leaked_moves:{species}:{sorted(known - allowed)}")
        item = (m["item"] or "").lower().replace(" ", "")
        if item not in ("", "unknown_item", "none") and item not in items.get(name or "", set()):
            if item != (own_items.get(name or "", "") or "<none>"):
                bad.append(f"item_odd:{species}:{item}")
            else:
                bad.append(f"leaked_item:{species}:{item}")
        if m["max_hp"] not in (100, None) and m["hp_fraction"] not in (None,):
            bad.append(f"exact_hp:{species}:{m['max_hp']}")
    return bad


def s2_root(args: tuple[int, str, int, int, int]) -> dict[str, Any]:
    root_id, opponent, n_worlds, n_seeds, seed = args
    from pokeai.cf.branching import rollout
    from pokeai.cf.cf1 import _unexpected, choose_candidates, seed_triple
    from pokeai.cf.recon_match import build_world_root, p2_pov_facts
    from pokeai.cf.recon_world import Unsupported, learner_probe, reconstruct_world
    from pokeai.cf.rollouts import phase_bucket

    sim = _S_["sim"]
    out: dict[str, Any] = {"root": root_id, "opponent": opponent}
    try:
        picked = s1.draw_root(sim, root_id, seed, opponent)
        info, _, _ = s1.learner_info(sim, picked)
        players = picked["players"]
        cp = picked["root"].checkpoint
        out.update(turn=picked["turn"], phase=phase_bucket(picked["turn"]),
                   pair="|".join(picked["names"]), forced_switch=picked["forced_switch"])
        own_items = {e["ident"].split(": ", 1)[1]: e["item"] for e in info.request["side"]["pokemon"]}
        # true-root digests of the learner's inputs (oracle, evaluation only)
        # (the root observation does not depend on the forced candidate)
        mask = picked["mask_pass1"]
        legal = [i for i, ok in enumerate(mask) if ok]
        true_rollout = rollout(sim, players, picked["root"], "p1", legal[0],
                               seed_triple(root_id, 0), keep_signature=False)
        true_model = true_rollout.root_decisions["p1"][0]["model"]
        out["true_digest"] = {k: true_model[k] for k in (
            "tokens_sha", "numbers_sha", "hidden_sha", "mask")}
        worlds: list[dict[str, Any]] = []
        candidates: list[dict[str, Any]] | None = None
        for w in range(n_worlds):
            rec: dict[str, Any] = {"world": w}
            tag = None
            try:
                recon = reconstruct_world(
                    sim, s1._S["sampler"], info, world_seed=seed + root_id * 100 + w,
                    battle_seed=seed + root_id)
                tag = recon.tag
                model = learner_probe(sim, players, cp.players["p1"], cp.tag, info, recon)
                if candidates is None:
                    candidates = choose_candidates(model, random.Random(seed * 7 + root_id))
                    out["candidates"] = candidates
                world_cp = build_world_root(sim, players, cp.players["p1"], cp.tag, info, recon)
                root_like = SimpleNamespace(checkpoint=world_cp)
                facts = p2_pov_facts(world_cp)
                names_species = {
                    e["ident"].split(": ", 1)[1]: e["details"].split(",")[0]
                    for e in info.request["side"]["pokemon"]
                }
                rec["pov_violations"] = pov_check(facts, info, own_items, names_species)
                rec["pov_active_ok"] = facts["own_active"] is not None
                outcomes, walls, retries, illegal, unexpected = [], [], Counter(), 0, []
                digests = []
                sig0 = None
                for ci, cand in enumerate(candidates):
                    row_o, row_w = [], []
                    for k in range(n_seeds):
                        res = rollout(sim, players, root_like, "p1", cand["slot"],
                                      seed_triple(root_id, k), keep_signature=False)
                        row_o.append(outcome(res.winner))
                        row_w.append(res.wall_s)
                        unexpected += _unexpected(res.errors)
                        retries.update(e for e in res.errors if e.startswith("protocol-retry:"))
                        illegal += res.illegal_actions
                        m = res.root_decisions["p1"][0]["model"]
                        digests.append({x: m[x] for x in (
                            "tokens_sha", "numbers_sha", "hidden_sha", "mask")})
                        if ci == 0 and k == 0:
                            sig0 = (res.sig_digest, res.winner, res.final_prng)
                    outcomes.append(row_o)
                    walls.append(row_w)
                again = rollout(sim, players, root_like, "p1", candidates[0]["slot"],
                                seed_triple(root_id, 0), keep_signature=False)
                rec["deterministic"] = sig0 == (again.sig_digest, again.winner, again.final_prng)
                unexpected += _unexpected(again.errors)
                rec.update(
                    supported=True, outcomes=outcomes, wall_s=walls,
                    protocol_retries=dict(retries), illegal=illegal + again.illegal_actions,
                    unexpected=unexpected,
                    root_inputs_equal_true=all(d == out["true_digest"] for d in digests),
                    root_inputs_equal_across_candidates=len({json.dumps(d, sort_keys=True)
                                                              for d in digests}) == 1,
                    world_digest=digests[0],
                )
            except Unsupported as e:
                rec.update(supported=False, reason=str(e))
            except Exception as e:  # noqa: BLE001
                rec.update(supported=False, reason="ERROR:" + repr(e)[:200],
                           trace=traceback.format_exc()[-1500:])
            finally:
                if tag:
                    with contextlib.suppress(Exception):
                        sim.destroy(tag)
            worlds.append(rec)
        out["worlds"] = worlds
    except Exception as e:  # noqa: BLE001
        out["error"] = repr(e)[:300]
        out["trace"] = traceback.format_exc()[-1500:]
    return out


_S_ = s1._S


def summarize(results: list[dict[str, Any]], wall: float) -> dict[str, Any]:
    ok = [r for r in results if "error" not in r]
    worlds = [(r, w) for r in ok for w in r["worlds"]]
    sup = [w for _, w in worlds if w.get("supported")]
    reasons = Counter(w["reason"][:90] for _, w in worlds if not w.get("supported"))
    rollouts = sum(len(row) for w in sup for row in w["outcomes"]) + len(sup)
    pov_bad = [(r["root"], w["pov_violations"]) for r, w in worlds
               if w.get("supported") and w["pov_violations"]]
    retries: Counter = Counter()
    for w in sup:
        retries.update(w["protocol_retries"])
    return {
        "roots": len(results), "root_errors": len(results) - len(ok),
        "worlds": len(worlds), "supported_worlds": len(sup),
        "unsupported_reasons": dict(reasons),
        "rollouts": rollouts,
        "deterministic_worlds": sum(w["deterministic"] for w in sup),
        "root_inputs_equal_true_worlds": sum(w["root_inputs_equal_true"] for w in sup),
        "root_inputs_equal_across_candidates": sum(
            w["root_inputs_equal_across_candidates"] for w in sup),
        "pov_violation_worlds": len(pov_bad), "pov_violation_examples": pov_bad[:10],
        "illegal_actions": sum(w["illegal"] for w in sup),
        "protocol_retries": dict(retries),
        "unexpected_errors": [e for w in sup for e in w["unexpected"]][:20],
        "unexpected_error_count": sum(len(w["unexpected"]) for w in sup),
        "mean_rollout_wall_s": float(
            sum(x for w in sup for row in w["wall_s"] for x in row)
            / max(sum(len(row) for w in sup for row in w["wall_s"]), 1)),
        "by_opponent": dict(Counter(r["opponent"] for r in ok)),
        "by_phase": dict(Counter(r["phase"] for r in ok)),
        "wall_s": wall,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", type=int, default=60)
    ap.add_argument("--worlds", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=S2_SEED)
    ap.add_argument("--out", default="runs/w5-bs0-2")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(i, ["GymLeader", "Grunt"][i % 2], args.worlds, args.seeds, args.seed)
            for i in range(args.roots)]
    started = time.perf_counter()
    results = []
    with mp.get_context("spawn").Pool(args.workers, initializer=s1.init) as pool:
        for r in pool.imap_unordered(s2_root, jobs):
            results.append(r)
            print(r["root"], r.get("phase"), [w.get("supported") for w in r.get("worlds", [])],
                  r.get("error", ""), flush=True)
    results.sort(key=lambda r: r["root"])
    (out / "roots.jsonl").write_text("\n".join(json.dumps(r) for r in results) + "\n")
    summary = summarize(results, time.perf_counter() - started)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
