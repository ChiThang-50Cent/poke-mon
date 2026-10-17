"""CF0.2: branching smoke over offline battles -> reports/w5_cf0_2_branching.json."""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from collections import Counter
from itertools import combinations

import pokeai  # noqa: F401  (import order: metamon.baselines before GenData)
from pokeai.cf.branching import candidate_actions, rollout, root_consistency
from pokeai.cf.offline import FRAMING_NOTES, SIDES, PlayerSpec, make_player, new_tag
from pokeai.cf.restore_check import (
    C_POLICY,
    PREVIEW,
    TRAP_PREFIXES,
    CheckpointRule,
    load_pool,
    play_with_checkpoints,
    signature_mechanics,
    write_json,
)
from pokeai.cf.sim import Sim

# (opponent policy, p1 team kind, p2 team kind)
BATTLES = [
    (C_POLICY, "cram", "rb"), (C_POLICY, "rb", "any"),
    ("GymLeader", "cram", "rb"), ("GymLeader", "any", "cram"),
]
PRIORITY = ("forced", "mid", "late", "early")


def root_message(rollout_result, side: str) -> str | None:
    """The side's order at the root, None if it only had a wait request."""
    decisions = rollout_result.root_decisions[side]
    return decisions[0]["message"] if decisions else None


def root_model(rollout_result, side: str) -> str:
    decisions = rollout_result.root_decisions[side]
    return json.dumps(decisions[0]["model"] if decisions else None, sort_keys=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teams", default="data/teams_w5/collect/gen9ou")
    parser.add_argument("--seed", type=int, default=20261018)
    parser.add_argument("--roots-per-battle", type=int, default=2)
    parser.add_argument("--rollouts", type=int, default=2)
    parser.add_argument("--out", default="reports/w5_cf0_2_branching.json")
    args = parser.parse_args()

    pool = load_pool(args.teams)
    names = sorted(pool)
    kinds = {
        "cram": [n for n in names if "cramorant" in pool[n][1].lower()],
        "rb": [n for n in names if "revival blessing" in pool[n][1].lower()],
        "any": names,
    }
    rng = random.Random(args.seed)
    learner = "p1"
    started = time.perf_counter()
    roots_report, all_rollouts = [], []
    criteria = Counter()
    walls, rounds_per, errors, illegal = [], [], [], 0
    mech: Counter = Counter()
    with Sim() as sim:
        for b_index, (opponent, kind1, kind2) in enumerate(BATTLES):
            name1, name2 = rng.choice(kinds[kind1]), rng.choice(kinds[kind2])
            specs = {
                "p1": PlayerSpec(C_POLICY, sample=True, preview=PREVIEW),
                "p2": PlayerSpec(
                    opponent, sample=True,
                    preview=PREVIEW if opponent == C_POLICY else None,
                ),
            }
            players = {s: make_player(specs[s], s) for s in SIDES}
            rule = CheckpointRule(
                seed=args.seed * 1000 + b_index, cap_remaining=[99], battle_index=b_index
            )
            match, taken = play_with_checkpoints(
                sim, players, new_tag(), {"p1": pool[name1][0], "p2": pool[name2][0]},
                sim_seed=args.seed + b_index,
                side_seeds={"p1": args.seed + 100 + b_index, "p2": args.seed + 200 + b_index},
                rule=rule,
            )
            chosen = 0
            ordered = sorted(
                taken,
                key=lambda t: min(
                    (PRIORITY.index(r) for r in t.reasons if r in PRIORITY), default=9
                ),
            )
            for root in ordered:
                if chosen >= args.roots_per_battle:
                    break
                cands = candidate_actions(sim, players, root, learner)
                if cands is None:
                    continue
                chosen += 1
                root_id = len(roots_report)
                per_k: dict[int, dict[int, list]] = {}
                for k in range(args.rollouts):
                    base = 7_000_000 + 1000 * root_id + 10 * k
                    seeds = {"sim": base + 1, "opponent": base + 2, "learner": base + 3}
                    per_k[k] = {}
                    for cand in cands["candidates"]:
                        first = rollout(sim, players, root, learner, cand, seeds)
                        again = rollout(sim, players, root, learner, cand, seeds)
                        same = (
                            first.signature == again.signature
                            and first.winner == again.winner
                            and first.final_prng == again.final_prng
                        )
                        criteria["a_pairs"] += 1
                        criteria["a_identical"] += int(same)
                        for r in (first, again):
                            walls.append(r.wall_s)
                            rounds_per.append(r.rounds)
                            errors.extend(
                                e for e in r.errors if not e.startswith(TRAP_PREFIXES)
                            )
                            illegal += r.illegal_actions
                        per_k[k][cand] = [first, again]
                        all_rollouts.append({
                            "root": root_id, "k": k, "candidate": cand,
                            "winner": first.winner, "rounds": first.rounds,
                            "turns": first.turns, "wall_s": round(first.wall_s, 3),
                            "deterministic_repeat": same,
                            "learner_root_message": root_message(first, learner),
                            "opponent_root_message": root_message(first, "p2"),
                        })
                        if "terastallize" in (root_message(first, learner) or ""):
                            mech["tera_candidate_rollouts"] += 1
                        mech["rollouts"] += 1
                        mech.update(signature_mechanics(first.signature))
                # (b)/(c) across candidates for each rollout seed k
                pair_rows = []
                for k, by_cand in per_k.items():
                    opp_msgs = {c: root_message(v[0], "p2") for c, v in by_cand.items()}
                    opp_models = [root_model(v[0], "p2") for v in by_cand.values()]
                    criteria["c_groups"] += 1
                    criteria["c_identical"] += int(
                        len(set(opp_msgs.values())) == 1 and len(set(opp_models)) == 1
                    )
                    for c1, c2 in combinations(by_cand, 2):
                        info = root_consistency(
                            by_cand[c1][0].signature, by_cand[c2][0].signature, learner
                        )
                        ok = (
                            info["root_inputs_identical"]
                            and info["opponent_root_decision_identical"]
                            and info["learner_root_observation_identical"]
                            and info["learner_root_choice_differs"]
                        )
                        criteria["b_pairs"] += 1
                        criteria["b_ok"] += int(ok)
                        pair_rows.append({"k": k, "candidates": [c1, c2], **info})
                roots_report.append({
                    "root": root_id, "battle": b_index, "opponent": opponent,
                    "p1_team": name1, "p2_team": name2,
                    "round": root.round_index, "turn": root.turn,
                    "reasons": root.reasons,
                    "mask": cands["mask"],
                    "policy_probs": [round(p, 4) for p in cands["probs"]],
                    "candidates": cands["candidates"],
                    "opponent_root_messages": {
                        str(k): {str(c): root_message(v[0], "p2") for c, v in by.items()}
                        for k, by in per_k.items()
                    },
                    "divergence_pairs": pair_rows,
                })
            for cp in taken:
                sim.drop_snapshot(cp.checkpoint.sim_key)
            match.close()
            print(f"battle {b_index}: roots so far {len(roots_report)}", flush=True)

    sorted_walls = sorted(walls)
    timing = {
        "rollouts_timed": len(walls),
        "mean_s": statistics.mean(walls), "median_s": statistics.median(walls),
        "p90_s": sorted_walls[int(0.9 * (len(walls) - 1))], "max_s": max(walls),
        "mean_rounds": statistics.mean(rounds_per),
        "mean_ms_per_round": 1000 * sum(walls) / sum(rounds_per),
        "note": "single process, CPU, both policies in-process, includes restore",
    }
    gate = {
        "roots": len(roots_report),
        "roots_in_5_10": 5 <= len(roots_report) <= 10,
        "a_same_candidate_same_seed_identical": criteria["a_identical"] == criteria["a_pairs"],
        "b_diverge_only_after_root": criteria["b_ok"] == criteria["b_pairs"],
        "c_opponent_root_action_identical": criteria["c_identical"] == criteria["c_groups"],
        "d_zero_unexpected_errors": not errors,
        "d_zero_illegal_actions": illegal == 0,
    }
    gate["PASS"] = all(v for k, v in gate.items() if isinstance(v, bool))
    report = {
        "phase": "CF0.2", "command": "scripts/cf0_branch_smoke.py", "seed": args.seed,
        "criteria_counts": dict(criteria), "gate": gate,
        "unexpected_error_lines": errors, "illegal_actions": illegal,
        "timing_e": timing, "mechanics": dict(mech),
        "candidate_rule": (
            "top-2 legal slots by C's masked policy probability + the most probable legal "
            "move+Tera slot (9-12) if not already included; learner = p1 = C "
            "(sampled T=1.0); seeds per (root, k): sim PRNG, opponent (python random + "
            "torch sampler) and learner streams are independent of the candidate"
        ),
        "root_rule": (
            "per battle: CheckpointRule decision rounds sorted by priority "
            "forced>mid>late>early; the first viable roots (learner has a represented "
            "model decision with >=2 legal slots)"
        ),
        "simultaneity": (
            "Both players decide inside their own frame delivery before any order is "
            "submitted to the sim; the learner's forced candidate replaces its sampled "
            "action after its forward pass; the opponent's order is never visible to "
            "the learner before commitment."
        ),
        "framing_differences_vs_real_server": FRAMING_NOTES,
        "roots": roots_report,
        "rollouts": all_rollouts,
        "wall_time_s": round(time.perf_counter() - started, 1),
    }
    write_json(args.out, report)
    print(json.dumps({"gate": gate, "timing": timing}, indent=2))
    if not gate["PASS"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
