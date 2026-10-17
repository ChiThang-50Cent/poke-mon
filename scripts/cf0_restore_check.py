"""CF0.1: offline-harness snapshot/restore determinism gate -> reports/w5_cf0_1_restore.json."""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter

import pokeai  # noqa: F401  (import order: metamon.baselines before GenData)
from pokeai.cf.offline import (
    FRAMING_NOTES,
    SIDES,
    PlayerSpec,
    make_player,
    new_tag,
)
from pokeai.cf.restore_check import (
    C_POLICY,
    PREVIEW,
    TRAP_PREFIXES,
    WINDOW,
    CheckpointRule,
    load_pool,
    match_mechanics,
    negative_control,
    play_with_checkpoints,
    verify_checkpoints,
    write_json,
)
from pokeai.cf.sim import Sim

# (p1 team kind, p2 team kind); rb = Revival Blessing, cram = Cramorant, any = pool draw
SCHEDULE = [
    ("cram", "rb"), ("rb", "cram"), ("cram", "any"), ("rb", "rb"), ("any", "any"),
    ("cram", "any"), ("rb", "any"), ("rb", "cram"), ("cram", "rb"), ("any", "any"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teams", default="data/teams_w5/collect/gen9ou")
    parser.add_argument("--battles", type=int, default=10)
    parser.add_argument("--cap", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument("--out", default="reports/w5_cf0_1_restore.json")
    args = parser.parse_args()

    pool = load_pool(args.teams)
    names = sorted(pool)
    kinds = {
        "cram": [n for n in names if "cramorant" in pool[n][1].lower()],
        "rb": [n for n in names if "revival blessing" in pool[n][1].lower()],
        "any": names,
    }
    rng = random.Random(args.seed)
    started = time.perf_counter()
    cap = [args.cap]
    battles, rows = [], []
    negative: dict[str, dict] = {}
    mech_total: Counter = Counter()
    illegal = unknown = tokens = 0
    unexpected_errors: list[str] = []
    expected_retries = 0
    fallbacks: Counter = Counter()
    with Sim() as sim:
        for index in range(args.battles):
            kind1, kind2 = SCHEDULE[index % len(SCHEDULE)]
            name1, name2 = rng.choice(kinds[kind1]), rng.choice(kinds[kind2])
            opponent = C_POLICY if index < args.battles // 2 else "GymLeader"
            specs = {
                "p1": PlayerSpec(C_POLICY, sample=True, temperature=1.0, preview=PREVIEW),
                "p2": PlayerSpec(
                    opponent, sample=True, temperature=1.0,
                    preview=PREVIEW if opponent == C_POLICY else None,
                ),
            }
            players = {s: make_player(specs[s], s) for s in SIDES}
            tag = new_tag()
            rule = CheckpointRule(
                seed=args.seed * 1000 + index, cap_remaining=cap, battle_index=index
            )
            began = time.perf_counter()
            match, taken = play_with_checkpoints(
                sim, players, tag, {"p1": pool[name1][0], "p2": pool[name2][0]},
                sim_seed=args.seed + index,
                side_seeds={"p1": args.seed + 100 + index, "p2": args.seed + 200 + index},
                rule=rule,
            )
            duration = time.perf_counter() - began
            mech = match_mechanics(match.rounds)
            mech_total.update(mech)
            errors = [e for r in match.rounds for e in r.errors]
            retries = [e for e in errors if e.startswith(TRAP_PREFIXES)]
            bad = [e for e in errors if not e.startswith(TRAP_PREFIXES)]
            expected_retries += len(retries)
            unexpected_errors += bad
            for side in SIDES:
                player = players[side]
                if hasattr(player, "neural_stats"):
                    stats = player.neural_stats()
                    illegal += stats["illegal_actions"]
                    unknown += stats["unknown_tokens"]
                    tokens += stats["total_tokens"]
                    fallbacks.update(stats["fallbacks"])
            row = {
                "battle": tag, "p1_team": name1, "p2_team": name2,
                "p2_policy": opponent, "winner": match.winner(), "turns": match.turn,
                "rounds": match.round_index, "duration_s": round(duration, 3),
                "checkpoints": [(t.round_index, t.turn, t.reasons) for t in taken],
                "mechanics": dict(mech), "error_lines": len(errors),
                "expected_trap_retries": len(retries),
                "unexpected_error_lines": bad,
            }
            battles.append(row)
            if taken and not negative:
                # Negative controls (first checkpoint of the first battle with one):
                # deliberately broken restores must be flagged by the same comparison.
                for sabotage in ("hidden", "prng"):
                    negative[sabotage] = negative_control(
                        sim, players, match, taken[-1], sabotage
                    )
            rows.extend(verify_checkpoints(sim, players, match, taken))
            for cp in taken:
                sim.drop_snapshot(cp.checkpoint.sim_key)
            match.close()
            print(
                f"{tag}: winner={row['winner']} turns={row['turns']} "
                f"checkpoints={len(taken)} mech={dict(mech)}", flush=True,
            )

    totals: Counter = Counter()
    for row in rows:
        for key, value in (
            ("channel_mismatch", sum(
                sum(m.get(k, 0) for k in ("channel_p1", "channel_p2", "channel_omni", "events_out"))
                for m in row["mismatch_vs_original"]
            )),
            ("sim_state_mismatch", sum(m.get("sim_state", 0) for m in row["mismatch_vs_original"])),
            ("observation_mismatch", sum(
                m.get("observation_tokens", 0) + m.get("observation_numbers", 0)
                + m.get("legality_mask", 0) for m in row["mismatch_vs_original"]
            )),
            ("hidden_mismatch", sum(m.get("hidden", 0) for m in row["mismatch_vs_original"])),
            ("choice_mismatch", sum(
                m.get("choices", 0) + m.get("model_action", 0)
                for m in row["mismatch_vs_original"]
            )),
            ("other_mismatch", sum(
                m.get("round_count", 0) for m in row["mismatch_vs_original"]
            )),
            ("restore_twice_mismatch", sum(row["mismatch_restore1_vs_restore2"].values())),
            ("immediate_state_not_equal", sum(
                not x for x in row["restore_immediate_state_equal"]
            )),
            ("script_mismatches", sum(row["script_mismatches"])),
            ("model_decisions_compared", row["model_decisions_compared"]),
            ("restore_error_lines", row["error_lines"]),
        ):
            totals[key] += value
    restore_unexpected = [
        e for row in rows for e in row["error_lines_detail"]
        if not e.startswith(TRAP_PREFIXES)
    ]
    gate = {
        "checkpoints": len(rows),
        "checkpoints_in_20_50": 20 <= len(rows) <= 50,
        "channel_outputs_identical": totals["channel_mismatch"] == 0,
        "sim_state_identical": totals["sim_state_mismatch"] == 0
        and totals["immediate_state_not_equal"] == 0,
        "observation_identical": totals["observation_mismatch"] == 0,
        "hidden_identical": totals["hidden_mismatch"] == 0,
        "choices_recomputed_equal_replayed": totals["script_mismatches"] == 0
        and totals["choice_mismatch"] == 0,
        "restore_twice_identical": totals["restore_twice_mismatch"] == 0
        and totals["other_mismatch"] == 0,
        "zero_unexpected_error_lines": not unexpected_errors
        and not restore_unexpected,
        "negative_controls_detected": bool(negative)
        and all(sum(v.values()) > 0 for v in negative.values()),
        "zero_model_illegal_actions": illegal == 0,
        "all_battles_completed_with_winner": all(b["winner"] for b in battles),
    }
    gate["PASS"] = all(v for k, v in gate.items() if isinstance(v, bool))
    end_to_end = {}
    for b in battles:
        group = end_to_end.setdefault(
            "C_vs_C" if b["p2_policy"] == C_POLICY else f"C_vs_{b['p2_policy']}",
            {"battles": 0, "with_winner": 0, "error_lines": 0, "turns": []},
        )
        group["battles"] += 1
        group["with_winner"] += int(bool(b["winner"]))
        group["error_lines"] += b["error_lines"]
        group["turns"].append(b["turns"])
    report = {
        "phase": "CF0.1",
        "end_to_end": end_to_end,
        "command": "scripts/cf0_restore_check.py",
        "seed": args.seed,
        "battles": battles,
        "selection_rule": (
            "Per battle a seeded RNG (seed*1000+battle index) draws early (turn 2-5), "
            "mid (6-14) and late (15-25) targets; snapshot the first decision round "
            "whose upcoming turn >= target. Also the first forced-switch round and up "
            "to 2 Revival-Blessing 'reviving' rounds (exempt from the per-battle "
            "budget of 5). One extra slot per battle: even battle index = first round "
            "with the learner's Cramorant active and Surf/Dive legal; odd index = first "
            "round (turn>=2) where p1 can Terastallize, in which the ORIGINAL run is "
            "forced to Terastallize (stress). "
            f"Global cap {args.cap}. Decision round = a pending non-wait, "
            f"non-teampreview request exists. Window = {WINDOW} rounds."
        ),
        "normalisations": [
            "battle id (frame header '>battle-gen9ou-N' -> '>battle-gen9ou-<TAG>')",
            (
                "|t:|<unix seconds> protocol lines -> '|t:|' (frames, omniscient "
                "channel, sim toJSON log)"
            ),
            "end-event `seed` field is dropped from the end payload comparison",
            "rqid is NOT normalised (the room-wide counter is part of the checkpoint)",
        ],
        "gate_totals": dict(totals),
        "gate": gate,
        "mechanics_exercised_all_battles": dict(mech_total),
        "mechanics_in_verified_windows": dict(sum(
            (Counter(r["mechanics"]) for r in rows), Counter()
        )),
        "neural_fallbacks": dict(fallbacks),
        "expected_trap_retry_error_lines": expected_retries,
        "unexpected_error_lines": unexpected_errors,
        "unexpected_error_lines_in_restored_windows": restore_unexpected,
        "negative_controls": negative,
        "model_illegal_actions": illegal,
        "unknown_token_rate_C": unknown / tokens if tokens else 0.0,
        "unknown_tokens": unknown, "total_tokens": tokens,
        "framing_differences_vs_real_server": FRAMING_NOTES,
        "checkpoint_rows": rows,
        "wall_time_s": round(time.perf_counter() - started, 1),
    }
    write_json(args.out, report)
    print(json.dumps({"gate": gate, "totals": dict(totals)}, indent=2))
    if not gate["PASS"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
