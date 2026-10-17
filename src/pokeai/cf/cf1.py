"""CF1: belief-aware counterfactual labels, per-root processing (QA + rollouts).

Variants (same roots, candidates and (world, seed) pairs):
  belief_greedy  primary: sampled opponent worlds, learner C greedy after the root
  oracle_greedy  true opponent state, 16 distinct seeds, learner C greedy
  belief_t1      control: sampled worlds, learner C sampled T=1.0
Rollout k = 2*world + seed_index for the belief variants (8 worlds x 2 seeds); the oracle
uses the same 16 seed triples on the true world.
"""

from __future__ import annotations

import copy
import random
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from pokeai.cf.belief import (
    BeliefSampler,
    MonKnowledge,
    annotate_categories,
    check_public_facts,
    field_distribution,
    parse_knowledge,
    sample_mon,
    to_id,
    validator_set,
)
from pokeai.cf.branching import candidate_actions, reseed, rollout
from pokeai.cf.offline import OfflineMatch, new_tag, normalise_line
from pokeai.cf.restore_check import (
    TRAP_PREFIXES,
    signature_mechanics,
)
from pokeai.cf.rollouts import phase_bucket, pick_root
from pokeai.cf.sim import Sim, gen5_seed

OPPONENTS = ["C", "GymLeader", "Grunt"]
N_WORLDS, N_SEEDS = 8, 2
N_ROLLOUTS = N_WORLDS * N_SEEDS
VARIANTS = {
    "belief_greedy": {"belief": True, "learner_sample": False},
    "oracle_greedy": {"belief": False, "learner_sample": False},
    "belief_t1": {"belief": True, "learner_sample": True},
}


def seed_triple(root_id: int, k: int) -> dict[str, int]:
    base = 50_000_000 + root_id * 1000 + k * 3
    return {"sim": base, "opponent": base + 1, "learner": base + 2}


def choose_candidates(info: dict[str, Any], rng: random.Random) -> list[dict[str, Any]]:
    """Outcome-blind rule: C greedy, next-highest prob, uniform random other, best Tera."""
    mask, probs = info["mask"], info["probs"]
    legal = [i for i, ok in enumerate(mask) if ok]
    ranked = sorted(legal, key=lambda i: (-probs[i], i))
    out: list[dict[str, Any]] = []

    def add(slot: int, reason: str, inclusion: float | None) -> None:
        out.append({
            "slot": slot, "reason": reason, "inclusion_prob": inclusion,
            "prob": probs[slot],
        })

    add(ranked[0], "c_greedy", 1.0)
    if len(ranked) > 1:
        add(ranked[1], "next_highest_prob", 1.0)
    chosen = {c["slot"] for c in out}
    others = [i for i in legal if i not in chosen]
    if others:
        pick = rng.choice(others)
        add(pick, "uniform_random_other", 1 / len(others))
        chosen.add(pick)
    tera = [i for i in ranked if i >= 9 and i not in chosen]
    if tera:
        add(tera[0], "best_tera", 1.0)
    else:
        rest = [i for i in legal if i not in chosen]
        if rest and len(out) < 4:
            pick = rng.choice(rest)
            add(pick, "uniform_random_extra", 1 / len(rest))
    return out[:4]


@dataclass
class RootCtx:
    root_id: int
    opponent: str
    names: tuple[str, str]
    picked: dict[str, Any]
    info: dict[str, Any]
    candidates: list[dict[str, Any]]
    roster: list[dict[str, Any]]
    know: dict[str, MonKnowledge]
    worlds: list[dict[str, Any]]
    public_lines: int

    @property
    def root(self) -> Any:
        return self.picked["root"]

    @property
    def players(self) -> dict[str, Any]:
        return self.picked["players"]


def build_root(
    sim: Sim, sampler: BeliefSampler, pool: dict[str, tuple[str, str]], seed: int,
    root_id: int, n_worlds: int = N_WORLDS, greedy_prefix: bool = False,
) -> RootCtx:
    names = sorted(pool)
    opponent = OPPONENTS[root_id % 3]
    for attempt in range(20):
        index = root_id * 20 + attempt
        rng = random.Random(seed * 100 + index)
        name1, name2 = rng.choice(names), rng.choice(names)
        seeds = {"sim": seed + index, "p1": seed + 1000 + index, "p2": seed + 2000 + index}
        picked = pick_root(
            sim, {"p1": pool[name1][0], "p2": pool[name2][0]}, opponent, seeds, rng,
            greedy_prefix=greedy_prefix,
        )
        if picked is not None:
            break
    else:
        raise RuntimeError(f"root {root_id}: no eligible decision in 20 battles")
    root, players = picked["root"], picked["players"]
    info = candidate_actions(sim, players, root, "p1")
    if info is None or info["mask"] != picked["mask_pass1"]:
        raise RuntimeError("replayed root differs from pass 1 (determinism bug)")
    cands = choose_candidates(info, random.Random(seed * 7 + root_id))
    probe = OfflineMatch.restore(sim, players, root.checkpoint, new_tag())
    try:
        roster = sim.roster(probe.tag)
        lines = sim.public_log(probe.tag)
        know = parse_knowledge(lines, roster)
        annotate_categories(know, sim)
    finally:
        probe.close()
    worlds = [
        sampler.sample_world(
            roster, know, np.random.default_rng([seed, root_id, 17, w])
        )
        for w in range(n_worlds)
    ]
    return RootCtx(
        root_id=root_id, opponent=opponent, names=(name1, name2), picked=picked,
        info=info, candidates=cands, roster=roster, know=know, worlds=worlds,
        public_lines=len(lines),
    )


def _unexpected(errors: list[str]) -> list[str]:
    return [
        e for e in errors
        if not e.startswith((*TRAP_PREFIXES, "protocol-retry:"))
    ]


def run_variant(
    sim: Sim, ctx: RootCtx, variant: str, repeat_rng: random.Random | None,
    repeat_frac: float,
) -> dict[str, Any]:
    spec = VARIANTS[variant]
    outcomes, walls = [], []
    unexpected: list[str] = []
    retries: dict[str, int] = {}
    illegal = 0
    determinism = {"pairs": 0, "identical": 0}
    mech: dict[str, int] = {}
    for ci, cand in enumerate(ctx.candidates):
        row_o, row_w = [], []
        for k in range(N_ROLLOUTS):
            world = ctx.worlds[k // N_SEEDS] if spec["belief"] else None
            args = (sim, ctx.players, ctx.root, "p1", cand["slot"], seed_triple(ctx.root_id, k))
            kwargs = {
                "world": world, "learner_sample": spec["learner_sample"],
                "keep_signature": ci == 0 and k < 2,
            }
            result = rollout(*args, **kwargs)
            row_o.append(0.5 if result.winner is None or result.winner == "tie"
                         else 1.0 if result.winner == "p1" else 0.0)
            row_w.append(result.wall_s)
            unexpected += _unexpected(result.errors)
            for e in result.errors:
                if e.startswith("protocol-retry:"):
                    retries[e] = retries.get(e, 0) + 1
            illegal += result.illegal_actions
            if result.signature:
                for key, value in signature_mechanics(result.signature).items():
                    mech[key] = mech.get(key, 0) + value
            if repeat_rng is not None and repeat_rng.random() < repeat_frac:
                again = rollout(*args, **kwargs)
                determinism["pairs"] += 1
                same = (
                    again.sig_digest == result.sig_digest
                    and again.winner == result.winner
                    and again.final_prng == result.final_prng
                )
                determinism["identical"] += int(same)
                unexpected += _unexpected(again.errors)
                illegal += again.illegal_actions
        outcomes.append(row_o)
        walls.append(row_w)
    return {
        "outcomes": outcomes, "wall_s": walls, "unexpected_errors": unexpected,
        "illegal": illegal, "determinism": determinism, "mechanics": mech,
        "protocol_retries": retries,
    }


def _first_p2_subject(lines: list[str]) -> int | None:
    for i, line in enumerate(lines):
        parts = line.split("|")
        if len(parts) > 2 and parts[2].startswith("p2"):
            return i
        if "[of] p2" in line:
            return i
    return None


def qa_root(sim: Sim, ctx: RootCtx, sampler: BeliefSampler) -> dict[str, Any]:
    """G1 checks for one root against the true world."""
    players, root = ctx.players, ctx.root
    true = OfflineMatch.restore(sim, players, root.checkpoint, new_tag())
    true_full = sim.p2_full(true.tag)
    true_view = sim.public_view(true.tag)
    true_state = sim.state(true.tag)
    true_prng = sim.get_prng(true.tag)
    true_errors = sim.validate([m["set"] for m in true_full])
    true.close()
    out: dict[str, Any] = {
        "true_team_legal": not true_errors, "worlds": [], "violations": [],
        "diversity": {}, "hit_rates": {},
    }
    candidate = ctx.candidates[0]["slot"]
    seeds = seed_triple(ctx.root_id, 0)
    ability_legal = {m["name"]: m["legal_abilities"] for m in ctx.roster}

    def probe(world: dict[str, Any] | None) -> dict[str, Any]:
        match = OfflineMatch.restore(sim, players, root.checkpoint, new_tag())
        try:
            players["p1"].sample = False
            result: dict[str, Any] = {}
            if world is not None:
                match.apply_world(world)
                result["full"] = sim.p2_full(match.tag)
                result["view"] = sim.public_view(match.tag)
                state = sim.state(match.tag)
                result["p1_state"] = state["sides"][0]
                result["prng"] = sim.get_prng(match.tag)
            sim.set_prng(match.tag, gen5_seed(seeds["sim"]))
            reseed(match, "p2", seeds["opponent"])
            reseed(match, "p1", seeds["learner"])
            match.forced["p1"].append(candidate)
            first = match.step()
            second = match.step()
            model = next(
                (d["model"] for d in first.decisions["p1"] if d["model"]), None
            )
            result["root_frames_p1"] = [
                normalise_line(x) for f in first.delivered["p1"]
                for x in f.split("\n")[1:]
            ]
            result["root_model"] = (
                None if model is None else {
                    k: model[k] for k in
                    ("tokens_sha", "numbers_sha", "mask", "hidden_sha", "value")
                }
            )
            result["turn_lines"] = [
                normalise_line(x) for f in second.delivered["p1"] for x in f.split("\n")[1:]
            ]
            result["errors"] = _unexpected(first.errors + second.errors)
            return result
        finally:
            match.close()

    reference = probe(None)
    ref_lines_state = true_state["sides"][0]
    for w, world in enumerate(ctx.worlds):
        res = probe(world)
        team = [validator_set(m, s) for m, s in zip(ctx.roster, world["mons"], strict=True)]
        legal = not sim.validate(team)
        view_equal = res["view"] == true_view
        p1_equal = res["p1_state"] == ref_lines_state
        facts = check_public_facts(ctx.know, res["full"], ability_legal)
        c_ok = (
            res["root_frames_p1"] == reference["root_frames_p1"]
            and res["root_model"] == reference["root_model"]
        )
        a, b = reference["turn_lines"], res["turn_lines"]
        diff = next(
            (i for i in range(min(len(a), len(b))) if a[i] != b[i]),
            None if len(a) == len(b) else min(len(a), len(b)),
        )
        firsts = [x for x in (_first_p2_subject(a), _first_p2_subject(b)) if x is not None]
        gate = min(firsts) if firsts else None
        # a learner move that fails/redirects (Sucker Punch, Thunderclap, accuracy) depends
        # on the opponent's chosen action or hidden item/ability: a legitimate outcome
        same_move = (
            diff is not None and diff < min(len(a), len(b))
            and a[diff].startswith("|move|p1") and b[diff].startswith("|move|p1")
            and a[diff].split("|")[2:4] == b[diff].split("|")[2:4]
        )
        leak_free = diff is None or same_move or (gate is not None and diff >= gate)
        out["worlds"].append({
            "world": w, "legal": legal, "public_view_equal": view_equal,
            "learner_state_equal": p1_equal, "prng_unchanged": res["prng"] == true_prng,
            "facts_violations": facts, "root_inputs_identical": c_ok,
            "turn_first_diff": diff, "turn_first_p2_subject": gate,
            "turn_leak_free": leak_free, "errors": res["errors"],
            "turn_diff_lines": None if diff is None else {
                "true": a[diff] if diff < len(a) else None,
                "world": b[diff] if diff < len(b) else None},
        })
        if not (legal and view_equal and p1_equal and not facts and c_ok and leak_free):
            out["violations"].append(w)
    # negative control: a world that drops a revealed move must be flagged
    bad = copy.deepcopy(ctx.worlds[0])
    target = next((m for m in ctx.roster if ctx.know[m["name"]].moves), None)
    if target is not None:
        revealed = {to_id(x) for x in ctx.know[target["name"]].moves}
        mon = bad["mons"][target["index"]]
        mon["moves"] = [m for m in mon["moves"] if to_id(m) not in revealed]
        mon["moves"] += ["Splash", "Tackle", "Growl", "Harden"][: 4 - len(mon["moves"])]
        neg = OfflineMatch.restore(sim, players, root.checkpoint, new_tag())
        try:
            neg.apply_world(bad)
            found = check_public_facts(
                ctx.know, sim.p2_full(neg.tag), ability_legal
            )
        finally:
            neg.close()
        out["negative_control"] = {"mon": target["name"], "violations_detected": len(found)}
    # (e) diversity + hit rate vs the true hidden fields
    true_by_name = {m["name"]: m for m in true_full}
    prior = sampler.prior
    stats = {"item": [0, 0, 0.0], "ability": [0, 0, 0.0], "tera": [0, 0, 0.0],
             "moves": [0, 0, 0.0]}
    for mon in ctx.roster:
        k = ctx.know[mon["name"]]
        sampled = [wd["mons"][mon["index"]] for wd in ctx.worlds]
        out["diversity"][mon["species"]] = {
            "distinct_movesets": len({tuple(sorted(s["moves"])) for s in sampled}),
            "distinct_items": len({s["item"] for s in sampled}),
            "distinct_abilities": len({s["ability"] for s in sampled}),
            "distinct_tera": len({s["tera_type"] for s in sampled}),
            "frozen": {"item": k.item_state != "unknown", "ability": k.ability is not None
                       or k.freeze_ability, "tera": k.tera is not None,
                       "moves_revealed": len(k.moves)},
        }
        truth = true_by_name[mon["name"]]
        if k.item_state == "unknown":
            dist = field_distribution(prior, mon, k, "item")
            stats["item"][0] += sum(s["item"] == _item_name(truth) for s in sampled)
            stats["item"][1] += len(sampled)
            stats["item"][2] += len(sampled) * dist.get(_item_name(truth), 0.0)
        if not (k.ability or k.freeze_ability):
            dist = field_distribution(prior, mon, k, "ability")
            true_ability = _ability_name(mon, truth)
            stats["ability"][0] += sum(s["ability"] == true_ability for s in sampled)
            stats["ability"][1] += len(sampled)
            stats["ability"][2] += len(sampled) * dist.get(true_ability, 0.0)
        if k.tera is None:
            dist = field_distribution(prior, mon, k, "tera_types")
            stats["tera"][0] += sum(s["tera_type"] == truth["tera_type"] for s in sampled)
            stats["tera"][1] += len(sampled)
            stats["tera"][2] += len(sampled) * dist.get(truth["tera_type"], 0.0)
        hidden = [m for m in truth["moves"]
                  if m not in {to_id(x) for x in k.moves}]
        if hidden:
            rng = np.random.default_rng([ctx.root_id, mon["index"], 99])
            mc = [sample_mon(prior, mon, k, rng) for _ in range(300)]
            for move in hidden:
                stats["moves"][0] += sum(
                    move in {_mid(x) for x in s["moves"]} for s in sampled
                )
                stats["moves"][1] += len(sampled)
                stats["moves"][2] += len(sampled) * float(np.mean([
                    move in {_mid(x) for x in s["moves"]} for s in mc
                ]))
    out["hit_rates"] = {
        key: {"hits": v[0], "trials": v[1], "expected_hits_under_prior": v[2]}
        for key, v in stats.items()
    }
    return out


def _mid(name: str) -> str:
    key = to_id(name)
    return "hiddenpower" if key.startswith("hiddenpower") else key


def _item_name(truth: dict[str, Any]) -> str:
    """True item as the prior names it (set.item is the display name)."""
    return truth["set"]["item"]


def _ability_name(mon: dict[str, Any], truth: dict[str, Any]) -> str:
    for ability in mon["legal_abilities"]:
        if to_id(ability) == truth["base_ability"]:
            return ability
    return truth["set"]["ability"]


def process_root(
    sim: Sim, sampler: BeliefSampler, pool: dict[str, tuple[str, str]], seed: int,
    root_id: int, variants: list[str], qa: bool, repeat_frac: float,
    greedy_prefix: bool = False, n_worlds: int = N_WORLDS,
) -> dict[str, Any]:
    started = time.perf_counter()
    ctx = build_root(
        sim, sampler, pool, seed, root_id, n_worlds=n_worlds, greedy_prefix=greedy_prefix
    )
    result: dict[str, Any] = {
        "root": root_id, "opponent": ctx.opponent, "p1_team": ctx.names[0],
        "p2_team": ctx.names[1], "pair": f"{ctx.names[0]}|{ctx.names[1]}",
        "turn": ctx.picked["turn"], "phase": phase_bucket(ctx.picked["turn"]),
        "forced_switch": ctx.picked["forced_switch"],
        "eligible_decisions": ctx.picked["eligible_decisions"],
        "mask": ctx.info["mask"], "policy_probs": ctx.info["probs"],
        "probe_value": ctx.info["value"],
        "a_c": ctx.candidates[0]["slot"], "candidates": ctx.candidates,
        "public_log_lines": ctx.public_lines, "variants": {},
        "greedy_prefix": greedy_prefix, "features": ctx.info["features"],
        "pass1_unexpected_errors": _unexpected(ctx.picked["pass1_errors"]),
    }
    if qa:
        result["qa"] = qa_root(sim, ctx, sampler)
    repeat_rng = random.Random(seed * 31 + root_id)
    for variant in variants:
        result["variants"][variant] = run_variant(
            sim, ctx, variant, repeat_rng, repeat_frac if variant == "belief_greedy" else 0.0
        )
    sim.drop_snapshot(ctx.root.checkpoint.sim_key)
    result["wall_total_s"] = time.perf_counter() - started
    return result
