"""ActionDescriptorEncoder: public semantic features of each legal action (CF1.5c).

One code path for the offline harness and the live `NeuralPlayer`: the input is ONLY the
learner's own poke-env/Metamon battle object (which includes its current request) plus
public Dex data. No sim state, no `p2_full`, no sampled world, no opponent hidden field.
The opponent contributes only its public typing (species types, or the revealed Tera type)
and public HP fraction. logπ_C(a) is NOT part of this vector (it is not known before the
policy runs); the head concatenates it from C's own output.

Layout of one descriptor row (all float32, see `FEATURE_NAMES`):
  common      weather(6) terrain(5) own hazards(4: SR, spikes/3, tspikes/2, web)
              opp hazards(4) own active hp, opp active hp  [kind(3) is the first block]
  move        type(18) category(3) bp/150 dynamic_bp accuracy priority/3 pp_frac
              flags: recovery setup hazard_set hazard_removal pivot status_inflict protect
              priority_move; stab; eff_vs_opp (log2 mult/2, 0 for status moves);
              eff_valid; disabled; maybe_disabled
  switch      incoming type multi-hot(18) hp status(6) hazard_damage_frac
              def_eff of the opp active's public types vs incoming (log2 max mult /2)
              off_eff best known damaging move vs opp active; n_known_moves/4
  tera        own tera type(18) stab_after_tera stab_mult_before stab_mult_after
              d_def_eff (opp STAB types vs tera type minus vs current types)
Normalisation is fixed (no fitted statistics): log2 multipliers clipped to [-2, 2] / 2,
base power /150 (clipped 2), priority /3, fractions in [0, 1].
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from poke_env.data import GenData
from poke_env.environment import (
    MoveCategory,
    PokemonType,
    SideCondition,
    Status,
    Weather,
)
from poke_env.environment.field import Field

TYPES = [t for t in PokemonType if t.name not in ("THREE_QUESTION_MARKS", "STELLAR")]
WEATHERS = [
    ("sun", {Weather.SUNNYDAY, Weather.DESOLATELAND}),
    ("rain", {Weather.RAINDANCE, Weather.PRIMORDIALSEA}),
    ("sand", {Weather.SANDSTORM}),
    ("snow", {Weather.SNOW, Weather.SNOWSCAPE, Weather.HAIL}),
    ("other", {Weather.DELTASTREAM}),
]
TERRAINS = [Field.ELECTRIC_TERRAIN, Field.GRASSY_TERRAIN, Field.MISTY_TERRAIN,
            Field.PSYCHIC_TERRAIN]
STATUSES = [Status.BRN, Status.FRZ, Status.PAR, Status.PSN, Status.SLP, Status.TOX]
HAZARDS = [SideCondition.STEALTH_ROCK, SideCondition.SPIKES, SideCondition.TOXIC_SPIKES,
           SideCondition.STICKY_WEB]
REMOVAL = {"rapidspin", "defog", "courtchange", "mortalspin", "tidyup"}
DYNAMIC_BP = {
    "gyroball", "electroball", "lowkick", "grassknot", "heavyslam", "heatcrash", "reversal",
    "flail", "return", "frustration", "acrobatics", "facade", "hex", "venoshock", "brine",
    "knockoff", "storedpower", "powertrip", "punishment", "crushgrip", "wringout",
    "eruption", "waterspout", "dragonenergy", "present", "magnitude", "naturalgift",
    "fling", "spitup", "trumpcard", "beatup", "avalanche", "revenge", "assurance",
    "payback", "boltbeak", "fishiousrend", "ragefist", "lastrespects", "terrainpulse",
    "weatherball", "solarbeam", "risingvoltage", "expandingforce", "psyblade",
}
MOVE_TYPES = [t.name for t in TYPES]


def _names() -> list[str]:
    n = ["kind_move", "kind_switch", "kind_tera"]
    n += [f"weather_{w}" for w, _ in WEATHERS] + ["weather_none"]
    n += [f"terrain_{t.name.lower()}" for t in TERRAINS] + ["terrain_none"]
    n += [f"own_hz_{h.name.lower()}" for h in HAZARDS]
    n += [f"opp_hz_{h.name.lower()}" for h in HAZARDS]
    n += ["own_active_hp", "opp_active_hp"]
    n += [f"move_type_{t.lower()}" for t in MOVE_TYPES]
    n += ["cat_physical", "cat_special", "cat_status", "bp", "dynamic_bp", "accuracy",
          "priority", "pp_frac", "recovery", "setup", "hazard_set", "hazard_removal",
          "pivot", "status_inflict", "protect", "priority_move", "stab", "eff", "eff_valid",
          "disabled", "maybe_disabled"]
    n += [f"sw_type_{t.lower()}" for t in MOVE_TYPES]
    n += ["sw_hp"] + [f"sw_status_{s.name.lower()}" for s in STATUSES]
    n += ["sw_status_none", "sw_hazard_dmg", "sw_def_eff", "sw_off_eff", "sw_known_moves"]
    n += [f"tera_type_{t.lower()}" for t in MOVE_TYPES]
    n += ["tera_stab_after", "tera_stab_mult_before", "tera_stab_mult_after", "tera_d_def_eff"]
    return n


FEATURE_NAMES = _names()
DIM = len(FEATURE_NAMES)
_IDX = {n: i for i, n in enumerate(FEATURE_NAMES)}


@dataclass
class Descriptor:
    """Rows for the 13 slots; rows of illegal slots stay all-zero."""

    x: np.ndarray  # [13, DIM] float32
    ids: list[tuple[str, str] | None]  # per slot: (kind, move/species name) for F2 lookups
    reasons: dict[int, list[str]] = field(default_factory=dict)  # slot -> missing reasons


def _chart() -> dict:
    return GenData.from_gen(9).type_chart


def _log_eff(attack: PokemonType, defender: list[PokemonType]) -> float:
    t1 = defender[0]
    t2 = defender[1] if len(defender) > 1 else None
    mult = attack.damage_multiplier(t1, t2, type_chart=_chart())
    if mult <= 0:
        return -1.0  # immune: the lowest bucket (same clip as 1/4 damage)
    return max(-2.0, min(2.0, math.log2(mult))) / 2.0


def public_types(mon: Any) -> list[PokemonType]:
    """Public typing: revealed Tera type once terastallized, else species/battle types."""
    if mon.is_terastallized and mon.tera_type is not None:
        return [mon.tera_type]
    return list(mon.types)


def _onehot_types(types: list[PokemonType], out: np.ndarray, base: int) -> None:
    for t in types:
        if t in TYPES:
            out[base + TYPES.index(t)] = 1.0


def _common(battle: Any, row: np.ndarray, reasons: list[str]) -> None:
    weather = set(battle.weather)
    hit = False
    for i, (_, group) in enumerate(WEATHERS):
        if weather & group:
            row[_IDX["weather_sun"] + i] = 1.0
            hit = True
    row[_IDX["weather_none"]] = 0.0 if hit else 1.0
    fields = set(battle.fields)
    on = False
    for t in TERRAINS:
        if t in fields:
            row[_IDX[f"terrain_{t.name.lower()}"]] = 1.0
            on = True
    row[_IDX["terrain_none"]] = 0.0 if on else 1.0
    for prefix, conds in (("own", battle.side_conditions), ("opp", battle.opponent_side_conditions)):
        scale = {SideCondition.SPIKES: 3.0, SideCondition.TOXIC_SPIKES: 2.0}
        for h in HAZARDS:
            layers = float(conds.get(h, 0))
            row[_IDX[f"{prefix}_hz_{h.name.lower()}"]] = min(1.0, layers / scale.get(h, 1.0))
    active, opp = battle.active_pokemon, battle.opponent_active_pokemon
    if active is None:
        reasons.append("own_active_missing")
    else:
        row[_IDX["own_active_hp"]] = float(active.current_hp_fraction)
    if opp is None:
        reasons.append("opp_active_missing")
    else:
        row[_IDX["opp_active_hp"]] = float(opp.current_hp_fraction)


def _request_moves(battle: Any) -> tuple[dict[str, bool], bool]:
    """Per move id: disabled flag, and the active-level maybeDisabled flag."""
    request = getattr(battle, "last_request", None) or getattr(battle, "_last_request", None)
    disabled: dict[str, bool] = {}
    maybe = False
    try:
        active = request["active"][0]
        maybe = bool(active.get("maybeDisabled", False))
        for m in active["moves"]:
            disabled[m["id"]] = bool(m.get("disabled", False))
    except (KeyError, IndexError, TypeError):
        pass
    return disabled, maybe


def _move_features(
    move: Any, active: Any, opp: Any, row: np.ndarray, reasons: list[str],
    disabled: dict[str, bool], maybe: bool, tera: PokemonType | None,
) -> None:
    mtype = move.type
    if mtype in TYPES:
        row[_IDX["move_type_" + mtype.name.lower()]] = 1.0
    else:
        reasons.append("move_type_unknown")
    cat = move.category
    row[_IDX["cat_physical"]] = float(cat == MoveCategory.PHYSICAL)
    row[_IDX["cat_special"]] = float(cat == MoveCategory.SPECIAL)
    row[_IDX["cat_status"]] = float(cat == MoveCategory.STATUS)
    bp = float(move.base_power)
    row[_IDX["bp"]] = min(2.0, bp / 150.0)
    dynamic = cat != MoveCategory.STATUS and (
        bp <= 1 or move.id in DYNAMIC_BP or move.entry.get("damage") is not None
    )
    row[_IDX["dynamic_bp"]] = float(dynamic)
    row[_IDX["accuracy"]] = float(move.accuracy)
    row[_IDX["priority"]] = move.priority / 3.0
    max_pp = move.max_pp
    if max_pp:
        row[_IDX["pp_frac"]] = float(move.current_pp) / float(max_pp)
    else:
        reasons.append("pp_missing")
    boosts = move.self_boost or (move.boosts if move.target and str(move.target.name) == "SELF" else None)
    row[_IDX["recovery"]] = float(move.heal > 0 or move.drain > 0)
    row[_IDX["setup"]] = float(bool(boosts) and any(v > 0 for v in boosts.values()))
    row[_IDX["hazard_set"]] = float(move.side_condition in HAZARDS)
    row[_IDX["hazard_removal"]] = float(move.id in REMOVAL)
    row[_IDX["pivot"]] = float(bool(move.self_switch))
    row[_IDX["status_inflict"]] = float(move.status is not None or move.volatile_status is not None)
    row[_IDX["protect"]] = float(move.is_protect_move)
    row[_IDX["priority_move"]] = float(move.priority > 0)
    if active is not None:
        types = list(active.types)
        stab_types = [tera] if tera is not None else types
        base_stab = mtype in types
        row[_IDX["stab"]] = float(mtype in stab_types or base_stab)
    if opp is None or cat == MoveCategory.STATUS:
        row[_IDX["eff_valid"]] = float(cat == MoveCategory.STATUS and opp is not None)
        if opp is None:
            reasons.append("opp_active_missing_eff")
    else:
        row[_IDX["eff"]] = _log_eff(mtype, public_types(opp))
        row[_IDX["eff_valid"]] = 1.0
    if move.id in disabled:
        row[_IDX["disabled"]] = float(disabled[move.id])
    else:
        reasons.append("request_move_missing")
    row[_IDX["maybe_disabled"]] = float(maybe)


def _tera_features(
    move: Any, active: Any, opp: Any, tera: PokemonType | None, row: np.ndarray,
    reasons: list[str],
) -> None:
    if tera is None or active is None:
        reasons.append("tera_type_missing")
        return
    _onehot_types([tera], row, _IDX["tera_type_" + MOVE_TYPES[0].lower()])
    mtype = move.type
    before = mtype in active.types
    after = mtype == tera or before
    row[_IDX["tera_stab_after"]] = float(after)
    row[_IDX["tera_stab_mult_before"]] = 1.5 if before else 1.0
    if mtype == tera and before:
        mult = 2.0
    elif after:
        mult = 1.5
    else:
        mult = 1.0
    row[_IDX["tera_stab_mult_after"]] = mult
    if opp is None:
        reasons.append("opp_active_missing_tera")
        return
    opp_types = public_types(opp)
    now = max(_log_eff(t, list(active.types)) for t in opp_types)
    then = max(_log_eff(t, [tera]) for t in opp_types)
    row[_IDX["tera_d_def_eff"]] = then - now


def _hazard_damage(mon: Any, battle: Any) -> float:
    conds = battle.side_conditions
    types = list(mon.types)
    dmg = 0.0
    if SideCondition.STEALTH_ROCK in conds:
        mult = PokemonType.ROCK.damage_multiplier(
            types[0], types[1] if len(types) > 1 else None, type_chart=_chart()
        )
        dmg += mult / 8.0
    layers = conds.get(SideCondition.SPIKES, 0)
    if layers:
        dmg += {1: 1 / 8, 2: 1 / 6, 3: 1 / 4}.get(int(layers), 1 / 4) * 0.5
    return min(1.0, dmg)


def _switch_features(
    mon: Any, opp: Any, battle: Any, row: np.ndarray, reasons: list[str]
) -> None:
    _onehot_types(list(mon.types), row, _IDX["sw_type_" + MOVE_TYPES[0].lower()])
    row[_IDX["sw_hp"]] = float(mon.current_hp_fraction)
    if mon.status in STATUSES:
        row[_IDX["sw_status_" + mon.status.name.lower()]] = 1.0
    else:
        row[_IDX["sw_status_none"]] = 1.0
    row[_IDX["sw_hazard_dmg"]] = _hazard_damage(mon, battle)
    known = [m for m in mon.moves.values() if m.category != MoveCategory.STATUS]
    row[_IDX["sw_known_moves"]] = len(mon.moves) / 4.0
    if opp is None:
        reasons.append("opp_active_missing_switch")
        return
    opp_types = public_types(opp)
    row[_IDX["sw_def_eff"]] = max(_log_eff(t, list(mon.types)) for t in opp_types)
    if known:
        row[_IDX["sw_off_eff"]] = max(_log_eff(m.type, opp_types) for m in known)
    else:
        reasons.append("switch_no_damaging_known_move")


def describe(battle: Any, mask: np.ndarray, legal_orders: dict[int, Any]) -> Descriptor:
    """Descriptor rows for each legal slot of the learner at its current decision.

    `legal_orders` maps slot -> BattleOrder from `UniversalAction(slot).to_BattleOrder`;
    the order carries the Move/Pokemon and the Tera flag, built from the battle object only.
    """
    x = np.zeros((13, DIM), dtype=np.float32)
    ids: list[tuple[str, str] | None] = [None] * 13
    reasons: dict[int, list[str]] = {}
    active, opp = battle.active_pokemon, battle.opponent_active_pokemon
    disabled, maybe = _request_moves(battle)
    tera = battle.can_tera if active is not None else None
    for slot in range(13):
        if not mask[slot]:
            continue
        order = legal_orders.get(slot)
        why: list[str] = []
        if order is None or order.order is None:
            reasons[slot] = ["order_missing"]
            continue
        row = x[slot]
        _common(battle, row, why)
        target = order.order
        if slot < 4 or slot >= 9:
            row[_IDX["kind_tera" if slot >= 9 else "kind_move"]] = 1.0
            _move_features(
                target, active, opp, row, why, disabled, maybe,
                tera if slot >= 9 else None,
            )
            if slot >= 9:
                _tera_features(target, active, opp, tera, row, why)
            ids[slot] = ("move", target.id)
        else:
            row[_IDX["kind_switch"]] = 1.0
            _switch_features(target, opp, battle, row, why)
            ids[slot] = ("species", target.species)
        if why:
            reasons[slot] = why
    return Descriptor(x=x, ids=ids, reasons=reasons)


def describe_battle(battle: Any, mask: np.ndarray) -> Descriptor:
    """Live/offline entry point: builds the orders from the battle object, then `describe`."""
    from metamon.interface import UniversalAction

    orders = {}
    for slot in range(13):
        if mask[slot]:
            orders[slot] = UniversalAction(slot).to_BattleOrder(battle)
    return describe(battle, mask, orders)


def coverage(descs: list[tuple[np.ndarray, Descriptor]]) -> dict[str, Any]:
    """Per slot type: candidates and fraction with complete features; counts by reason."""
    kinds = ("move", "switch", "move+tera")
    total = Counter()
    complete = Counter()
    reason = Counter()
    for mask, d in descs:
        for slot in range(13):
            if not mask[slot]:
                continue
            kind = kinds[0] if slot < 4 else kinds[1] if slot < 9 else kinds[2]
            total[kind] += 1
            if slot in d.reasons:
                for r in d.reasons[slot]:
                    reason[f"{kind}:{r}"] += 1
            else:
                complete[kind] += 1
    return {
        "legal_slots": dict(total),
        "complete": {k: complete[k] for k in total},
        "fraction_complete": {k: complete[k] / total[k] for k in total},
        "missing_by_reason": dict(reason),
    }
