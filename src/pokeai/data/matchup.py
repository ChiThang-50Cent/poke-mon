"""Public-information damage/type/speed features per decision (encoder v2 extras).

One stateless function of `UniversalState`, used by replay encoding and live play.
Inputs are only what the acting player sees (own set, opponent's revealed
moves/item/ability, field) plus the public Dex. Unknown opponent abilities use a
uniform prior over the species' legal abilities; unknown items are ignored.
Stats come from base stats with one fixed spread (both sides), so the values are
approximations of the Showdown formula, not exact calcs.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
from metamon.backend.showdown_dex.dex import Dex
from metamon.interface import UniversalMove, UniversalPokemon, UniversalState

MOVE_F = 6  # per own move: log2 eff, stab, p_immune, dmg, p_ko, damaging
OPP_MOVE_F = 3  # per opp revealed move: log2 eff vs own, dmg to own, p_ko own
THREAT_F = 2  # opp STAB threat vs own active: log2 eff, dmg
BENCH_F = 5  # per bench: best dmg, best p_ko, max incoming, def log2 eff, outspeeds max
SPEED_F = 3  # own > opp min, own > opp max, log speed ratio vs opp mid
EXTRAS_DIM = 4 * MOVE_F + 4 * OPP_MOVE_F + THREAT_F + 5 * BENCH_F + SPEED_F
EXTRAS_VERSION = "matchup-v0"
PAD = -2.0

EV = 128  # fixed assumed investment for every stat, both sides
THREAT_BP = 90
IMMUNE_ABILITIES = {
    "levitate": "ground",
    "eartheater": "ground",
    "waterabsorb": "water",
    "stormdrain": "water",
    "dryskin": "water",
    "voltabsorb": "electric",
    "lightningrod": "electric",
    "motordrive": "electric",
    "flashfire": "fire",
    "wellbakedbody": "fire",
    "sapsipper": "grass",
}
UNKNOWN_ABILITIES = {"", "unknownability", "noability"}
ITEM_MULT = {
    "choiceband": ("physical", 1.5),
    "choicespecs": ("special", 1.5),
    "lifeorb": ("any", 1.3),
}


@lru_cache(None)
def _dex() -> Dex:
    return Dex.from_gen(9)


def _norm(name: str) -> str:
    return "".join(c for c in name.lower() if c.isalnum())


@lru_cache(None)
def _ability_names(species: str) -> tuple[str, ...]:
    entry = _dex().pokedex.get(_norm(species))
    if entry is None:
        return ()
    return tuple(_norm(a) for a in entry.get("abilities", {}).values())


def type_mult(atk_type: str, def_types: tuple[str, ...]) -> float:
    chart = _dex().type_chart
    mult = 1.0
    for t in def_types:
        row = chart.get(t.upper())
        if row is not None:
            mult *= row.get(atk_type.upper(), 1.0)
    return mult


def _types(p: UniversalPokemon) -> tuple[str, ...]:
    return tuple(t for t in p.types.split() if t not in ("notype", "<blank>"))


def p_immune_ability(defender: UniversalPokemon, atk_type: str) -> float:
    ability = _norm(defender.ability)
    if ability not in UNKNOWN_ABILITIES:
        return float(IMMUNE_ABILITIES.get(ability) == atk_type)
    options = _ability_names(defender.base_species or defender.name) or _ability_names(
        defender.name
    )
    if not options:
        return 0.0
    return sum(IMMUNE_ABILITIES.get(a) == atk_type for a in options) / len(options)


def _stat(base: int, lvl: int, boost: int = 0) -> float:
    raw = math.floor((2 * base + 31 + EV // 4) * lvl / 100) + 5
    mult = (2 + boost) / 2 if boost >= 0 else 2 / (2 - boost)
    return raw * mult


def _hp(base: int, lvl: int) -> float:
    if base == 1:  # Shedinja
        return 1.0
    return math.floor((2 * base + 31 + EV // 4) * lvl / 100) + lvl + 10


def _speed(p: UniversalPokemon, ev: int, nature: float, scarf: bool) -> float:
    raw = (math.floor((2 * p.base_spe + 31 + ev // 4) * p.lvl / 100) + 5) * nature
    b = p.spe_boost
    raw *= (2 + b) / 2 if b >= 0 else 2 / (2 - b)
    if scarf:
        raw *= 1.5
    if p.status == "par":
        raw *= 0.5
    return raw


@lru_cache(None)
def _move_info(name: str) -> tuple[float, str | None, str | None, bool, int | None]:
    """(hits, offensive stat override, defensive override, foul play, fixed damage)."""
    m = _dex().moves.get(_norm(name), {})
    hits = m.get("multihit")
    if isinstance(hits, list):
        n = 3.0 if hits == [2, 5] else float(sum(hits)) / 2
    else:
        n = float(hits or 1)
    fixed = m.get("damage")
    return (
        n,
        m.get("overrideOffensiveStat"),
        m.get("overrideDefensiveStat"),
        m.get("overrideOffensivePokemon") == "target",
        None if fixed is None else (-1 if fixed == "level" else int(fixed)),
    )


def damage_frac(
    move_type: str,
    category: str,
    base_power: float,
    move_name: str,
    attacker: UniversalPokemon,
    defender: UniversalPokemon,
    weather: str,
    defender_conditions: str,
    attacker_item: str | None = None,
) -> tuple[float, float]:
    """Expected max-roll damage as a fraction of defender max HP, and effectiveness."""
    def_types = _types(defender)
    eff = type_mult(move_type, def_types) * (1.0 - p_immune_ability(defender, move_type))
    hits, off_override, def_override, foul, fixed = _move_info(move_name)
    max_hp = _hp(defender.base_hp, defender.lvl)
    if fixed is not None:
        dmg = attacker.lvl if fixed == -1 else fixed
        return (dmg / max_hp if eff > 0 else 0.0), eff
    if category not in ("physical", "special") or base_power <= 0:
        return 0.0, eff
    physical = category == "physical"
    src = defender if foul else attacker
    if off_override == "def":
        a = _stat(src.base_def, src.lvl, src.def_boost)
    elif physical:
        a = _stat(src.base_atk, src.lvl, src.atk_boost)
    else:
        a = _stat(src.base_spa, src.lvl, src.spa_boost)
    use_def = physical if def_override is None else def_override == "def"
    if use_def:
        d = _stat(defender.base_def, defender.lvl, defender.def_boost)
    else:
        d = _stat(defender.base_spd, defender.lvl, defender.spd_boost)
    base = math.floor(math.floor(2 * attacker.lvl / 5 + 2) * base_power * a / d / 50) + 2
    mod = eff * hits
    if move_type in _types(attacker):
        mod *= 1.5
    if physical and attacker.status == "brn":
        mod *= 0.5
    if weather in ("sunnyday", "desolateland"):
        mod *= 1.5 if move_type == "fire" else 0.5 if move_type == "water" else 1.0
    elif weather in ("raindance", "primordialsea"):
        mod *= 1.5 if move_type == "water" else 0.5 if move_type == "fire" else 1.0
    if (physical and defender_conditions == "reflect") or (
        not physical and defender_conditions == "lightscreen"
    ) or defender_conditions == "auroraveil":
        mod *= 0.5
    if attacker_item in ITEM_MULT:
        kind, m = ITEM_MULT[attacker_item]
        if kind == "any" or kind == category:
            mod *= m
    return base * mod / max_hp, eff


def p_ko(max_frac: float, hp_pct: float, accuracy: float) -> float:
    """P(KO) over the uniform 85-100% roll, times accuracy."""
    if max_frac <= 0 or hp_pct <= 0:
        return 0.0
    lo = 0.85 * max_frac
    if lo >= hp_pct:
        p = 1.0
    elif max_frac < hp_pct:
        p = 0.0
    else:
        p = (max_frac - hp_pct) / (max_frac - lo)
    return p * min(max(accuracy, 0.0), 1.0)


def _log_eff(eff: float) -> float:
    return math.log2(eff) / 2 if eff > 0 else -2.0


def _real(m: UniversalMove) -> bool:
    return m.name not in ("", "<blank>", "nomove")


def _attack(
    move: UniversalMove,
    atk: UniversalPokemon,
    dfn: UniversalPokemon,
    state: UniversalState,
    def_cond: str,
    item: str | None,
) -> tuple[float, float]:
    return damage_frac(
        move.move_type,
        move.category,
        float(move.base_power),
        move.name,
        atk,
        dfn,
        state.weather,
        def_cond,
        item,
    )


def _threat(opp: UniversalPokemon, target: UniversalPokemon, state: UniversalState):
    best_eff, best_dmg = 0.0, 0.0
    category = "physical" if opp.base_atk >= opp.base_spa else "special"
    for t in _types(opp):
        dmg, eff = damage_frac(
            t, category, THREAT_BP, "", opp, target, state.weather, state.player_conditions
        )
        if dmg >= best_dmg:
            best_eff, best_dmg = eff, dmg
    return best_eff, best_dmg


def matchup_extras(state: UniversalState) -> np.ndarray:
    out = np.full(EXTRAS_DIM, PAD, dtype=np.float32)
    me = state.player_active_pokemon
    opp = state.opponent_active_pokemon
    if me is None or opp is None or opp.name in ("", "<blank>"):
        return out
    my_item = _norm(me.item)
    i = 0
    for k in range(4):
        if k < len(me.moves) and _real(me.moves[k]):
            mv = me.moves[k]
            dmg, eff = _attack(mv, me, opp, state, state.opponent_conditions, my_item)
            damaging = float(mv.category in ("physical", "special") and dmg > 0)
            out[i : i + MOVE_F] = (
                _log_eff(eff) if damaging else 0.0,
                float(mv.move_type in _types(me)),
                p_immune_ability(opp, mv.move_type),
                min(dmg, 2.0),
                p_ko(dmg, opp.hp_pct, mv.accuracy),
                damaging,
            )
        i += MOVE_F
    opp_moves = [m for m in opp.moves if _real(m)][:4]
    for k in range(4):
        if k < len(opp_moves):
            mv = opp_moves[k]
            dmg, eff = _attack(mv, opp, me, state, state.player_conditions, None)
            out[i : i + OPP_MOVE_F] = (
                _log_eff(eff),
                min(dmg, 2.0),
                p_ko(dmg, me.hp_pct, mv.accuracy),
            )
        i += OPP_MOVE_F
    eff, dmg = _threat(opp, me, state)
    out[i : i + THREAT_F] = (_log_eff(eff), min(dmg, 2.0))
    i += THREAT_F
    opp_max = _speed(opp, 252, 1.1, _norm(opp.item) == "choicescarf")
    opp_min = _speed(opp, 0, 0.9, False)
    for k in range(5):
        if k < len(state.available_switches):
            sw = state.available_switches[k]
            best_dmg, best_ko = 0.0, 0.0
            for mv in sw.moves:
                if not _real(mv):
                    continue
                d, _ = _attack(mv, sw, opp, state, state.opponent_conditions, _norm(sw.item))
                if d > best_dmg:
                    best_dmg, best_ko = d, p_ko(d, opp.hp_pct, mv.accuracy)
            incoming = _threat(opp, sw, state)[1]
            for mv in opp_moves:
                incoming = max(incoming, _attack(mv, opp, sw, state, state.player_conditions, None)[0])
            def_eff = max((type_mult(t, _types(sw)) for t in _types(opp)), default=1.0)
            sw_speed = _speed(sw, EV, 1.0, _norm(sw.item) == "choicescarf")
            out[i : i + BENCH_F] = (
                min(best_dmg, 2.0),
                best_ko,
                min(incoming, 2.0),
                _log_eff(def_eff),
                float(sw_speed > opp_max),
            )
        i += BENCH_F
    mine = _speed(me, EV, 1.0, my_item == "choicescarf")
    opp_mid = _speed(opp, EV, 1.0, _norm(opp.item) == "choicescarf")
    out[i : i + SPEED_F] = (
        float(mine > opp_min),
        float(mine > opp_max),
        max(min(math.log(mine / opp_mid), 2.0), -2.0) if opp_mid > 0 and mine > 0 else 0.0,
    )
    return out
