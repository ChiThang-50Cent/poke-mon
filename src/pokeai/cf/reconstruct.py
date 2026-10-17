"""BS0.1 public-to-simulator reconstruction (learner point of view only).

Inputs, and ONLY these: the learner's own packed team, its current request, the public event
history it received (the p1-channel protocol lines), team preview (part of that history),
public Dex data and the CF1 belief sampler's opponent worlds. This module never imports the
simulator oracle (see `recon_oracle.py`, which a test keeps out of this import graph).

`Tracker` folds the history into a `Public` state; `build_spec` turns it plus the request and
a sampled world into the spec consumed by `sim_driver.js: reconstruct`.

What is exact / sampled / unsupported is listed in `SUPPORT` and in the report. Everything
that is not public is either drawn from a documented rule (seeded RNG) or the root is marked
unsupported with a reason - never copied from the true battle.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

SIDES = ("p1", "p2")
BOOSTS = ("atk", "def", "spa", "spd", "spe", "accuracy", "evasion")
TERRAINS = {
    "electricterrain": "electricterrain", "grassyterrain": "grassyterrain",
    "mistyterrain": "mistyterrain", "psychicterrain": "psychicterrain",
}
WEATHER_IDS = {
    "sunnyday": "sunnyday", "raindance": "raindance", "sandstorm": "sandstorm",
    "snow": "snowscape", "snowscape": "snowscape", "hail": "hail",
    "desolateland": "desolateland", "primordialsea": "primordialsea",
    "deltastream": "deltastream",
}
WEATHER_ITEM = {
    "sunnyday": "heatrock", "raindance": "damprock", "sandstorm": "smoothrock",
    "snowscape": "icyrock", "hail": "icyrock",
}
TERRAIN_ITEM = "terrainextender"
PSEUDO_DURATION = {
    "trickroom": 5, "gravity": 5, "magicroom": 5, "wonderroom": 5, "mudsport": 5,
    "watersport": 5, "iondeluge": 1, "fairylock": 2,
}
SIDE_DURATION = {  # id -> (duration, item that extends it, extended duration)
    "reflect": (5, "lightclay", 8), "lightscreen": (5, "lightclay", 8),
    "auroraveil": (5, "lightclay", 8), "tailwind": (4, None, 4), "safeguard": (5, None, 5),
    "mist": (5, None, 5), "luckychant": (5, None, 5),
}
STACKING = {"spikes": 3, "toxicspikes": 2}
PERMANENT_SIDE = {"stealthrock", "stickyweb", "spikes", "toxicspikes"}
# volatiles with documented state; everything else seen is `unsupported_volatile:<id>`
BOOST_STATS = {"atk", "def", "spa", "spd", "spe"}
SIMPLE_VOLATILES = {
    "substitute", "confusion", "leechseed", "taunt", "encore", "disable", "torment",
    "yawn", "aquaring", "ingrain", "magnetrise", "focusenergy", "charge", "curse",
    "flashfire", "embargo", "healblock", "smackdown", "telekinesis", "powertrick",
    "laserfocus", "noretreat", "destinybond_never",
}
VOLATILE_NAMES = {
    "substitute": "substitute", "confusion": "confusion", "leech seed": "leechseed",
    "taunt": "taunt", "encore": "encore", "disable": "disable", "torment": "torment",
    "yawn": "yawn", "aqua ring": "aquaring", "ingrain": "ingrain",
    "magnet rise": "magnetrise", "focus energy": "focusenergy", "charge": "charge",
    "curse": "curse", "ability: flash fire": "flashfire", "embargo": "embargo",
    "heal block": "healblock", "bind": "partiallytrapped", "clamp": "partiallytrapped",
    "fire spin": "partiallytrapped", "infestation": "partiallytrapped",
    "magma storm": "partiallytrapped", "sand tomb": "partiallytrapped",
    "snap trap": "partiallytrapped", "thunder cage": "partiallytrapped",
    "whirlpool": "partiallytrapped", "wrap": "partiallytrapped", "salt cure": "saltcure", "smack down": "smackdown", "telekinesis": "telekinesis",
    "power trick": "powertrick", "laser focus": "laserfocus", "no retreat": "noretreat",
}
UNSUPPORTED_MOVE_STATE = {
    "outrage", "thrash", "petaldance", "rollout", "iceball", "uproar", "bide", "furycutter",
    "wish", "futuresight", "doomdesire", "healingwish", "lunardance", "perishsong",
    "transform", "mimic", "imprison", "substitute_never",
}
CHOICE = {"choiceband", "choicespecs", "choicescarf"}
PROTECT_MOVES = {
    "protect", "detect", "kingsshield", "spikyshield", "banefulbunker", "obstruct",
    "silktrap", "burningbulwark", "endure", "wideguard", "quickguard", "maxguard",
}
SLEEP_TURNS = (2, 5)  # random(2, 5) in the sim -> 2..4
CONFUSION_TURNS = (2, 6)


def to_id(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def parse_ident(token: str) -> tuple[str, str] | None:
    m = re.match(r"^(p[12])[a-z]?: (.*)$", token.strip())
    return (m.group(1), m.group(2)) if m else None


def parse_hp(token: str) -> tuple[int, int, str]:
    """`63/100 par` -> (63, 100, 'par'); `0 fnt` -> (0, 100, 'fnt')."""
    parts = token.split(" ")
    status = parts[1] if len(parts) > 1 else ""
    if parts[0].startswith("0") and "/" not in parts[0]:
        return 0, 100, status
    cur, mx = parts[0].split("/")
    return int(cur), int(mx), status


def gender_of(details: str) -> str:
    parts = [p.strip() for p in details.split(",")]
    return next((p for p in parts[1:] if p in ("M", "F")), "N")


def species_of(details: str) -> str:
    return details.split(",")[0].strip()


@dataclass
class Volatile:
    id: str
    turn: int
    post: bool
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Mon:
    name: str | None = None
    species: str = ""
    base_species: str = ""
    level: int = 100
    gender: str = ""
    shiny: bool = False
    hp: int = 100
    maxhp: int = 100
    status: str = ""
    status_turn: int = 0
    status_post: bool = True
    cant_slp: int = 0
    tox_ticks: int = 0
    boosts: dict[str, int] = field(default_factory=lambda: dict.fromkeys(BOOSTS, 0))
    volatiles: dict[str, Volatile] = field(default_factory=dict)
    active: bool = False
    fainted: bool = False
    tera: str | None = None
    switch_turn: int = 0
    switch_post: bool = True
    stint_moves: list[str] = field(default_factory=list)  # ids, first executed first
    move_actions: int = 0
    last_move: str | None = None
    last_move_turn: int = -1
    pp_used: dict[str, int] = field(default_factory=dict)
    protect_run: int = 0  # consecutive successful protects ending last turn
    protect_turn: int = -9
    recharge_turn: int = -9
    unsupported: list[str] = field(default_factory=list)
    first_seen_index: int = -1
    hidden_forme: bool = False
    booster_pending: bool = False
    ability_state: dict[str, int] = field(default_factory=dict)
    types: list[str] | None = None
    added_type: str = ""
    exits_since_hp: int = 0  # voluntary switch-outs since the HP was last observed
    exits_total: int = 0
    item_lost_stint: bool = False  # item used/taken while in the current stint


@dataclass
class Public:
    turn: int = 0
    post: bool = True  # after the residual of the current turn
    mons: dict[str, list[Mon]] = field(default_factory=lambda: {"p1": [], "p2": []})
    names: dict[str, dict[str, Mon]] = field(default_factory=lambda: {"p1": {}, "p2": {}})
    preview: dict[str, list[str]] = field(default_factory=lambda: {"p1": [], "p2": []})
    weather: dict[str, Any] | None = None
    terrain: dict[str, Any] | None = None
    pseudo: dict[str, dict[str, Any]] = field(default_factory=dict)
    conditions: dict[str, dict[str, dict[str, Any]]] = field(
        default_factory=lambda: {"p1": {}, "p2": {}})
    faint_turn: dict[str, int] = field(default_factory=lambda: {"p1": -9, "p2": -9})
    total_fainted: dict[str, int] = field(default_factory=lambda: {"p1": 0, "p2": 0})
    unsupported: list[str] = field(default_factory=list)
    last_move_line: tuple[str, str, str] | None = None  # (side, mon name, move id)
    recent: list[tuple[str, int, str]] = field(default_factory=list)  # (move, turn, side)
    baton: dict[str, bool] = field(default_factory=lambda: {"p1": False, "p2": False})
    switch_order: dict[str, list[str]] = field(default_factory=lambda: {"p1": [], "p2": []})
    last_active: dict[str, str | None] = field(default_factory=lambda: {"p1": None, "p2": None})
    tera_used: dict[str, bool] = field(default_factory=lambda: {"p1": False, "p2": False})

    def active(self, side: str) -> Mon | None:
        return next((m for m in self.mons[side] if m.active), None)


class Tracker:
    """Fold the learner-channel protocol into a `Public` state."""

    def __init__(self) -> None:
        self.s = Public()
        self.unhandled: dict[str, int] = {}

    # -- helpers -----------------------------------------------------------------------
    def _mon(self, token: str) -> tuple[str, Mon] | None:
        ident = parse_ident(token)
        if ident is None:
            return None
        side, name = ident
        mon = self.s.names[side].get(name)
        if mon is None:  # nickname not yet bound to a preview entry
            mon = next((m for m in self.s.mons[side] if m.name is None
                        and to_id(m.base_species) == to_id(name)), None)
            if mon is None:
                mon = next((m for m in self.s.mons[side] if m.name is None), None)
            if mon is None:
                return None
            mon.name = name
            self.s.names[side][name] = mon
        return side, mon

    def ticks_since(self, turn: int, post: bool) -> int:
        """End-of-turn residual phases executed since an event at (turn, post)."""
        if turn == 0:  # before the first |turn|: residual 1 is the end of turn 1
            return max(self.s.turn - 1, 0)
        return max(self.s.turn - turn - (1 if post else 0), 0)

    def unhandled_event(self, kind: str) -> None:
        self.unhandled[kind] = self.unhandled.get(kind, 0) + 1

    # -- feed ----------------------------------------------------------------------------
    def feed(self, lines: list[str]) -> Public:
        for line in lines:
            if line.startswith("|"):
                self._line(line.split("|")[1:])
        return self.s

    def _preview(self, side: str, details: str) -> None:
        species = species_of(details)
        mon = Mon(species=species.removesuffix("-*"), base_species=species.removesuffix("-*"))
        mon.hidden_forme = species.endswith("-*")  # preview hides the forme: base forme assumed
        m = re.search(r"L(\d+)", details)
        mon.level = int(m.group(1)) if m else 100
        mon.gender = "M" if ", M" in details else "F" if ", F" in details else ""
        mon.shiny = "shiny" in details
        mon.first_seen_index = len(self.s.mons[side])
        self.s.mons[side].append(mon)
        self.s.preview[side].append(mon.base_species)

    def _line(self, parts: list[str]) -> None:
        kind, args = parts[0], parts[1:]
        s = self.s
        tags = [a for a in args if a.startswith("[")]
        frm = next((t[7:].strip() for t in tags if t.startswith("[from]")), None)
        of = next((t[5:].strip() for t in tags if t.startswith("[of]")), None)
        handler = getattr(self, "_h_" + kind.lstrip("-"), None)
        if kind == "poke" and len(args) >= 2:
            self._preview(args[0], args[1])
        elif kind == "turn":
            s.turn = int(args[0])
            s.post = False
            for side in SIDES:
                for m in s.mons[side]:
                    if m.protect_turn != s.turn - 1:
                        m.protect_run = 0
        elif kind == "upkeep":
            s.post = True
        elif handler is not None:
            handler(args, tags, frm, of)
        elif kind in {"player", "teamsize", "gametype", "gen", "tier", "rule", "start",
                      "clearpoke", "teampreview", "j", "l", "c", "raw", "html", "init",
                      "title", "inactive", "inactiveoff", "rated", "t:", "split", "debug",
                      "uhtml", "message", "request", "error", "win", "tie", "cant", "n", "b",
                      "move_ignore"}:
            if kind == "cant" and len(args) >= 2:
                self._h_cant(args, tags, frm, of)
        elif kind.startswith("-"):
            # informational transient lines (-hint, -miss, -crit, -resisted, -supereffective,
            # -immune, -fail, -block, -notarget, -nothing, -center, -singleturn, ...)
            pass
        else:
            self.unhandled_event(kind)

    # -- handlers (name = protocol kind without leading '-') -------------------------------
    def _h_switch(self, args, tags, frm, of) -> None:
        side, mon_name = parse_ident(args[0]) or (None, None)
        if side is None:
            return
        s = self.s
        details = args[1]
        species = species_of(details)
        mon = s.names[side].get(mon_name)
        if mon is None:
            mon = next((m for m in s.mons[side] if m.name is None
                        and to_id(m.base_species) == to_id(species.split("-")[0])
                        or m.name is None and to_id(m.base_species) == to_id(species)), None)
            if mon is None:
                mon = next((m for m in s.mons[side] if m.name is None), None)
            if mon is None:  # preview hid it (should not happen in OU preview)
                mon = Mon(species=species, base_species=species)
                s.mons[side].append(mon)
            mon.name = mon_name
            s.names[side][mon_name] = mon
        previous = s.active(side)
        carry: tuple[dict[str, int], dict[str, Volatile]] | None = None
        if previous is not None:
            if s.baton[side]:
                carry = (dict(previous.boosts), {
                    k: v for k, v in previous.volatiles.items()
                    if k in ("substitute", "confusion", "leechseed", "focusenergy",
                             "aquaring", "ingrain", "magnetrise", "powertrick", "embargo",
                             "healblock", "telekinesis", "laserfocus")})
            previous.exits_since_hp += 1
            previous.exits_total += 1
            self._deactivate(previous)
        s.baton[side] = False
        mon.active = True
        s.last_active[side] = mon_name
        mon.species = species
        if mon.hidden_forme:
            mon.base_species, mon.hidden_forme = species, False
        mon.switch_turn, mon.switch_post = s.turn, s.post
        mon.stint_moves = []
        mon.item_lost_stint = False
        mon.move_actions = 0
        mon.tox_ticks = 0
        mon.cant_slp = 0
        mon.last_move = None
        mon.boosts = dict.fromkeys(BOOSTS, 0)
        mon.volatiles = {}
        mon.protect_run = 0
        if carry is not None:
            mon.boosts, mon.volatiles = carry
        hp, mx, status = parse_hp(args[2]) if len(args) > 2 else (100, 100, "")
        mon.hp, mon.maxhp = hp, mx
        mon.exits_since_hp = 0
        if status and status != "fnt" and not mon.status:
            mon.status, mon.status_turn, mon.status_post = status, s.turn, s.post
        if side not in s.switch_order:
            s.switch_order[side] = []
        if mon_name not in s.switch_order[side]:
            s.switch_order[side].append(mon_name)

    _h_drag = _h_switch

    def _deactivate(self, mon: Mon) -> None:
        mon.active = False
        mon.boosts = dict.fromkeys(BOOSTS, 0)
        mon.volatiles = {}
        mon.stint_moves = []
        mon.ability_state = {}
        mon.types, mon.added_type = None, ""

    def _h_replace(self, args, tags, frm, of) -> None:
        self.s.unsupported.append("illusion")

    def _h_detailschange(self, args, tags, frm, of) -> None:
        found = self._mon(args[0])
        if found:
            found[1].species = species_of(args[1])

    def _h_formechange(self, args, tags, frm, of) -> None:
        found = self._mon(args[0])
        if found:
            found[1].species = args[1].split(",")[0].strip()

    def _h_move(self, args, tags, frm, of) -> None:
        found = self._mon(args[0])
        if not found:
            return
        side, mon = found
        move = to_id(args[1])
        if move.startswith("hiddenpower"):
            move = "hiddenpower"
        own_choice = frm is None or frm == "lockedmove"
        if frm and frm.startswith("lockedmove"):
            mon.unsupported.append("lockedmove")
        if own_choice and move != "struggle":
            mon.move_actions += 1
            mon.stint_moves.append(move)
            mon.last_move, mon.last_move_turn = move, self.s.turn
            mon.pp_used[move] = mon.pp_used.get(move, 0) + 1
            if any(t == "[miss]" for t in tags):
                pass
            # remove two-turn state when the second turn executes
            mon.volatiles.pop("twoturnmove", None)
            mon.volatiles.pop("lockedmove", None)
        if move in UNSUPPORTED_MOVE_STATE and own_choice:
            mon.unsupported.append(f"move_state:{move}")
        if move == "batonpass":
            self.s.baton[side] = True
        self.s.last_move_line = (side, mon.name or "", move)
        if move in ("wish", "futuresight", "doomdesire", "healingwish", "lunardance"):
            self.s.recent.append((move, self.s.turn, side))
        if move == "healingwish" or move == "lunardance":
            mon.unsupported.append("move_state:" + move)

    def _h_cant(self, args, tags, frm, of) -> None:
        found = self._mon(args[0])
        if not found:
            return
        _, mon = found
        reason = args[1] if len(args) > 1 else ""
        if reason == "slp":
            mon.cant_slp += 1
        elif reason == "recharge":
            mon.volatiles.pop("mustrecharge", None)

    def _hp_update(self, mon: Mon, token: str) -> None:
        hp, mx, _status = parse_hp(token)
        mon.hp, mon.maxhp = hp, mx
        mon.exits_since_hp = 0
        if hp == 0:
            return

    def _h_damage(self, args, tags, frm, of) -> None:
        found = self._mon(args[0])
        if not found:
            return
        _, mon = found
        self._hp_update(mon, args[1])
        if frm == "psn" and mon.status == "tox":
            mon.tox_ticks += 1

    _h_heal = _h_damage
    _h_sethp = _h_damage

    def _h_status(self, args, tags, frm, of) -> None:
        found = self._mon(args[0])
        if not found:
            return
        _, mon = found
        mon.status, mon.status_turn, mon.status_post = args[1], self.s.turn, self.s.post
        mon.cant_slp = 0
        mon.tox_ticks = 0

    def _h_curestatus(self, args, tags, frm, of) -> None:
        found = self._mon(args[0])
        if found:
            found[1].status = ""
            found[1].cant_slp = 0
            found[1].tox_ticks = 0

    def _h_cureteam(self, args, tags, frm, of) -> None:
        found = self._mon(args[0])
        if found:
            for m in self.s.mons[found[0]]:
                m.status = ""

    def _h_faint(self, args, tags, frm, of) -> None:
        found = self._mon(args[0])
        if not found:
            return
        side, mon = found
        mon.fainted, mon.active, mon.hp = True, False, 0
        mon.boosts = dict.fromkeys(BOOSTS, 0)
        mon.volatiles = {}
        mon.status = ""
        mon.tera = mon.tera  # tera type is kept for the set; `terastallized` flag is dropped
        self.s.total_fainted[side] += 1
        self.s.faint_turn[side] = self.s.turn

    def _boost(self, args, sign: int) -> None:
        found = self._mon(args[0])
        if found and args[1] in BOOSTS:
            b = found[1].boosts
            b[args[1]] = max(-6, min(6, b[args[1]] + sign * int(args[2])))

    def _h_boost(self, args, tags, frm, of) -> None:
        self._boost(args, 1)

    def _h_unboost(self, args, tags, frm, of) -> None:
        self._boost(args, -1)

    def _h_setboost(self, args, tags, frm, of) -> None:
        found = self._mon(args[0])
        if found and args[1] in BOOSTS:
            found[1].boosts[args[1]] = int(args[2])

    def _h_clearboost(self, args, tags, frm, of) -> None:
        found = self._mon(args[0])
        if found:
            found[1].boosts = dict.fromkeys(BOOSTS, 0)

    def _h_clearallboost(self, args, tags, frm, of) -> None:
        for side in SIDES:
            for m in self.s.mons[side]:
                m.boosts = dict.fromkeys(BOOSTS, 0)

    def _h_clearpositiveboost(self, args, tags, frm, of) -> None:
        found = self._mon(args[0])
        if found:
            found[1].boosts = {k: min(v, 0) for k, v in found[1].boosts.items()}

    def _h_clearnegativeboost(self, args, tags, frm, of) -> None:
        found = self._mon(args[0])
        if found:
            found[1].boosts = {k: max(v, 0) for k, v in found[1].boosts.items()}

    def _h_invertboost(self, args, tags, frm, of) -> None:
        found = self._mon(args[0])
        if found:
            found[1].boosts = {k: -v for k, v in found[1].boosts.items()}

    def _h_swapboost(self, args, tags, frm, of) -> None:
        a, b = self._mon(args[0]), self._mon(args[1])
        if a and b:
            stats = args[2].split(", ") if len(args) > 2 and not args[2].startswith("[") else BOOSTS
            for k in stats:
                a[1].boosts[k], b[1].boosts[k] = b[1].boosts[k], a[1].boosts[k]

    def _h_copyboost(self, args, tags, frm, of) -> None:
        a, b = self._mon(args[0]), self._mon(args[1])
        if a and b:
            b[1].boosts = dict(a[1].boosts)

    def _h_weather(self, args, tags, frm, of) -> None:
        name = to_id(args[0])
        if name == "none":
            self.s.weather = None
            return
        if any(t == "[upkeep]" for t in tags):
            return
        wid = WEATHER_IDS.get(name)
        if wid is None:
            self.s.unsupported.append(f"weather:{name}")
            return
        self.s.weather = self._effect_record(wid, frm, of, args)

    def _effect_record(self, eid: str, frm: str | None, of: str | None, args) -> dict[str, Any]:
        rec: dict[str, Any] = {
            "id": eid, "turn": self.s.turn, "post": self.s.post, "from": frm, "source": None,
        }
        token = of or None
        if token:
            found = self._mon(token)
            if found:
                rec["source"] = {"side": found[0], "name": found[1].name}
        return rec

    def _h_fieldstart(self, args, tags, frm, of) -> None:
        name = to_id(args[0].replace("move: ", ""))
        rec = self._effect_record(name, frm, of, args)
        if name in TERRAINS:
            self.s.terrain = rec
        elif name in PSEUDO_DURATION:
            self.s.pseudo[name] = rec
        else:
            self.s.unsupported.append(f"field:{name}")

    def _h_fieldend(self, args, tags, frm, of) -> None:
        name = to_id(args[0].replace("move: ", ""))
        if name in TERRAINS:
            self.s.terrain = None
        else:
            self.s.pseudo.pop(name, None)

    def _h_sidestart(self, args, tags, frm, of) -> None:
        ident = parse_ident(args[0])
        if ident is None:
            return
        side = ident[0]
        name = to_id(args[1].replace("move: ", ""))
        cond = self.s.conditions[side]
        if name in STACKING:
            rec = cond.setdefault(name, {"layers": 0})
            rec["layers"] = min(STACKING[name], rec["layers"] + 1)
        elif name in PERMANENT_SIDE or name in SIDE_DURATION:
            rec = self._effect_record(name, frm, of, args)
            rec["layers"] = 1
            # who set it: the mon active on the opposite... the user is the acting mon
            last = self.s.last_move_line
            rec["setter"] = {"side": last[0], "name": last[1]} if last else None
            cond[name] = rec
        else:
            self.s.unsupported.append(f"sidecondition:{name}")

    def _h_sideend(self, args, tags, frm, of) -> None:
        ident = parse_ident(args[0])
        if ident:
            self.s.conditions[ident[0]].pop(to_id(args[1].replace("move: ", "")), None)

    def _h_start(self, args, tags, frm, of) -> None:
        if not args:  # the battle-start marker
            return
        found = self._mon(args[0])
        if not found:
            return
        side, mon = found
        raw = args[1] if len(args) > 1 else ""
        key = raw.replace("move: ", "").strip().lower()
        if key.startswith("fallen") and key[6:].isdigit():  # Supreme Overlord ability state
            mon.ability_state = {"fallen": int(key[6:])}
            return
        if key in ("typechange", "typeadd"):
            source = (frm or "").lower()
            if any(x in source for x in ("burn up", "double shock", "roost", "tera")):
                mon.unsupported.append("typechange_volatile")
            elif key == "typechange" and len(args) > 2:
                mon.types = [t.strip() for t in args[2].split("/")]
            elif key == "typeadd" and len(args) > 2:
                mon.added_type = args[2].strip()
            return
        if key in ("dynamax",):
            mon.unsupported.append("dynamax")
            return
        vid = VOLATILE_NAMES.get(key)
        extra: dict[str, Any] = {}
        for prefix in ("protosynthesis", "quarkdrive"):
            if key.startswith(prefix) and key[len(prefix):] in BOOST_STATS:
                mon.volatiles[prefix] = Volatile(prefix, self.s.turn, self.s.post, {
                    "bestStat": key[len(prefix):], "fromBooster": mon.booster_pending})
                mon.booster_pending = False
                return
        if key == "disable" and len(args) > 2:
            extra["move"] = to_id(args[2])
        if key == "encore":
            extra["move"] = mon.last_move
        if key == "confusion":
            extra["fatigue"] = "[fatigue]" in tags
        if vid is None:
            if key.startswith(("perish", "stockpile", "attract", "protosynthesis", "quarkdrive",
                               "ability: protosynthesis", "ability: quark drive",
                               "slow start", "ability: slow start", "bide", "nightmare",
                               "ability: ", "future", "doom", "rage", "uproar", "sky drop")):
                if key.startswith("ability: ") and key not in VOLATILE_NAMES:
                    mon.unsupported.append(f"volatile:{key}")
                else:
                    mon.unsupported.append(f"volatile:{key}")
            else:
                mon.unsupported.append(f"volatile:{key}")
            return
        mon.volatiles[vid] = Volatile(vid, self.s.turn, self.s.post, extra)
        if vid == "leechseed":
            mon.volatiles[vid].extra["source_side"] = "p2" if side == "p1" else "p1"

    def _h_enditem(self, args, tags, frm, of) -> None:
        found = self._mon(args[0])
        if found and found[1].active:
            found[1].item_lost_stint = True

    def _h_end(self, args, tags, frm, of) -> None:
        found = self._mon(args[0])
        if not found:
            return
        _, mon = found
        key = (args[1] if len(args) > 1 else "").replace("move: ", "").strip().lower()
        if key.startswith("fallen"):
            mon.ability_state = {}
        vid = VOLATILE_NAMES.get(key)
        if vid:
            mon.volatiles.pop(vid, None)
        if key in ("protosynthesis", "quark drive", "quarkdrive"):
            mon.volatiles.pop(key.replace(" ", ""), None)
        mon.unsupported = [u for u in mon.unsupported if u != f"volatile:{key}"]

    def _h_activate(self, args, tags, frm, of) -> None:
        found = self._mon(args[0])
        if not found or len(args) < 2:
            return
        _, mon = found
        what = args[1].replace("move: ", "").strip().lower()
        v = mon.volatiles.get("confusion")
        if what == "confusion" and v is not None:
            v.extra["activations"] = v.extra.get("activations", 0) + 1
        vid = VOLATILE_NAMES.get(what)
        if vid == "partiallytrapped":
            src = of or ""
            side_src = parse_ident(src)
            mon.volatiles[vid] = Volatile(vid, self.s.turn, self.s.post, {
                "move": to_id(what), "source_side": side_src[0] if side_src else
                ("p2" if found[0] == "p1" else "p1")})
        if what in ("ability: protosynthesis", "ability: quark drive"):
            mon.booster_pending = "[fromitem]" in tags

    def _h_singleturn(self, args, tags, frm, of) -> None:
        found = self._mon(args[0])
        if not found or len(args) < 2:
            return
        _, mon = found
        move = to_id(args[1].replace("move: ", ""))
        if move in PROTECT_MOVES and move != "endure" or move == "endure":
            mon.protect_run = mon.protect_run + 1 if mon.protect_turn == self.s.turn - 1 else 1
            mon.protect_turn = self.s.turn

    def _h_mustrecharge(self, args, tags, frm, of) -> None:
        found = self._mon(args[0])
        if found:
            found[1].volatiles["mustrecharge"] = Volatile(
                "mustrecharge", self.s.turn, self.s.post)

    def _h_prepare(self, args, tags, frm, of) -> None:
        found = self._mon(args[0])
        if found:
            found[1].unsupported.append("twoturnmove")

    def _h_terastallize(self, args, tags, frm, of) -> None:
        found = self._mon(args[0])
        if found:
            side, mon = found
            mon.tera = args[1]
            self.s.tera_used[side] = True

    def _h_swapsideconditions(self, args, tags, frm, of) -> None:
        self.s.unsupported.append("courtchange")

    def _h_mega(self, args, tags, frm, of) -> None:
        self.s.unsupported.append("mega")

    def _h_burst(self, args, tags, frm, of) -> None:
        self.s.unsupported.append("burst")

    def _h_zpower(self, args, tags, frm, of) -> None:
        self.s.unsupported.append("zmove")

    def _h_transform(self, args, tags, frm, of) -> None:
        self.s.unsupported.append("transform")

    def _h_swap(self, args, tags, frm, of) -> None:
        self.s.unsupported.append("swap")


def build_public(lines: list[str]) -> tuple[Public, Tracker]:
    tracker = Tracker()
    tracker.feed(lines)
    return tracker.s, tracker


# ------------------------------------------------------------------------------ spec
def unpack_team(packed: str) -> list[dict[str, str]]:
    """Minimal Showdown packed-team reader: name, species, item (the learner's own team)."""
    from poke_env.data import GenData

    out = []
    for chunk in packed.split("]"):
        f = chunk.split("|")
        if len(f) < 4:
            continue
        species = f[1] or f[0]
        entry = GenData.from_gen(9).pokedex.get(to_id(species))
        e = entry or {}
        base = e["baseSpecies"] if e.get("forme") else e.get("name", species)
        species = (entry or {}).get("name", species)
        nick = f[0] if f[0] and to_id(f[0]) != to_id(species) else base
        out.append({"name": nick, "species": species, "item": f[2]})
    return out


def legal_abilities(species: str) -> list[str]:
    from poke_env.data import GenData

    entry = GenData.from_gen(9).pokedex.get(to_id(species))
    return list(entry["abilities"].values()) if entry else []


def reveal_order(public: Public, side: str) -> list[Mon]:
    """Party order used for the opponent: active, other revealed (reveal order), unrevealed."""
    mons = public.mons[side]
    for m in mons:
        if m.name is None:  # never seen: identified by its preview species
            m.name = m.base_species
            public.names[side][m.name] = m
    revealed = [public.names[side][n] for n in public.switch_order[side]]
    active = [m for m in revealed if m.active]
    if not active and public.last_active[side]:  # fainted, replacement pending
        active = [public.names[side][public.last_active[side]]]
    rest = [m for m in revealed if m not in active]
    hidden = [m for m in mons if m not in revealed]
    return active + rest + hidden


def own_order(public: Public, request: dict[str, Any]) -> list[tuple[Mon, dict[str, Any]]]:
    """Bind each request pokemon (exact, the learner's party order) to its tracked Mon."""
    out = []
    used: set[int] = set()
    for entry in request["side"]["pokemon"]:
        name = entry["ident"].split(": ", 1)[1]
        mon = public.names["p1"].get(name)
        if mon is None:
            species = species_of(entry["details"])
            mon = next((m for m in public.mons["p1"] if id(m) not in used and m.name is None
                        and to_id(m.base_species) in (to_id(species), to_id(species.split("-")[0]))),
                       None)
            if mon is None:
                mon = Mon(species=species, base_species=species)
            mon.name = name
        used.add(id(mon))
        out.append((mon, entry))
    return out


class Unsupported(Exception):
    pass


def _remaining(duration: int, ticks: int) -> int:
    return max(duration - ticks, 1)


def build_spec(
    tracker: Tracker, request: dict[str, Any], own_team: str, world: dict[str, Any],
    know: dict[str, Any], rng: np.random.Generator, seed: str, p1_name: str, p2_name: str,
    p2_sets: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Spec for `reconstruct`. `world` is a CF1 belief world over `reveal_order(p2)`."""
    s = tracker.s
    reasons: list[str] = list(s.unsupported)
    if request.get("wait") or request.get("teamPreview"):
        raise Unsupported("request_not_a_decision")
    forced = "forceSwitch" in request
    switch_sides: list[str] = []
    pairs1 = own_order(s, request)
    order2 = reveal_order(s, "p2")
    if forced:
        if not pairs1[0][1]["condition"].endswith(" fnt"):
            raise Unsupported("pivot_switch_request")  # mid-turn queue cannot be rebuilt
        switch_sides = ["p1"] + (["p2"] if order2[0].fainted else [])
    own_items = {e["name"]: e["item"] for e in unpack_team(own_team)}
    items: dict[tuple[str, str], str] = {}
    for mon, entry in pairs1:
        items[("p1", mon.name or "")] = entry.get("item", "")
    for mon, w in zip(order2, world["mons"], strict=True):
        items[("p2", mon.name or "")] = to_id(w["item"] or "")

    def mon_unsupported(mon: Mon, side: str) -> None:
        if mon.active:
            reasons.extend(f"{side}:{u}" for u in mon.unsupported)
        if "zoroark" in to_id(mon.base_species) or "zorua" in to_id(mon.base_species):
            reasons.append("illusion_possible")
        if to_id(mon.base_species) == "palafin" and mon.exits_total:
            reasons.append("palafin_zero_to_hero")

    for side in SIDES:
        for m in s.mons[side]:
            mon_unsupported(m, side)
    for move, turn, side in s.recent:
        if s.turn - turn <= 3:
            reasons.append(f"pending_move:{move}")
    if reasons:
        raise Unsupported(min(set(reasons)) + (
            "" if len(set(reasons)) == 1 else f" (+{len(set(reasons)) - 1})"))

    def vol_specs(
        side: str, mon: Mon, item: str, ability: str, moves: list[str]
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        if not mon.active:
            return out
        if to_id(ability or "") == "unburden" and mon.item_lost_stint:
            out.append({"id": "unburden", "state": {}})
        for vid, v in mon.volatiles.items():
            ticks = tracker.ticks_since(v.turn, v.post)
            st: dict[str, Any] = {}
            if vid == "substitute":
                st = {"hp_quarter": True}
            elif vid == "confusion":
                lo = max(CONFUSION_TURNS[0], v.extra.get("activations", 0) + 1)
                hi = CONFUSION_TURNS[1]
                t0 = int(rng.integers(lo, max(hi, lo + 1)))
                st = {"time": t0 - v.extra.get("activations", 0)}
            elif vid == "taunt":
                st = {"duration": _remaining(3, ticks)}
            elif vid == "encore":
                st = {"move": v.extra.get("move"), "duration": _remaining(3, ticks)}
                if not st["move"] or st["move"] not in {to_id(m) for m in moves}:
                    raise Unsupported("encore_move_not_in_moveset")
            elif vid == "disable":
                st = {"move": v.extra.get("move"), "duration": _remaining(4, ticks)}
            elif vid == "yawn":
                if ticks >= 2:
                    continue
                st = {"duration": 2 - ticks}
            elif vid == "magnetrise":
                st = {"duration": _remaining(5, ticks)}
            elif vid == "leechseed":
                st = {"source_side": v.extra["source_side"]}
            elif vid == "mustrecharge":
                if v.turn != s.turn - 1:
                    continue
                st = {"duration": 1}
            elif vid in ("protosynthesis", "quarkdrive"):
                st = dict(v.extra)
            elif vid == "partiallytrapped":
                lo = ticks + 1
                total = int(rng.integers(max(5, lo), max(7, lo + 1)))
                st = {"duration": total - ticks, "move": v.extra["move"],
                      "source_side": v.extra["source_side"]}
            elif vid == "charge":
                st = {"duration": 1}
            elif vid in ("healblock", "embargo"):
                st = {"duration": _remaining(5, ticks)}
            elif vid == "telekinesis":
                st = {"duration": _remaining(3, ticks)}
            out.append({"id": vid, "state": st})
        # protect counter: a protect that succeeded last turn leaves `stall`
        if mon.protect_turn == s.turn - 1 and mon.protect_run >= 1:
            out.append({"id": "stall", "state": {
                "counter": 3 ** mon.protect_run, "duration": 1}})
        # choice lock: the learner knows its own item; the opponent's is the sampled one
        if item in CHOICE and mon.stint_moves:
            out.append({"id": "choicelock", "state": {"move": mon.stint_moves[0]}})
        return out

    def status_state(mon: Mon, status: str) -> dict[str, Any]:
        if status == "slp":
            c = mon.cant_slp
            lo = max(SLEEP_TURNS[0], c + 1)
            t0 = int(rng.integers(lo, max(SLEEP_TURNS[1], lo + 1)))
            return {"startTime": t0, "time": max(t0 - c, 1)}
        if status == "tox":
            return {"stage": mon.tox_ticks}
        return {}

    def mon_base(mon: Mon) -> dict[str, Any]:
        return {
            "boosts": {k: v for k, v in mon.boosts.items() if v},
            "active_turns": (tracker.ticks_since(mon.switch_turn, mon.switch_post)
                             if mon.active else 0),
            "active_move_actions": mon.move_actions if mon.active else 0,
            "last_move": mon.last_move if mon.active else None,
            "pp_used": dict(mon.pp_used),
            "types": mon.types if mon.active else None,
            "added_type": mon.added_type if mon.active else "",
            "ability_state": dict(mon.ability_state) if mon.active else {},
        }

    mons: dict[str, list[dict[str, Any]]] = {"p1": [], "p2": []}
    index_of: dict[tuple[str, str], int] = {}
    for i, (mon, entry) in enumerate(pairs1):
        cur, _mx, status = parse_hp(entry["condition"])
        item = entry.get("item", "")
        d = mon_base(mon)
        d.update({
            "hp_exact": cur, "fainted": cur == 0, "status": "" if status == "fnt" else status,
            "ability": entry.get("ability") or entry.get("baseAbility"),
            "forme": species_of(entry["details"]), "name": mon.name,
            "terastallized": entry.get("terastallized") or "",
            "item_gone": item == "" and own_items.get(mon.name or "", "") != "",
            "item_exact": item,
            "active": bool(entry["active"]),
        })
        d["status_state"] = status_state(mon, d["status"]) if d["status"] else {}
        d["volatiles"] = vol_specs("p1", mon, item, d["ability"], entry.get("moves", [])) if not d["fainted"] else []
        if d["active"]:
            moves = (request.get("active") or [{}])[0].get("moves", [])
            d["pp_exact"] = {m["id"]: m["pp"] for m in moves if "pp" in m}
        mons["p1"].append(d)
        index_of[("p1", mon.name or "")] = i
    for i, (mon, w) in enumerate(zip(order2, world["mons"], strict=True)):
        k = know[mon.name or ""]
        d = mon_base(mon)
        status = mon.status if not mon.fainted else ""
        d.update({
            "pct": 0 if mon.fainted else max(1, mon.hp), "fainted": mon.fainted,
            "status": status, "ability": w["ability"], "name": mon.name,
            "forme": mon.species if mon.species != mon.base_species else None,
            "terastallized": mon.tera if mon.tera and mon.active and not mon.fainted else "",
            "item_gone": k.item_state == "gone", "active": mon.active,
        })
        d["regen"] = (mon.exits_since_hp if to_id(w["ability"]) == "regenerator"
                      and not mon.active and not mon.fainted else 0)
        d["status_state"] = status_state(mon, status) if status else {}
        d["volatiles"] = vol_specs("p2", mon, items[("p2", mon.name or "")], w["ability"], w["moves"])
        mons["p2"].append(d)
        index_of[("p2", mon.name or "")] = i

    def ref(src: dict[str, Any] | None) -> dict[str, Any] | None:
        if not src:
            return None
        idx = index_of.get((src["side"], src["name"]))
        return None if idx is None else {"side": src["side"], "index": idx}

    def extends(src: dict[str, Any] | None, item: str, default: str | None = None) -> bool:
        if not src:
            return False
        return items.get((src["side"], src["name"]), "") == item

    field_spec: dict[str, Any] = {"weather": None, "terrain": None, "pseudo": {}}
    if s.weather:
        w = s.weather
        src = w["source"] or (
            {"side": s.last_move_line[0], "name": s.last_move_line[1]}
            if s.last_move_line and not (w["from"] or "").startswith("ability:") else None)
        ticks = tracker.ticks_since(w["turn"], w["post"])
        if w["id"] in ("desolateland", "primordialsea", "deltastream"):
            duration = 0
        else:
            base = 8 if extends(src, WEATHER_ITEM.get(w["id"], "")) else 5
            if ticks >= base:
                base = 8
            duration = _remaining(base, ticks)
        field_spec["weather"] = {"id": w["id"], "duration": duration, "source": ref(src)}
    if s.terrain:
        t = s.terrain
        src = t["source"]
        ticks = tracker.ticks_since(t["turn"], t["post"])
        base = 8 if extends(src, TERRAIN_ITEM) else 5
        if ticks >= base:
            base = 8
        field_spec["terrain"] = {"id": t["id"], "duration": _remaining(base, ticks),
                                 "source": ref(src)}
    for pid, rec in s.pseudo.items():
        ticks = tracker.ticks_since(rec["turn"], rec["post"])
        field_spec["pseudo"][pid] = {
            "duration": _remaining(PSEUDO_DURATION[pid], ticks), "source": ref(rec["source"])}
    sides: dict[str, Any] = {}
    for side in SIDES:
        conds: dict[str, Any] = {}
        for cid, rec in s.conditions[side].items():
            if cid in STACKING:
                conds[cid] = {"layers": rec["layers"]}
            elif cid in PERMANENT_SIDE:
                conds[cid] = {"layers": 1}
            else:
                base, item, ext = SIDE_DURATION[cid]
                ticks = tracker.ticks_since(rec["turn"], rec["post"])
                if item and extends(rec.get("setter"), item):
                    base = ext
                if ticks >= base:
                    raise Unsupported(f"side_condition_duration_exhausted:{cid}")
                conds[cid] = {"duration": _remaining(base, ticks), "layers": 1}
        sides[side] = {
            "conditions": conds, "total_fainted": s.total_fainted[side],
            "tera_used": s.tera_used[side],
            "fainted_last_turn": s.faint_turn[side] == s.turn - 1,
        }
    return {
        "formatid": "gen9ou", "seed": seed, "prng": seed, "turn": s.turn,
        "request_state": "switch" if forced else "move",
        "switch_sides": switch_sides,
        "p1": {"name": p1_name, "team": own_team,
               "order": [m.name for m, _ in pairs1],
               "genders": [gender_of(e["details"]) for _, e in pairs1]},
        "p2": {"name": p2_name, "team": p2_sets if p2_sets is not None else
               p2_team_sets(order2, world)},
        "mons": mons, "sides": sides, "field": field_spec,
    }


def p2_team_sets(
    order2: list[Mon], world: dict[str, Any]
) -> list[dict[str, Any]]:
    """Showdown set objects for the opponent party (belief world over `order2`)."""
    from pokeai.cf.belief import validator_set

    out = []
    for i, (mon, w) in enumerate(zip(order2, world["mons"], strict=True)):
        roster = {
            "name": mon.name, "species": mon.base_species, "gender": mon.gender,
            "level": mon.level, "shiny": mon.shiny, "index": i,
        }
        out.append(validator_set(roster, w))
    return out


def roster_for(order2: list[Mon]) -> list[dict[str, Any]]:
    """Belief-sampler roster from public information (preview + reveals)."""
    return [{
        "index": i, "name": m.name, "species": m.base_species, "level": m.level,
        "gender": m.gender, "shiny": m.shiny,
        "legal_abilities": legal_abilities(m.base_species), "is_active": m.active,
        "fainted": m.fainted,
    } for i, m in enumerate(order2)]
