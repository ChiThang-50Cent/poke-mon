"""CF1.1 belief sampler b(H | I_t) over the opponent's unrevealed team details.

Input = ONLY what the learner has observed: the learner (p1) channel of the battle log
(`Sim.public_log`) and the public roster (species / name / level / gender / legal ability
list from the dex). The sampler never calls `Sim.p2_full` (true hidden sets); that call is
reserved for QA and oracle labels. Everything unrevealed is drawn from Metamon's
NaiveUsagePredictor distributions (Smogon gen9ou usage, marginal per field, reimplemented
with a seeded numpy Generator because the vendor class uses global ``np.random``).

Inference rules APPLIED (history constraints, all from the learner channel):
  * revealed moves (`|move|` without `[from]` except lockedmove) stay in the moveset
  * item: `-item`, `-enditem` (consumed/knocked/stolen => gone), `[from] item: X`,
    `-activate|..|item: X` reveal / remove; revealed item or "gone" is frozen
  * ability: `-ability`, `-activate|..|ability: X`, `[from] ability: X` (owner = `[of]`
    else the subject); ambiguous swap/copy abilities freeze the TRUE ability (conservative)
  * Tera: `-terastallize` freezes the Tera type; unrevealed Tera type is sampled
  * Choice rule: two different moves in one stint with no item event => no Choice item
  * Assault Vest rule: a revealed status move => no Assault Vest
  * public HP % / status / boosts / volatiles / fainted / species / preview are untouched
    (the world is the true battle state with only hidden fields replaced)
Knowingly SKIPPED: damage-calc and speed-order inference, teammate/team-level
correlations (marginal sampling => unlikely move/item combinations possible),
Focus-Sash / unrevealed-consumed-item inference, forme-hiding species (Zacian etc. read
from the sim roster), onStart effects of newly assigned unannounced abilities,
Substitute HP / other hidden volatile values are kept from the true state.
"""

from __future__ import annotations

import datetime
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

CHOICE_ITEMS = {"Choice Band", "Choice Specs", "Choice Scarf"}
AMBIGUOUS_ABILITIES = {
    "Trace", "Pickpocket", "Magician", "Symbiosis", "Receiver", "Power of Alchemy",
    "Mummy", "Lingering Aroma", "Wandering Spirit", "Skill Swap", "Neutralizing Gas",
}
AMBIGUOUS_MOVES = {
    "Skill Swap", "Role Play", "Worry Seed", "Simple Beam", "Gastro Acid", "Entrainment",
    "Doodle", "Transform", "Mimic", "Imprison", "Roar of Time",
}
USAGE_START = datetime.date(2026, 1, 1)
USAGE_END = datetime.date(2026, 4, 1)
USAGE_RANK = 1500
NATURES_DEFAULT = "Hardy"
OGERPON_TERA = {"Wellspring": "Water", "Hearthflame": "Fire", "Cornerstone": "Rock"}


def to_id(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def move_key(name: str) -> str:
    key = to_id(name)
    return "hiddenpower" if key.startswith("hiddenpower") else key


def _ident(token: str) -> tuple[str, str] | None:
    """`p2a: Name` -> ('p2', 'Name'); None for non-p2 or malformed."""
    match = re.match(r"^(p[12])[a-z]?: (.*)$", token.strip())
    return (match.group(1), match.group(2)) if match else None


@dataclass
class MonKnowledge:
    name: str
    moves: list[str] = field(default_factory=list)
    status_move_seen: bool = False
    item_state: str = "unknown"  # unknown | known | gone
    item: str | None = None
    ability: str | None = None
    freeze_ability: bool = False
    tera: str | None = None
    stint_moves: set[str] = field(default_factory=set)
    stint_item_event: bool = False
    choice_excluded: bool = False
    move_categories: dict[str, str] = field(default_factory=dict)


def parse_knowledge(lines: list[str], roster: list[dict[str, Any]]) -> dict[str, MonKnowledge]:
    """Fold the learner channel into per-opponent-Pokemon knowledge (by roster name)."""
    names = {m["name"] for m in roster}
    know = {name: MonKnowledge(name) for name in names}

    def mon(token: str) -> MonKnowledge | None:
        ident = _ident(token)
        if ident is None or ident[0] != "p2" or ident[1] not in know:
            return None
        return know[ident[1]]

    def owner_of(parts: list[str], subject: str) -> MonKnowledge | None:
        for tail in parts:
            if tail.startswith("[of] "):
                return mon(tail[5:])
        return mon(subject)

    def item_event(k: MonKnowledge, state: str, item: str | None) -> None:
        k.item_state, k.stint_item_event = state, True
        if item:
            k.item = item

    for line in lines:
        if not line.startswith("|"):
            continue
        parts = line.split("|")[1:]
        kind = parts[0]
        args = parts[1:]
        tags = [a for a in args if a.startswith("[")]
        frm = next((t[7:].strip() for t in tags if t.startswith("[from]")), None)
        if kind in ("switch", "drag") and args:
            k = mon(args[0])
            if k:
                k.stint_moves, k.stint_item_event = set(), False
        elif kind == "move" and len(args) >= 2:
            k = mon(args[0])
            own = frm is None or frm == "lockedmove"
            if k and own and args[1] != "Struggle":
                if move_key(args[1]) not in {move_key(x) for x in k.moves}:
                    k.moves.append(args[1])
                if frm is None:
                    k.stint_moves.add(move_key(args[1]))
                    if len(k.stint_moves) >= 2 and not k.stint_item_event:
                        k.choice_excluded = True
            if k and frm and frm.startswith(("move:", "ability:")):
                name = frm.split(":", 1)[1].strip()
                if name in AMBIGUOUS_ABILITIES | AMBIGUOUS_MOVES:
                    k.freeze_ability = True
        elif kind == "-item" and len(args) >= 2:
            k = mon(args[0])
            if k:
                item_event(k, "known", args[1])
        elif kind == "-enditem" and len(args) >= 2:
            k = mon(args[0])
            if k:
                item_event(k, "gone", args[1])
        elif kind == "-ability" and len(args) >= 2:
            k = mon(args[0])
            if k:
                k.ability = args[1]
                if frm and frm.startswith("ability:") and (
                    frm.split(":", 1)[1].strip() in AMBIGUOUS_ABILITIES
                ):
                    k.freeze_ability = True
        elif kind == "-endability" and args:
            k = mon(args[0])
            if k:
                k.freeze_ability = True
        elif kind == "-terastallize" and len(args) >= 2:
            k = mon(args[0])
            if k:
                k.tera = args[1]
        elif kind == "-activate" and len(args) >= 2:
            k = mon(args[0])
            detail = args[1]
            if k and detail.startswith("item: "):
                k.item_state, k.item = "known", detail[6:]
            elif k and detail.startswith("ability: "):
                k.ability = detail[9:]
            if detail.startswith("move: ") and detail[6:] in AMBIGUOUS_MOVES:
                for token in [args[0], *[t[5:] for t in tags if t.startswith("[of] ")]]:
                    kk = mon(token)
                    if kk:
                        kk.freeze_ability = True
        if frm and args:
            if frm.startswith("item:") and kind != "-enditem":
                k = owner_of(tags, args[0])
                if k:
                    k.item_state, k.item = "known", frm.split(":", 1)[1].strip()
            elif frm.startswith("ability:"):
                name = frm.split(":", 1)[1].strip()
                if name in AMBIGUOUS_ABILITIES:
                    for token in [args[0], *[t[5:] for t in tags if t.startswith("[of] ")]]:
                        kk = mon(token)
                        if kk:
                            kk.freeze_ability = True
                else:
                    k = owner_of(tags, args[0])
                    if k:
                        k.ability = name
    return know


def annotate_categories(know: dict[str, MonKnowledge], sim: Any) -> None:
    """Mark Pokemon that revealed a status move (dex category, public knowledge)."""
    names = sorted({m for k in know.values() for m in k.moves})
    if not names:
        return
    category = {x["id"]: x["category"] for x in sim.move_info(names)}
    for k in know.values():
        k.status_move_seen = any(
            category.get(to_id(m)) == "Status" for m in k.moves
        )


class UsagePrior:
    """Per-species marginal usage distributions (Smogon gen9ou, Jan-Apr 2026, rank 1500)."""

    def __init__(self) -> None:
        from metamon.backend.team_prediction.usage_stats import get_usage_stats

        self.stat = get_usage_stats(
            "gen9ou", USAGE_START, USAGE_END, rank=USAGE_RANK
        )
        self._cache: dict[str, dict[str, dict[str, float]]] = {}

    def entry(self, species: str) -> dict[str, dict[str, float]]:
        if species not in self._cache:
            raw = self.stat[species]
            clean = {}
            for key in ("abilities", "items", "spreads", "moves", "tera_types"):
                dist = {
                    k: float(v) for k, v in dict(raw[key]).items()
                    if k not in ("Other", "Nothing", "") and v > 0
                }
                clean[key] = dist
            self._cache[species] = clean
        return self._cache[species]


def _draw(rng: np.random.Generator, dist: dict[str, float]) -> str | None:
    if not dist:
        return None
    keys = list(dist)
    p = np.array([dist[k] for k in keys], float)
    return keys[int(rng.choice(len(keys), p=p / p.sum()))]


def field_distribution(
    prior: UsagePrior, mon: dict[str, Any], k: MonKnowledge, name: str
) -> dict[str, float]:
    """Normalised constrained marginal for 'item' / 'ability' / 'tera_types'."""
    entry = prior.entry(mon["species"])
    if name == "ability":
        dist = {a: w for a, w in entry["abilities"].items() if a in mon["legal_abilities"]}
        if not dist:
            dist = {a: 1.0 for a in mon["legal_abilities"]}
    elif name == "item":
        dist = dict(entry["items"])
        if k.choice_excluded:
            dist = {i: w for i, w in dist.items() if i not in CHOICE_ITEMS}
        if k.status_move_seen:
            dist.pop("Assault Vest", None)
    else:
        dist = dict(entry["tera_types"])
    total = sum(dist.values())
    return {a: w / total for a, w in dist.items()} if total else {}


def parse_spread(spread: str) -> tuple[str, dict[str, int]]:
    nature, evs = spread.split(":")
    values = [int(x) for x in evs.split("/")]
    if not any(values):
        values = [84] * 6
    keys = ("hp", "atk", "def", "spa", "spd", "spe")
    return nature.strip(), dict(zip(keys, values, strict=True))


def sample_mon(
    prior: UsagePrior, mon: dict[str, Any], k: MonKnowledge, rng: np.random.Generator
) -> dict[str, Any]:
    entry = prior.entry(mon["species"])
    nature, evs = parse_spread(_draw(rng, entry["spreads"]) or f"{NATURES_DEFAULT}:84/84/84/84/84/84")
    moves = list(k.moves)
    pool = {
        m: w for m, w in entry["moves"].items()
        if move_key(m) not in {move_key(x) for x in moves} and not m.startswith("Hidden Power")
    }
    while len(moves) < 4 and pool:
        choice = _draw(rng, pool)
        moves.append(choice)
        pool.pop(choice)
    item_dist = field_distribution(prior, mon, k, "item")
    keep_item = k.item_state in ("known", "gone")
    item = k.item if keep_item and k.item else (_draw(rng, item_dist) or "")
    keep_ability = k.freeze_ability or k.ability is not None
    ability_dist = field_distribution(prior, mon, k, "ability")
    # a frozen (copied/swapped) current ability is not the species' own: the sim keeps
    # the true value, the validator only sees a legal placeholder
    # (Tera-forme abilities such as Embody Aspect are likewise not set abilities)
    known_legal = (
        k.ability and not k.freeze_ability and k.ability in mon["legal_abilities"]
    )
    ability = k.ability if known_legal else _draw(rng, ability_dist)
    keep_tera = k.tera is not None
    tera = k.tera if keep_tera else _draw(rng, field_distribution(prior, mon, k, "tera_types"))
    if not keep_tera and mon["species"].startswith("Ogerpon"):
        tera = OGERPON_TERA.get(mon["species"].split("-")[1] if "-" in mon["species"] else "", "Grass")
    return {
        "index": mon["index"], "moves": moves, "item": item, "keep_item": keep_item,
        "ability": ability, "keep_ability": keep_ability, "nature": nature, "evs": evs,
        "ivs": dict.fromkeys(("hp", "atk", "def", "spa", "spd", "spe"), 31),
        "tera_type": tera, "keep_tera": keep_tera,
    }


def validator_set(mon: dict[str, Any], sampled: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": mon["name"], "species": mon["species"], "gender": mon["gender"] or "N",
        "level": mon["level"], "shiny": mon["shiny"], "item": sampled["item"],
        "ability": sampled["ability"], "moves": sampled["moves"],
        "nature": sampled["nature"], "evs": sampled["evs"], "ivs": sampled["ivs"],
        "teraType": sampled["tera_type"],
    }


@dataclass
class WorldStats:
    worlds: int = 0
    mon_attempts: int = 0
    mon_rejections: int = 0
    team_attempts: int = 0
    team_rejections: int = 0
    rejection_reasons: dict[str, int] = field(default_factory=dict)


class BeliefSampler:
    """Draw complete legal opponent worlds from public information only."""

    def __init__(self, sim: Any, prior: UsagePrior | None = None) -> None:
        self.sim = sim
        self.prior = prior or UsagePrior()
        self.stats = WorldStats()

    def sample_world(
        self, roster: list[dict[str, Any]], know: dict[str, MonKnowledge],
        rng: np.random.Generator, max_mon_attempts: int = 200,
    ) -> dict[str, Any]:
        for _ in range(50):
            self.stats.team_attempts += 1
            mons = []
            for mon in roster:
                k = know[mon["name"]]
                for _attempt in range(max_mon_attempts):
                    sampled = sample_mon(self.prior, mon, k, rng)
                    self.stats.mon_attempts += 1
                    errors = self.sim.validate([validator_set(mon, sampled)])
                    if not errors:
                        break
                    self.stats.mon_rejections += 1
                    reason = re.sub(r"[^:]*'s ", "", errors[0])[:80]
                    self.stats.rejection_reasons[reason] = (
                        self.stats.rejection_reasons.get(reason, 0) + 1
                    )
                else:
                    raise RuntimeError(f"no legal set found for {mon['species']}")
                mons.append(sampled)
            team = [validator_set(m, s) for m, s in zip(roster, mons, strict=True)]
            errors = self.sim.validate(team)
            if not errors:
                self.stats.worlds += 1
                return {"mons": mons}
            self.stats.team_rejections += 1
        raise RuntimeError("no legal team found")


def check_public_facts(
    know: dict[str, MonKnowledge], full: list[dict[str, Any]]
) -> list[str]:
    """Violations of observed facts by a world's true-state dump (empty = consistent)."""
    bad = []
    for mon in full:
        k = know[mon["name"]]
        have = {move_key(m) for m in mon["current_moves"]}
        for move in k.moves:
            if move_key(move) not in have:
                bad.append(f"{mon['name']}: revealed move {move} missing")
        if k.item_state == "known" and k.item and to_id(k.item) != mon["item"]:
            bad.append(f"{mon['name']}: item {k.item} != {mon['item']}")
        if k.item_state == "gone" and mon["item"] != "":
            bad.append(f"{mon['name']}: item should be gone, has {mon['item']}")
        if k.ability and not k.freeze_ability and to_id(k.ability) != mon["ability"]:
            bad.append(f"{mon['name']}: ability {k.ability} != {mon['ability']}")
        if k.tera and mon["tera_type"] != k.tera:
            bad.append(f"{mon['name']}: tera {k.tera} != {mon['tera_type']}")
        if k.choice_excluded and to_id("Choice") in mon["item"][:6] and (
            k.item_state == "unknown"
        ):
            bad.append(f"{mon['name']}: Choice item despite two-move stint")
    return bad
