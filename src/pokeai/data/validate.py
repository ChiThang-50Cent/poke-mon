"""Fail-closed schema, legality, terminal consistency and public-view QA."""

from __future__ import annotations

import copy
import json
from collections import Counter
from dataclasses import fields
from pathlib import Path
from typing import Any

import lz4.frame
import numpy as np
from metamon.backend.replay_parser.str_parsing import (
    clean_name,
    clean_no_numbers,
    move_name,
)
from metamon.interface import DefaultObservationSpace, UniversalState
from poke_env.data import GenData

from pokeai.data.dataset import (
    TrajectoryDataset,
    default_human_root,
    human_entries,
    local_entries,
    resolve_run_path,
)

REQUIRED_META = {
    "schema_version",
    "battle_id",
    "run_id",
    "format",
    "player_side",
    "policy_version",
    "observation_space_version",
    "action_space",
    "reward_function",
    "final_result",
    "n_steps",
    "action_masks",
    "chosen_actions",
    "unsupported_reasons",
    "team_file",
    "decision_turns",
    "public_events",
    "decision_event_counts",
}
UNKNOWN = {
    "",
    "unknownitem",
    "unknownability",
    "notype",
    "noteratype",
    "noitem",
    "noability",
    "nomove",
    "nothing",
    "none",
}


def read_native(path: Path) -> dict[str, Any]:
    if path.name.endswith(".lz4"):
        return json.loads(lz4.frame.decompress(path.read_bytes()))
    return json.loads(path.read_text())


def schema_errors(data: dict[str, Any], meta: dict[str, Any]) -> list[str]:
    errors = []
    absent = REQUIRED_META - meta.keys()
    if absent:
        errors.append(f"Missing metadata fields: {sorted(absent)}")
    if meta.get("schema_version") != "1.0" or meta.get("format") != "gen9ou":
        errors.append("Unsupported schema or format")
    if meta.get("player_side") not in {"p1", "p2"}:
        errors.append("Invalid player_side")
    if meta.get("final_result") not in {"win", "loss", "tie"}:
        errors.append("Invalid final_result")
    for key, value in (
        ("action_space", "DefaultActionSpace"),
        ("reward_function", "DefaultShapedReward"),
    ):
        if meta.get(key) != value:
            errors.append(f"Invalid {key}")
    states, actions = data.get("states", []), data.get("actions", [])
    count = len(actions) - 1
    if count < 1 or len(states) != len(actions) or actions[-1:] != [-1]:
        errors.append("Native format requires N+1 states/actions and terminal -1")
    if meta.get("n_steps") != count:
        errors.append("n_steps differs from native decisions")
    for key in (
        "action_masks",
        "chosen_actions",
        "unsupported_reasons",
        "decision_turns",
        "decision_event_counts",
    ):
        if len(meta.get(key, [])) != max(0, count):
            errors.append(f"Unaligned {key}")
    for mask in meta.get("action_masks", []):
        if len(mask) != 13 or any(
            type(value) is not int or value not in (0, 1) for value in mask
        ):
            errors.append("Mask must contain exactly 13 binary integers")
    if actions[:-1] != meta.get("chosen_actions"):
        errors.append("Native and metadata actions differ")
    for action, reason in zip(
        meta.get("chosen_actions", []), meta.get("unsupported_reasons", [])
    ):
        if type(action) is not int or action < -1 or action > 12:
            errors.append("Action out of range")
        if (action == -1) != bool(reason):
            errors.append("Unsupported reason must accompany exactly missing actions")
    allowed = {field.name for field in fields(UniversalState)}
    for state in states:
        if not isinstance(state, dict) or set(state) != allowed:
            errors.append("Unexpected/missing observation state fields")
            continue
        try:
            UniversalState.from_dict(copy.deepcopy(state))
        except (TypeError, ValueError, KeyError) as exc:
            errors.append(f"Invalid UniversalState: {exc}")
    return errors


def invalid_actions(data: dict[str, Any], meta: dict[str, Any]) -> int:
    invalid = 0
    for index, action in enumerate(data.get("actions", [])[:-1]):
        masks = meta.get("action_masks", [])
        if action == -1:
            continue
        if (
            type(action) is not int
            or not 0 <= action < 13
            or index >= len(masks)
            or len(masks[index]) != 13
            or masks[index][action] != 1
        ):
            invalid += 1
    return invalid


def _public_knowledge(events: list[list[str]], side: str) -> dict[str, dict[str, set]]:
    """Reconstruct reveal sets from protocol events, not private team sets."""
    known: dict[str, dict[str, set]] = {}
    aliases: dict[str, str] = {}

    def pokemon(ident: str) -> dict[str, set] | None:
        species = aliases.get(ident)
        if species is None:
            return None
        return known[species]

    def reveal(field: str, owner: str, value: str) -> None:
        holder = pokemon(owner)
        if holder is not None:
            holder[field].add(clean_no_numbers(value))

    for event in events:
        if len(event) < 2:
            continue
        kind = event[1]
        ident = event[2] if len(event) > 2 else ""
        if kind in {
            "switch", "drag", "replace", "detailschange", "-formechange",
        } and ident.startswith(side):
            species = clean_name(event[3].split(",")[0])
            if (
                kind in {"detailschange", "-formechange", "replace"}
                and ident in aliases
            ):
                known[species] = known[aliases[ident]]
            aliases[ident] = species
            known.setdefault(
                species, {k: set() for k in ("moves", "item", "ability", "tera_type")}
            )
        entry = pokemon(ident)
        if entry is not None and len(event) > 3:
            if kind == "move":
                entry["moves"].add(move_name(event[3]))
            elif kind in {"-item", "-enditem"}:
                entry["item"].add(clean_no_numbers(event[3]))
            elif kind == "-ability":
                entry["ability"].add(clean_no_numbers(event[3]))
            elif kind == "-terastallize":
                entry["tera_type"].add(clean_no_numbers(event[3]))
        source = next(
            (part[5:] for part in event[3:] if part.startswith("[of] ")), ident
        )
        # Skill Swap publicly names the abilities each participant receives.
        # Both revelations belong to this exact public protocol prefix.
        if (
            kind == "-activate" and len(event) > 5
            and event[3] == "Skill Swap" and source != ident
        ):
            reveal("ability", ident, event[4])
            reveal("ability", source, event[5])

        for part in event[3:]:
            if part.startswith("[from] ability: "):
                value = part.removeprefix("[from] ability: ")
                ability = clean_no_numbers(value)
                # Item theft names the donor in [of], not the ability holder.
                # Its silent -enditem companion does not identify the thief.
                if ability in {"pickpocket", "magician"}:
                    if kind == "-item":
                        reveal("ability", ident, value)
                    continue
                if kind == "-ability" and ability == "trace":
                    reveal("ability", ident, value)
                    reveal("ability", source, event[3])
                else:
                    # Absorption heals name the attacker in [of]. Reactive
                    # damage (Rough Skin/Rocky Helmet) instead names the holder.
                    owner = ident if kind == "-heal" else source
                    reveal("ability", owner, value)
            elif part.startswith("[from] item: "):
                owner = ident if kind == "-heal" else source
                reveal("item", owner, part.removeprefix("[from] item: "))
            elif part.startswith("ability: "):
                owner = source if kind == "-block" else ident
                value = part.removeprefix("ability: ")
                reveal("ability", owner, value)
                # Mummy/Lingering Aroma both reveal the source and replace the
                # contacted Pokémon's ability, without emitting -ability.
                if (
                    kind == "-activate"
                    and clean_no_numbers(value) in {"mummy", "lingeringaroma"}
                    and len(event) > 4
                ):
                    reveal("ability", event[4], value)
            elif part.startswith("item: "):
                owner = source if kind == "-block" else ident
                reveal("item", owner, part.removeprefix("item: "))
    return known


def leakage_errors(
    data: dict[str, Any],
    meta: dict[str, Any],
    other_data: dict[str, Any] | None = None,
    other_meta: dict[str, Any] | None = None,
) -> list[str]:
    errors = []
    states = data.get("states", [])[:-1]
    counts = meta.get("decision_event_counts", [])
    events = meta.get("public_events", [])
    opponent = "p2" if meta.get("player_side") == "p1" else "p1"
    if len(counts) != len(states):
        return ["Missing decision-time public evidence"]
    previous = 0
    space = DefaultObservationSpace()
    space.reset()
    for index, (state, limit) in enumerate(zip(states, counts)):
        if not isinstance(limit, int) or not previous <= limit <= len(events):
            errors.append(f"{index}: invalid event prefix")
            continue
        previous = limit
        # The terminal flags exist in the storage schema but must be false at
        # every decision, and may never enter the encoded policy observation.
        forbidden = {"final_result", "winner", "reward", "battle_result", "done"}
        if (
            forbidden.intersection(state)
            or state.get("battle_won")
            or state.get("battle_lost")
        ):
            errors.append(f"{index}: terminal result in decision state")
        if any(
            "opponent" in key and ("reserve" in key or "switches" in key)
            for key in state
        ):
            errors.append(f"{index}: opponent reserves in observation")
        knowledge = _public_knowledge(events[:limit], opponent)
        for field, side in (
            ("opponent_prev_move", opponent),
            ("player_prev_move", meta["player_side"]),
        ):
            previous_move = move_name(state[field]["name"])
            public_moves = {
                move_name(event[3])
                for event in events[:limit]
                if len(event) > 3 and event[1] == "move" and event[2].startswith(side)
            }
            if previous_move not in UNKNOWN and previous_move not in public_moves:
                errors.append(f"{index}: future/unrevealed {field} {previous_move}")
        # Cross-POV states constrain species identity; own private move sets are
        # never treated as revelation. Actual public 'move' events remain truth
        # even when a chosen move was prevented by flinch/sleep/paralysis.
        if other_data is not None and other_meta is not None:
            turn = meta["decision_turns"][index]
            own_roster = {
                clean_name(p["name"])
                for s, t in zip(other_data["states"], other_meta["decision_turns"])
                if t <= turn
                for p in [s["player_active_pokemon"], *s["available_switches"]]
            }
            active = clean_name(state["opponent_active_pokemon"]["name"])
            # Public switches remain authoritative when one side never receives
            # another decision (KO/pivot), and Illusion may change public species.
            if active not in own_roster and active not in knowledge:
                errors.append(
                    f"{index}: opponent species absent from public/POV history"
                )
        pokemon = state.get("opponent_active_pokemon", {})
        name = clean_name(pokemon.get("name", ""))
        evidence = knowledge.get(name)
        if evidence is not None and other_data is not None and other_meta is not None:
            turn = meta["decision_turns"][index]
            # A private chosen move alone is not a reveal (sleep/flinch). For
            # choices whose execution is publicly confirmed, recover the move
            # from the opponent's own action slot and state, independently of
            # the current observation. Public called/copied moves are retained.
            chosen_moves = set()
            for own_state, action, own_turn in zip(
                other_data["states"],
                other_data["actions"][:-1],
                other_meta["decision_turns"],
            ):
                if own_turn > turn or action < 0:
                    continue
                own = own_state["player_active_pokemon"]
                slot = action - 9 if action >= 9 else action
                moves = sorted(own["moves"], key=lambda move: move_name(move["name"]))
                if 0 <= slot < min(4, len(moves)) and (action < 4 or action >= 9):
                    value = move_name(moves[slot]["name"])
                    if value in evidence["moves"]:
                        chosen_moves.add(value)
            called_moves = {
                move_name(event[3])
                for event in events[:limit]
                if len(event) > 3
                and event[1] == "move"
                and event[2].startswith(opponent)
                and any(part.startswith("[from]") for part in event[4:])
            }
            evidence = dict(
                evidence,
                moves=evidence["moves"] & (chosen_moves | called_moves | {"struggle"}),
            )
        if evidence is None:
            errors.append(f"{index}: opponent active not publicly revealed: {name}")
            continue
        for move in pokemon.get("moves", []):
            value = move_name(move["name"])
            if value not in UNKNOWN and value not in evidence["moves"]:
                errors.append(f"{index}: unrevealed opponent move {value}")
        for field in ("item", "ability", "tera_type"):
            value = clean_no_numbers(pokemon.get(field, ""))
            dex = GenData.from_gen(9).pokedex.get(name, {})
            abilities = {clean_no_numbers(a) for a in dex.get("abilities", {}).values()}
            if field == "ability" and len(abilities) == 1 and value in abilities:
                continue
            # Publicly known forms imply their required items and Tera types,
            # just like a species with only one possible ability. poke-env's
            # dex uses forceTeraType; the simulator uses requiredTeraType.
            required = {
                "item": dex.get("requiredItem"),
                "tera_type": dex.get("forceTeraType", dex.get("requiredTeraType")),
            }.get(field)
            if required and value == clean_no_numbers(required):
                continue
            if value not in UNKNOWN and value not in evidence[field]:
                errors.append(f"{index}: unrevealed opponent {field} {value}")
        try:
            observation = space.state_to_obs(
                UniversalState.from_dict(copy.deepcopy(state))
            )
            if forbidden.intersection(observation):
                errors.append(f"{index}: result field in encoded observation")
            text = str(observation.get("text", "")).lower()
            if any(
                token in text for token in ("final_result", "battle_result", "winner")
            ):
                errors.append(f"{index}: encoded terminal result")
        except (TypeError, ValueError, KeyError) as exc:
            errors.append(f"{index}: invalid observation: {exc}")
    return errors


def validate_run(run: Path, human_root: Path | None = None) -> dict[str, Any]:
    """Return a machine-readable report; every observed failure affects status."""
    report: dict[str, Any] = {
        "stage": "W2",
        "format": "gen9ou",
        "completed_battles": 0,
        "expected_player_trajectories": 0,
        "saved_player_trajectories": 0,
        "total_decision_steps": 0,
        "invalid_action_count": 0,
        "missing_observation_count": 0,
        "terminal_result_mismatches": 0,
        "hidden_information_leakage_failures": 0,
        "schema_validation": "FAIL",
        "dataset_load_test": "FAIL",
        "overall_status": "FAIL",
        "human_replay_sample": {},
        "unsupported_action_counts": {},
        "errors": [],
    }
    errors = report["errors"]
    try:
        manifest = json.loads((run / "manifest.json").read_text())
        battles = [
            json.loads(line)
            for line in (run / "battles.jsonl").read_text().splitlines()
            if line
        ]
    except (OSError, ValueError) as exc:
        errors.append(str(exc))
        return report
    completed = {
        b["battle_id"]: b
        for b in battles
        if b.get("error") is None and b.get("winner") is not None
    }
    report["completed_battles"] = len(completed)
    report["expected_player_trajectories"] = 2 * len(battles)
    if len(completed) != len(battles) or not battles:
        errors.append("Run contains incomplete battles or no battles")
    if manifest.get("missing") or manifest.get("expected_trajectories") != 2 * len(
        battles
    ):
        errors.append("Manifest reports missing trajectories or wrong expected count")
    if manifest.get("saved_trajectories") != len(manifest.get("trajectory_files", [])):
        errors.append("Manifest saved count differs from trajectory entries")
    loaded: dict[tuple[str, str], tuple[dict, dict]] = {}
    rows = []
    schema_ok = True
    unsupported: Counter[str] = Counter()
    for entry in manifest.get("trajectory_files", []):
        try:
            data = read_native(resolve_run_path(run, entry["file"]))
            meta = json.loads(resolve_run_path(run, entry["meta"]).read_text())
            issues = schema_errors(data, meta)
            if issues:
                schema_ok = False
                errors.extend(f"{entry['file']}: {issue}" for issue in issues)
            key = (entry["battle_id"], entry["player_side"])
            if key in loaded:
                schema_ok = False
                errors.append(f"Duplicate POV: {key}")
            if (meta.get("battle_id"), meta.get("player_side")) != key:
                schema_ok = False
                errors.append(f"Manifest/sidecar identity differs: {key}")
            loaded[key] = (data, meta)
            count = max(0, len(data.get("actions", [])) - 1)
            report["total_decision_steps"] += count
            report["invalid_action_count"] += invalid_actions(data, meta)
            report["missing_observation_count"] += sum(
                not s for s in data.get("states", [])[:-1]
            ) + max(0, count + 1 - len(data.get("states", [])))
            unsupported.update(
                reason for reason in meta.get("unsupported_reasons", []) if reason
            )
            battle = completed.get(key[0])
            if battle is None:
                errors.append(f"Trajectory has no completed battle: {key}")
            else:
                winner = battle["winner"]
                expected = (
                    "tie"
                    if winner == "tie"
                    else ("win" if (winner == "player") == (key[1] == "p1") else "loss")
                )
                terminal = data["states"][-1]
                actual = (
                    "win"
                    if terminal.get("battle_won")
                    else "loss"
                    if terminal.get("battle_lost")
                    else "tie"
                )
                if meta.get("final_result") != expected or actual != expected:
                    report["terminal_result_mismatches"] += 1
        except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
            schema_ok = False
            errors.append(f"{entry}: {exc}")
    report["saved_player_trajectories"] = len(loaded)
    for battle in completed:
        for side in ("p1", "p2"):
            if (battle, side) not in loaded:
                errors.append(f"Missing POV: {battle} {side}")
                schema_ok = False
    for (battle, side), (data, meta) in loaded.items():
        other = loaded.get((battle, "p2" if side == "p1" else "p1"))
        try:
            issues = leakage_errors(data, meta, *(other or (None, None)))
            report["hidden_information_leakage_failures"] += len(issues)
            errors.extend(f"{battle}/{side}: {issue}" for issue in issues)
        except (TypeError, ValueError, KeyError, IndexError) as exc:
            report["hidden_information_leakage_failures"] += 1
            errors.append(f"{battle}/{side}: leakage check failed: {exc}")
    try:
        rows = local_entries(run)
        local = TrajectoryDataset(rows)
        if len(local) != len(loaded) or not len(local):
            raise ValueError("Local dataset size mismatch or empty dataset")
        for sample in local:
            n = len(sample["actions"])
            if n == 0 or sample["action_masks"].shape != (n, 13):
                raise ValueError("Invalid dataset masks")
            if any(len(v) != n for v in sample["observations"].values()):
                raise ValueError("Dataset observation/action misalignment")
            if not np.isfinite(sample["rewards"]).all() or not sample["dones"][-1]:
                raise ValueError("Invalid rewards/dones")
            sign = np.sign(sample["rewards"][-1])
            expected_sign = {"win": 1, "loss": -1, "tie": 0}[sample["battle_result"]]
            # Ties may have nonzero HP/status shaping, but never terminal +/-100.
            if (expected_sign and sign != expected_sign) or (
                not expected_sign and abs(sample["rewards"][-1]) >= 90
            ):
                report["terminal_result_mismatches"] += 1
        root = human_root or default_human_root()
        stats = json.loads((root / "sample.json").read_text())
        human = TrajectoryDataset(human_entries(root))
        if not len(human):
            raise ValueError("Human sample is empty")
        decisions = missing = 0
        for sample in human:
            decisions += len(sample["actions"])
            missing += int(sample["missing"].sum())
            if sample["action_masks"].shape != (len(sample["actions"]), 13):
                raise ValueError("Invalid human action masks")
        report["human_replay_sample"] = dict(
            stats, decision_steps=decisions, missing_actions=missing, load_status="PASS"
        )
        report["dataset_load_test"] = "PASS"
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        IndexError,
        AssertionError,
    ) as exc:
        errors.append(f"Dataset load failed: {exc}")
    report["schema_validation"] = "PASS" if schema_ok else "FAIL"
    report["unsupported_action_counts"] = dict(unsupported)
    if (
        not errors
        and report["schema_validation"] == "PASS"
        and report["dataset_load_test"] == "PASS"
        and report["saved_player_trajectories"]
        == report["expected_player_trajectories"]
        and not any(
            report[key]
            for key in (
                "invalid_action_count",
                "missing_observation_count",
                "terminal_result_mismatches",
                "hidden_information_leakage_failures",
            )
        )
    ):
        report["overall_status"] = "PASS"
    return report
