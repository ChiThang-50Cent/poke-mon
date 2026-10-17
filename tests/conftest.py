"""Small native-format two-POV battles for offline data-pipeline tests."""

import copy
import json
from dataclasses import fields
from pathlib import Path

import lz4.frame
import pytest
from metamon.interface import UniversalMove, UniversalPokemon, UniversalState

from pokeai.data.dataset import SPACE_VERSIONS


def make_move(name: str) -> dict:
    return UniversalMove(name, "normal", "physical", 40, 1.0, 0, 35, 35).__dict__


def make_pokemon(name: str, own: bool = False) -> dict:
    values = {field.name: 0 for field in fields(UniversalPokemon)}
    values.update(
        name=name,
        base_species=name,
        hp_pct=1.0,
        types="normal notype",
        item="unknownitem",
        ability="unknownability",
        lvl=100,
        status="nostatus",
        effect="noeffect",
        tera_type="notype",
        moves=[make_move("tackle")] if own else [],
    )
    return values


@pytest.fixture
def synthetic_run(tmp_path: Path) -> tuple[Path, Path]:
    run = tmp_path / "run"
    root = run / "trajectories"
    (root / "gen9ou").mkdir(parents=True)
    meta_root = run / "trajectory_meta" / "gen9ou"
    meta_root.mkdir(parents=True)
    human = tmp_path / "human"
    (human / "gen9ou").mkdir(parents=True)
    events = [
        ["", "switch", "p1a: Pikachu", "Pikachu, L100", "100/100"],
        ["", "switch", "p2a: Eevee", "Eevee, L100", "100/100"],
        ["", "move", "p1a: Pikachu", "Tackle", "p2a: Eevee"],
        ["", "move", "p2a: Eevee", "Tackle", "p1a: Pikachu"],
    ]
    entries = []
    for side, own, opponent, result in (
        ("p1", "pikachu", "eevee", "win"),
        ("p2", "eevee", "pikachu", "loss"),
    ):
        state = UniversalState(
            format="gen9ou",
            player_active_pokemon=UniversalPokemon(**make_pokemon(own, True)),
            opponent_active_pokemon=UniversalPokemon(**make_pokemon(opponent)),
            available_switches=[],
            player_prev_move=UniversalMove.blank_move(),
            opponent_prev_move=UniversalMove.blank_move(),
            opponents_remaining=1,
            player_conditions="noconditions",
            opponent_conditions="noconditions",
            weather="noweather",
            battle_field="nofield",
            forced_switch=False,
            battle_won=False,
            battle_lost=False,
            can_tera=False,
            opponent_teampreview=[opponent],
        ).to_dict()
        second = copy.deepcopy(state)
        second["opponent_active_pokemon"]["moves"] = [make_move("tackle")]
        terminal = copy.deepcopy(second)
        terminal["battle_won"] = result == "win"
        terminal["battle_lost"] = result == "loss"
        data = {"states": [state, second, terminal], "actions": [0, 0, -1]}
        stem = f"local-gen9ou-1_Unrated_{side}_vs_other_10-16-2026_{result.upper()}"
        file = root / "gen9ou" / f"{stem}.json.lz4"
        file.write_bytes(lz4.frame.compress(json.dumps(data).encode()))
        meta_path = meta_root / f"{stem}.meta.json"
        meta = {
            "schema_version": "1.0",
            "battle_id": "battle-gen9ou-1",
            "run_id": "run",
            "format": "gen9ou",
            "player_side": side,
            "policy_version": "synthetic-test",
            **SPACE_VERSIONS,
            "final_result": result,
            "n_steps": 2,
            "action_masks": [[1] + [0] * 12] * 2,
            "chosen_actions": [0, 0],
            "unsupported_reasons": [None, None],
            "team_file": "test-team",
            "decision_turns": [1, 2],
            "decision_event_counts": [2, 4],
            "public_events": events,
        }
        meta_path.write_text(json.dumps(meta))
        entries.append(
            {
                "file": str(file),
                "meta": str(meta_path),
                "battle_id": "battle-gen9ou-1",
                "player_side": side,
            }
        )
        (human / "gen9ou" / file.name).write_bytes(file.read_bytes())
    (human / "sample.json").write_text(json.dumps({"trajectory_count": 2}))
    (run / "manifest.json").write_text(
        json.dumps(
            {
                "trajectory_files": entries,
                "expected_trajectories": 2,
                "saved_trajectories": 2,
                "missing": [],
            }
        )
    )
    (run / "battles.jsonl").write_text(
        json.dumps(
            {
                "battle_id": "battle-gen9ou-1",
                "winner": "player",
                "error": None,
            }
        )
        + "\n"
    )
    return run, human
