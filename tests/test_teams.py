"""Check the fixed team pool and its local Showdown compatibility."""

import json
import subprocess
from pathlib import Path

import pytest
from metamon.env.wrappers import TeamSet

from pokeai.env.teams import load_teamset

ROOT = Path(__file__).resolve().parents[1]
TEAM_ROOT = ROOT / "data/teams"
FORMAT_ROOT = TEAM_ROOT / "gen9ou"


@pytest.fixture
def manifest() -> list[dict[str, object]]:
    return json.loads((FORMAT_ROOT / "manifest.json").read_text(encoding="utf-8"))


def test_manifest_matches_files(manifest: list[dict[str, object]]) -> None:
    filenames = [entry["file"] for entry in manifest]
    assert filenames
    assert len(filenames) == len(set(filenames))
    assert set(filenames) == {path.name for path in FORMAT_ROOT.glob("*.gen9ou_team")}
    assert sum(entry["source"] == "competitive" for entry in manifest) >= 10
    assert any(entry["source"] == "hl_05_26" for entry in manifest)
    for entry in manifest:
        assert str(entry["file"]).startswith(f"{entry['source']}-")


def test_every_team_has_six_species(manifest: list[dict[str, object]]) -> None:
    for entry in manifest:
        path = FORMAT_ROOT / str(entry["file"])
        parsed = TeamSet.parse_showdown_team(path.read_text(encoding="utf-8"))
        species = [mon.species or mon.nickname for mon in parsed]
        assert len(species) == 6, path.name
        assert all(species), path.name
        assert species == entry["species"], path.name


def test_load_teamset(manifest: list[dict[str, object]]) -> None:
    teamset = load_teamset(TEAM_ROOT)
    assert isinstance(teamset, TeamSet)
    assert len(teamset.team_files) == len(manifest)
    assert {Path(path).name for path in teamset.team_files} == {
        entry["file"] for entry in manifest
    }
    assert teamset.yield_team()


def test_missing_team_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="build_teamset.py"):
        load_teamset(tmp_path / "missing")


def test_empty_team_directory(tmp_path: Path) -> None:
    (tmp_path / "gen9ou").mkdir()
    with pytest.raises(ValueError, match="No Gen9 OU teams"):
        load_teamset(tmp_path)


@pytest.mark.server
def test_team_passes_showdown_validation(manifest: list[dict[str, object]]) -> None:
    showdown = ROOT / "third_party/metamon/server/pokemon-showdown/pokemon-showdown"
    team = FORMAT_ROOT / str(manifest[0]["file"])
    result = subprocess.run(
        ["node", str(showdown), "validate-team", "gen9ou"],
        input=team.read_text(encoding="utf-8"),
        text=True,
        capture_output=True,
        timeout=60,
        cwd=showdown.parent,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
