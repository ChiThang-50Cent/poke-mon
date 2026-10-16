"""Load the fixed, locally validated Gen9 OU team pool."""

from pathlib import Path

from metamon.env.wrappers import TeamSet


def load_teamset(path: str | Path = "data/teams") -> TeamSet:
    """Load Gen9 OU teams, failing before battle startup if none are available."""
    root = Path(path)
    if not root.is_dir():
        raise FileNotFoundError(
            f"Team directory does not exist: {root}. "
            "Run `uv run python scripts/build_teamset.py` first."
        )
    teamset = TeamSet(team_file_dir=str(root), battle_format="gen9ou")
    if not teamset.team_files:
        raise ValueError(
            f"No Gen9 OU teams found in {root}. "
            "Run `uv run python scripts/build_teamset.py` first."
        )
    return teamset
