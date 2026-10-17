import io
import json
import tarfile
import urllib.request
from pathlib import Path

import lz4.frame
import pytest

from pokeai.data import download
from pokeai.data.download import HumanDownloadConfig, download_human_replays


def _archive() -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for index in range(40):
            rating = "Unrated" if index % 4 == 0 else str(1000 + 20 * index)
            name = (
                f"gen9ou/2024/01/gen9ou-{index}_{rating}_a_vs_b_01-02-2024_WIN.json.lz4"
            )
            actions = [0, -1, 1, -1]
            payload = lz4.frame.compress(json.dumps({"actions": actions}).encode())
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    return buffer.getvalue()


class _RangeResponse(io.BytesIO):
    def __init__(self, blob: bytes, start: int) -> None:
        super().__init__(blob[start:])
        self.status = 206
        self.headers = {
            "Content-Range": f"bytes {start}-{len(blob) - 1}/{len(blob)}",
            "ETag": '"fixed"',
        }


@pytest.fixture
def fake_remote(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    blob = _archive()
    starts: list[int] = []

    def urlopen(request: urllib.request.Request, timeout: float) -> _RangeResponse:
        start = int(request.get_header("Range").removeprefix("bytes=").rstrip("-"))
        starts.append(start)
        return _RangeResponse(blob, start)

    monkeypatch.setattr(download.urllib.request, "urlopen", urlopen)
    return starts


def _kept(root: Path) -> list[str]:
    return sorted(p.name for p in (root / "gen9ou").iterdir())


def test_resume_after_byte_cap_matches_single_pass(
    tmp_path: Path, fake_remote: list[int]
) -> None:
    config = HumanDownloadConfig(
        target_valid_decisions=40, min_rating=1100, rate_limit_bps=1e12
    )
    single = download_human_replays(tmp_path / "single", config)

    resumed_root = tmp_path / "resumed"
    with pytest.raises(ValueError, match="No trajectories kept"):
        # Cap inside the first members so nothing qualifying is kept yet.
        download_human_replays(
            resumed_root,
            HumanDownloadConfig(
                max_files=1, min_rating=10_000, rate_limit_bps=1e12, max_bytes=300
            ),
        )
    prefix = (resumed_root / "gen9ou.prefix.tar.gz").stat().st_size
    assert prefix == 300
    resumed = download_human_replays(resumed_root, config)

    assert fake_remote[-1] == prefix  # second session continued, not restarted
    assert _kept(resumed_root) == _kept(tmp_path / "single")
    assert resumed["valid_decisions"] == single["valid_decisions"] >= 40
    assert resumed["skipped"] == single["skipped"]
    assert resumed["archive_prefix_bytes"] == single["archive_prefix_bytes"]


def test_byte_cap_is_hard_ceiling(tmp_path: Path, fake_remote: list[int]) -> None:
    cap = len(_archive()) // 2
    stats = download_human_replays(
        tmp_path,
        HumanDownloadConfig(
            target_valid_decisions=10_000, rate_limit_bps=1e12, max_bytes=cap
        ),
    )
    assert stats["stopped_reason"] == "max_bytes"
    assert stats["archive_prefix_bytes"] == cap
    assert 0 < stats["valid_decisions"] < 10_000


def test_finished_sample_rejects_different_config(
    tmp_path: Path, fake_remote: list[int]
) -> None:
    config = HumanDownloadConfig(max_files=2, rate_limit_bps=1e12)
    download_human_replays(tmp_path, config)
    assert download_human_replays(tmp_path, config)["trajectory_count"] == 2
    with pytest.raises(ValueError, match="different config"):
        download_human_replays(tmp_path, HumanDownloadConfig(max_files=3))
