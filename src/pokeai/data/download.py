"""Bounded, rate-limited, resumable prefix download of Metamon human replays."""

from __future__ import annotations

import io
import json
import statistics
import tarfile
import time
import urllib.request
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, BinaryIO

import lz4.frame

from pokeai.data.dataset import SPACE_VERSIONS

HUMAN_URL = (
    "https://huggingface.co/datasets/jakegrigsby/metamon-parsed-replays/"
    "resolve/v6/gen9ou.tar.gz"
)
CHUNK = 64 * 1024


@dataclass(frozen=True)
class HumanDownloadConfig:
    """Stop at whichever limit is hit first; ``max_bytes`` is a hard ceiling."""

    max_files: int | None = None
    target_valid_decisions: int | None = None
    min_rating: int | None = None
    rate_limit_bps: float = 12_000_000
    max_bytes: int = 1_000_000_000
    url: str = HUMAN_URL

    def __post_init__(self) -> None:
        if self.max_files is None and self.target_valid_decisions is None:
            raise ValueError("Set max_files or target_valid_decisions")
        if self.rate_limit_bps <= 0 or self.max_bytes <= 0:
            raise ValueError("rate_limit_bps and max_bytes must be positive")


class ByteCapReached(Exception):
    """The archive prefix hit ``max_bytes`` before the sample target was met."""


class PrefixReader:
    """Replay the saved compressed prefix, then extend it via HTTP Range.

    Remote bytes are appended to ``part`` so an interrupted run resumes from its
    last byte. The remote object's ETag is pinned in ``part_meta`` so a resume
    never splices bytes from a different archive version. Remote reads are paced
    to ``rate_bps`` and refused beyond ``max_bytes`` total prefix length.
    """

    def __init__(self, part: Path, url: str, rate_bps: float, max_bytes: int) -> None:
        self.part = part
        self.part_meta = part.with_name(part.name + ".json")
        self.url = url
        self.rate_bps = rate_bps
        self.max_bytes = max_bytes
        self.local: BinaryIO = part.open("rb") if part.exists() else io.BytesIO()
        self.offset = part.stat().st_size if part.exists() else 0
        # Remote bytes are appended to the same file, so replay only what existed
        # at open time; otherwise the local handle would re-read fresh appends.
        self.local_remaining = self.offset
        self.remote: Any = None
        self.sink: BinaryIO | None = None
        self.fetched = 0
        self.started = time.monotonic()
        pinned = json.loads(self.part_meta.read_text()) if self.part_meta.exists() else {}
        self.etag: str | None = pinned.get("etag")
        if pinned and pinned.get("url") != url:
            raise RuntimeError(f"{part} belongs to {pinned.get('url')}, not {url}")

    def _open_remote(self) -> None:
        request = urllib.request.Request(
            self.url, headers={"Range": f"bytes={self.offset}-"}
        )
        self.remote = urllib.request.urlopen(request, timeout=120)
        content_range = self.remote.headers.get("Content-Range", "")
        if self.remote.status != 206 or not content_range.startswith(
            f"bytes {self.offset}-"
        ):
            raise RuntimeError(
                f"Server ignored Range {self.offset}-: status {self.remote.status}, "
                f"Content-Range {content_range!r}"
            )
        etag = self.remote.headers.get("ETag")
        if self.etag is not None and etag != self.etag:
            raise RuntimeError(f"Archive changed (ETag {self.etag} -> {etag}); delete {self.part}")
        self.etag = etag
        self.part_meta.write_text(json.dumps({"url": self.url, "etag": etag}) + "\n")
        self.sink = self.part.open("ab")

    def read(self, size: int = -1) -> bytes:
        want = size if size > 0 else CHUNK
        if self.local_remaining:
            data = self.local.read(min(want, self.local_remaining))
            if not data:
                raise RuntimeError(f"{self.part} shrank while resuming")
            self.local_remaining -= len(data)
            return data
        if self.offset >= self.max_bytes:
            raise ByteCapReached(f"archive prefix reached max_bytes={self.max_bytes}")
        if self.remote is None:
            self._open_remote()
        data = self.remote.read(min(want, CHUNK, self.max_bytes - self.offset))
        assert self.sink is not None
        self.sink.write(data)
        self.offset += len(data)
        self.fetched += len(data)
        # Pace the session average so it never runs ahead of rate_bps.
        ahead = self.fetched / self.rate_bps - (time.monotonic() - self.started)
        if ahead > 0:
            time.sleep(ahead)
        return data

    def close(self) -> None:
        self.local.close()
        if self.sink is not None:
            self.sink.close()
        if self.remote is not None:
            self.remote.close()


def parse_replay_name(name: str) -> tuple[int | None, str]:
    """Return (rating or None if unrated, ``YYYY-MM``) from a replay filename."""
    parts = name.removesuffix(".json.lz4").split("_")
    rating = int(parts[1]) if parts[1].isdigit() else None
    month, _, year = parts[-2].split("-")
    return rating, f"{year}-{month}"


def download_human_replays(root: Path, config: HumanDownloadConfig) -> dict[str, Any]:
    """Stream the archive prefix, keep filtered trajectories, write ``sample.json``.

    Kept files land in ``root/gen9ou/`` (ParsedReplayDataset layout). The compressed
    prefix stays at ``root/gen9ou.prefix.tar.gz`` so reruns resume instead of
    re-downloading; a finished ``sample.json`` with the same config is returned.
    """
    record = root / "sample.json"
    if record.exists():
        stats = json.loads(record.read_text())
        if stats.get("config") == asdict(config):
            return stats
        raise ValueError(f"{record} was built with a different config; use a new root")
    target = root / "gen9ou"
    target.mkdir(parents=True, exist_ok=True)
    for stale in target.glob("*.tmp"):
        stale.unlink()
    reader = PrefixReader(
        root / "gen9ou.prefix.tar.gz",
        config.url,
        config.rate_limit_bps,
        config.max_bytes,
    )
    skipped: Counter[str] = Counter()
    months: Counter[str] = Counter()
    ratings: list[int] = []
    kept = valid = missing = seen = 0
    stopped = "target_reached"
    try:
        with tarfile.open(fileobj=reader, mode="r|gz") as archive:
            for member in archive:
                if not member.isfile() or not member.name.endswith(".json.lz4"):
                    continue
                seen += 1
                name = Path(member.name).name
                rating, month = parse_replay_name(name)
                if config.min_rating is not None and (
                    rating is None or rating < config.min_rating
                ):
                    skipped["unrated" if rating is None else "below_min_rating"] += 1
                    continue
                source = archive.extractfile(member)
                if source is None:
                    raise ValueError(f"Unreadable archive member: {member.name}")
                payload = source.read()
                actions = json.loads(lz4.frame.decompress(payload))["actions"][:-1]
                file_valid = sum(action != -1 for action in actions)
                if not file_valid:
                    skipped["no_valid_decisions"] += 1
                    continue
                tmp = target / (name + ".tmp")
                tmp.write_bytes(payload)
                tmp.rename(target / name)
                kept += 1
                valid += file_valid
                missing += len(actions) - file_valid
                months[month] += 1
                if rating is not None:
                    ratings.append(rating)
                if (config.max_files and kept >= config.max_files) or (
                    config.target_valid_decisions
                    and valid >= config.target_valid_decisions
                ):
                    break
            else:
                stopped = "archive_end"
    except ByteCapReached:
        stopped = "max_bytes"
    finally:
        reader.close()
    if not kept:
        raise ValueError(f"No trajectories kept (stopped: {stopped})")
    elapsed = time.monotonic() - reader.started
    stats = {
        "source": config.url,
        "revision": "v6",
        "etag": reader.etag,
        "config": asdict(config),
        "stopped_reason": stopped,
        "archive_prefix_bytes": reader.offset,
        "downloaded_bytes_this_session": reader.fetched,
        "elapsed_s": round(elapsed, 2),
        "avg_rate_bps": round(reader.fetched / elapsed) if elapsed else None,
        "members_seen": seen,
        "trajectory_count": kept,
        "skipped": dict(skipped),
        "valid_decisions": valid,
        "missing_decisions": missing,
        "stored_bytes": sum(p.stat().st_size for p in target.iterdir()),
        "rating_deciles": (
            [int(q) for q in statistics.quantiles(ratings, n=10)]
            if len(ratings) >= 10
            else None
        ),
        "months": dict(sorted(months.items())),
        **SPACE_VERSIONS,
    }
    record.write_text(json.dumps(stats, indent=2) + "\n")
    return stats
