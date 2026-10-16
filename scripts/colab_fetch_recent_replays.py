"""Stream Metamon's gen9ou.tar.gz once and keep only 2025-2026 replays.

Self-contained (stdlib + ``lz4``) so it runs on Google Colab:

    !pip -q install lz4
    from google.colab import drive; drive.mount("/content/drive")
    !python colab_fetch_recent_replays.py --out /content/drive/MyDrive/metamon-gen9ou-recent

The 20.4 GB archive is gzip, so it cannot be seeked: every byte is streamed and
decompressed in memory, but only matching trajectories are written. Dropped
connections resume via HTTP Range without restarting decompression. Output is
uncompressed tar shards (members are already lz4) plus ``stats.json``.

Locally, extract the shards so files land in ``<root>/gen9ou/*.json.lz4``:

    mkdir -p .cache/metamon/w4-human/gen9ou
    for f in shard-*.tar; do tar -xf "$f" -C .cache/metamon/w4-human/gen9ou; done
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

import lz4.frame

URL = (
    "https://huggingface.co/datasets/jakegrigsby/metamon-parsed-replays/"
    "resolve/v6/gen9ou.tar.gz"
)
CHUNK = 1 << 20


class ResumingStream:
    """File-like HTTP body that reconnects with Range on errors, optionally paced."""

    def __init__(self, url: str, rate_bps: float | None, max_bytes: int | None):
        self.url = url
        self.rate_bps = rate_bps
        self.max_bytes = max_bytes
        self.offset = 0
        self.etag: str | None = None
        self.started = time.monotonic()
        self.response = None
        self.retries = 0
        self.total: int | None = None

    def at_cap(self) -> bool:
        return self.max_bytes is not None and self.offset >= self.max_bytes

    def _connect(self) -> None:
        request = urllib.request.Request(
            self.url, headers={"Range": f"bytes={self.offset}-"}
        )
        response = urllib.request.urlopen(request, timeout=60)
        if response.status != 206:
            raise RuntimeError(f"Range not honoured (status {response.status})")
        etag = response.headers.get("ETag")
        if self.etag is not None and etag != self.etag:
            raise RuntimeError(f"Archive changed mid-download: {self.etag} -> {etag}")
        self.etag = etag
        self.total = int(response.headers["Content-Range"].rsplit("/", 1)[1])
        self.response = response

    def read(self, size: int = -1) -> bytes:
        if self.at_cap() or (self.total is not None and self.offset >= self.total):
            return b""
        want = CHUNK if size is None or size < 0 else min(size, CHUNK)
        if self.max_bytes is not None:
            want = min(want, self.max_bytes - self.offset)
        while True:
            try:
                if self.response is None:
                    self._connect()
                data = self.response.read(want)
                if not data and self.total is not None and self.offset < self.total:
                    # The CDN occasionally closes a 206 body early; an empty read
                    # before the known end must reconnect, never look like EOF.
                    raise ConnectionError("empty read before end of archive")
                break
            except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as exc:
                self.retries += 1
                if self.retries > 50:
                    raise
                print(f"[retry {self.retries}] at byte {self.offset}: {exc}", flush=True)
                self.response = None
                time.sleep(min(60, 2 ** min(self.retries, 6)))
        self.offset += len(data)
        if self.rate_bps:
            ahead = self.offset / self.rate_bps - (time.monotonic() - self.started)
            if ahead > 0:
                time.sleep(ahead)
        return data


class ShardWriter:
    """Write kept members into ~``shard_bytes`` uncompressed tar files."""

    def __init__(self, out: Path, shard_bytes: int):
        self.out = out
        self.shard_bytes = shard_bytes
        self.index = 0
        self.size = 0
        self.tar: tarfile.TarFile | None = None

    def add(self, name: str, payload: bytes) -> None:
        if self.tar is None or self.size >= self.shard_bytes:
            self.close()
            path = self.out / f"shard-{self.index:03d}.tar.part"
            self.tar = tarfile.open(path, "w")  # noqa: SIM115 - spans add() calls; close()
            self.size = 0
        info = tarfile.TarInfo(name)
        info.size = len(payload)
        self.tar.addfile(info, io.BytesIO(payload))
        self.size += len(payload) + 1024

    def close(self) -> None:
        if self.tar is not None:
            self.tar.close()
            part = self.out / f"shard-{self.index:03d}.tar.part"
            part.rename(part.with_suffix(""))
            self.index += 1
            self.tar = None


def read_actions(payload: bytes) -> list[int]:
    """Parse only the trailing ``"actions"`` list (~40x faster than the full JSON).

    Metamon writes ``{"states": [...], "actions": [...]}`` so the key is last;
    anything unexpected falls back to a full parse rather than guessing.
    """
    raw = lz4.frame.decompress(payload)
    key = raw.rfind(b'"actions"')
    if key != -1:
        tail = raw[raw.find(b"[", key) : raw.rfind(b"]") + 1]
        try:
            actions = json.loads(tail)
        except ValueError:
            actions = None
        if isinstance(actions, list) and all(type(a) is int for a in actions):
            return actions
    return json.loads(raw)["actions"]


def parse_name(name: str) -> tuple[int | None, int, int]:
    """(rating or None, year, month) from ``id_rating_pov_vs_opp_MM-DD-YYYY_RES``."""
    parts = name.removesuffix(".json.lz4").split("_")
    rating = int(parts[1]) if parts[1].isdigit() else None
    month, _, year = parts[-2].split("-")
    return rating, int(year), int(month)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--years", type=int, nargs="+", default=[2025, 2026])
    parser.add_argument("--min-rating", type=int, default=None,
                        help="drop unrated and lower-rated replays (default: keep all)")
    parser.add_argument("--sample-fraction", type=float, default=1.0,
                        help="keep this share of battles, chosen by battle-id hash "
                        "(deterministic; both POVs of a battle stay together)")
    parser.add_argument("--rate-limit-mbps", type=float, default=None,
                        help="MB/s cap (default: unlimited)")
    parser.add_argument("--shard-mb", type=int, default=2000)
    parser.add_argument("--max-bytes", type=int, default=None,
                        help="stop after this many compressed bytes (smoke tests)")
    parser.add_argument("--url", default=URL)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if any(args.out.glob("shard-*.tar")):
        raise SystemExit(f"{args.out} already has shards; use an empty directory")

    stream = ResumingStream(
        args.url,
        args.rate_limit_mbps * 1e6 if args.rate_limit_mbps else None,
        args.max_bytes,
    )
    writer = ShardWriter(args.out, args.shard_mb * 1_000_000)
    years = set(args.years)
    if not 0 < args.sample_fraction <= 1:
        raise SystemExit("--sample-fraction must be in (0, 1]")
    sample_cutoff = int(args.sample_fraction * 2**64)
    seen = kept = kept_bytes = valid = missing = 0
    skipped: Counter[str] = Counter()
    months: Counter[str] = Counter()
    ratings: Counter[str] = Counter()
    last_log = time.monotonic()
    completed = False
    try:
        try:
            archive = tarfile.open(fileobj=stream, mode="r|gz")  # noqa: SIM115 - with below
        except tarfile.ReadError as exc:
            raise SystemExit(f"Archive did not start as gzip: {exc}") from exc
        with archive:
            for member in archive:
                if not member.isfile() or not member.name.endswith(".json.lz4"):
                    continue
                seen += 1
                name = Path(member.name).name
                rating, year, month = parse_name(name)
                if year not in years:
                    skipped["other_year"] += 1
                    continue
                battle_id = name.split("_", 1)[0]
                digest = hashlib.sha1(battle_id.encode()).digest()
                if int.from_bytes(digest[:8], "big") >= sample_cutoff:
                    skipped["not_sampled"] += 1
                    continue
                if args.min_rating is not None and (
                    rating is None or rating < args.min_rating
                ):
                    skipped["unrated" if rating is None else "below_min_rating"] += 1
                    continue
                source = archive.extractfile(member)
                if source is None:
                    raise ValueError(f"Unreadable member {member.name}")
                payload = source.read()
                actions = read_actions(payload)[:-1]
                file_valid = sum(a != -1 for a in actions)
                writer.add(name, payload)
                kept += 1
                kept_bytes += len(payload)
                valid += file_valid
                missing += len(actions) - file_valid
                months[f"{year}-{month:02d}"] += 1
                bucket = "unrated" if rating is None else f"{rating // 100 * 100}+"
                ratings[bucket] += 1
                if time.monotonic() - last_log > 30:
                    last_log = time.monotonic()
                    rate = stream.offset / (last_log - stream.started) / 1e6
                    print(
                        f"{stream.offset / 1e9:6.2f} GB read ({rate:5.1f} MB/s) | "
                        f"seen {seen:,} kept {kept:,} ({kept_bytes / 1e9:.2f} GB) "
                        f"valid decisions {valid:,}",
                        flush=True,
                    )
        completed = stream.total is not None and stream.offset == stream.total
    except (tarfile.ReadError, EOFError):
        # Only a deliberate --max-bytes cut may end mid-member; anything else is real.
        if not stream.at_cap():
            raise
        print(f"stopped at --max-bytes {args.max_bytes}; last member dropped")
    finally:
        writer.close()
        elapsed = time.monotonic() - stream.started
        stats = {
            "source": args.url,
            "etag": stream.etag,
            "completed_full_archive": completed,
            "compressed_bytes_read": stream.offset,
            "elapsed_s": round(elapsed, 1),
            "avg_rate_mbps": round(stream.offset / elapsed / 1e6, 2) if elapsed else None,
            "retries": stream.retries,
            "filters": {
                "years": sorted(years),
                "min_rating": args.min_rating,
                "sample_fraction": args.sample_fraction,
            },
            "members_seen": seen,
            "kept": kept,
            "kept_bytes": kept_bytes,
            "valid_decisions": valid,
            "missing_decisions": missing,
            "skipped": dict(skipped),
            "months": dict(sorted(months.items())),
            "rating_buckets": dict(sorted(ratings.items())),
            "shards": writer.index,
        }
        (args.out / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
        print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
