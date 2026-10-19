"""One lock protocol for the one-shot heldout claim (BS1.4-P).

Every verify and every claim of a heldout pool happens inside ONE critical section owned by
`pool_lock(pool)` (an exclusive flock on `<pool>/final.lock`; the only place that takes it):

    lane:    with pool_lock(pool) as lock:
                 verify_unlocked(lock, ...)      # code/config/team files + pool entries
                 claim_unlocked(lock, ...)       # final.started.json + this window
    driver:  with pool_lock(pool) as lock:
                 verify_unlocked(lock, ...)      # before any lane, and before every block

Claim files are staged OUTSIDE the pool, in `<pool.parent>/.<pool.name>.claim-staging/` (same
filesystem): write + fsync, `os.link` into the pool (fails if the target exists), fsync the pool
directory, unlink the staged file. A crash leaves an orphan in the staging dir; the verifier
refuses while any exists (fail closed, manual review); nothing is ever auto-deleted."""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import re
import tempfile
from pathlib import Path

CLAIM_STARTED = "final.started.json"
WINDOW_RE = re.compile(r"final\.window\.(\d+)of(\d+)\.(\d+)-(\d+)\.json")


def staging_dir(pool: Path | str) -> Path:
    pool = Path(pool)
    return pool.parent / f".{pool.name}.claim-staging"


class PoolLock:
    """Token proving the caller holds `<pool>/final.lock` exclusively."""

    def __init__(self, pool: Path, stream) -> None:
        self.pool = pool.resolve()
        self._stream = stream

    @property
    def held(self) -> bool:
        return self._stream is not None

    def require(self, pool: Path | str) -> None:
        if not self.held or self.pool != Path(pool).resolve():
            raise RuntimeError(f"the lock of pool {pool} is not held by the caller")


@contextlib.contextmanager
def pool_lock(pool: Path | str):
    """Exclusive critical section on the pool (blocks until the holder releases)."""
    pool = Path(pool)
    with (pool / "final.lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        lock = PoolLock(pool, stream)
        try:
            yield lock
        finally:
            lock._stream = None


def _claim_step(stage: str, path: Path) -> None:
    """Interleaving point (no-op); tests monkeypatch it inside a child process."""


def _fsync_dir(directory: Path) -> None:
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _create_exclusive(path: Path, payload: dict) -> None:
    """Create `path` with its COMPLETE content, or raise FileExistsError; never a partial file."""
    staging = staging_dir(path.parent)
    staging.mkdir(exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=staging, prefix=f"{path.name}.")
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(payload, stream)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        _claim_step("staged", path)
        os.link(tmp, path)
    except BaseException:
        os.unlink(tmp)
        raise
    _claim_step("linked", path)
    _fsync_dir(path.parent)
    os.unlink(tmp)


def claim_unlocked(lock: PoolLock | None, pool: Path, final: bool, config: dict,
                   window: tuple[int, int, int, int] | None = None) -> None:
    """Claim heldout globally, not relative to the user-selected report path.

    One frozen config may be executed as several (lane, lanes, pair-start, pair-stop) windows of
    the same schedule: the first creates the one-shot claim, later windows must carry the
    IDENTICAL config and each window can be taken exactly once. The caller MUST hold `pool_lock`
    (dev pools need no lock and no claim)."""
    manifest = json.loads((pool / "gen9ou/manifest.json").read_text())
    role = manifest["role"]
    if role not in ("dev", "heldout"):
        raise ValueError("Benchmark requires a dev or heldout pool")
    if role == "heldout":
        if not final:
            raise ValueError("heldout pool requires --final; never use for selection")
        if lock is None:
            raise RuntimeError("claiming a heldout pool requires the pool lock")
        lock.require(pool)
        claim = pool / CLAIM_STARTED
        try:
            _create_exclusive(claim, {"purpose": "one-shot final W5 benchmark", "config": config})
        except FileExistsError:
            if window is None or json.loads(claim.read_text())["config"] != config:
                raise
        if window is not None:
            lane, lanes, start, stop = window
            _create_exclusive(pool / f"final.window.{lane}of{lanes}.{start}-{stop}.json",
                              {"config": config})
    elif final:
        raise ValueError("--final requires the heldout pool")


def verify_unlocked(lock: PoolLock, config: dict, config_path: Path | str,
                    pool: Path | str | None = None, claim_config: dict | None = None) -> None:
    """A final (heldout) run must match its frozen bundle exactly, before any claim: code and
    model hashes, the config FILE actually passed (path, sha256, parsed content), the exact team
    files of the pool actually used, and a pool root holding only valid claim artifacts. The
    caller MUST hold `pool_lock` of that pool (so no claim is in flight while it is listed)."""
    path = config.get("run", {}).get("release_manifest")
    if not path:
        raise ValueError("--final requires run.release_manifest in the config")
    pool = pool if pool is not None else config["pool"]
    lock.require(pool)
    from bs14_manifest import verify_bundle

    problems = verify_bundle(Path(path), config_path, config, pool, claim_config=claim_config)
    if problems:
        raise RuntimeError("release manifest mismatch; refusing the heldout claim:\n"
                           + "\n".join(problems))


def verify_and_claim(pool: Path | str, config: dict, config_path: Path | str, effective: dict,
                     window: tuple[int, int, int, int] | None) -> None:
    """Lane entry: ONE critical section (never verify, release, then re-lock to claim)."""
    pool = Path(pool)
    with pool_lock(pool) as lock:
        verify_unlocked(lock, config, config_path, pool, claim_config=effective)
        claim_unlocked(lock, pool, True, effective, window)


def verify_only(pool: Path | str, config: dict, config_path: Path | str) -> None:
    """Driver entry: verify under the lock, before any lane and before every block."""
    with pool_lock(pool) as lock:
        verify_unlocked(lock, config, config_path, pool)
