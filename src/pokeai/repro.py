"""Deterministic source identity when the working tree has no Git metadata."""

import hashlib
from pathlib import Path


def source_fingerprint(root: str | Path | None = None) -> str:
    root = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    digest = hashlib.sha256()
    paths = sorted([*root.glob("src/**/*.py"), *root.glob("scripts/*.py")])
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()
