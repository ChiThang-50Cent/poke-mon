"""Frozen lane watchdog for one-shot heldout runs (hashed in the bundle manifest).

  bs_watchdog.py --unit w5-bs14.service --prefix w5-bs14 --log runs/w5-bs14-watchdog.log

A lane has FAILED iff its evaluate_w5.py process is gone while its benchmark.json does not say
"valid": true, observed on two consecutive checks. Then the whole main unit is stopped (all lanes,
workers, driver), the heldout claim is left in place, and the reason is logged ("LANE FAILURE").
Exits when the main unit is no longer active.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PERIOD = 20


def unit_active(unit: str) -> bool:
    return subprocess.run(
        ["systemctl", "--user", "is-active", "--quiet", unit], check=False).returncode == 0


def stop_unit(unit: str) -> None:
    subprocess.run(["systemctl", "--user", "stop", unit], check=False)


def lane_cmdlines() -> list[str]:
    out = []
    for proc in Path("/proc").iterdir():
        if proc.name.isdigit():
            try:
                cmd = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode()
            except OSError:
                continue
            if "evaluate_w5.py" in cmd and "--out" in cmd:
                out.append(cmd)
    return out


def suspects(repo: Path, prefix: str, cmdlines: list[str]) -> set[str]:
    """Lane reports with no running lane process and no `"valid": true`."""
    running = " ".join(cmdlines)
    bad = set()
    for lane_dir in sorted(repo.glob(f"runs/{prefix}-b*/lane*")):
        report = lane_dir / "benchmark.json"
        rel = str(report.relative_to(repo))
        if rel in running:
            continue
        try:
            valid = json.loads(report.read_text()).get("valid") is True
        except (OSError, ValueError):
            valid = False
        if not valid:
            bad.add(rel)
    return bad


def watch(unit: str, prefix: str, log_path: Path, *, repo: Path = REPO,
          is_active: Callable[[str], bool] = unit_active,
          cmdlines: Callable[[], list[str]] = lane_cmdlines,
          stop: Callable[[str], None] = stop_unit,
          sleep: Callable[[float], None] = time.sleep, period: float = PERIOD) -> int:
    """Return 1 after stopping the unit on a confirmed lane failure, else 0."""
    def log(msg: str) -> None:
        with log_path.open("a") as fh:
            fh.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}\n")

    log(f"watchdog start, watching {unit}")
    previous: set[str] = set()
    while is_active(unit):
        current = suspects(repo, prefix, cmdlines())
        confirmed = current & previous
        if confirmed:
            log(f"LANE FAILURE confirmed: {sorted(confirmed)}; stopping {unit}; claim kept")
            stop(unit)
            return 1
        previous = current
        sleep(period)
    log(f"{unit} no longer active; watchdog exit")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--unit", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--log", required=True, type=Path)
    args = ap.parse_args()
    sys.exit(watch(args.unit, args.prefix, args.log))


if __name__ == "__main__":
    main()
