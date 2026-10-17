"""Synchronous wrapper around the Node simulator driver (``sim_driver.js``)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Self

DRIVER = Path(__file__).with_name("sim_driver.js")


class SimError(RuntimeError):
    """The Node driver rejected a command (not a battle-level |error|)."""


def gen5_seed(value: int) -> str:
    """Deterministic Gen5-RNG seed string (``gen5,<16 hex>``) from an integer."""
    import random

    return "gen5," + format(random.Random(value).getrandbits(64), "016x")


class Sim:
    """One driver subprocess hosting any number of battles keyed by id."""

    def __init__(self) -> None:
        self._proc = subprocess.Popen(
            ["node", str(DRIVER)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

    def _call(self, **request: Any) -> dict[str, Any]:
        assert self._proc.stdin is not None and self._proc.stdout is not None
        self._proc.stdin.write(json.dumps(request) + "\n")
        self._proc.stdin.flush()
        line = self._proc.stdout.readline()
        if not line:
            raise SimError("simulator driver exited")
        response = json.loads(line)
        if not response["ok"]:
            raise SimError(response["error"])
        return response

    def create(
        self,
        battle_id: str,
        seed: str,
        p1: dict[str, str],
        p2: dict[str, str],
        formatid: str = "gen9ou",
    ) -> list[dict[str, Any]]:
        return self._call(
            cmd="create", id=battle_id, seed=seed, p1=p1, p2=p2, formatid=formatid
        )["events"]

    def reconstruct(self, battle_id: str, spec: dict[str, Any]) -> list[dict[str, Any]]:
        """New battle at a decision point from a public-information spec (BS0.1)."""
        return self._call(cmd="reconstruct", id=battle_id, **spec)["events"]

    def choose(self, battle_id: str, side: str, choice: str) -> dict[str, Any]:
        return self._call(cmd="choose", id=battle_id, side=side, choice=choice)

    def snapshot(self, battle_id: str) -> str:
        return self._call(cmd="snapshot", id=battle_id)["key"]

    def restore(self, key: str, battle_id: str) -> None:
        self._call(cmd="restore", key=key, id=battle_id)

    def set_prng(self, battle_id: str, seed: str) -> None:
        self._call(cmd="set_prng", id=battle_id, seed=seed)

    def get_prng(self, battle_id: str) -> str:
        return self._call(cmd="get_prng", id=battle_id)["seed"]

    def state(self, battle_id: str) -> dict[str, Any]:
        return self._call(cmd="state", id=battle_id)["state"]

    def info(self, battle_id: str) -> dict[str, Any]:
        return self._call(cmd="info", id=battle_id)

    def destroy(self, battle_id: str) -> None:
        self._call(cmd="destroy", id=battle_id)

    def drop_snapshot(self, key: str) -> None:
        self._call(cmd="drop_snapshot", key=key)

    def stats(self) -> dict[str, int]:
        """Driver bookkeeping: live battles, stored snapshots, process memory."""
        response = self._call(cmd="stats")
        return {k: response[k] for k in ("battles", "snapshots", "rss", "heap")}

    def roster(self, battle_id: str) -> list[dict[str, Any]]:
        return self._call(cmd="roster", id=battle_id)["roster"]

    def public_log(self, battle_id: str) -> list[str]:
        """The p1-channel lines of the battle log (what the learner has seen)."""
        return self._call(cmd="public_log", id=battle_id)["lines"]

    def public_view(self, battle_id: str) -> dict[str, Any]:
        return self._call(cmd="public_view", id=battle_id)["view"]

    def p2_full(self, battle_id: str) -> list[dict[str, Any]]:
        """TRUE opponent sets/state. QA and oracle labels only, never the sampler."""
        return self._call(cmd="p2_full", id=battle_id)["mons"]

    def apply_world(self, battle_id: str, world: dict[str, Any]) -> dict[str, Any]:
        return self._call(cmd="apply_world", id=battle_id, world=world)

    def validate(self, team: list[dict[str, Any]], formatid: str = "gen9ou") -> Any:
        return self._call(cmd="validate", team=team, formatid=formatid)["errors"]

    def move_info(self, names: list[str]) -> list[dict[str, Any]]:
        return self._call(cmd="move_info", names=names)["moves"]

    def close(self) -> None:
        if self._proc.poll() is None:
            assert self._proc.stdin is not None
            self._proc.stdin.close()
            self._proc.wait(timeout=10)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
