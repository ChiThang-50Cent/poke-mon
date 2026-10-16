"""CF0.1 parity: offline frames vs frames captured from the real local Showdown server.

Plays one GymLeader-vs-GymLeader battle on the already-running local server (:8000),
records every raw websocket message each client receives, plays the same teams through
the offline harness, and compares the frame STRUCTURE (kinds, ordering, request JSON key
order). Contents differ (different RNG); only what the policy can observe structurally.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import uuid
from typing import Any

import pokeai  # noqa: F401  (import order: metamon.baselines before GenData)
from pokeai.cf.offline import SIDES, OfflineMatch, PlayerSpec, make_player, new_tag
from pokeai.cf.restore_check import load_pool, write_json
from pokeai.cf.sim import Sim


def classify(frame: str) -> str | None:
    """Frame kind for battle frames; None for lobby/pm traffic."""
    if not frame.startswith(">battle"):
        return None
    body = frame.split("\n", 1)[1] if "\n" in frame else ""
    lines = body.split("\n")
    if lines[0].startswith("|request|"):
        return "request"
    if lines[0].startswith("|init|"):
        return "init"
    if lines[0].startswith("|j|"):
        return "join"
    if lines[0] in ("|deinit",) or lines[0].startswith("|l|"):
        return "leave"
    kinds = [x.split("|")[1] for x in lines if x.startswith("|") and x != "|"][:2]
    return "update:" + "+".join(kinds)


def request_keys(frames: list[str]) -> list[list[str]]:
    out = []
    for frame in frames:
        body = frame.split("\n", 1)[1] if "\n" in frame else ""
        if body.startswith("|request|"):
            out.append(list(json.loads(body[len("|request|"):])))
    return out


def live_capture(teams: tuple[str, str]) -> dict[str, list[str]]:
    from poke_env import AccountConfiguration
    from poke_env.concurrency import POKE_LOOP
    from poke_env.ps_client import PSClient
    from poke_env.ps_client.server_configuration import ServerConfiguration
    from poke_env.teambuilder import Teambuilder

    from pokeai.env.runner import _counted_baseline

    frames: dict[str, list[str]] = {}
    original = PSClient._handle_message

    async def spy(self: Any, message: str) -> None:
        frames.setdefault(self.username, []).append(message)
        return await original(self, message)

    class Fixed(Teambuilder):
        def __init__(self, packed: str) -> None:
            self.packed = packed

        def yield_team(self) -> str:
            return self.packed

    server = ServerConfiguration(
        "ws://localhost:8000/showdown/websocket", "http://localhost:8000/action.php?"
    )

    async def go() -> list[str]:
        cls = _counted_baseline("GymLeader")
        players = [
            cls(
                battle_format="gen9ou", max_concurrent_battles=1,
                account_configuration=AccountConfiguration(
                    "w" + uuid.uuid4().hex[:17], None
                ),
                server_configuration=server, log_level=logging.ERROR,
            )
            for _ in range(2)
        ]
        for player, packed in zip(players, teams, strict=True):
            player.update_team(Fixed(packed))
        await asyncio.wait_for(
            players[0].battle_against(players[1], n_battles=1), 120
        )
        return [p.username for p in players]

    PSClient._handle_message = spy
    try:
        users = asyncio.run_coroutine_threadsafe(go(), POKE_LOOP).result(150)
    finally:
        PSClient._handle_message = original
    return {f"p{i + 1}": frames[u] for i, u in enumerate(users)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teams", default="data/teams_w5/collect/gen9ou")
    parser.add_argument("--out", default="reports/w5_cf0_1_frame_parity.json")
    args = parser.parse_args()
    pool = load_pool(args.teams)
    names = sorted(pool)
    teams = (pool[names[0]][0], pool[names[1]][0])
    live = live_capture(teams)

    with Sim() as sim:
        players = {s: make_player(PlayerSpec("GymLeader"), s) for s in SIDES}
        match = OfflineMatch(sim, players, new_tag())
        match.start(1, {"p1": teams[0], "p2": teams[1]}, {"p1": 1, "p2": 2})
        match.play()
        offline = {s: [f for r in match.rounds for f in r.delivered[s]] for s in SIDES}

    report: dict[str, Any] = {"phase": "CF0.1-frame-parity", "sides": {}}
    ok = True
    for side in SIDES:
        live_kinds = [k for f in live[side] if (k := classify(f))]
        off_kinds = [k for f in offline[side] if (k := classify(f))]
        # the live stream ends with deinit/leave frames the offline harness omits
        live_core = [k for k in live_kinds if k != "leave"]
        prefix = min(9, len(live_core), len(off_kinds))
        live_keys = request_keys([f for f in live[side] if f.startswith(">battle")])
        off_keys = request_keys(offline[side])
        alternation_live = _alternation(live_core)
        alternation_off = _alternation(off_kinds)
        checks = {
            "first_frames_identical": live_core[:prefix] == off_kinds[:prefix],
            "request_key_order_no_conflict": not _conflicts(live_keys, off_keys),
            "update_then_request_pattern": alternation_live == alternation_off,
        }
        ok &= all(checks.values())
        report["sides"][side] = {
            "checks": checks,
            "live_first_kinds": live_core[:prefix],
            "offline_first_kinds": off_kinds[:prefix],
            "request_key_orders_live": _dedupe(live_keys),
            "request_key_orders_offline": _dedupe(off_keys),
            "live_frames": len(live_core), "offline_frames": len(off_kinds),
        }
    report["PASS"] = ok
    write_json(args.out, report)
    print(json.dumps(report, indent=2)[:3000])
    if not ok:
        raise SystemExit(1)


def _conflicts(live: list[list[str]], offline: list[list[str]]) -> list[Any]:
    """Request key sets seen in both streams must use the same key order."""
    live_orders = {frozenset(k): tuple(k) for k in live}
    return [
        (live_orders[frozenset(k)], tuple(k)) for k in offline
        if frozenset(k) in live_orders and live_orders[frozenset(k)] != tuple(k)
    ]


def _dedupe(items: list[list[str]]) -> list[list[str]]:
    out: list[list[str]] = []
    for item in items:
        if item not in out:
            out.append(item)
    return out


def _alternation(kinds: list[str]) -> list[str]:
    """Collapse to the update/request/other pattern: no request without a prior update."""
    pattern = []
    for kind in kinds:
        label = "update" if kind.startswith("update") else kind
        if not pattern or pattern[-1] != label:
            pattern.append(label)
    # trailing request-less end (win frame) is allowed on both sides identically
    return pattern[:4] + ["..."] + pattern[-2:]


if __name__ == "__main__":
    main()
