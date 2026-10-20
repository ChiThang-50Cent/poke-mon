"""Play a NeuralPlayer on the PUBLIC Pokémon Showdown ladder ([Gen 9] OU, rated) against humans.

User decision (W8): play from a normal account. The PS bot registry rejected registration
("Punished users may not register bots", likely a shared-IP punishment); the user accepted that
risk. The account status is set to say it is an AI. Credentials come from the environment
(PS_USERNAME, PS_PASSWORD), never from argv or the repo.

One battle at a time, a team sampled from --teams per search, greedy, Metamon team preview.
A search that does not start a battle within --search-timeout-s is cancelled and logged with any
server popup (e.g. team rejected by the current banlist), then retried with a new team.
battles.jsonl (result, turns, opponent, ratings) and HTML replays go to --out.

    set -a; . .omp/ps/credentials.env; set +a
    uv run python scripts/ladder_bot.py --games 50 --out runs/ladder_kk_v4
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
from pathlib import Path

import pokeai  # noqa: F401  (must precede metamon imports)
from pokeai.env.teams import load_teamset

PREVIEW = "third_party/metamon/metamon/backend/team_preview/gen9ou_high_elo_v4/best_model.pt"
STATUS = "AI bot (research), not a human"


class Popups(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        if "Popup" in record.getMessage():
            self.messages.append(record.getMessage())


async def run() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="models/pokedecision-v4-kk")
    ap.add_argument("--teams", default="data/teams_w5/dev")
    ap.add_argument("--games", type=int, default=50)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--search-timeout-s", type=float, default=300.0)
    ap.add_argument("--battle-timeout-s", type=float, default=3600.0)
    ap.add_argument("--max-failed-searches", type=int, default=5)
    args = ap.parse_args()
    username, password = os.environ["PS_USERNAME"], os.environ["PS_PASSWORD"]
    from metamon.backend.team_preview.preview import TeamPreviewModel
    from poke_env import AccountConfiguration, ShowdownServerConfiguration

    from pokeai.agents.neural_player import NeuralPlayer

    out = Path(args.out)
    (out / "replays").mkdir(parents=True, exist_ok=False)
    player = NeuralPlayer(
        model_dir=args.model, device=args.device,
        team_preview_model=TeamPreviewModel.load_from_checkpoint(
            PREVIEW, device=args.device, use_argmax=True),
        battle_format="gen9ou", team=load_teamset(args.teams), max_concurrent_battles=1,
        account_configuration=AccountConfiguration(username, password),
        server_configuration=ShowdownServerConfiguration,
        save_replays=str(out / "replays"), log_level=logging.WARNING,
    )
    popups = Popups()
    player.ps_client.logger.addHandler(popups)
    await player.ps_client.logged_in.wait()
    await player.ps_client.send_message(f"/status {STATUS}")
    games = failed = 0
    with (out / "battles.jsonl").open("x") as log:
        while games < args.games:
            start, before = time.time(), len(player.battles)
            popups.messages.clear()
            task = asyncio.ensure_future(player.ladder(1))
            while len(player.battles) == before and not task.done():
                if time.time() - start > args.search_timeout_s:
                    break
                await asyncio.sleep(1)
            if len(player.battles) == before:
                task.cancel()
                await player.ps_client.send_message("/cancelsearch")
                failed += 1
                row = {"event": "search_failed", "popups": list(popups.messages),
                       "seconds": round(time.time() - start)}
                log.write(json.dumps(row) + "\n")
                log.flush()
                print(json.dumps(row), flush=True)
                if failed >= args.max_failed_searches:
                    raise SystemExit(f"{failed} failed searches in a row; stopping")
                continue
            failed = 0
            await asyncio.wait_for(task, timeout=args.battle_timeout_s)
            tag, battle = next(reversed(player.battles.items()))
            games += 1
            row = {"i": games, "battle": tag, "opponent": battle.opponent_username,
                   "result": "win" if battle.won else "loss" if battle.lost else "tie",
                   "turns": battle.turn, "rating": battle.rating,
                   "opponent_rating": battle.opponent_rating,
                   "illegal_actions": player.illegal_actions,
                   "fallbacks": dict(player.fallbacks), "popups": list(popups.messages),
                   "seconds": round(time.time() - start)}
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(json.dumps(row), flush=True)
    wins = sum(b.won for b in player.battles.values())
    print(f"done: {wins}/{len(player.battles)} wins", flush=True)


async def main() -> None:
    from poke_env.concurrency import handle_threaded_coroutines

    # poke-env's websocket and events live on its own loop (POKE_LOOP); run everything there.
    await handle_threaded_coroutines(run())


if __name__ == "__main__":
    asyncio.run(main())
