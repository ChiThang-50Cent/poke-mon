"""Play a NeuralPlayer on the PUBLIC Pokémon Showdown ladder ([Gen 9] OU, rated) against humans.

Policy (Smogon "Initial policy on bots", Oct 2026): every bot must be registered (in PS chat:
/ht -> "Something else" -> "I want to register my bot"); running an unregistered bot is a
bannable offense. This script refuses to start without --registered-bot. Credentials come from
the environment (PS_USERNAME, PS_PASSWORD), never from argv or the repo.

One battle at a time, a team sampled from --teams per battle, greedy, Metamon team preview.
HTML replays and a battles.jsonl (result, turns, opponent, rating info) go to --out for error
analysis. Stops after --games battles.

    PS_USERNAME=... PS_PASSWORD=... uv run python scripts/ladder_bot.py --registered-bot \
        --model models/pokedecision-v4-kk --games 20 --out runs/ladder_kk_v4
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


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="models/pokedecision-v4-kk")
    ap.add_argument("--teams", default="data/teams_w5/dev")
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--registered-bot", action="store_true",
                    help="confirm the account is registered as a bot with PS staff")
    args = ap.parse_args()
    if not args.registered_bot:
        raise SystemExit("Refusing: register the bot on PS first (/ht), then pass --registered-bot")
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
    with (out / "battles.jsonl").open("x") as log:
        for i in range(args.games):
            start = time.time()
            await player.ladder(1)
            tag, battle = next(reversed(player.battles.items()))
            row = {"i": i, "battle": tag, "opponent": battle.opponent_username,
                   "result": "win" if battle.won else "loss" if battle.lost else "tie",
                   "turns": battle.turn, "rating": battle.rating,
                   "opponent_rating": battle.opponent_rating,
                   "illegal_actions": player.illegal_actions,
                   "fallbacks": dict(player.fallbacks), "seconds": round(time.time() - start)}
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(json.dumps(row), flush=True)
    wins = sum(b.won for b in player.battles.values())
    print(f"done: {wins}/{len(player.battles)} wins", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
