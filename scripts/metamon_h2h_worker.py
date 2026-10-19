"""Metamon side of a paired head-to-head on the LOCAL Showdown server.

Launched by ``run_metamon_h2h.py``. Loads a pretrained Metamon/AMAGO agent and accepts
one challenge per scheduled battle from the neural player's username, using the packed
teams from a frozen schedule file (battle i -> team i). Never touches the public server:
``ChallengeByUsername`` uses ``LocalhostServerConfiguration``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from functools import partial
from pathlib import Path


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", type=int)
    parser.add_argument("--username", required=True)
    parser.add_argument("--opponent-username", required=True)
    parser.add_argument("--schedule", required=True, help="JSON list of {packed, name}")
    parser.add_argument("--info-out", required=True)
    parser.add_argument("--team-log", required=True)
    parser.add_argument("--ready-file", required=True)
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()
    os.environ.setdefault("METAMON_CACHE_DIR", str(Path(".cache/metamon").resolve()))

    import warnings

    warnings.filterwarnings("ignore")
    from metamon.env import TeamSet, wrappers
    from metamon.env.metamon_player import PokeAgentPlayer
    from metamon.rl.metamon_to_amago import make_challenge_env
    from metamon.rl.pretrained import get_pretrained_model
    from poke_env.teambuilder import Teambuilder

    import pokeai  # noqa: F401  (must precede metamon imports: GenData constructor)

    # Vendored wrappers.py references PokeAgentPlayer without importing it (the
    # 'pokeagent' backend raises NameError). We never patch third_party/metamon, so bind
    # the name at runtime.
    wrappers.PokeAgentPlayer = PokeAgentPlayer

    schedule = json.loads(Path(args.schedule).read_text())

    class ScheduledTeamSet(TeamSet):
        """Yield schedule[i] on the i-th call; log every yield for validation."""

        def __init__(self) -> None:
            Teambuilder.__init__(self)
            self.battle_format = "gen9ou"
            self.index = 0
            self._most_recent_team_file = None

        def yield_team(self) -> str:
            if self.index >= len(schedule):
                raise RuntimeError("team requested beyond the frozen schedule")
            entry = schedule[self.index]
            self.index += 1
            self._most_recent_team_file = entry["name"]
            with open(args.team_log, "a") as log:
                log.write(json.dumps({
                    "yield": self.index - 1, "name": entry["name"],
                    "sha256": hashlib.sha256(entry["packed"].encode()).hexdigest(),
                }) + "\n")
            return entry["packed"]

    pretrained = get_pretrained_model(args.model)
    checkpoint = args.checkpoint if args.checkpoint is not None else pretrained.default_checkpoint
    agent = pretrained.initialize_agent(
        checkpoint=checkpoint, log=False, action_temperature=args.temperature
    )
    agent.env_mode = "sync"
    agent.parallel_actors = 1
    agent.verbose = False
    ckpt_path = pretrained.get_path_to_checkpoint(checkpoint)
    info = {
        "model": args.model, "checkpoint": checkpoint,
        "checkpoint_path": os.path.realpath(ckpt_path),
        "checkpoint_sha256": sha256_file(ckpt_path),
        "backend": pretrained.battle_backend,
        "temperature": args.temperature,
        "observation_space": type(pretrained.observation_space.base_obs_space).__name__,
    }
    Path(args.info_out).write_text(json.dumps(info, indent=2) + "\n")

    def make_env():
        env = make_challenge_env(
            battle_format="gen9ou",
            num_battles=len(schedule),
            observation_space=pretrained.observation_space,
            action_space=pretrained.action_space,
            reward_function=pretrained.reward_function,
            player_team_set=ScheduledTeamSet(),
            player_username=args.username,
            opponent_username=args.opponent_username,
            role="acceptor",
            battle_backend=pretrained.battle_backend,
            print_battle_bar=False,
        )
        Path(args.ready_file).write_text("ready\n")
        return env

    results = agent.evaluate_test(
        [partial(make_env)], timesteps=len(schedule) * 1000, episodes=len(schedule)
    )
    print("METAMON_RESULTS " + json.dumps(results, sort_keys=True, default=float))


if __name__ == "__main__":
    main()
