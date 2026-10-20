# poke-mon

A small decision model that plays Pokémon Showdown **Gen 9 OU Singles**: a structured-state
policy + value network (≤ 4.7M params, not an LLM), built on
[Metamon](https://github.com/UT-Austin-RPL/metamon), poke-env and a pinned local Showdown.

- **Production:** KK, `models/pokedecision-v4-kk`. A 1.01M entity-pointer network distilled
  from Metamon's Kakuna. It wins 43.8% vs Kakuna on dev.
- **Status:** W1–W7 done. W8 E1 (matchup features) closed with no gain, and W8 is paused.
- **Docs:**
  - `AGENTS.md`: setup, commands, invariants, code map. It is authoritative for contributors and agents.
  - `research.md`: design, roadmap and decisions.
  - `reports/`: the evidence for every milestone.

Quick start (details in `AGENTS.md`):

```bash
uv sync
(cd third_party/metamon/server/pokemon-showdown && node pokemon-showdown start --no-security)
uv run python scripts/run_battles.py --player neural:models/pokedecision-v4-kk --opponent GymLeader --battles 20
uv run pytest -m "not server"
```

Training and evaluation run on the local server only. `scripts/ladder_bot.py` plays the
production model on the public ladder as a strength read-out.
