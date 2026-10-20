# KK v4 on the public ladder ([Gen 9] OU, rated, humans), 20/10/2026

- Setup:
  - account `PokeDecisionKK`, `scripts/ladder_bot.py`, greedy, one battle at a time;
  - teams sampled from `data/teams_w5/dev`, Metamon team preview.
- Stopped by the user after 24 model games, between battles: 13 wins / 13 losses on the account.
  - This includes 2 losses forfeited during debugging, not played by the model.
  - Model-played record: 13–11. 0 illegal actions, 0 fallbacks.
- Final account rating: **Elo 1091, GXE 44.3%, Glicko 1456 ± 63**. The peak Elo was ~1203.
- Reference: Metamon reports a human-ladder GXE in Gen 9 OU (sample teams) of:
  - Abra 50%;
  - Kadabra3 64%;
  - Kakuna 71%.

  Their "#1 on the ladder" claim is KakunaEnsemble/TaurosEnsemble in **Gen 1 OU**.
- Gap: KK beats Kakuna 43.8% head-to-head (dev), yet sits ~27 GXE points below it against humans.
  - Untested causes:
    - within-family h2h overstates strength vs humans;
    - team pool quality (Metamon-built sets vs expert sample teams);
    - small sample.
  - Evidence for analysis: `runs/ladder_kk_v4/battles.jsonl` and `replays/` (gitignored).
