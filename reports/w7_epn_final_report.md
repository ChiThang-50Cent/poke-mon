# W7: EPN final, promoted as `models/pokedecision-v3-epn`

**Verdict: PASS, promoted.** On a fresh heldout pool, EPN (entity-pointer network, 1.01M
parameters, 22% fewer than P) beats the previous champion P by **+19.6 pp pooled
[+17.1, +22.1]**. Every per-opponent CI is above 0.

## Candidate

- Architecture: `autoresearch/train.py` at `f1304d9` (EPN), described in
  `reports/ar_autoresearch_report.md` §4.
- Training (`checkpoints/epn/epn20`):
  - from scratch, 20 minutes, BC plus value loss;
  - 285,142 `pac-base` Alakazam-family POVs (`datasets/w6p` + `w6p_b`);
  - pile val NLL 0.808 (P: 1.190).
- Selected on dev only (`configs/ar_epn_dev.yaml`): +23.8 pp pooled vs P.

## Final (one-shot, fresh pool)

- Pool `data/teams_w7/heldout`, now **spent**:
  - built with `build_teamset.py --w5-fresh-heldout --seed 20262401`;
  - excluded dirs: `teams/gen9ou`, `teams_w5/{collect,dev,heldout}`, `teams_w5_bs14/heldout`,
    `teams_w5_bs15/heldout` and `teams_w6p/heldout`;
  - 200 server-validated teams.
- Bundle: `configs/w7_epn_final.yaml` (seed 20262411) and `reports/w7_epn_final_manifest.json`,
  verified with `verify_bundle` before the claim (no problems found).
- Run: 3,000 battles, 500 per cell (250 pair clusters, both directions). All cells valid,
  0 illegal actions, 0 errors.
- Pooled CI: stratified cluster bootstrap (`reports/w7_epn_final_pooled.json`).

| Opponent | P win rate | EPN win rate | EPN − P [95% CI] |
|---|---:|---:|---|
| P | 0.508 | 0.744 | **+0.236 [+0.188, +0.284]** |
| Grunt | 0.726 | 0.930 | **+0.204 [+0.166, +0.244]** |
| GymLeader | 0.784 | 0.932 | **+0.148 [+0.110, +0.188]** |
| pooled | | | **+0.196 [+0.171, +0.221]** |

Gate (`w6_0_distill_feasibility.md` §5, with P as the reference):

| Condition | Result |
|---|---|
| Pooled CI > 0 | ✔ |
| vs-P CI > 0 | ✔ |
| No opponent CI below 0 | ✔ |
| 0 illegal actions, 0 errors | ✔ |

## Promotion

- `models/pokedecision-v3-epn` contains `best.pt`, `model.py`, `tokenizer.json`, `summary.json`
  and `metrics.json`. The weights are byte-identical to `checkpoints/epn/epn20`.
- It is loaded by `load_pokedecision` through `load_research_model`.
- Smoke: 20/20 wins vs GymLeader via `run_battles.py`, 0 errors. Tag `w7-epn`.

## Against Metamon (dev pool, not part of the gate)

| Opponent | Params | EPN win rate |
|---|---:|---:|
| Minikazam | 4.7M | 0.762 |
| Kakuna (flex attention) | 142M | 0.290 |
