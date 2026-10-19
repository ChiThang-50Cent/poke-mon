# W5.BS1.5 stopped by decision; W5 closed

**Status: STOPPED BY USER DECISION. There is no heldout estimate.** The pool `data/teams_w5_bs15/heldout` is spent (claimed). The 40 battles that completed are **sealed**: their win/loss was not computed or viewed.

## What happened (UTC+7, 2026-10-19)

| Time | Event |
|---|---|
| 11:03 | Read-only preflight all PASS: clean tree, HEAD = `w5-bs15-eval` (`4fecfa5`), `src/` = RC4, no claim/windows, `verify_bundle` 0 problems, server ready, no old units, staging empty. Launched `w5-bs15` + `w5-bs15-watch` |
| 11:03 | Driver and lanes verified under the pool lock; `final.started.json` + block-1 windows (0of2 / 1of2, pairs 0–12) claimed |
| 11:20:10 | **Unit stopped on the user's decision**: cost/benefit rather than a technical fault. No block had completed and no error had occurred. The watchdog saw the unit inactive and exited normally (no LANE FAILURE) |

Totals: 17 min wall, 2.3 CPU-hours, peak memory 11.2 GB. 4 run dirs with 40 completed battles, all with `error` null, plus in-progress battles that were killed. Staging was empty and no process was left.

Claim markers are committed; they contain only `purpose`/`config`, with no gameplay:

| File | sha256 |
|---|---|
| `final.started.json` | `4a3f51f0897d49bd819c81a6cd1d172722769d303ecd5550bf609bba2682fbff` |
| `final.window.0of2.0-12.json`, `final.window.1of2.0-12.json` | `beb7ce81d86fee91b4960368c4241d5fa3de06c80761a50d551ab092f9f8fdae` |

## Why it was stopped

The user decided that a further ~6 h of CPU did not match the value of the evidence:
- C stays production whatever the outcome;
- W6 plans to make search much cheaper than 5 s/decision, so the confirmed artifact would be replaced and need its own final evaluation anyway.

The decision was taken before any BS1.5 result was viewed. The alternative of shortening to 300 battles (blocks 1–2) was offered and declined.

## W5 conclusion

| Item | Status |
|---|---|
| C (`models/pokedecision-v1-value`) | **Production** (Fast Mode). Unchanged since `w5-baseline` |
| C+Belief Search (RC4, 5 s/decision) | **Dev-validated, independent confirmation unavailable.** Dev benchmark BS1.2, 600 paired battles: pooled Δ +0.267 [0.210, 0.323]; wins per 100, Grunt 64→96, GymLeader 65→95, C 54→72. The dev pool was also used for QA/requal during fixes, so this estimate may be optimistic. It covers these three opponents only |
| Heldout attempts | BS1.3 harness stall (fixed in RC4 from forensic payload only); BS1.4 verify/claim race (fixed in BS1.4-P, I1–I5 PASS); BS1.5 stopped by decision. All three pools are spent; all partial gameplay stays sealed and is never merged |
| Follow-up | No BS1.6 and no more W5 benchmarks. W6 uses a lighter process: unit tests → smoke → bounded dev benchmark → one final evaluation |
