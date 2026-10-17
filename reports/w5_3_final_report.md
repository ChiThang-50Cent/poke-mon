# W5.3 AWBC pilot final report

Dev-only exploratory evaluation. Production C is unchanged; heldout was not used.

## Greedy confirmation (400 battles/opponent/arm; three training seeds)

| Comparison | Pooled delta (pp) | Paired 95% CI (pp) | Verdict | Positive seeds | Severe regressions |
|---|---:|---|---|---:|---|
| E1-C | +1.63 | [-0.32, +3.68] | inconclusive | 3/3 | none |
| E2-C | +0.45 | [-1.48, +2.35] | inconclusive | 2/3 | none |
| E2-E1 | -1.18 | [-2.60, +0.22] | inconclusive | 0/3 | none |
| E3-C | +1.08 | [-0.87, +3.08] | inconclusive | 3/3 | none |
| E3-E1 | -0.55 | [-2.02, +0.88] | inconclusive | 0/3 | none |
| E3-shuffled | +0.37 | [-1.05, +1.78] | inconclusive | 2/3 | none |
| E3-shuffled-C | +0.72 | [-1.25, +2.70] | inconclusive | 2/3 | none |

## Absolute greedy win rates (Wilson 95% intervals)

- C: 1356/2000 (67.80%, [65.72, 69.81]%).
- E1-none-s1: 1392/2000 (69.60%, [67.55, 71.58]%).
- E1-none-s2: 1381/2000 (69.05%, [66.99, 71.04]%).
- E1-none-s3: 1393/2000 (69.65%, [67.60, 71.63]%).
- E2-0.5-s1: 1377/2000 (68.85%, [66.79, 70.84]%).
- E2-0.5-s2: 1351/2000 (67.55%, [65.47, 69.57]%).
- E2-0.5-s3: 1367/2000 (68.35%, [66.28, 70.35]%).
- E3-2.0-s1: 1385/2000 (69.25%, [67.19, 71.23]%).
- E3-2.0-s2: 1364/2000 (68.20%, [66.13, 70.20]%).
- E3-2.0-s3: 1384/2000 (69.20%, [67.14, 71.18]%).

## Per-opponent and seed evidence

### E1 vs C
Seed paired deltas (pp): E1-none-s1: +1.80, E1-none-s2: +1.25, E1-none-s3: +1.85
- RandomBaseline: +1.17 pp [-0.17, +2.75], inconclusive.
- Grunt: +0.17 pp [-4.59, +5.08], inconclusive.
- GymLeader: +1.00 pp [-3.75, +5.75], inconclusive.
- EmeraldKaizo: +3.58 pp [-1.42, +8.58], inconclusive.
- v1: +2.25 pp [-2.67, +7.33], inconclusive.

### E2 vs C
Seed paired deltas (pp): E2-0.5-s1: +1.05, E2-0.5-s2: -0.25, E2-0.5-s3: +0.55
- RandomBaseline: +1.00 pp [-0.50, +2.58], inconclusive.
- Grunt: +0.25 pp [-4.42, +5.09], inconclusive.
- GymLeader: -1.75 pp [-6.58, +3.08], inconclusive.
- EmeraldKaizo: +2.50 pp [-2.42, +7.42], inconclusive.
- v1: +0.25 pp [-4.50, +4.92], inconclusive.

### E3 vs C
Seed paired deltas (pp): E3-2.0-s1: +1.45, E3-2.0-s2: +0.40, E3-2.0-s3: +1.40
- RandomBaseline: +0.92 pp [-0.50, +2.50], inconclusive.
- Grunt: -1.17 pp [-6.17, +3.92], inconclusive.
- GymLeader: +0.50 pp [-4.00, +5.00], inconclusive.
- EmeraldKaizo: +3.58 pp [-1.50, +8.75], inconclusive.
- v1: +1.58 pp [-2.83, +6.09], inconclusive.

## Sampling at T=1.0 (200 battles/opponent/arm)

- E1-C: +0.93 pp [-1.77, +3.70], inconclusive.
- E2-C: +0.87 pp [-2.13, +3.80], inconclusive.
- E3-C: +0.20 pp [-2.73, +3.13], inconclusive.

## Training policy diagnostics

- E1-none-s1/train: KL(C||actor) 0.014765 nats; entropy 1.299743 vs C 1.326455; greedy move/switch/Tera {"move": 0.7324977817213842, "switch": 0.26133540372670805, "tera": 0.00616681455190772}; sampled move/switch/Tera {"move": 0.6521117832349694, "switch": 0.3269445773025689, "tera": 0.020943639800944155}.
- E1-none-s1/val: KL(C||actor) 0.011205 nats; entropy 1.317532 vs C 1.338745; greedy move/switch/Tera {"move": 0.739871472478346, "switch": 0.25407469497997576, "tera": 0.006053832541678308}; sampled move/switch/Tera {"move": 0.6516126810510676, "switch": 0.3259816538791278, "tera": 0.022405677771250763}.
- E1-none-s2/train: KL(C||actor) 0.014982 nats; entropy 1.304902 vs C 1.326455; greedy move/switch/Tera {"move": 0.7283274179236913, "switch": 0.26535048802129546, "tera": 0.006322094055013309}; sampled move/switch/Tera {"move": 0.6477709461337396, "switch": 0.3310697259369844, "tera": 0.021159322978970004}.
- E1-none-s2/val: KL(C||actor) 0.011327 nats; entropy 1.322795 vs C 1.338745; greedy move/switch/Tera {"move": 0.7364254447238521, "switch": 0.25724131507869985, "tera": 0.0063332401974480765}; sampled move/switch/Tera {"move": 0.647336159914224, "switch": 0.32997974368394994, "tera": 0.022684108481523076}.
- E1-none-s3/train: KL(C||actor) 0.014723 nats; entropy 1.301781 vs C 1.326455; greedy move/switch/Tera {"move": 0.732187222715173, "switch": 0.26157941437444543, "tera": 0.006233362910381544}; sampled move/switch/Tera {"move": 0.6518134359987626, "switch": 0.3272123274172738, "tera": 0.020974239080271684}.
- E1-none-s3/val: KL(C||actor) 0.011022 nats; entropy 1.319320 vs C 1.338745; greedy move/switch/Tera {"move": 0.7396852007078327, "switch": 0.254260966750489, "tera": 0.006053832541678308}; sampled move/switch/Tera {"move": 0.650956487748902, "switch": 0.32658310910004906, "tera": 0.02246039622298741}.
- E2-0.5-s1/train: KL(C||actor) 0.017571 nats; entropy 1.289983 vs C 1.326455; greedy move/switch/Tera {"move": 0.7285048802129548, "switch": 0.2650399290150843, "tera": 0.006455190771960958}; sampled move/switch/Tera {"move": 0.6507708653582867, "switch": 0.32810739248118365, "tera": 0.02112173887090336}.
- E2-0.5-s1/val: KL(C||actor) 0.013886 nats; entropy 1.307229 vs C 1.338745; greedy move/switch/Tera {"move": 0.7365185806091087, "switch": 0.25714817919344324, "tera": 0.0063332401974480765}; sampled move/switch/Tera {"move": 0.6513736082142576, "switch": 0.3262512002386971, "tera": 0.022375185551607397}.
- E2-0.5-s2/train: KL(C||actor) 0.017677 nats; entropy 1.292878 vs C 1.326455; greedy move/switch/Tera {"move": 0.7277284826974268, "switch": 0.2660603371783496, "tera": 0.006211180124223602}; sampled move/switch/Tera {"move": 0.6491616128076995, "switch": 0.3298444945937041, "tera": 0.020993894185232925}.
- E2-0.5-s2/val: KL(C||actor) 0.014006 nats; entropy 1.309221 vs C 1.338745; greedy move/switch/Tera {"move": 0.7344695911334638, "switch": 0.2593834404396014, "tera": 0.006146968426934898}; sampled move/switch/Tera {"move": 0.6492810067689051, "switch": 0.32846223956059506, "tera": 0.022256765927839425}.
- E2-0.5-s3/train: KL(C||actor) 0.017215 nats; entropy 1.292112 vs C 1.326455; greedy move/switch/Tera {"move": 0.7305013309671695, "switch": 0.26308784383318545, "tera": 0.006410825199645076}; sampled move/switch/Tera {"move": 0.6520370138146334, "switch": 0.32680602594086205, "tera": 0.02115695956753965}.
- E2-0.5-s3/val: KL(C||actor) 0.013580 nats; entropy 1.308177 vs C 1.338745; greedy move/switch/Tera {"move": 0.7367979882648784, "switch": 0.2566824997671603, "tera": 0.006519511967961256}; sampled move/switch/Tera {"move": 0.652011339475936, "switch": 0.32558280644252935, "tera": 0.02240585337096423}.
- E3-2.0-s1/train: KL(C||actor) 0.015085 nats; entropy 1.297521 vs C 1.326455; greedy move/switch/Tera {"move": 0.7317435669920141, "switch": 0.26224489795918365, "tera": 0.006011535048802129}; sampled move/switch/Tera {"move": 0.6518957623783846, "switch": 0.32717210656261697, "tera": 0.020932126478907697}.
- E3-2.0-s1/val: KL(C||actor) 0.011472 nats; entropy 1.315208 vs C 1.338745; greedy move/switch/Tera {"move": 0.7387538418552668, "switch": 0.25509918971779827, "tera": 0.006146968426934898}; sampled move/switch/Tera {"move": 0.6517207289670248, "switch": 0.32595384357427876, "tera": 0.022325429057479848}.
- E3-2.0-s2/train: KL(C||actor) 0.015266 nats; entropy 1.302079 vs C 1.326455; greedy move/switch/Tera {"move": 0.72877107364685, "switch": 0.26506211180124223, "tera": 0.00616681455190772}; sampled move/switch/Tera {"move": 0.6483370641320811, "switch": 0.3305859048702914, "tera": 0.021077034657469025}.
- E3-2.0-s2/val: KL(C||actor) 0.011570 nats; entropy 1.319562 vs C 1.338745; greedy move/switch/Tera {"move": 0.7369842600353916, "switch": 0.2567756356524169, "tera": 0.006240104312191488}; sampled move/switch/Tera {"move": 0.6479583666335051, "switch": 0.3295153312047145, "tera": 0.02252631113273178}.
- E3-2.0-s3/train: KL(C||actor) 0.014935 nats; entropy 1.299639 vs C 1.326455; greedy move/switch/Tera {"move": 0.7323646850044365, "switch": 0.26157941437444543, "tera": 0.006055900621118012}; sampled move/switch/Tera {"move": 0.6522629786087754, "switch": 0.32677061356795817, "tera": 0.020966405580820188}.
- E3-2.0-s3/val: KL(C||actor) 0.011236 nats; entropy 1.316721 vs C 1.338745; greedy move/switch/Tera {"move": 0.7393126571668064, "switch": 0.2546335102915153, "tera": 0.006053832541678308}; sampled move/switch/Tera {"move": 0.6514668151565556, "switch": 0.32616273990676226, "tera": 0.022370462967406395}.

## Runtime diagnostics

- smoke: runtime errors 0; model-illegal {"player": 0, "opponent": 0}; protocol retries {"player": 0, "opponent": 0}.
- screen: runtime errors 0; model-illegal {"player": 0, "opponent": 0}; protocol retries {"player": 1, "opponent": 0}.
- confirm: runtime errors 0; model-illegal {"player": 0, "opponent": 0}; protocol retries {"player": 15, "opponent": 23}.
- sampling: runtime errors 0; model-illegal {"player": 0, "opponent": 0}; protocol retries {"player": 15, "opponent": 3}.
- control: runtime errors 0; model-illegal {"player": 0, "opponent": 0}; protocol retries {"player": 23, "opponent": 26}. Control-stage totals include reused confirmation records.

## Frozen data and training

C-only decisions: 55,817; re-encoding probability max difference 0.0.
Split hash: `139710deb7a18a64de5491a367a3337e6f4f4c17ef757612fecee51e1446b3b8` (unchanged from W5.2; 45,080 train / 10,737 validation decisions).
Base actor runs: 21, each exactly five epochs; final weights retained. Value heads were removed.
Screen beta choices: {"E2": "0.5", "E3": "2.0"}; highest pooled seed-1 dev delta, deterministic beta-order tie break.
W5.2 V was fitted on these same train decisions (a 385-parameter linear head; minimal but nonzero overfit); C's probe was fitted on human data.
- E2, beta=0.5: A mean -0.007037, std 0.462781; normalized ESS 0.612806, clipped weight fraction 0.001720.
- E3, beta=2.0: A mean -0.010070, std 0.475542; normalized ESS 0.948226, clipped weight fraction 0.000000.

## Answer and limitations

The pilot provides a pre-registered AWBC positive signal, but does not establish that C is beaten.
Stage-2 results are dev-selection evidence, not unbiased final estimates.
Limitations: noisy Monte Carlo advantages credit every decision in a winning battle; T=1 sampled collection differs from greedy production; only the existing 1,000-battle pilot was used; beta/config selection on dev creates selection bias; W5.2 V was fitted on the same data.
Actor sequences retained 49 non-model observations without loss, as required. Frozen critic/reference features preserve serving parity, which skips fallback memory updates; this is an additional train/serve memory caveat.
No new data, heldout pool, W3 test, or W4 test was used. C remains production; promotion is reserved for W5.4/W5.5.

## Reliability, drift, and decision

All completed stages require zero runtime errors and zero model-illegal actions on both neural sides.
Training policy-drift flags: [].
Pilot gate: **expansion_eligible: positive, seed-stable exploratory signal; not production promotion**.
Candidate: **E1-none-s1**, copied to `models/pokedecision-v2-candidate` with artifact hashes and selection manifest.
Candidate selection is not evidence of production superiority. A CI containing zero is inconclusive.
Intervals use paired team-pair bootstrap within opponent, averaging the shared schedule across seeds; they condition on fitted policies. Wilson intervals and full seed/opponent results are in the JSON.
Severe regression: strictly more than 5 pp loss with opponent CI upper bound below zero. Comparisons are exploratory, without multiplicity correction.

## Wall time

- smoke: 156.2 s.
- screen: 1612.2 s.
- confirm: 3962.5 s.
- sampling: 1922.7 s.
- control: 1214.0 s.

## Pre-registered experiment decisions

- E2: positive_signal=True; weighting_matters (mean)=False; weighting supported by paired CI=False; evidence=not_positive; shuffled control evaluated=False.
- E3: positive_signal=True; weighting_matters (mean)=False; weighting supported by paired CI=False; evidence=not_positive; shuffled control evaluated=True.

## Execution and verification

- 21 registered actors plus 3 triggered shuffled-control actors, five epochs each.
- Unique battle logs across all stages: **44,800 completed battles; 0 runtime errors; 0 model-illegal actions**. The control addendum reuses confirmation's C and candidate logs; those are counted once.
- Grid training wall time: 233.47 s; shuffled-control training: 28.48 s.
- Affected tests: **79 passed in 1.03 s**. `ruff check src scripts tests`: **All checks passed!** Exact resource-limited commands and the initial missing-cache/lint corrections are recorded in `w5_3_training_metrics.json` → `execution.verification`.
- Both critics meet the pre-registered positive-signal definition, but neither beats unweighted E1 on the seed-mean comparison. E3 versus shuffled is also inconclusive. **Outcome weighting itself is therefore not established as beneficial.**
- Gate interpretation: positive but wide CIs permits scaling this experiment; it is not a strength claim or evidence that AWBC beats C. The saved E1 candidate does not establish superiority over C.

