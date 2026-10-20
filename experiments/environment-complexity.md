# MTG-122: learning across environment complexity

## Split GPU-training/Mini-evaluation plan — 2026-10-19

**Complete.** All nine fits and all 36,000 games; every pool learns well past
the ten-point threshold against greedy (see Results below).

Jack Heart directed GPU training with evaluation run separately on the Mini, no
inline diagnostic evaluation, and all 36,000 scored games. The 104-game
amendment below is not selected. MTG-135 protocol017 finished; for the
64-stream plan it recommends 64x2, which is the architecture already shared by
L1/L2/L3 (final package from source `3c3681d2`, recipe `small-s64.json`).

The plan is `SplitComplexityPlan` (`schema_version: 3`) in
`experiments.runners.complexity_campaign`; its frozen values are in
[environment-complexity-split-plan.json](environment-complexity-split-plan.json).

- **Recipe:** the tested 64-stream horizon. 128 iterations x 64 streams x 64
  transitions = 524,288 learner transitions per fit, with
  `iteration_schedule: {steps: 128, reference_steps: 2048}`, so learning rate
  stays 1e-4 and tau falls from 0.05 to 0.005076577477226473. Each fit has its
  protocol017 watchdog, 1,740 seconds. CUDA, w6, shared research catalog,
  fresh initialization. Nothing else in the learner changes.
- **GPU job:** trains nine fits (three seeds x three pools, rotated order) and
  retains initialization, iteration 32 (unscored), midpoint 64 and endpoint 128
  exports. It runs no games. The job checks that each fit's exports exist and
  that initialization tensors match across pools.
- **Mini evaluation:** after the outputs are fetched into a campaign directory,
  run `uv run python -m experiments.runners.complexity_campaign evaluate
  <campaign>` on the Mini. It scores the unchanged cohort below (50 deals x 8
  legs per anchor; initialization, midpoint and endpoint candidates plus
  same-deal frozen-initialization controls) from the hash-checked exports. A
  rerun skips completed cells and stops at a retained incomplete one.
- **Cost:** the L4 rental is capped at $0.69/hour for 19,800 seconds ($3.80), which includes
  the in-job native build.
  The plan counts $4.65 already spent or reserved (including
  failed attempt `a53e679aef2a`, below), plus the $3 settlement
  reserve, against the $35 cap. The horizon is fixed by the tested recipe, so
  no update count is calibrated. The corrected calibration's slowest block,
  5.41 s/update, projects about 11.5 minutes per fit.

First attempt `a53e679aef2a` (source `90275997`, $0.0914 SDK estimate) trained
L1 seed 12251 and then failed its initialization check: the preflight digest,
computed with the GPU host's default thread count, differed from the producer's
1-thread initial tensors. Orthogonal initialization uses a QR decomposition
whose x86 MKL result depends on thread count. The preflight now uses the
recipe's thread count. Its outputs are retained under
`.runs/mtg122/gpu-split-training/retained`; that fit is not a scored candidate.

**Training complete — 2026-10-19.** Attempt `9ed2ffaa2d2d` (source `56dff0b1`)
trained all nine fits in 3,235 seconds on one L4, SDK estimate $0.1957. All 104
outputs were fetched with digest verification to
`.runs/mtg122/gpu-split-training-2/retained` and copied byte-identically to the
Mini. Every fit completed 128 iterations, 524,288 learner transitions and
8,192 optimizer updates, retaining exports at 32 and 64.

| Pool | Seconds per fit (12251/12252/12253) | Training games |
|---|---|---|
| L1 | 342 / 331 / 333 | 14,411 / 13,384 / 14,218 |
| L2 | 336 / 332 / 337 | 14,953 / 17,920 / 15,737 |
| L3 | 400 / 392 / 431 | 9,509 / 10,891 / 10,254 |

**Results — 2026-10-19.** The Mini scored all 36,000 games (90 cells, nine
concurrent per-fit processes from source `56dff0b1`; first cell 07:56 UTC,
last 13:35 UTC, 180,343 summed evaluation seconds). No cell was incomplete,
failed or rerun. The fetched campaign is byte-identical to the Mini's (18,193
files) at `.runs/mtg122/gpu-split-training-2/mini-evaluation`. The report,
analysis JSON and editable notebook are in
[data/environment-complexity](data/environment-complexity/comparison.html).

Paired gain over the same seed's frozen initialization on the same deals, in
points. Three-seed mean with 95% t interval (df 2); per-seed values in
parentheses are means with conditional whole-deal intervals in the report.

| Pool | Anchor | Midpoint (64) | Endpoint (128) | Endpoint seeds | Endpoint candidate / control |
|---|---|---|---|---|---|
| L1 | greedy | +28.6 [+22.3, +34.9] | **+35.3 [+34.4, +36.3]** | +35.8 / +35.0 / +35.2 | 54.9% / 19.6% |
| L1 | random | +31.2 [+23.6, +38.9] | +31.7 [+25.0, +38.5] | +34.0 / +28.7 / +32.5 | 83.2% / 51.5% |
| L2 | greedy | +49.5 [+46.7, +52.3] | **+49.8 [+41.9, +57.7]** | +53.5 / +47.8 / +48.2 | 71.5% / 21.7% |
| L2 | random | +37.2 [+34.8, +39.7] | +38.5 [+34.4, +42.6] | +40.0 / +38.8 / +36.8 | 91.4% / 52.9% |
| L3 | greedy | +27.5 [+23.6, +31.4] | **+31.7 [+25.8, +37.7]** | +30.8 / +34.5 / +30.0 | 55.3% / 23.6% |
| L3 | random | +29.1 [+25.1, +33.0] | +28.3 [+23.7, +33.0] | +28.2 / +30.2 / +26.5 | 81.1% / 52.9% |

- **L1 ≥ 10 points against greedy, positive in all seeds:** confirmed.
- **L2 ≥ 5 points:** confirmed.
- **L3 gains less than L1:** not established. The paired seed difference
  L1 − L3 against greedy is +3.6 [−3.0, +10.2]. As frozen above, raw
  cross-pool scores are confounded, so this is not a difficulty ranking.
- The useful-positive diagnostic holds in every pool: all three greedy seed
  effects positive, mean ≥ 10, seed lower bound > 0, and no random-anchor
  regression.
- Most of the gain arrives by the midpoint. From 64 to 128 the greedy gain
  rises 6.7 points in L1, 0.3 in L2 and 4.2 in L3; random-anchor gains are flat.

**Progress.** The frozen clock, `active_training_seconds`, is unavailable for
every fit because initialization checkpoints record that clock as null rather
than zero; the analysis correctly refuses to impute it. On the `updates` and
`learner_transitions` clocks (supplementary, computed with the same analysis
owner), all nine fits cross ten points against greedy within (0, 64] updates,
i.e. (0, 262,144] transitions, and stay above at the endpoint. Midpoint active
learning took 162–227 seconds per fit.

**Frozen controls.** Untrained networks act almost uniformly: the action-score
head is initialized at gain 0.01 (`manabot/model/agent.py`) and the arena seeds
each game's policy. Different seeds' controls therefore play nearly identical
games (L1 greedy midpoint: 398 of 400 games with the same decisions across two
seeds), so control scores are effectively one near-random baseline per pool.
The seed spread reflects training variation only. This is a property of the
design, not a defect, but it means "gain over initialization" here is "gain over
a near-uniform policy."

**Cost.** GPU for this plan: $0.0914 (failed `a53e679aef2a`) + $0.1957
(`9ed2ffaa2d2d`), SDK estimates pending settlement. The plan's $4.65 prior already includes
the failed attempt, so the inclusive total is about $4.85 against the $35 cap. Training
executor time 3,233 seconds over nine fits. Evaluation ran on the Mini with no
GPU spend.

**Supplementary heuristic anchor — 2026-10-20.** Greedy was near the scoring
ceiling for some checkpoints, so the same retained exports were also scored
against MTG-152's `heuristic`. This anchor was chosen after the frozen results,
so it is supplementary, not a frozen prediction test. Same deals, legs and
frozen controls as above: 18,000 games (45 cells), source `102509ac`, an engine
binary byte-identical to the frozen cells', 00:52–03:41 UTC on the Mini, no
failures or reruns. Before scoring, the heuristic was pointed at the research
catalog (`complexity.source.json`), so it plays Lightning Bolt by effect.
Previously it would have fallen back to v1's seat-ordered target and could Bolt
its own side. The product-card programs are identical in both catalogs.

| Pool | Midpoint gain | Endpoint gain | Endpoint seeds | Endpoint candidate / control | Greedy − heuristic endpoint gain |
|---|---|---|---|---|---|
| L1 | +23.1 [+14.0, +32.2] | **+29.1 [+27.5, +30.6]** | +28.5 / +29.8 / +29.0 | 48.1% / 19.0% | +6.2 [+3.8, +8.7] |
| L2 | +25.4 [+22.6, +28.2] | **+30.2 [+27.0, +33.4]** | +31.2 / +30.5 / +28.7 | 35.4% / 5.3% | +19.7 [+13.4, +25.9] |
| L3 | +21.5 [+17.7, +25.3] | **+24.7 [+11.0, +38.5]** | +25.2 / +30.0 / +19.0 | 33.8% / 9.1% | +7.0 [−1.7, +15.7] |

- Learning holds against the stronger anchor: every seed in every pool gains
  at least 19 points. All three pools pass the useful-positive diagnostic
  against `heuristic` as well.
- The heuristic is much stronger than greedy where it has rules to apply. In
  L2 it holds the near-uniform control to 5% (22% for greedy), and the
  candidate's gain shrinks by about 20 points. In L1, with vanilla creatures
  only, it barely beats the control harder than greedy does.
- No trained endpoint beats the heuristic: candidates score 34–48%. Against
  this anchor, gains rise from midpoint to endpoint in every pool (+6.0, +4.8
  and +3.2 points), including L2, whose greedy gain was flat (+0.3).
- L3 has the widest seed spread (seed 12253 gains 19 points; the others gain
  25 and 30).

Mini evaluation logs are retained at
`.runs/mtg122/gpu-split-training-2/heuristic-evaluation-logs`. The fetched
campaign, now 27,238 files, is byte-identical to the Mini's.

Historical plans (schema 1 and 2) keep their meaning; the deal families are the
same by milestone ordinal.

## Current disposition — 2026-10-17

**Reactivated for GPU preparation; not scientifically complete or launch-admitted.**
Jack Heart selected 64 streams, following MTG-135's findings, and GPU rather than
Mini for the three-pool experiment. This supersedes the baseline-first deferral.
Existing baseline execution/recovery/evaluation remains MTG-137 (`fc765e`);
its running jobs, sources and allocations are not changed or borrowed here.
Jack Heart authorized a separate **$35 inclusive GPU restart cap** on
2026-10-17. Charge calibration, setup, training, all evaluation, failed attempts,
artifact transfer/storage and cleanup to it. Historical baseline allocations
remain separately accounted; this is not another $35 for MTG-137.
Reserve at most $2 for initial calibration and $3 for artifact/settlement costs,
leaving at most $30 for admitted scientific execution. No automatic paid retries.
The full three-seed/three-pool evaluation matrix must fit measured admission;
otherwise report the capacity shortfall before renting for science.

Use one identical 64-stream model/learner configuration across L1/L2/L3.
Do not silently inherit the old 1,200-update target or CPU calibration: at
64 transitions per stream, one update collects 4,096 learner transitions,
16 times the historical four-stream batch. Training horizon, absolute-iteration
schedules, timestep-filtered optimizer exposure and evaluation milestones must
be explicitly frozen with the GPU plan. MTG-135's exact recipe and numerical
findings must be checked before treating its 64-stream result as transferable.
Calibration and CPU arena work on the GPU host count toward the inclusive GPU
allocation; no Mini or local scientific execution is selected for this restart.

The implementation includes vanilla simple-pool content, shared semantic catalog
binding, ordinary checkpoint admission, an Etude runner and analysis/reporting
support. Original calibration completed all three pools. The stopped campaign
`063198864bfb` retained three pool fits for seed12251, but not the completed
three-seed comparison. Its pre-fix world, interrupted cohort and original receipts
remain historical evidence, not corrected-world learning conclusions.

Remaining outcomes:

- Re-admit the three-pool contrast on a pinned corrected world, with an explicit
  64-stream GPU configuration and separately bounded resource allocation. The old allocation
  below is historical, not a fresh reservation.
- Execute the complete independent-seed/control/evaluation matrix and analyze
  within-pool improvement, uncertainty, costs and failures before concluding
  whether simpler games provide a useful learning diagnostic.
- Design and separately admit the requested catalog-size contrast, distinguishing
  definition count, semantic-token vocabulary and program lengths. The common
  full catalog in the original contrast holds this factor fixed; it does not
  measure catalog-size effects.
- Keep curriculum transfer separately selectable; no transfer outcome is claimed.

The frozen protocol below preserves the original declaration and predictions.
A future corrected-world plan must retain this lineage rather than overwrite it.

## Corrected GPU calibration — 2026-10-17

Run `0df9b17970bf`, source `0cd95fa8`, completed three reserved-seed12250
fits of100 updates/409,600 learner transitions each and six eight-game replayed
anchor cohorts (48 games). All60 published outputs were fetched with SDK digest
verification under `.runs/mtg122/gpu-complexity-startup1/retained`.
The L4 rental ended; estimated SDK cost $0.24563316, not an invoice.
No calibration checkpoint is a scientific candidate.

| Pool | Training seconds (100 updates) | Worst 20-update block, seconds/update |
|---|---:|---:|
| L1 vanilla |375.51|4.031|
| L2 interaction |379.63|4.034|
| L3 Allies/Lessons |462.00|5.410|

The frozen historical count resolver rejects this calibration: its two-hour
per-fit cap admits800, below its1000-update minimum. Moreover, the retained
worst-cell/coordinator costs with1.5× margin project51.53 hours for36,000
assessment games. A1000-update target needs2.287 hours/fit with margin; nine fits,
full assessments and30 minutes setup/publication total72.62 hours, exceeding
this study's allocation. These are projections, not sustained-throughput claims.

**Proposed amendment, awaiting Jack Heart's decision:** retain1000 updates,
three seeds, all three pools, all four deck-role cells, initialization/midpoint/
endpoint and same-deal frozen controls. Use13 paired deals per cell per anchor
(26 games/cell;104 games/anchor/checkpoint), totaling9360 assessment games.
The same margin projects13.40 assessment hours and34.48 inclusive hours before
separately retained historical costs/reserves; at the $0.69 ceiling, $23.80.
This retains at least100 balanced greedy games per checkpoint, but loses
per-cell precision. Do not silently overwrite the earlier50-deal protocol or
claim the original cohort completed. No scientific launch follows until the
amendment and exact plan are frozen.

## Frozen scientific protocol — 2026-10-14, before scored execution

Jack Heart authorized Mini for up to 48 hours or $25 GPU spending, without local
experiment execution. This cohort selects Mini only. Reserve one hour inclusive
for calibration/failures, 46.75 hours for the scientific SDK job, and 15 minutes
for admission/transfer/settlement outside that job. No GPU budget is added.

The executable declaration is `experiments.runners.complexity_campaign` and the
exact count/cost plan is [environment-complexity-plan.json](environment-complexity-plan.json).
Etude owns remote execution; TrainingRun and the replayed arena own evidence.

## Treatments and predictions

Three independent seeds: 12251, 12252, 12253. Each trains 1,200 updates (307,200
learner transitions before filtering), retaining initialization, update 600 and
update 1,200. Pool order rotates across seeds. Cross-balanced current-policy
self-play uses four streams, 64 transitions per stream per update. All arms use
the same full research semantic catalog, width64/two-layer/four-head encoder,
scalar masked-mean value and the same Ataraxos-move learning recipe. The source
recipe is `experiments/regimes/early-learning-rate-mini-source.json`; no LR-screen
variant is selected. Exact resolved Experiment receipts are exported before work.

- L1: Mountain9/Plains8; swift has Savannah Lions8/Glory Seeker8/Hill Giant7,
  heavy has 4/8/11. All creatures are ability-free.
- L2: the same lands; swift creatures6/5/4, heavy2/5/8; both add four Lightning
  Bolts and four Yip Yip! cards. Yip Yip!'s real Ally clause is not stripped;
  these creatures have no Ally subtype. Simple decks have empty sideboards.
- L3: unchanged UR Lessons41/GW Allies40 product decks and sideboards, under
  the explicitly selected shared `mtg122-complexity-v1` research content binding.

Exact definitions/decks are generated by `scripts/compile_complexity_content.py`.
Original product source/IR/schema bytes remain unchanged. Research content is a
new binding; existing product checkpoints are not retroactively admitted.

**Predictions:** L1 gains at least ten points against greedy over frozen
initialization, positive in all three seeds; L2 gains at least five; L3 gains
less than L1. Endpoint is primary. Midpoint is descriptive, not a stopping test.
Different curves, colors, action mixes and opponent competence confound raw
cross-world scores. No result here automatically promotes the product baseline.

## Cohort and analysis

For each pool/seed/milestone, both fixed greedy and random anchors receive four
role cells (A/A, A/B, B/A, B/B), each with 50 held-out paired deals in both play
orders: 400 games per anchor, 800 per checkpoint cohort. Initialization/mid/end
families are disjoint. Mid/end reevaluate exact frozen initialization on the
candidate's same deals and action streams. Initial weights are hash-checked
across pools for each seed. Evaluation is stochastic policy, CPU/batch1, no search.

Nine fits × five cohorts (initial; mid candidate/control; end candidate/control)
×800 = **36,000 games**. No replacement seed, selected checkpoint or silent
cohort reduction. Calibration uses seed12250 and separate deals; its scores do
not enter scientific estimates. Complete-game watchdog is120 seconds/10,000
Commands. Incomplete cells suppress their rates and seed-level method claims.

Report per-cell endpoint scores and paired improvement. Whole-deal bootstrap
(10,000 replicates, seed12290) conditions on a checkpoint; the three paired
training-seed effects use a separate 95% t interval, df2. Never pool games as
independent training seeds. Progress threshold is ten points against greedy,
confirmed at endpoint; distinguish interval/right censoring and unconfirmed
crossings. Report collected transitions, optimizer exposures, active learning,
executor elapsed and inclusive experiment cost separately. Frozen controls have
zero learning exposures. A useful positive diagnostic requires all three greedy
seed effects positive, mean≥10 points, seed interval lower bound>0 and no
established random-anchor regression; nonsignificance is not equivalence.

The analysis owner is `environment_complexity_analysis.py`; metric meanings are
in `docs/experiment-metrics.md`. Editable notebook/read-only HTML and archived
originals remain deliverables, not implied by successful execution. Catalog-size
and curriculum-transfer comparisons are separate, not part of this cohort.

## Calibration and cost admission

Etude calibration `60cfb89655fa` (source330d365b) completed all three100-update
runs and48 complete replay-verified games. First attempt `ff9f4cd90d03` failed
before learning because source archives lacked Git metadata. The repaired
source-bundle admission verifies submitted exact source hashes plus native bytes;
both attempts remain retained. Local retained outputs: `.runs/mtg122/calibration`.
The plan binds all nine calibration TrainingRun/evaluation JSON hashes.

The initial cost model charged every evaluation game at the globally slowest
cell and rejected the plan. **Pre-scoring amendment:** reserve each pool/anchor
panel at its own slowest measured cell mean or full coordinator/replay mean,
whichever is greater, with1.5× margin. Each of six panels has6,000 games.
This preserves all36,000 games and all seeds, rather than shrinking the cohort.
The resulting evaluation reserve is26.833692 hours, plus18 hours for nine
bounded learners and600 seconds setup/publication reserve:45.000359 hours.
Common1,200 updates use1.5× the worst observed20-update block with a120-second
per-run overhead reserve. Each learner remains capped at two hours.

These sparse timings are extrapolations, not sustained throughput guarantees.
Every failure and overrun counts against the fixed allocation. Unknown/failed
attempts are not silently restarted. The SDK guardian bounds the scientific job;
remaining-time checks cannot extend it. An incomplete or negative result is
retained and analyzed honestly, not reported as successful learning.
