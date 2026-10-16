# Corrected-world baseline — MTG-122

2026-10-15. Sustained-horizon amendment: Jack Heart initially approved a 72-hour inclusive
Mini ceiling and 10,000 updates per seed with reduced evaluation. The prior
3,200-update attempt is stopped and retained; it is not a continuation source.

Jack Heart authorized stopping the w4 complexity study, deferring
card-pool comparisons, and launching an Allies/Lessons baseline after PR288/289.
Both are merged. This is a fresh w6/layout2 study, never checkpoint continuation.

## Question and recipe

Does the evidence-backed small masked-mean Ataraxos-move recipe learn useful
full-game play with attack, target, mode and ability choices now distinguishable?
ETU-118's three-seed sustained monitoring improved against both anchors; that
supports this recipe as a baseline, not a claim that it wins every treatment.
`experiments/current-baseline.md` retains that evidence and its acceptance limits.
The exact recipe is the original `early-learning-rate-mini-source.json` (not
its LR-screen treatments): width64, depth2, heads4, scalar masked mean, four
streams, 64 transitions/stream/update, terminal reward, Adam, original iteration
LR/tau schedules, quantile/floor/filter/reference settings, raw evaluation.

Explicit differences from historical execution: w6/layout2, corrected action and
permanent inputs, current numerical repairs, explicit cross-balanced collection,
new seeds/deals, one step-target stage without recovery, and the public Etude
launcher. Decklists, programs and product semantic catalog remain unchanged.
The larger research catalog is not used. No mirror training or curriculum.
Thus this is corrected-world baseline establishment, not a causal estimate of
one bug fix or byte-identical historical reproduction. Historical w4 scores are
context only. Fresh raw/EMA exports preserve ordinary checkpoint admission.

## Predeclared execution and count rule

Runner: `experiments.runners.corrected_baseline`, through the existing shared
TrainingRun/VerifyStore/arena execution and Etude job, on Mini only. No laptop
experiment, GPU expenditure, automatic retry or resume. A single SDK hard deadline
includes setup, native build, calibration, all learning/evaluation and cleanup.

After the conservative admission exceeded the53.5h job, Jack Heart requested
launch without further budget fine-tuning. The bounded operational ceiling is now
**80 inclusive Mini hours**:18h prior attempts/preparation,61.5h new allocation,
0.5h settlement. Successful calibration from `1ebb94c24ef2` is reused, not rerun.
Its100 updates and16 completed games bind the committed
`corrected-world-baseline-plan.json` through exact receipt hashes; current recipe
equality was checked. It reserves48.6918h training and4.8623h evaluation plus
600s overhead. The60h plan allowance sits inside the61.5h job. No GPU spending is added.
The stopped w4 and short w6 attempts retain original receipts and artifacts.
The earlier w4 calibration reserves and failed admissions remain charged.

Calibrate with reserved seed12260:100 updates (1200-second learner cap), plus
one complete eight-leg deal against each anchor. Calibration scores cannot choose
the recipe, seeds, or count. Retain all failures; incomplete calibration stops.
The scientific plan is saved durably before its first fit, with **10000 updates
per seed fixed in advance**. Admission reserves1.5 times the slowest measured
20-update block times10000, plus120 seconds per learner. It never shortens the
horizon to fit. Evaluation reserves1800 games per anchor times1.5 times the larger
of its slowest role-cell mean or full coordinator/replay seconds/game. All three
learners plus evaluation and600 seconds must fit the remaining SDK time; another
120 seconds remain outside the scientific plan. Insufficient capacity stops.
These timings extrapolate, not guarantee sustained performance. The SDK hard
deadline remains binding; no automatic extension or replacement seed.

Science seeds12261/12262/12263 each begin independently from initialization.
Each fit collects256 learner transitions/update before filtering:2,560,000
per seed and7,680,000 total. Score0 and10000 only; the5000-update export remains
unscored. Raw only, stochastic one-forward-pass CPU policy. Each scored cohort
has25 held-out deal blocks ×8 legs covering AA/AB/BA/BB balanced over play order,
against fixed scripted-greedy and random anchors:200 games per anchor and
50 games per role cell/anchor. Endpoint also re-evaluates exact frozen
initialization on the candidate's same deal/action streams. Three cohorts/fit
×400 games ×3fits =3600 games. Initialization and endpoint families begin at
1226100000 and1226101000; calibration uses1226000000. The short attempt's scores
did not select this change; no best-checkpoint or seed replacement is permitted.
Old attempted outputs stay separate, not pooled with the new cohort.

## Analysis and stopping

Prediction before scoring: all three endpoint greedy changes positive, mean at
least10 percentage points over same-deal initialization; random improvement
positive on average. This is a prediction, not a stopping rule. Report all four
role cells, per-seed paired effects, mean and95% t interval(df2), separately from
conditional whole-deal bootstrap intervals(10000 replicates, seed12290).
Retain the existing paired analysis/censoring semantics and cost/exposure axes.
Only endpoint progress is observed; any10-point threshold crossing is interval-
censored between initialization and endpoint, not dated to the unscored export;
missing endpoints stay unavailable. No method-level conclusion from one seed.

Every completed cohort requires exact replay and ordinary legality. Game watchdog
120seconds/10000Commands. Missing cells, failures, nonfinite learning or identity
mismatch remain retained and prevent complete-cohort claims. This plan neither
promotes a product baseline nor establishes chapter/human-play acceptance.
Scientific outputs stay private; editable notebook/read-only HTML and inclusive
cost settlement follow retained results. Complexity, catalog-size and curriculum
comparisons are deferred, not automatically queued after this baseline.

## Prospective cadence amendment (2026-10-15)

Jack Heart requested exports every 500 updates and diagnostic evaluation every
1,000 for future seeds, while evaluating only the existing 5,000 export for the
current seed. Seed 12261 continues unchanged to 10,000 on source `09dc662d`.
A bounded Mini sidecar evaluates its 5,000 export (13 paired deals × eight
deck/play-order legs per greedy/random anchor; 208 games, 30-minute ceiling).
These diagnostic deals are disjoint from the fixed scientific cohorts.

The sidecar uses the existing stop-file contract only after the first complete
TrainingRun, holding the old three-seed job before another learner starts. The
replacement job retains that producer and completes its fixed evaluations, then
runs seeds 12262/12263 with 500-update exports and diagnostics at 1,000 through
9,000; the 10,000 endpoint keeps the larger original evaluation. Original partial
evaluation attempts remain retained, not rewritten as successes. No first-seed
optimizer restart, checkpoint port or changed learning recipe is involved.

Each future seed receives at most 2.75 evaluator hours (900 seconds per anchor
attempt) through the existing evaluator lease/queue. Budget exhaustion retains
missing diagnostics, never changes endpoint selection. Concurrent evaluation may
slow learning; export/cadence and source changes must remain explicit in seed
comparisons. The original scientific plan, seeds, targets and endpoint deals are
unchanged. The new job has a 44-hour cap covering two training allowances, all
three fixed evaluation cohorts, diagnostics and setup; prior 18 hours, first-fit
allowance and settlement remain inside the 80-hour operational ceiling.

Seven focused software checks passed; no local training or evaluation ran.

## Historical GPU context (2026-10-15)

Jack Heart recalled roughly 50% piloting Lessons against scripted greedy Allies.
Read-only retrieval of the versioned ETU-103/126 monitoring artifacts confirms
that result for the small GPU seed10351 and its continuations, not the ETU-118
Mini baseline. Each historical deck split contains50 games; all are completed
monitoring cohorts, not independent training seeds or held-out confirmation.

| Absolute update | Lessons wins /50 | Recorded training hours | Active training hours |
| --- | --- | --- | --- |
| 26,000 (original endpoint) | 21 | 4.93 | 4.41 |
| 31,165 | 22 | 5.93 | 5.22 |
| 54,645 | 23 | 9.93 | 8.98 |
| 60,467 | 24 | 10.93 | 9.92 |
| 63,000 | 18 | 11.36 | 10.33 |
| 68,014 | 24 | 12.36 | 11.13 |
| 73,797 | 24 | 13.36 | 12.07 |
| 79,478 | 20 | 14.36 | 13.00 |
| 85,145 | 25 | 15.36 | 13.93 |
| 90,732 | 23 | 16.36 | 14.86 |
| 96,324 | 26 | 17.36 | 15.80 |
| 100,000 | 25 | 18.02 | 16.41 |

These are checkpoint clocks, not inclusive rental/queue time. The first observed
50% point is85,145; fluctuations preclude a sustained-threshold claim. The100k
monitor digest is `de1f2ea2885063fdf42e671b64e9a5d81bd58f12fff504b63602f74b37df6d6e`;
the85,145 digest is `f3d9035bb905fc85a54ecd33adf57f6bcea7ec9ef3971f3f7d20314063a4ef01`.
Versioned manifests and verified monitor bytes remain in
`.runs/mtg122/historical-gpu/cache`; `summary.json` retains all retrieved points.
No historical artifacts or jobs were changed.

The current5k seed scores12/26 (46.2%) in that cell after5.87 recorded training
hours (5.76 active) on Mini. It collects256 learner transitions/update versus512
for the old GPU run:1.28M transitions now,43.59M at the old85,145 point. Worlds,
value aggregation, collection balance, hardware, seeds and evaluation deals differ.
This is descriptive context, not a causal speedup or matched-compute comparison.
