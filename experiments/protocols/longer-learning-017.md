# Longer learning across capacity and batching — 017

Frozen 2026-10-18T00:11:37.592499+00:00. This fixes the allocation and analysis before any paid admission or new-cohort scoring.

## Decision to support

Recommend a concrete starting recipe for MTG-122: architecture, streams, total
transitions, LR/tau schedule mapping, raw checkpoint cadence, expected training
cost, and observed playing-strength limits. This remains a comparison of whole
training recipes. Larger collection batches also change Adam update frequency,
filtering, sample grouping, staleness and EMA updates per transition. It does not
isolate the optimal optimizer minibatch or establish an architecture's ceiling.

## Preserved treatments

Use fresh training seeds 13591 and 13592. Each seed includes all five capacities
at both 64 and 256 streams, plus small 64x2 at four streams: 11 cells per seed,
22 fresh TrainingRuns. No architecture is dropped because of screen013 scores.

Capacities are 64x2, 64x4, 192x4, 384x2 and 384x8, with four attention heads,
FF width four times hidden width and the baseline masked-mean scalar output.
Preserve corrected w6, exact decks/content/action/observation identities,
cross-balanced current-self, terminal rewards, 64 transitions per stream,
ordinary Ataraxos MOVE/Adam, FP32, TF32 off, one CPU producer/thread and RTX5090.
No warm-start or inherited optimizer/game/RNG state is permitted.

All cells have one common experience horizon N and raw checkpoints at 0,
131072, 262144 and N transitions. Require exact completed iteration boundaries;
no rounding, substituted checkpoint, best-checkpoint selection or scoring of a
partial endpoint as if complete. Retain the final raw, EMA and Adam exports, but
score only raw policies. Four streams retain the original iteration schedule;
other configurations map their full run endpoints to N/256 reference updates.
EMA stays 0.999 per iteration; current-self uses raw collection/bootstrap targets.

Training source is 3c3681d2b26a7fc8167928fa63fe24a6cd3d058a. Compared with the
added-shape feasibility source 9a1faddb96ba518d9b6e63eed545f157d9771c6c, native
sources, model/learning code and dependency locks are unchanged. The only changed
manabot module is training/execution.py, which handles process RSS sampling when
an observed child has exited. Runner changes expose exact milestone selection
and grouped execution; preserve this diff as provenance. The 131072 point within
a longer normalized schedule is not a new replicate of screen013's short run.

## Horizon and cost selection, before admission

Use only fully verified 131072-transition timing receipts, including required
new-shape 64/256 cells. Run price_validation.py after current provider deletion
and ledger settlement. Its engineering budget uses measured linear extrapolation,
a 1.35 slowdown allowance, 90 seconds per-cell export/launch allowance, a 60-second
supervisor allowance, setup/CUDA/retention bounds and a $1.25/hour quote ceiling.
These are bounded allocations, not statistical runtime guarantees.

Candidate N values are 524288, 655360, 786432, 1048576 and 1310720. Select the
largest fully priced candidate that leaves at least $1 uncommitted after its
two-seed reservation, all settled/failed rentals and the existing ancillary
reserve. If even 524288 fails, do not silently remove architectures, seeds or
the reference; report the exact budget conflict. Do not change N after admission.
The independent MTG-121 $5 allowance cannot finance this study.

One group per seed shares setup but starts every cell in a fresh process and
output directory. Seed13591 order: small64, small256, small-deep256,
small-deep64, middle64, middle256, large-shallow256, large-shallow64, large64,
large256, small4. Seed13592 reverses that complete order. Capture shared CPU,
driver, GPU and setup evidence. This blocks host differences within a seed and
counterbalances order across two rentals; it does not estimate host variability.

Use one scheduling owner, max two rentals including other tasks' observed
capacity, bounded local sleep protection and fresh-claim admission. Each group
has one attempt. Each cell gets its preregistered learner/supervisor deadline;
the outer job and rental budgets preserve artifact retention. Failure stops that
group and new group admissions. Diagnose it before any separately authorized,
budgeted replacement; never extend an admitted endpoint or timeout. Retain all
failed/censored/unstarted cells and settle provider deletion before releasing caps.

## Evaluation allocation

Use the verified local CPU route012 with four independent CLI processes and one
CPU thread per policy/game. Freeze 13 new paired deal seeds, 1357201000 through
1357201012 inclusive, with all eight seat/deck/mirror legs. Evaluate every retained
point against scripted_greedy and random: 104 games per anchor/checkpoint,
18304 games across the complete 22-run, four-checkpoint cohort. The deal panel is
shared across configurations, training seeds and checkpoints, and is disjoint
from screen013. Use the ordinary MonitorProtocol, 120-second game watchdog and
10000-command game ceiling. Bind the exact final study protocol SHA in every panel.

Declare finite command and aggregate evaluation deadlines from the retained
route012/013 timing evidence before evaluation starts. Preserve partial output on
failure; no automatic rerun, omitted loss or shorter replacement panel. Evaluate
initializations individually even if some checkpoint identities or scores agree.
All source/runtime, checkpoint-coordinate, artifact-hash, per-role, trace and
replay checks must pass before reporting a complete panel. No GPU evaluation cost.

## Analysis and recommendation rules

Report every cell and both training seeds separately before averaging. Primary
learning observations are initialization-relative greedy score and random score
at the common experience checkpoints. Show all four deck-role cells. Compare
64 versus 256 within each architecture and architectures within each stream count;
small4 is the reference. Keep paired training-seed/deal identities throughout.

For descriptive uncertainty, retain all 13 deal-block means per training seed.
Bootstrap paired deal blocks within each seed with 10000 draws and fixed analysis
seed135917001, sharing resampled indices across compared configurations and
checkpoints. Show the two training-seed estimates and their range. Two training
seeds do not support a precise population training-variance claim; deal-bootstrap
intervals do not replace training-seed replication or correct the grid comparisons.
Do not claim noninferiority or a universal architecture ranking.

Plot scores at each checkpoint's observed inclusive training seconds and active
collection/learning seconds. Report optimizer updates, selected exposures, games,
memory and occupied rental/setup cost separately. Do not interpolate checkpoints
and label them an exact fixed-wall-time experiment. The earlier and later retained
points bracket any observed threshold crossing.

For a practical competence screen, a retained point qualifies only if both
training seeds score at least 50% against greedy and 75% against random, improve
greedy score by at least 20 percentage points over their own initializations,
and keep those thresholds at all later retained points. These are operational
selection thresholds, not statistical claims or human-level competence.

Among qualifying recipes, prefer the lowest worst-seed inclusive time to its
first qualifying retained point. Within 10% of that time, prefer fewer parameters,
then lower measured peak CUDA memory. Report the full score/time Pareto tradeoff,
including any larger model that has higher endpoint skill. Recommend the full
tested N-transition recipe and its exact schedule/cadence; an early qualifying
checkpoint does not silently authorize a different schedule or truncated recipe.
If no recipe qualifies, report that result and the strongest measured alternative
with its specific shortfall; do not relabel the threshold as passed.

An infrastructure success, finite checkpoint or short-screen win cannot finish
MTG-135. Completion requires the retained full comparison, honest failures and
limits, the exact recommendation and evidence delivered into MTG-122's checkout,
and fresh Task verification. Keep MTG-122's explicit pause in force.


## Exact allocation

N = 524288 learner transitions for every cell; raw checkpoints at 0, 131072, 262144 and 524288. Two groups reserve $17.50 total, $8.75 each. Each job is bounded to 23715 seconds and its rental to 24600 seconds. No allocation is admitted by writing this protocol.

| Cell | Learner limit (seconds) | Supervisor limit (seconds) |
|---|---:|---:|
| small:64 | 1740 | 1800 |
| small:256 | 600 | 660 |
| small-deep:256 | 720 | 780 |
| small-deep:64 | 1620 | 1680 |
| middle:64 | 1320 | 1380 |
| middle:256 | 720 | 780 |
| large-shallow:256 | 660 | 720 |
| large-shallow:64 | 1200 | 1260 |
| large:64 | 2220 | 2280 |
| large:256 | 1020 | 1080 |
| small:4 | 9720 | 9780 |

Source transport SHA256: e4a6b0e6f71d840f96892549af1f2f94c5e419dd83ebb9b1967365c968031ec5. Full provenance and cost receipts are bound in the allocation inputs.

Local evaluation uses 4 processes, 4500 seconds per command and 25800 seconds of aggregate work, plus 240 seconds for retention. These limits come from the retained screen013 command timings, scaled to the larger panel with the declared engineering allowances; they are not promises about game duration.
