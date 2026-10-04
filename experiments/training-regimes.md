# Training regimes: learning-speed protocol

2026-10-04. Jack Heart requested this comparison. ETU-91 owns the study;
ETU-89 supplies run infrastructure and ETU-90 supplies learning treatments.
Jack Heart authorized scientific execution after integration and landing, with
the root session owning the single launch. Numerical allocations remain proposed
until integrated calibration; smoke does not freeze them.
The executable smoke freezes `EvaluationProtocol` in `protocol.json` before
scoring. It uses one seed, two real checkpoints, a fixed random anchor, and
one four-leg block for each paired-recipe and anchor comparison. Its width-16,
four-simulation/one-world teacher settings are workflow checks only. The
CPU calibration and scientific profiles require an explicit resolved plan.
MPS is unsupported. The resumed execution section defines the inclusive campaign
ceiling and launch commands; earlier numerical examples are not frozen counts.

```bash
uv run --extra notebook experiments/runners/run_training_regimes.py --study learning-speed --profile smoke --out .runs/learning-speed
uv run --extra notebook experiments/runners/run_training_regimes.py --report-only .runs/learning-speed
```

Reports list every retained comparison, exact-replay result, cost and source
checkpoint. Missing or incomplete evidence stays unavailable. One training seed
cannot provide cross-seed uncertainty; four legs of one deal are not independent
replicates. The plot uses fixed-anchor scores, and the paired recipes' scores are
reported separately. No strength decision follows from this smoke.

## First experiment: proposed executable recipes

Question: at equal end-to-end laptop training cost, which recipe is learning
faster now, and which produces the stronger final policy? This is a recipe
comparison; a win cannot identify which individual treatment caused it. The
ablation study below is also a required executable use of the delivered system.

Common settings: current pinned world, authored UR Lessons/GW Allies with
sideboards, equal deck/seat training exposure, viewer-safe inputs, width-64
attention Agent with four heads, identical observation capacity, fresh weights,
terminal +1/-1 and 0 for authoritative draws, and no reward shaping. Training
aborts on omitted legal offers or observation truncation. Distinguish draws
from engine truncation throughout training and evaluation.

**A: search distillation.** Generate complete teacher games with existing
uniform-prior determinized PUCT: 64 total simulations/root, four worlds,
random terminal continuations, 2,000-step continuation limit. Train the policy
on normalized root visits using existing supervised training, Adam 1e-3,
batch 128, at most ten epochs, 10% whole-game validation, value weight zero.
Preserve the unused value head to keep the common network structure. Execute
four collect/train rounds: at most 4.5 hours collecting fresh games and one
hour fitting/exporting each round. Fit the cumulative corpus with permanent
whole-game split assignments, continuing model and optimizer from the previous
round. Each round allows at most ten epochs; unused time is recorded, not
fabricated as compute. The teacher remains fixed, so this is repeated
distillation, not an expert-iteration claim. Final completed epoch per round
is selected. Incomplete teacher games remain charged and
excluded with explicit counts. No free pre-existing corpus or warm start.

**B: direct self-play.** Current policy plays both sides; train a joint
policy/value model for at most 18 hours/run including export. Pilot defaults:
16 streams, 256 learner transitions/stream, gamma=1, policy GAE lambda=.95,
value lambda=1 with rollout-boundary bootstrapping, four epochs, four
minibatches, ratio clip .1, value weight .5, gradient norm .5. Separate the
two estimators and make filtering and schedules independently configurable.
Start with highest-|advantage| 50% of rows selected before normalization for
both objectives as a labelled MTG pilot choice. Also support quantile plus
absolute-threshold filtering for the paper-derived treatment. Log selection
by action type and retained fraction; an empty selection skips and records
the update rather than silently training on all rows.

Proposed damped objective: clipped policy loss + value loss +
`tau * KL(current || uniform_legal)` + `beta * KL(current || collection)`.
Store full legal behavior distributions for the collection-policy penalty.
Use Adam LR `2.5e-4 * (1 + 9p)^-1` and tau `.01 * (1 + 9p)^-1`, beta=.1,
where p is elapsed training-budget fraction. Exact coefficients are pilot
choices, not paper-reproduction claims. Freeze after finite-loss/throughput
calibration without searching for a favorable final result. No search or
belief model supplies B's training data.

Add a structured-uniform reference option: choose uniformly among legal
action types, then uniformly among canonical offers of that type. This has
full legal support, is invariant to offer order, and avoids giving an action
type more mass merely because it has more offers. The grouping is an MTG
design choice, not Stratego's piece-first prior. Make it a measured treatment
before replacing B's flat-uniform pilot. Test masks, normalization and
semantic grouping on target/combat/priority decisions.

Maintain optional EMA outputs initialized from the learner and updated once
per collect/update iteration. Default collection uses current learner weights;
evaluate raw and EMA variants on identical development deals, charging both
evaluations. Do not silently collect with EMA: that is a different behavior
policy and needs explicitly recorded likelihoods. The final raw/EMA selection
rule is frozen before final scoring. Default remains raw for both main arms;
EMA is a paired diagnostic, not post-hoc selection of whichever wins.

The initial comparison is proposed as policy-only for both arms. The later
pipeline freezes B's policy, generates ground-truth-labelled self-play,
trains a belief model, and evaluates search with learned versus uniform
beliefs at matched decision time. Whether that pipeline belongs in this first
week remains an explicit scope question.

## Budget and evaluation protocol

Draft allocation assumes **168 active laptop hours total**, not one week per
arm: 4 hours calibration, 15 for the five-arm screen, 108 main training
(three seeds per recipe x 18 hours), 30 evaluation/analysis and 11 reserve. Laptop unavailability extends calendar time;
it does not create compute credit. If seven elapsed days is the hard limit,
availability must reduce these allocations before freezing. Engineering work
precedes the experiment; setup and failed scientific attempts are accounted.
No paid hardware. Proposed baseline profile: CPU, four collection workers maximum,
one Torch thread/worker, 32 GiB process-tree limit, one active training run.
CPU is the supported profile. Freeze batching and workload counts using complete
collect/update/export and recorded/replayed evaluation timing; MPS requires a
separate implementation and certification. Alternate arm order across seed pairs. Record sleep, contention and throttling.

Final selection is the last completely saved checkpoint within each run's
allowance; never select the best seed. Planned seed IDs: 197, 198, 199, with
separate named RNG streams for initialization, collection, minibatches and
evaluation. Reuse initial weights across paired arms where shapes match;
different visited trajectories are expected.

Use the existing four-leg arena blocks: both deck assignments and both first
players. Primary evaluation is all nine A-seed/B-seed matchups, 128 untouched
deal blocks each: 4,608 games. Secondary evaluation uses every candidate
against fixed random, scripted-greedy and PUCT-64 anchors, 32 blocks each:
2,304 games. Keep anchor results separate from direct head-to-head strength.
The main run has four export stages within an 18-hour ceiling. Freeze interior
cost cutoffs from integrated calibration, with actual elapsed cost attached. Capture the latest complete checkpoint at each cutoff;
never use a future checkpoint or interpolate weights. Compare all six runs at
the first three cutoffs against the same three anchors on 16 development blocks
each (3,456 additional games), plus paired-seed A/B matches on 32 blocks each
(1,152 games). Total proposed schedule is 11,520 games. Evaluation stops the
training clock and is charged to the evaluation allocation for both arms.
Include source loading, data generation, failed work and exports in the
training cost axis. No endpoint-only answer to the learning-speed question.
Before final scoring, calibration must show the complete schedule including
recording/replay fits 24 hours with a 25% timing margin. Otherwise amend the
document before scoring; never trim games after seeing outcomes.

Both candidates use one stochastic policy pass, CPU, one thread, batch one,
no search. Freeze identical 120-second game / 10,000-Command limits and
report p50/p95 latency and memory. A forfeit, crash or truncation is a failed
execution, not an authoritative draw. Retain every scheduled attempt and
report unresolved-game bounds; an incomplete cohort cannot establish a win.

Report mean B score (win=1, draw=.5) across nine cells with equal cell weight,
all cell/deck/seat scores, and uncertainty resampling A seeds, B seeds and
deal blocks as crossed clusters. Thousands of games do not turn three seeds
into thousands of training replicates. Label intervals descriptive at this
seed count. Proposed decision: B is promising if mean score >=.55 and its
95% interval lies above .50; symmetric criterion for A. Otherwise unresolved.
Anchor reversals or catastrophic deck-specific weaknesses qualify the claim,
not disappear into an aggregate. This is matchup-specific playing strength,
not exploitability or general Magic superiority.

Learning-speed analysis plots every seed's fixed-anchor mean score against
actual cumulative laptop hours, environment decisions and unique complete
games. Cost is primary; decision/game axes diagnose sample efficiency. Report
paired-seed A/B scores at each cutoff, score gains/hour over observed intervals,
and first observed crossing of a preregistered .60 equal-weight anchor score
(unreached is censored). For area-under-learning-curve comparisons use a shared
cost horizon and the last actually available checkpoint between evaluations;
never backfill strength over data-collection time. Do not compare RL returns
to distillation cross-entropy as a shared progress measure. A crossing of the
curves is a useful result: one recipe can learn faster early and lose late.


For treatments see [Ataraxos-inspired ablations](ataraxos-mtg-ablations.md).
For later mechanisms see [follow-up protocols](training-regime-followups.md).

## Saved smoke evidence and cost analysis

The resolved small recipes are saved before execution in `recipes.json` and
bound by digest in `protocol.json`. Run exports bind source/runtime identities,
seed streams and stage artifacts. Each attempted run and arena cell enters
`study.json` before execution, so failure is retained even before a run export
or completed cell exists. Partial arena traces remain in that cell's directory.
The 900-second deadline includes notebook execution. An interrupted notebook
can be regenerated with `--report-only`; that does not complete a failed smoke.

`metrics.json` retains every paired and fixed-anchor measurement and separates
collection, optimization and export time. Total training cost also includes
setup and measured stage overhead. The current preliminary runner sums setup
and stage durations; inter-stage persistence is omitted. New execution requires ETU-89's cumulative checkpoint clock and fresh integrated
runs. The runner now consumes that field; historical costs are never replaced. `cost-comparison.json` uses the
intersection of observed cost ranges across recipes and the last checkpoint
available at each cost. It neither interpolates weights nor credits a later
checkpoint during earlier label generation. When ranges do not overlap, an
equal-cost comparison is unavailable. Smoke checkpoint counts are fixed; their
elapsed costs need not match. Their step plots describe those measurements,
not a statistically established learning curve.

Offline regeneration checks protocol, resolved recipe, run-export and measured
checkpoint digests. Preserve the complete output directory and its referenced
artifact paths. Notebook/report generation reads saved evidence only; it does
not contact a service or invoke a trainer. The original inherited smoke folders
remain preliminary evidence from before final contract integration.


## Resumed scientific execution (2026-10-04)

Jack Heart authorized the local experiment after integration and landing. The
root session owns the single scientific launch. Earlier implementation-only
restrictions are superseded; CPU remains the supported profile. Neither the
smoke nor a timing pilot replaces the multi-seed experiment.

The existing runner accepts `--profile calibration` and `--profile scientific`
only with a resolved `--plan` JSON. Each plan contains complete TrainingRegime
recipes, their digests, training seeds, separate paired/anchor deals, checkpoint
count, declared cost cutoffs, process deadline, allocation, prior campaign cost,
and calibration evidence. Scientific plans require at least three seeds.
Recipes specify real stage counts and budgets; increasing a deadline alone does
not increase training. Every stage exporting raw weights is a checkpoint.
The root must budget enough stages to cover the declared cost cutoffs, using
measured full-loop throughput and leaving export margin. A missed cutoff cannot
be filled with future weights. Fixed update counts are not equal-cost evidence.

After the final source lands, prepare and run the two timing cohorts:

```sh
uv run --extra notebook experiments/runners/run_training_regimes.py --study learning-speed --profile calibration --write-plan .runs/learning-calibration-plan.json
uv run --extra notebook experiments/runners/run_training_regimes.py --study learning-speed --profile calibration --plan .runs/learning-calibration-plan.json --out .runs/learning-calibration-1
uv run --extra notebook experiments/runners/run_training_regimes.py --study ataraxos-ablations --profile calibration --write-plan .runs/ablation-calibration-plan.json
uv run --extra notebook experiments/runners/run_training_regimes.py --study ataraxos-ablations --profile calibration --plan .runs/ablation-calibration-plan.json --out .runs/ablation-calibration-1
```

The generated plans each allow one CPU hour, use full width-64 models, two
checkpoints, eight RL updates per stage and four teacher games per collection.
They measure collection, optimization, export, ordinary reload, four-leg arena,
replay and notebook execution together. Before each launch set
`prior_campaign_seconds` to all actual prior campaign time, including failures;
commit the resolved plan before running. These plans are timing probes, not
promises that this workload fits. Keep failed attempts and reduce counts in a
new plan if a stage deadline fires. Do not weaken legality or replay checks.

Proposed campaign envelope, to freeze after those measurements: at most four
hours calibration including failures, fifteen hours for the five-arm screen
(three seeds each, one hour/run), 108 hours for the main comparison (three seeds
per recipe, eighteen hours/run), thirty hours evaluation/analysis and eleven
hours reserve. Total: 168 active laptop hours, inclusive of both studies and
all failed work. This is a ceiling, not an instruction to spend unused time.
If the measured cohort cannot fit with 25% timing margin, reduce training
allowances or the pre-scoring evaluation cohort and commit an amendment before
scoring; never trim a cohort after seeing outcomes.

Screen seed family: 601, 1601, 2601. Main comparison: 5601, 6601, 7601.
Calibration uses 397, separate from smoke 197. Reserve screen paired deals
1000000–1000031 and anchor deals 1100000–1100031; reserve main paired deals
1200000–1200063 and anchor deals 1300000–1300063 for development. Reserve untouched screen
endpoint paired/anchor families 1400000–1400031 / 1500000–1500031, and main
endpoint paired/anchor families 1600000–1600127 / 1700000–1700031. Freeze the exact subsets
before scoring. All four deck/seat legs run for every selected deal. Scientific plans require random, scripted-greedy and PUCT-64 fixed baselines,
with separate results for each. Development comparisons pair initialization
seeds; endpoint learning-speed comparisons cover the full nine seed pairings.
Endpoint ablations retain matched seeds. All endpoint deal families must be
disjoint from all development deals. Three seeds support exploratory inference
only.
Main candidates remain raw outputs, with no post-hoc best-seed selection.

Generate both scientific plans from the completed calibration evidence. These
commands write plans only; they do not train. The default allocation is frozen
as a ceiling: 4 hours calibration, 21 hours screen (15 training + 6 evaluation),
132 hours comparison (108 training + 24 evaluation), and 11 hours recovery.
Use `--prior-campaign-seconds` for actual prior time if it exceeds the reserved
4-hour / 25-hour minima. If extra failures consume the recovery allocation,
amend the envelope before scoring; the generator refuses to spend it silently.

```sh
uv run --extra notebook experiments/runners/run_training_regimes.py --study ataraxos-ablations --profile scientific --calibration .runs/ablation-calibration-1 --write-plan .runs/frozen-ablation-plan.json
uv run --extra notebook experiments/runners/run_training_regimes.py --study learning-speed --profile scientific --calibration .runs/learning-calibration-1 --companion-plan .runs/frozen-ablation-plan.json --write-plan .runs/frozen-learning-plan.json
```

The generator keeps the calibrated width, streams, transitions, optimizer,
epochs and teacher strength. It scales **updates and teacher games**, not only
timeouts. It preserves optimizer continuation and cumulative teacher datasets.
Counts target 75% of each stage allowance using the slower observed per-unit
rate; cumulative fitting is priced at the final corpus size. Three screen
checkpoints and four main checkpoints remain distinct from nominal elapsed
cutoffs. Predicted interior cost cutoffs span 110% of the slowest first export
to 90% of the earliest predicted final export. Actual overlap is checked after
execution; predictions never manufacture equal-cost points.

The generator prices each of the three anchors from its measured four-leg cost,
adds 25% timing margin and ten minutes for analysis, then selects development
blocks up to 16, endpoint anchor blocks at twice that count, and endpoint paired
blocks at twice that count for the screen or eight times for the comparison.
Selection uses timing only. Even the smallest cohort must fit before a plan
can be written. Inspect and commit the exact resulting counts, identities,
seeds, deals and predictions before root starts scoring. Count extrapolation
is not evidence of long-run reliability or adequate statistical power.

Storage is also a pre-scoring gate. Calibration measures actual bytes for shards,
checkpoints, training records and arena/report artifacts. Projected growth follows
game/update/checkpoint/evaluation counts, with 25% margin plus a 4 GiB free-disk
reserve. The main plan additionally reserves the screen's projected bytes via
`--companion-plan`. With only 14 GiB free, a plan may fail this gate; retain the
error and archive evidence to an explicitly chosen larger volume or amend the
workload before scoring. No automatic deletion or omitted evidence is allowed.
Scientific execution checks free disk again and requires the calibrated native,
content, ABI and training-source identities. Regenerate calibration after source
drift; do not relabel old measurements.

After committing the two generated plans, root alone runs:

```sh
uv run --extra notebook experiments/runners/run_training_regimes.py --study ataraxos-ablations --profile scientific --plan .runs/frozen-ablation-plan.json --out .runs/ablation-scientific-1
uv run --extra notebook experiments/runners/run_training_regimes.py --study learning-speed --profile scientific --plan .runs/frozen-learning-plan.json --out .runs/learning-scientific-1
uv run --extra notebook experiments/runners/run_training_regimes.py --report-only .runs/learning-scientific-1
```

Existing output directories are never overwritten by a new execution. Failed
training retains SQLite state and partial artifacts; model checkpoints are not
collector/process snapshots. An interrupted training cohort needs a new plan
and output directory, charging all previous work. Do not restart the complete
108-hour comparison casually: its failure may require a smaller amended study.

If **all training runs completed**, `--resume` can finish unattempted arena cells
and reporting from verified immutable run/checkpoint artifacts, without training:

```sh
uv run --extra notebook experiments/runners/run_training_regimes.py --study learning-speed --profile scientific --plan .runs/frozen-learning-plan.json --out .runs/learning-scientific-1 --resume
```

An unclean kill leaving status `running` has no final elapsed-cost receipt and
cannot auto-resume; account for the lost interval before a new cohort.
Resume verifies source/runtime and protocol identities and uses only remaining
time in the original process allowance; downtime creates no compute credit.
Completed cells are reused. Failed or interrupted cells remain unresolved and
are never retried or replaced; the remaining schedule may finish, but that
cohort stays failed with unresolved-game bounds. Offline `--report-only` works
without training or network access. Run exports, checkpoint bytes and compressed
Command traces are digest-checked. Root owns cross-study time accounting; the
runner cannot infer other processes or the campaign's external ledger. No
parallel scientific training is authorized.

## Integrated workflow evidence (2026-10-04)

ETU-91 consumed ETU-89/ETU-90 and ETU-75's semantic checkpoint contract.
The final learning-speed smoke completed in 91.08 seconds with 24 games;
the final five-arm smoke completed in 204.20 seconds with 72 games. Both
exported two checkpoints per arm, reloaded ordinary w4 checkpoints, evaluated
all four deck/seat legs and exactly replayed every completed game. The learning
arm retained its collect/train handoff. All six saved metrics/report outputs
were unchanged after offline notebook/report regeneration for each final cohort.
Equal-cost windows exist in these final smoke cohorts; one seed and one deal
block still cannot establish method-level strength or select treatments.

Retained attempts remain local under this Task checkout's ignored `.runs`;
Jack Heart requested keeping the Task and checkout open to preserve them.
They are not archived model releases. The preceding inherited-ABI cohorts
remain preliminary and are never relabeled as current-world evidence.

| Directory under `.runs` | Status | Seconds | Completed stored game rows |
| --- | --- | ---: | ---: |
| `integrated-learning-speed-1` | completed | 83.24 | 24 |
| `integrated-ataraxos-ablations-1` | completed | 208.87 | 72 |
| `integrated-learning-speed-final` | completed | 91.08 | 24 |
| `integrated-ataraxos-ablations-final` | failed | 203.62 | 48 |
| `integrated-ataraxos-ablations-final-2` | completed | 204.17 | 72 |

The failed five-arm attempt was interrupted after a child process read transient
merge-conflict markers during main integration. Its partial artifacts and failed
report remain intact; the subsequent run used a new directory. No failed row was
replaced in that cohort. Do not change imported source while child processes
are running.

Validation: 14 study tests and six debug native vector tests passed; Ruff and
whitespace checks passed. The broader training/collector/supervised/arena gate
had 95 passes and four unchanged historical artifact failures: INT-7's retained
shards lack semantic tensors, INT-8's frozen loader/source digests reject the
new world, and the old arena fixture lacks a w4 binding. No frozen artifact or
admission check was rewritten to make these pass. CI owns its separate matrix.

No calibration or scientific cohort ran in this Task. Root must run integrated
calibration on the landed code and commit generated plans before scoring.
Study evidence does not satisfy ETU-82 demo completion, ETU-85 improvement,
current-world S1–S5 certification, or the human-challenger outcome.
