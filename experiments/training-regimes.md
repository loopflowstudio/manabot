# Training regimes: learning-speed protocol

2026-10-04. Jack Heart requested this comparison. ETU-91 owns the study;
ETU-89 supplies run infrastructure and ETU-90 supplies learning treatments.
The numerical scientific protocol below remains proposed. Only bounded smoke
execution is authorized; a completed smoke does not freeze or fund this protocol.
The executable smoke freezes `EvaluationProtocol` in `protocol.json` before
scoring. It uses one seed, two real checkpoints, a fixed random anchor, and
one four-leg block for each paired-recipe and anchor comparison. Its width-16,
four-simulation/one-world teacher settings are workflow checks only. The
scientific CPU/MPS calibration and final-scoring profiles are not implemented;
the CLI accepts only `--profile smoke` so it cannot silently launch the draft.

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
policy/value model for at most 22 hours/run including export. Pilot defaults:
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
arm: 6 hours calibration, 132 training (three seeds per arm x 22 hours), 24
evaluation/analysis and 6 reserve. Laptop unavailability extends calendar time;
it does not create compute credit. If seven elapsed days is the hard limit,
availability must reduce these allocations before freezing. Engineering work
precedes the experiment; setup and failed scientific attempts are accounted.
No paid hardware. Proposed baseline profile: CPU, four collection workers maximum,
one Torch thread/worker, 32 GiB process-tree limit, one active training run.
Calibration additionally measures MPS with in-process batched inference on
the real collect/update path; exp-07 found separate MPS processes problematic.
Use the faster validated complete-loop profile for each recipe, with the same
laptop/time/memory allowance, and freeze device/batching before scoring.
Report algorithm settings and hardware profile separately. A CPU-only reference
remains runnable; MPS microbenchmark wins do not qualify a training profile.
Alternate arm order across seed pairs. Record sleep, contention and throttling.

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
Save checkpoints at cumulative 5.5, 11, 16.5 and 22 hour ceilings, with actual
elapsed cost attached. Capture the latest complete checkpoint at each cutoff;
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
setup and other measured stage overhead. `cost-comparison.json` uses the
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
