# Corrected-world baseline — MTG-122

2026-10-15. Jack Heart authorized stopping the w4 complexity study, deferring
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

The original 48-hour ceiling is not reset: reserve 16 hours for all prior attempts
(including cancelled science), 31.5 hours for this allocation and 0.5 hour for
external transfer/report/settlement. Prior SDK planned-to-finished durations total
about 15.15 hours; the larger 16-hour reservation also covers earlier admission
and preparation charges. Original receipts, including the cancellation and stale
campaign JSON, are retained without rewriting their status. No old scores are
combined with this cohort.

Calibrate with reserved seed12260: 100 updates (1200-second learner cap), plus
one complete eight-leg deal against each anchor. Calibration scores cannot choose
the recipe, seeds, or count. Retain all failures; incomplete calibration stops.
The scientific plan is written durably before its first fit. Its count is the
largest multiple of100, capped at10000, fitting 1.5 times the slowest measured
20-update block within each equal learner allowance, reserving120 seconds/run.
The update floor is1000. Lower capacity fails admission, not a smaller study.

Evaluation reserves 6000 games per anchor times1.5 times the larger of its
slowest role-cell mean or full coordinator/replay seconds/game. From the remaining
SDK time subtract evaluation and601 seconds; divide the rest equally among three
learners. Another120 seconds remain outside the scientific plan. These are timing
projections, not sustained-throughput proof. The SDK deadline remains binding.

Science seeds12261/12262/12263 each begin independently from initialization.
Each fit collects256 learner transitions/update before filtering. Score0,N/2,N;
raw only, stochastic one-forward-pass CPU policy. Each milestone has50 held-out
deal blocks, eight legs covering AA/AB/BA/BB balanced over play order, against
fixed scripted-greedy and random anchors. Midpoint and endpoint each re-evaluate
exact frozen initialization on the candidate's same deal/action streams. Five
cohorts/fit ×800 games ×3fits =12000 games. Separate milestone families start at
1226100000,1226101000,1226102000; calibration uses1226000000. No score-selected
checkpoint, replacement seed, sample reduction or early strength stopping.

## Analysis and stopping

Prediction before scoring: all three endpoint greedy changes positive, mean at
least10 percentage points over same-deal initialization; random improvement
positive on average. This is a prediction, not a stopping rule. Report all four
role cells, per-seed paired effects, mean and95% t interval(df2), separately from
conditional whole-deal bootstrap intervals(10000 replicates, seed12290).
Retain the existing paired analysis/censoring semantics and cost/exposure axes.
Threshold attainment means10-point greedy improvement confirmed at endpoint;
missing endpoints stay unavailable. No method-level conclusion from one seed.

Every completed cohort requires exact replay and ordinary legality. Game watchdog
120seconds/10000Commands. Missing cells, failures, nonfinite learning or identity
mismatch remain retained and prevent complete-cohort claims. This plan neither
promotes a product baseline nor establishes chapter/human-play acceptance.
Scientific outputs stay private; editable notebook/read-only HTML and inclusive
cost settlement follow retained results. Complexity, catalog-size and curriculum
comparisons are deferred, not automatically queued after this baseline.
