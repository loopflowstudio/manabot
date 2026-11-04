# Batch closure and lambda v1

Task fc6ffb, Jack Heart, 2026-11-04. Authorized: $25 total, a hard cumulative cap
covering training, evaluation, failures and machine setup for both studies.
Spending beyond it needs Jack Heart's explicit approval. This protocol is committed
before any scored run. Calibration runs are unscored and charged to the cap.

Two separable studies share only the evaluation in
`experiments/learning_signal.py`. Each conclusion stands alone; neither study's arms
feed the other's comparison.

## Prior evidence

`experiments/ataraxos-technique-screen.md` lists `policy-trace` (.5 vs .95) and
`value-trace` (.8 vs 1) as candidates and `experiments/ataraxos-omitted-controls.md`
lists trace controls, but neither file reports a trace result that narrows this sweep,
and both predate the current world. Trace arms are therefore chosen from the weight
the game result receives, not from an earlier winner.

## Study 1: batch closure (`experiments/batch_closure.py`)

Question: at matched learner decisions, wall time and cost, does one-seat learning
from complete games differ from one-seat learning at an exact quota?

Recipe: `experiments.batched_baseline.closure_comparison`. It is
`seat_comparison("one")` (width 64, depth 2, 256 cross-balanced Allies/Lessons
streams, damped policy learning with absolute-iteration schedules and EMA .999,
CUDA) on the current world, from scratch. Resolved regimes differ only in ID,
`batch_closure` and `transitions`: the exact arm's quota is 106 rows per stream, and
the complete arm's floor is set from calibration so that realized rows per update
match. Seeds 65101–65104 give both arms identical initial weights per seed.

Predictions at the last update every run reached:

- Complete games score at least as well as exact quota: difference +.02, with a
  seed-level interval that includes zero.
- Complete games spend 10–25% more collection time per learner decision, from idle
  streams; idle stream steps are 15–35% of stream slots.

## Study 2: lambda (`experiments/lambda_sweep.py`)

Question: on complete games, does giving the game result more direct weight help?

Recipe: `experiments.batched_baseline.lambda_sweep`, the two-seat complete-game
`seat_comparison("both")` with only `policy_lambda` and `value_lambda` varied.
Seeds 65201–65203.

| Arm (policy / value) | Result weight 20 decisions out (policy / value) | Purpose |
|---|---|---|
| .5 / .8 | 1e-6 / .012 | Current recipe, baseline |
| .8 / .8 | .012 / .012 | Policy trace alone |
| .5 / .95 | 1e-6 / .36 | Value trace alone |
| .8 / .95 | .012 / .36 | Both raised |
| .95 / 1 | .36 / 1 | Most direct; the value target is the game result |

Predictions at the last update every run reached:

- Value .95 beats the baseline by +.02 to +.05 and lowers critic error far from game
  end less than it raises variance near it.
- .95 / 1 is no better than .8 / .95: variance offsets the added directness.
- Policy .8 alone moves the score less than .02 in either direction.

Critic quality is the absolute lambda-target residual by distance from game end, from
the `selection_groups` metrics, read per arm at the same updates.

## Evaluation

The same 25 held-out deals (seeds 6510000000–6510000024, reserved here) × all 8
deck/seat legs against the fixed heuristic, through the replay-checked arena on rented
CPU, never the laptop. Raw exports are scored, not EMA, matching seat-comparison v1.
Scored updates: 0, 20, 40, 75, 150, 300, and each run's endpoint. Deals are fixed
before any scoring.

Analysis per study: mean over seeds per arm on the learner-decision clock, then at
matched wall time and cost. Seed-level variation and deal-resampled intervals are
reported separately. Also reported per arm: rows, games, optimizer steps, environment
decisions, collection, idle-stream and learning time, and cost. Complete-game batches
vary in size, so rows per update and schedule position at matched decisions are
reported, and divergence between arms is named rather than corrected.

## Budget

| Line | Ceiling |
|---|---:|
| Calibration | $1.50 |
| Study 1 training | $9.00 |
| Study 2 training | $12.00 |
| Evaluation | $2.50 |
| **Total** | **$25.00** |

Ceilings are conservative reservations, not estimates. An unspent line may cover a
failed attempt in another line; the total may not move. Evaluation stays a small
fraction of training.

## Kill criteria

- Stop launching when billed plus reserved spend would exceed $25. Keep every failure
  and its cost.
- If a calibration arm fails, or complete-game and exact rows per update cannot be
  matched within 10%, stop study 1 before scoring and report why.
- A run stopped by its watchdog keeps its exports; matched comparisons extend only to
  the learner decisions every run reached. Missing runs stay in the denominator.
- If the budget supports fewer seeds, cut seeds equally across a study's arms before
  launch and record the change here, never after scoring.

Deleting exact quota is out of scope and waits for this report.

## Calibration

Pending.

## Results

Pending. Reports go to `experiments/data/closure-lambda-v1/`.
