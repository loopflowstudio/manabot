# Batch closure and lambda v1

Task fc6ffb, Jack Heart, 2026-11-04. Authorized: $25 total, a hard cumulative cap
covering training, evaluation, failures and machine setup for both studies.
Spending beyond it needs Jack Heart's explicit approval. This protocol is committed
before any scored run. Calibration runs are unscored and charged to the cap.
The 2026-11-05 calibration amendment below changed the seeds, updates and per-run
limits before any scored run; predictions stand as first written.

Two separable studies share only the evaluation and launch scaffolding in
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
`batch_closure` and `transitions`: the complete arm's floor is 76 rows per stream,
and the exact arm's quota is 106, the rows that floor realized in calibration once
game length settled. Seeds 65101–65103 give both arms identical initial weights per
seed. Target: 200 updates.

Predictions at the last update every run reached:

- Complete games score at least as well as exact quota: difference +.02, with a
  seed-level interval that includes zero.
- Complete games spend 10–25% more collection time per learner decision, from idle
  streams; idle stream steps are 15–35% of stream slots.

## Study 2: lambda (`experiments/lambda_sweep.py`)

Question: on complete games, does giving the game result more direct weight help?

Recipe: `experiments.batched_baseline.lambda_sweep`, the two-seat complete-game
`seat_comparison("both")` with only `policy_lambda` and `value_lambda` varied.
Seeds 65201–65203. Target: 100 updates.

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
Scored updates: 20, 40, 75, 150 where a run reaches them, and each run's endpoint.
Update 0 is not scored: a seed's arms share their initial weights, so it compares
nothing. Deals are fixed before any scoring.

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

Per-run limits fit those lines at the calibrated L4 rate. The launcher reserves a
job's window plus 15 minutes of setup at the hourly rate, so launch passes these limits
over the frozen job's own. Limits sit outside a job's identity, and each window still
covers the run's wall stop and finalization.

| Study | Runs | Window hours | Job dollars | Reserved |
|---|---:|---:|---:|---:|
| 1, exact quota | 3 | 1.85 | $1.24 | $3.72 |
| 1, complete games | 3 | 2.45 | $1.60 | $4.78 |
| 2, each of 5 arms | 15 | 1.0 | $.74 | $11.06 |

Reserved in total with calibration's $1.49 and evaluation's $2.50: $23.55. Evaluation
jobs pass theirs from `experiments/batch_closure.py` ($.72) and
`experiments/lambda_sweep.py` ($.96), inside the line, on a 32-vCPU rental.

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

Run 2026-11-04 on three RunPod L4 rentals at $.59/hour: one seed (65000), 30 updates,
unscored, at a quota of 106. All three completed, and every rental ended. Each pod ran
45 minutes for a 12–19 minute job: setup took 2–6 minutes, then the laptop lost its
network from about 23:22 to 23:59 UTC, so no controller ended the idle rentals before
their windows closed. Provider bills through 00:00 UTC total $1.21, about $.08 more
was still to post, and the launcher's per-run `usd` understates this. The $1.49
reservation stays charged until the bills settle. Scored launches keep a controller
running so a finished rental ends promptly.

| Run | Arm | Rows/update | Games/update | Optimizer steps per 1k rows | Collection s/update | Learning s/update | Idle share of stream slots |
|---|---|---:|---:|---:|---:|---:|---:|
| `d349ae373c72` | Exact quota 106 | 27,136 | 523 | 3.91 | 16.8 | 6.2 | .47 |
| `c9b784359edb` | Complete games, floor 76 | 30,461 | 503 | 3.92 | 27.4 | 11.1 | .54 |
| `adc13704d466` | Lambda .5 / .8 | 30,090 | 256 | 3.92 | 14.1 | 17.6 | .65 |

The table's means are 30-update means, and they mislead for the complete arm, whose
rows fall as games shorten:

| Run | Rows, updates 1–8 | Rows, updates 11–30 | Collection s, 11–30 | Learning s, 11–30 | Assembly s, 1–8 / 11–30 |
|---|---:|---:|---:|---:|---:|
| Exact quota 106 | 27,136 | 27,136 | 15.7 | 6.2 | 1.0 / 1.0 |
| Complete games, floor 76 | 37,825 | 27,264 | 15.5 | 9.9 | 32 / .9 |
| Lambda .5 / .8 | 46,269 | 23,654 | 10.7 | 13.9 | 2.2 / 1.1 |

The whole-run gap of 12.3% is past the 10% kill line, but it is the opening transient:
after it the arms sit .5% apart, and 8 of 200 updates move the run's mean by about 1.5%.
A first amendment moved the quota to 119 to match the 30-update mean. It was reversed
before any freeze, because 119 would fit only those opening updates and leave the arms
about 12% apart for the rest of the run. The quota stays 106, and each run's counters
report cumulative drift.

Against the predictions above, which stay as written: complete games took 45% more
collection time per learner decision, not 10–25%, and idle slots were .54, not
15–35%. Exact quota idles too (.47), so idle share does not isolate complete games.

The 45% is almost entirely batch assembly, 287 s of 820. That is not a steady cost: it
ran about 32 s per update for updates 1–8, then fell to .9 s and stayed there, when
collection matched the exact arm. The two-seat run packs more rows through the same
code in 2.2 s, so row count alone does not produce it. The cause is not established;
the scored runs test it, since each records assembly per update on separate hosts.
A wall-time result must name it. Packed complete-game batches also carry a measured
implementation cost: targets and health diagnostics scan one `[rows, 1]` column, 3–7 s
and 1–2 s per update against about .05 s for exact quota's `[quota, 256]`. That is the
learning-time gap, and it comes from the batch shape, not from the closure.

At the 30-update pace 300 updates and the original seeds would cost about $41, so
the studies were cut before launch: study 1 to three seeds per the kill criteria, and
both to fewer updates, since three seeds of study 2 could not reach 300 inside its line.
Runs wall-stop at their hour limit and keep their exports. Both windows are early
training, 6M and 3M learner decisions: a null here says nothing about later learning.

## Results

Pending. Reports go to `experiments/data/closure-lambda-v1/`.
