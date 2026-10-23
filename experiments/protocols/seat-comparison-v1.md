# Seat comparison v1: one-seat vs two-seat self-play from scratch

Jack Heart, 2026-10-22 (MTG-165). Approved: four training seeds per arm, about
64M learner decisions per run, $100 ceiling including calibration.

## Question

Does heuristic score at N learner decisions match between learning from one
seat and from both? If it does, two-seat at E environment decisions matches
one-seat at 2E, because two-seat learns from every environment decision and
one-seat from half of them. Both clocks are reported; neither alone answers it.

The warm-started $25 attempt (Etude run `896f6f66ad7e`, stopped at $4.63) began
at one-seat update 1000, on the flat part of the curve, and could not separate
the arms. This attempt starts from scratch, where v1's curve rises from 0.11 to
0.55 within 16M learner decisions.

## Recipe

`experiments/batched_baseline.py::seat_comparison`, bound to seeds and RunPod
allocations by `experiments/seat_comparison.py`. It is the scale-ladder recipe
(width 64, depth 2, 256 cross-balanced Allies/Lessons streams, Ataraxos move
learning with absolute-iteration LR/tau and EMA .999, CUDA) on the current
world, from scratch, with complete games:

- two-seat: `learning_seats: both`, one game per stream per update;
- one-seat: `learning_seats: one`, two games per stream per update.

Resolved regimes differ only in those two fields and the ID. Within a seed both
arms initialize identical weights (verified tensor-for-tensor in a CPU smoke).
Learner decisions per update match in expectation, so the iteration schedule
sits at the same position at equal learner decisions; Ataraxos minibatches of
256 rows give equal optimizer steps per learner decision. CPU smoke (32
streams, 4 updates): 23,456 vs 22,786 learner decisions, 28.3 vs 27.8 optimizer
steps per thousand, 25.0% retained rows in both arms. Recorded counters verify
these at full scale; divergence in learner decisions per update is reported.

Seeds 64001–64004; up to 2,700 updates (about 70M learner decisions at the
calibrated rate); raw exports every 20 updates. One L4 job per arm and seed, each
an independent job, 11.9-hour lifetime and $7.27 limit, submitted one after
another. A run that pauses at its rental deadline keeps its exports; matched
comparisons extend only to the learner decisions every run reached.

## Evaluation

The exports nearest 1, 2, 4, 8, 16, 32 and 64M learner decisions, plus the
shared-seed initial models, each play the same 25 fresh deals (seeds
6400000000–6400000024) × 8 deck/seat legs against the heuristic on CPU, off the
rental. Deals are fixed before any scoring.

Analysis per milestone: mean over four seeds per arm on the learner-decision
clock and on the environment-decision clock (one-seat interpolated on log
learner decisions between adjacent exports). Seed-level variation and
deal-resampled intervals are reported separately. Also reported: games,
environment decisions, collection, learning and wall time, and cost per arm.

## Calibration

One short L4 job per arm (seed 64000, 60 updates), completed and deleted
2026-10-22 for $0.21 and $0.17:

| | One-seat | Two-seat |
|---|---:|---:|
| Learner decisions | 1,611,504 | 1,619,566 |
| Environment decisions | 3,226,042 | 1,619,566 |
| Optimizer steps per thousand learner decisions | 3.9 | 3.9 |
| Retained rows | 25.0% | 25.0% |
| Learner decisions per hour: collection / learning / overall | 10.6M / 13.8M / 6.0M | 19.7M / 13.9M / 8.2M |

At these rates one-seat reaches about 64M within its 11.9-hour rental and
pauses at the deadline; two-seat reaches the 2,700-update cap in about nine
hours. A first two-seat submission returned RunPod HTTP 500 and remains an
unresolved ambiguous creation with no pod in the provider inventory; its $0.93
reservation counts against the ceiling.

## Results

All eight jobs ran 2026-10-22/23 on source 0aa4999. Two-seat runs reached the
2,700-update cap; one-seat runs paused at their rental deadline after
2,390–2,589 updates. Every run passed 64M learner decisions, and all 60
scheduled evaluations completed. Scores and per-deal rows are in
`experiments/data/seat-comparison-v1/`; reproduce the tables with
`uv run python -m experiments.seat_comparison_report --directory experiments/data/seat-comparison-v1`.

Matched learner decisions (mean over four seeds; hours are training hours to
the scored checkpoint, excluding pod setup):

| Learner decisions | One-seat | Two-seat | Two minus one | Seeds two-seat ahead | One-seat hours | Two-seat hours |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 0.085 | 0.085 | — | — | 0 | 0 |
| 1M | 0.300 | 0.348 | +0.048 | 3/4 | 0.20 | 0.14 |
| 2M | 0.350 | 0.415 | +0.065 | 4/4 | 0.38 | 0.26 |
| 4M | 0.453 | 0.495 | +0.042 | 3/4 | 0.71 | 0.50 |
| 8M | 0.530 | 0.576 | +0.046 | 4/4 | 1.39 | 0.98 |
| 16M | 0.557 | 0.586 | +0.029 | 3/4 | 2.69 | 1.97 |
| 32M | 0.613 | 0.645 | +0.032 | 3/4 | 5.34 | 3.92 |
| 64M | 0.671 | 0.659 | −0.012 | 1/4 | 10.74 | 7.81 |

Seed standard deviations are 0.02–0.06 per arm. At matched learner decisions,
where one-seat has seen twice the environment decisions and twice the games,
two-seat is at least as strong: about +0.05 from 1M to 8M, smaller by 32M, and
level at 64M (seed 64004 favours one-seat by 0.085, deal interval −0.14 to
−0.04). With four seeds this supports "no worse per learner decision", not a
reliable advantage.

Matched environment decisions (one-seat interpolated in log environment
decisions): two-seat leads in all 19 seed comparisons, by about +0.15 near 4M,
+0.12 at 8M, +0.06 at 16M, +0.09 at 32M and +0.05 at 64M. This is what the
hypothesis predicts on a flattening curve.

Time and cost: two-seat reaches each learner-decision milestone about 1.4×
faster in training time, all from collection; learning time per learner
decision is the same. RunPod billed $49.42 for the eight jobs (two-seat
$5.22–5.80, one-seat $6.81–6.89), so about $54.40 including the stopped
warm-start attempt and calibration. Etude's cleanup estimates (about $9 per
job) run to the observer's confirmation of deletion, not to the pod's end.

Conclusion: learning from both seats costs nothing per learner decision and
reaches the same strength with half the games and about 1.4× less training
time on this recipe.
