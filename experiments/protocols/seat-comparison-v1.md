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
an independent single-job cohort, 11.9-hour lifetime and $7.27 limit. A run
that pauses at its deadline before 64M continues as a new segment; collector
and RNG streams restart there, identically for both arms.

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
