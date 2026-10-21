# Training scale ladder

Jack Heart set this baseline on 2026-10-20 (MTG-159). It defines default recipes
at fixed scales, each bounded by both money and wall time:

| Tier | Money | Wall time |
|---|---|---|
| small | $10 | 1 day |
| medium | $50 | 7 days |
| large | $100 | 30 days (deferred until the Elemental multi-deck world, MTG-157, has its row) |

The aim is to establish clear scale points and track performance at them, not
to tune any one scale. A ladder row is one world. Each row comes from **one run**:
the run's checkpoint at a tier's cumulative spend supplies that tier. The learning-
rate and tau schedules use absolute iterations, so a prefix of the run is the same
recipe at a smaller budget. Jack chose one run over independent runs per tier
because many checkpoints over time give the smoothing that repeat runs would.

## Recipe

[`regimes/scale-ladder-allies-lessons-v1.json`](regimes/scale-ladder-allies-lessons-v1.json):
the protocol017 recipe (64×2, 256 streams × 64 transitions, Ataraxos move
gradient, the 017 learning settings) with the iteration schedule removed. The
learning rate is 1e-4 until about update 2,300, then decays as 0.5/i^1.1 to its
5e-6 floor near update 35,000. `updates` is an upper bound; the run's length is
set by its allocation, which pauses training and exports raw, EMA and optimizer
state at the money/time deadline.

**Machine.** A calibration on 2026-10-20 ran 32 updates of this recipe on three
secure RunPod GPUs ([data](data/scale-ladder/calibration-20261020.json), ≈$0.40):

| GPU | $/h | Host CPU | Transitions/s | Transitions per $ |
|---|---:|---|---:|---:|
| L4 | 0.59 | EPYC 9254 | 2,632 | 16.1M |
| A40 | 0.59 | Xeon Gold 6342 | 1,697 | 10.4M |
| RTX 4090 | 0.89 | EPYC 7532 | 1,849 | 7.5M |

The policy runs at small batch sizes, so host CPU speed appears to matter more
than the GPU. A run must also reach its first tier inside that tier's time
limit, which excludes machines under about $0.42/h for a $10 / 1 day tier.
The L4 gives the most learning per dollar and reaches $10 in about 17 hours.

## Allies vs Lessons row (v1)

- One L4 pod (EU-RO-1, EPYC 9254, 6 vCPU), created 2026-10-20 21:50 UTC, seed
  15902, source `e6a2aab1`. [Allocation](data/scale-ladder/allocation-allies-lessons-v1.json):
  $50 at $0.59/h plus $0.02/h storage allowance, training pause at
  2026-10-24 06:59 UTC; the pod deletes itself at 07:44 UTC (≤ $48.32 of pod time).
- Raw checkpoints every 250 updates (about 26 minutes).
- The first launch was rejected before training: allocation pauses require
  `schedule_clock: iteration_fraction`, which Ataraxos move updates ignore. The
  recipe was changed to it and training started at about 22:07 UTC; the
  rejected logs remain on the pod's `failed-attempt1/`. The earlier 17 minutes
  of setup count against the $50.
- Tier checkpoints: the last checkpoint whose cumulative pod time × $0.59/h is at
  most $10 (small) and the final paused export (medium).
- If the pod is lost, the run continues from its last retained checkpoint as a
  new stage with fresh optimizer state; that deviation would be recorded here.

### Evaluation (declared before any checkpoint is scored)

On CPU, after the tier checkpoint exists, costed separately from the tiers.
Opponents: `heuristic` (primary), the frozen Mini 12261@10k checkpoint, and
scripted greedy for continuity. 25 fresh deals (seeds 1590100000–1590100024)
with mirrors, 200 games per opponent per checkpoint. Report Lessons vs Allies
first, then AvL, LvL, AvA; bootstrap over whole deals. Also score a log-spaced
curve (updates 0, 250, 1,000, 4,000, 16,000 and the tier checkpoints) against
heuristic.

Predictions: the small-tier checkpoint scores at least 0.40 against heuristic
as Lessons vs Allies (MTG-137's 25k-update GPU checkpoints reached 0.46–0.52 from
about 6.4M transitions; this tier sees roughly 150M). The medium tier scores at
least 0.05 higher than the small tier in mean heuristic score over the four
cells. A miss is retained as the result.

Scale is reported in learner decisions (see the README's Scale section);
here one iteration is 16,384 decisions. The learning-rate schedule counts
iterations, so v1's tiers compare directly only with runs at the same
decisions per iteration. A later version may put the schedule on a decisions
clock.

v1 records only one seat per game: the opponent seat is played by the same
live policy, but its decisions are not trained on. Jack Heart decided on
2026-10-20 that self-play should learn from both sides (MTG-165). v1 runs to its
$50 checkpoint unchanged. A second Allies vs Lessons run, **v2**, follows MTG-165
at the same tiers and machine class, training on both seats' decisions, and is
reported as its own rows. Its recipe, allocation and predictions are committed
before it launches.

| World | Tier | Iterations | Learner decisions | Hours | vs heuristic (LvA) | vs heuristic (mean) |
|---|---|---:|---:|---:|---|---|
| Allies vs Lessons v1 | $10 / 1 day | pending | | | | |
| Allies vs Lessons v1 | $50 / 7 days | pending | | | | |
