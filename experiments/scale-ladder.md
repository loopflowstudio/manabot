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

`scale_ladder()` in [`recipes.py`](recipes.py)
(`uv run manabot train --regime-name scale-ladder`; the v1 run's
[exact file](https://github.com/loopflowstudio/manabot/blob/c667feac/experiments/regimes/scale-ladder-allies-lessons-v1.json)
pinned w6):
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

The small-tier checkpoint is update 9,250 (`monitor-00000036`, saved at
2026-10-21 14:40 UTC, $9.88 of pod time): 151.6M learner decisions, 308.9M
environment decisions. Recorded before scoring: the monitor evaluator plays only
code opponents, so the frozen Mini 12261@10k checkpoint (SHA-256 `1a9933b0…`, from
the Mini's MTG-153 retained copy) is played through the head-to-head tournament
runner instead. That runner uses its 100 fixed cross-deck deals × 4 legs (400
games), not the 25 mirror deals. Heuristic and greedy units use a 600-second game
budget on rented CPU; head-to-head keeps its fixed 120 seconds.

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

### Result (closed 2026-10-21)

Jack Heart closed v1 at 20:20 UTC on 2026-10-21, after deciding that one-seat
learning will not be used going forward. The $50 tier was never reached:
training stopped at update 12,000 (196.6M learner decisions, the last retained
checkpoint, saved at 20:09 UTC) after $13.34 of pod time. Training ignored
SIGINT and SIGTERM and was killed, so there is no final paused export. All 64
run files were copied back byte for byte before the pod was deleted. The second
prediction (medium tier ≥ small tier + 0.05) is therefore **untested**. The
log-spaced curve's 16,000 point was never reached.

| World | Tier | Iterations | Learner decisions | Pod hours | vs heuristic (LvA) | vs heuristic (mean) |
|---|---|---:|---:|---:|---|---|
| Allies vs Lessons v1 (one seat) | $10 / 1 day | 9,250 | 151.6M | 16.8 | 0.70 [0.58, 0.82] | 0.69 [0.64, 0.74] |
| Allies vs Lessons v1 (one seat) | $50 / 7 days | not reached (closed at 12,000) | | | | |

First prediction: **held**. As Lessons against heuristic Allies, the small tier
scored 0.70, above the predicted 0.40, with a lower bound of 0.58.

Heuristic curve, 200 games per checkpoint, all games complete and terminal
([data](data/scale-ladder/v1-evaluation.json)):

| Updates | Learner decisions | Mean | LvA | AvL | LvL | AvA |
|---:|---:|---|---|---|---|---|
| 0 | 0 | 0.11 [0.07, 0.16] | 0.00 | 0.26 | 0.12 | 0.06 |
| 250 | 4.1M | 0.47 [0.45, 0.51] | 0.22 | 0.82 | 0.38 | 0.48 |
| 1,000 | 16.4M | 0.55 [0.50, 0.59] | 0.36 | 0.86 | 0.42 | 0.54 |
| 4,000 | 65.5M | 0.68 [0.61, 0.73] | 0.60 | 0.96 | 0.52 | 0.62 |
| 9,250 | 151.6M | 0.69 [0.64, 0.74] | 0.70 | 0.94 | 0.54 | 0.58 |

- The mean flattens between 65M and 152M decisions, while Lessons vs Allies keeps
  rising (0.60 → 0.70). Allies vs Lessons is saturated against heuristic
  (0.94–0.96), as MTG-153 found, so the mean understates late progress. The
  mirror cells have the most headroom left.
- Against the frozen Mini 12261@10k checkpoint (head-to-head, 400 games, 0 invalid),
  the small tier scored 0.81 [0.78, 0.84]: 0.74 as Lessons, 0.88 as Allies.
- Scripted greedy was also played at 9,250 for continuity only; heuristic is the
  tracking anchor (Jack Heart, 2026-10-21).
- One seed, so the intervals resample deals, not training runs.

Costs: calibration $0.40, training pod $13.34, evaluation about $2.60, of which
about $2.25 was wasted. The waste was two EU-SE-1 pods that never accepted SSH,
then two pods left idle for 4.3 hours because an endpoint-polling loop failed
unnoticed. The first evaluation attempt also crashed every game, because the
unit script lacked a `__main__` guard for spawned workers; it was rerun in full
and its output kept separately. Raw run and evaluation archives remain outside
the repository in `~/manabot-runs/mtg159/`, with SHA-256 sums in the data file.
