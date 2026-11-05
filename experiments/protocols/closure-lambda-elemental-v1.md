# Batch closure and lambda on the ten-deck Elemental recipe, v1

Task 07c490, Jack Heart, 2026-11-05. Authorized: $25 total, a hard cumulative cap
covering training, evaluation, failures and machine setup for both studies, separate
from MTG-251's $25. Spending beyond it needs Jack Heart's explicit approval. This
protocol is committed before any scored run. Calibration runs are unscored and charged
to the cap.

MTG-251 (`experiments/protocols/closure-lambda-v1.md`, report under
`experiments/data/closure-lambda-v1/`) asked both questions on Allies/Lessons. Jack
Heart said on 2026-11-05 that this is not the current recipe, which is the ten-deck
Elemental setup at a minimum, so both comparisons repeat here with that recipe. The
Allies/Lessons results set no expectation for these; each conclusion below stands on its
own.

## Recipe

`experiments/elemental_variants.py` varies MTG-181/183's frozen recipe
(`experiments/regimes/elemental-long-183-v1.json`): the 64-wide, two-layer, four-head
scalar masked-mean policy, Adam, damped policy learning with absolute-iteration LR/tau
schedules and EMA .999, 256 streams, at least 64 decisions per stream per update,
independent uniform random ordered deck pairs at game roots including mirrors, both
seats learning, CUDA float32, one producer. Two compatibility changes, neither chosen
for effect: the frozen file names world w13 and these variants relabel it to the
current world w17, as MTG-181 did for w10, with new regime IDs; and the launch sets
the watchdog from each run's window, as `experiments/elemental.py` does. The recipe's
policy input (v1) and every learning parameter stay. The CUDA numerics gate is off, as
in MTG-251's launches; it is a job setting, not part of the recipe.

Both studies share the evaluation and launch scaffolding in
`experiments/learning_signal.py`; neither study's arms feed the other's comparison.

## Study 1: batch closure (`experiments/elemental_closure.py`)

Question: at matched learner decisions, wall time and cost, does learning from
complete games differ from learning at an exact quota, on the Elemental recipe?

Recipe: `elemental_variants.closure_comparison`. Exact quota needs one learning seat,
because at a pause only the acting seat has a next observation, and Elemental's recipe
learns from both seats. **Both arms therefore learn from one seat**, seat-balanced
across streams (stream s seats the learner at s % 2), and both drop the recipe's
game-error recovery, which exact quota rejects, so the arms differ only in
`batch_closure` and `transitions`. Everything else of the recipe stays. The departure
means the conclusion is about closure under one-seat learning on the ten-deck world: it
does not measure two-seat training, and a one-seat run takes about half the learner
decisions per update of the two-seat recipe over the same games. The two-seat idle cost
that exact quota would recover is reported separately (below), not built.

The complete arm keeps the recipe's floor of 64 rows per stream; the exact arm's quota
is what that floor realized per stream in calibration once game length settled (set by
the calibration amendment below). Seeds 70101–70103 give both arms identical initial
weights per seed. Target: 150 updates.

Predictions at the last update every run reached:

- Complete games score within .03 of exact quota, with a seed-level interval that
  includes zero.
- Complete games spend 0–20% more collection time per learner decision than exact
  quota, and idle stream steps are .40–.60 of stream slots in the complete arm.

## Study 2: lambda (`experiments/elemental_lambda.py`)

Question: on the two-seat Elemental recipe, does giving the game result more direct
weight help?

Recipe: `elemental_variants.lambda_sweep`, the recipe itself with only `policy_lambda`
and `value_lambda` varied. Seeds 70201–70203. Target: 75 updates.

| Arm (policy / value) | Result weight 20 decisions out (policy / value) | Purpose |
|---|---|---:|
| .5 / .8 | 1e-6 / .012 | The recipe, baseline |
| .8 / .8 | .012 / .012 | Policy trace alone |
| .5 / .95 | 1e-6 / .36 | Value trace alone |
| .8 / .95 | .012 / .36 | Both raised |
| .95 / 1 | .36 / 1 | Most direct; the value target is the game result |

Predictions at the last update every run reached:

- No arm beats the baseline by more than .03; .95 / 1 scores at least .08 below it.
- Policy .8 alone moves the score less than .03 in either direction.
- The critic's absolute residual 5+ decisions from game end rises with value lambda:
  at least 1.5× the baseline's for the .95 arms and 2× for 1.

Critic quality is the absolute lambda-target residual by distance from game end, from
the `selection_groups` metrics, read per arm at the same updates.

## Two-seat idle streams, reported not built

Jack Heart asked (2026-11-05) how much wall time two-seat Elemental runs lose to idle
streams, since that is the only case for building two-seat exact quota. Every two-seat
run of study 2 records `idle_stream_steps`; the report gives the idle share of stream
slots and an upper bound on the wall time it costs: collection seconds less batch
assembly, times that share, as if an idle slot cost as much as a stepped one (batched
inference is sublinear in live streams, so the true loss is smaller). Study 1's
one-seat arms give the direct measurement of finishing games against pausing.

## Evaluation

Jack Heart, 2026-11-05: not the full 10×10 grid; random deck-versus-deck pairings
instead. `learning_signal.ELEMENTAL_PAIRS` fixes **50 pairings** before scoring: each a
deck pair drawn with replacement from the 45 distinct Elemental pairs (seed 7070001;
mirrors excluded, since exchanging a mirror's decks changes nothing) with its own deal
(7070000000–7070000049, reserved here, disjoint from the grid's, the pairing graph's and
every other protocol's). The digest `9bff7292…503f` is pinned in code. Each pairing
plays the model's four cross legs against the fixed heuristic: both decks, on the play
and on the draw, of the same deal, 200 games per checkpoint. Identical across arms and
seeds; the 600-second and 10,000-command game limits are the grid's. Played through the
replay-checked arena on rented 32-vCPU machines, never the laptop, with journals in
memory. Raw exports are scored, not EMA.

Scored updates: 20, 40, 75, 150 where a run reaches them, and each run's endpoint.
Update 0 is not scored: a seed's arms share their initial weights.

Analysis per study, as MTG-251: mean over seeds per arm on the learner-decision clock,
then at matched wall time and cost. Seed-level t intervals and pairing-resampled
intervals are reported separately. Also reported per arm: rows, games, optimizer steps,
environment decisions, collection, assembly, idle-stream and learning time, and cost.
Complete-game batches vary in size, so rows per update and schedule position at matched
decisions are reported, and divergence between arms is named rather than corrected.

## Budget

L4 at $.59/hour; the launcher reserves a run's window plus 15 minutes of setup at the
ceiling rate of $.60. Windows are sized from MTG-181's measured pace (28–36 s per
two-seat update at 256 streams; 4.8–5.9M learner decisions per hour) so every run should
reach its target.

| Line | Ceiling |
|---|---:|
| Calibration | $.60 |
| Study 1 training | $6.60 |
| Study 2 training | $12.60 |
| Evaluation | $3.00 |
| Reserve for failed attempts | $2.20 |
| **Total** | **$25.00** |

| Study | Runs | Window hours | Job dollars | Reserved |
|---|---:|---:|---:|---:|
| Calibration, complete games, 30 updates | 1 | .6 | $.36 | $.51 |
| 1, each arm | 3 | 1.5 | $.89 | $3.15 |
| 2, each of 5 arms | 3 | 1.15 | $.68 | $2.52 |

Each arm launches as its own all-or-nothing cohort (MTG-251's queue mode undersized
windows). Evaluation passes pass their own limits from the experiment files ($1.20 and
$1.80) inside the line. Ceilings are reservations; an unspent line may cover a failed
attempt elsewhere, and the total may not move.

## Kill criteria

- Stop launching when billed plus reserved spend would exceed $25. Keep every failure
  and its cost. A machine that has not booted within 15 minutes is terminated and its
  window counted.
- If the calibration run fails, or the complete arm's realized rows per stream after
  the opening updates cannot be matched by a quota within 10%, stop study 1 before
  scoring and report why.
- A run stopped by its watchdog keeps its exports; matched comparisons extend only to
  the learner decisions every run reached. Missing runs stay in the denominator.
- If the budget supports fewer seeds, cut seeds equally across a study's arms before
  launch and record the change here, never after scoring.

Deleting exact quota is out of scope and waits for this report.

## Calibration

Pending: one seed (70100) of the complete arm, 30 updates, unscored.
