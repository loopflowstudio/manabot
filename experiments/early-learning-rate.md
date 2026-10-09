# Early learning rate on the mini

Status: queued on the M1 mini on 2026-10-09 to start after the ETU-118 final
evaluation releases the host. No results yet. This is **exploratory monitoring,
not a held-out comparison**: every score below is on monitoring deals that have
been inspected repeatedly, with two training seeds per arm.

## Question

The self-play recipe sets `lr = clip(0.5 / iteration^1.1, 5e-6, 1e-4)`, so the
rate sits at the `1e-4` cap for roughly the first 2,300 updates. One GPU run of
the same small model (width 64, depth 2) showed the policy barely moving per
update at that cap: collection KL about 0.0014 and clip fraction about 1.25%.
That suggests the early rate could be several times higher. It is an inference
from one seed. This screen tests it directly: does a higher constant rate learn
faster over the first 1,500 updates, and does it stay stable?

## Design

| | |
| --- | --- |
| Recipe | ETU-118 mini recipe ([source](regimes/early-learning-rate-mini-source.json)): width 64, depth 2, 4 heads, masked-mean scalar value, Ataraxos move rule, 4 streams x 64 transitions, current-self behavior, CPU, one learner thread |
| Arms | constant learning rate `1e-4` (control), `3e-4`, `1e-3` |
| How the rate is held constant | `learning_rate_min = learning_rate_max = rate`; nothing else differs between arms |
| Seeds | 11851, 11852, each run under all three arms |
| Length | 1,500 updates per run, fresh initialization |
| Monitoring | 25 deals x 4 seat/deck legs = 100 games vs scripted greedy at updates 0, 300, 600, 900, 1,200, 1,500 |
| Order | seed 11851: `1e-4`, `3e-4`, `1e-3`; then seed 11852 in the same order |
| Runner | [`early_learning_rate.py`](runners/early_learning_rate.py), through the shared `run_experiment` coordinator |

Pairing. Initialization and collection streams derive from the training seed
alone, so the three arms of a seed start from identical weights and identical
first rollouts, and every checkpoint is scored on the same 25 deals
(1,911,183,000-024, ETU-118's monitoring deals). The laptop smoke confirmed the
initial weights are tensor-for-tensor equal across arms. The three update-0
evaluations of a seed should therefore agree exactly; a disagreement means the
pairing is broken and the comparison should not be read.

The control arm is what the unmodified schedule does for all 1,500 updates, so
it is also the reference for "no change". The seeds and deals match ETU-118 so
the control can be read loosely against that baseline's early curve (about 30%
untrained, about 46% at update 1,240). That reading is loose: ETU-118 ran an
older source revision with recoverable collection, and this screen does not.

Tau still follows its own iteration schedule in every arm. Gradient clipping
stays at `max_grad_norm = 0.267`.

## What is recorded

Per checkpoint: score vs scripted greedy (a draw counts half) with its
deal-block bootstrap interval, and the same score split by the deck the policy
played (as Lessons, as Allies; 50 games each).

Per update, summarized in 300-update windows: collection KL, clip fraction,
entropy, value loss, median gradient norm before clipping, the share of updates
whose gradient was clipped, the largest legal-logit gap, and counts of skipped
steps, rejected steps and non-finite log-probabilities. The trainer already
exports all of these; nothing was added to the learner. The first three and the
gradient norm are the value at the last optimizer step of each update.

`uv run python -m experiments.runners.early_learning_rate report <run>` prints
both tables from retained evidence and is safe to run while training continues.

## Reading rule

Fixed before any result exists.

**Which arm is better.** For each arm and seed, average the score over the five
trained checkpoints (updates 300 through 1,500). Compare an arm with the control
by the paired difference of that average within each seed.

- Call an arm *faster* only if both seeds favor it and the mean paired
  difference is at least 10 points.
- Call an arm *slower* by the same rule with the sign reversed.
- Otherwise the arms are *not distinguished*.

A faster arm is a candidate for a properly powered confirmation. It is not a
finding, and it does not change the recipe by itself.

**What counts as instability.** Any of these, reported per arm and seed:

- a rejected optimizer step, a non-finite count above zero, or a run that fails;
- clip fraction averaging above 20% over any 300-update window (the recipe's
  ratio clip is 0.2, so this means a fifth of sampled actions hit it);
- mean entropy in the final window below half the control's in the same seed;
- a score that drops 15 points or more between consecutive checkpoints in both
  seeds, or ends below its own update-0 score.

An arm that is faster and also trips an instability flag is reported as both.
Do not net them against each other.

**Gradient clipping.** Report the share of updates clipped in each arm. If
nearly every update is clipped in every arm, clipping rather than the rate is
the first thing to vary next. Adam's normalization means a uniform rescale of
the gradient changes the step less than it would under plain SGD, so this is a
pointer, not a measured effect.

**What this cannot show.** One 100-game score has a standard error near 5
points, and two seeds give no usable estimate of seed-to-seed spread. Only
large effects are visible. Differences under 10 points, per-deck splits (50
games each) and single-checkpoint orderings are descriptive only. Nothing here
says what happens after update 1,500, where the real schedule begins to decay.

## Execution

- Host: M1 mini, CPU only, one learner thread plus one evaluator thread. No
  cloud calls, no tracker.
- Expected cost: about 4.7 s per update in ETU-118, so about 2 hours per run and
  about 12 hours for the six runs. Evaluations (about 5 minutes each, 36 total)
  overlap learning on the second thread.
- Bounds: 3 hours per run, 16 hours for the whole experiment. The coordinator
  stops and marks the experiment incomplete at either bound. It also stops if
  free disk falls below 40 GiB; projected use is about 3 GiB.
- Runs are not recoverable mid-run. A failed or timed-out run is retained as
  failed and is not retried.
- Source: a fresh clone on the mini at this document's commit, which is `main`
  at `4850e500` plus this experiment and one coordinator fix (below).

### Coordinator fix found by the smoke

The laptop smoke crashed once at the end of an evaluation. On macOS, signalling
a process group that holds only exited, unreaped processes returns `EPERM`
instead of `ESRCH`, and the coordinator treated that as fatal. A crash there
kills the live learner, and the lost run is never retried. `stop_process_group`
now treats `EPERM` as "already stopped" once the group leader is known to have
exited. ETU-118 did not hit this in more than 95 evaluations on the mini, so it
is rare there; the fix removes the exposure for an unattended run.
