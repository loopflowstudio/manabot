# Training ownership and learning rules

RL entry points resolve a `TrainingRegime` and call
`manabot.training.execution.execute_regime`. Bounded diagnostics may use the
same `SeatRoutedCollector` and `update_iteration` primitives directly. A new
loop or optimizer implementation needs a concrete unsupported behavior and a
current consumer; an old import or test is not a reason to retain one.

`Learner` owns the admitted train-mode `ActorCritic`, Adam (epsilon 1e-5), device and
native-stream minibatch width. The executor owns collection, EMA, schedules,
tracking, checkpoint exports, failure receipts and recovery. `PairedLearners`
uses two independent owners with one collector. Supervised, search and belief
stages retain their separate data/target contracts.

## Credit and behavior

`model.returns.transition_gae` consumes rewards and end flags from the same
transition. Each column follows the same player until terminal. Complete-game
collection groups seat histories before packing them; terminal flags separate
all seats and games. An interleaved native A/B/A stream must never go directly
into the estimator. Changing reward signs cannot repair an incorrect trajectory.

Gamma and lambda advance per retained same-player decision, including consecutive
decisions. Opponent and skipped forced decisions add no discount steps. One-seat
exact-quota tails bootstrap from the same player's paused observation. Both-seat
self-play finishes current games after the combined decision floor, so actual
rows may exceed the quota. Explicit games-per-stream controls work in either arm.
Frozen/random opponents contribute no learner rows. Native winner identity sets
terminal win/loss/draw labels; draws stay zero. Strict collection rejects
truncation because continuing-task bootstrap from a missing final observation
is unsupported. Opted-in game-error recovery instead quarantines the entire
failed or truncated game, discards both seats' uncommitted rows and regathers.
It finishes games even with one learning seat; that mode uses a decision floor,
not the strict one-seat exact quota.

Collection freezes behavior within each batch. Complete games end before weights
change; exact-quota games may span updates. Stored full legal-action probabilities,
log probabilities and sampled-action likelihoods belong to the behavior that
actually generated each row. Minibatch updates change current weights, so these
likelihoods remain necessary even without replay. Targets and references are
constants; gradients flow through current policy and value outputs only.

## Objectives

`Learning(gradient="ppo")` uses the clipped likelihood-ratio surrogate,
configurable discount and separate policy/value lambdas. It normalizes advantages
over selected actor rows, shuffles minibatches over multiple epochs and uses
half squared scalar value error with a configurable coefficient. Reverse KL to
the reference and collection behavior are explicit regularizers. This control
is used by the full reward-driven learnability proof.

`DampedPolicyLearning(gradient="damped_policy")` uses undiscounted same-player
returns, separate policy/value lambdas, absolute-advantage selection, one ordered
epoch, native-stream-sized minibatches and no advantage normalization. Its
clipped surrogate includes reverse KL to behavior and an action reference.
Scalar value loss is squared error; categorical value loss is cross-entropy to
loss/draw/win lambda-return mixtures. Actor-only selection retains critic rows;
actor-and-critic selection removes both. Inclusive quantile selection keeps ties.
Method provenance, equations and MTG adaptations are in
[the source fidelity contract](../../docs/ataraxos.md).

Iteration schedules count completed collection/update boundaries, including
empty-filter skips. EMA advances once at each such boundary. Optimizer steps,
learner transitions and environment decisions are recorded separately. Wall-time
limits fail the run; they do not shorten the declared schedule.

## Admission and recovery

Policy checkpoint admission binds world, setup, input schema and architecture.
The loader returns eval mode; `Learner` explicitly enables train mode, including
recurrent encoders whose CUDA backward requires a training-mode forward. Starting
from admitted weights is a new run, not exact recovery.

Supported private recovery restores learner, Adam, EMA, collector, sampler,
minibatch, Python, NumPy and Torch RNG state at a committed update boundary.
It preserves attempts, prior costs and absolute schedule coordinates. No update
is applied from an incomplete collection. Recovery remains subject to source,
recipe and backend admission; CPU proof does not certify CUDA recovery.

Automatic failure capture does not enable tolerance. Opted-in game-error
recovery retains its attempt ledger independently of optimizer snapshots, so
restoring a committed update cannot erase later failures or replenish the error
allowance. Source repair requires explicit restart admission; replay advancement
alone neither admits training nor establishes the original game's outcome.
See [game crash recovery](../../docs/game-crash-recovery.md) for bounds and limits.

Current recipes use `damped_policy` directly. The historical `ataraxos_move`
discriminator is rejected: old recipes require pinned source. No compatibility
decoder, serialized-tag override or old preset execution path remains.

## Entry points and retired routes

`uv run manabot train` resolves `local-smoke-v1`: two CPU updates, two streams,
16 combined decisions per stream as a floor, both seats and complete games,
with a 120-second failure limit. It uses the baseline model and horizon-scaled
damped objective. `--learning-seats one` selects exact quotas. `--recipe baseline`
adds the ordinary baseline's independent evaluation. Neither smoke nor successful
export certifies playing strength.

The old standalone `Trainer` and `NetOpponentTrainer` loops, fake environment
shim, training presets and step1–4/first-light/exp11 launchers are retired.
They used different shaping/update contracts and are not silently translated.
Historical evidence and first-light report/store readers remain available;
reproduction of retired executions requires their pinned source revision.
