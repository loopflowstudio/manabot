# Training loops

An update learns from one collected batch, a loop keeps playing and updating
until it is done, and a recipe is our loops with our settings. Write your own
recipe or use one of ours, and use Etude to launch, track and evaluate it
([remote training](training.md#remote-training)), or bring your own.

This guide covers the updates we maintain, the pieces they are made of, and a
complete loop to copy. [How a manabot learns](learning.md) explains the
method; [training](training.md) explains running a recipe.

- Train a manabot: the damped policy update, which every current recipe uses
  ([the updates we maintain](#the-updates-we-maintain)).
- Compare a new method: PPO is the control
  ([the updates we maintain](#the-updates-we-maintain)).
- Change one step: [the player's pieces](#the-players-pieces) and
  [standard methods](#standard-methods-kept-local).
- Mix fresh games with stored decision continuations:
  [snapshot-aware self-play](#snapshot-aware-self-play).
- Write your own loop: [a complete self-play loop](#a-complete-self-play-loop).

## The updates we maintain

A recipe's loop ([training](training.md#write-a-recipe)) runs one update
each iteration; Etude's run adds saving, resuming, tracking and checkpoints
around it. Its learning record selects the update; both live in
`manabot/training/updates/`:

- **Damped policy** (`DampedPolicyLearning`, `update_damped_iteration` in
  `manabot.training.updates.damped_policy`): the update every current
  recipe selects. A clipped surrogate with reverse KL penalties toward the
  behavior policy and a reference magnet, over native-stream-wide minibatches
  in batch order, with scalar or categorical value targets and no advantage
  normalization ([the loss](learning.md#4-update)). Use it to train a manabot.
- **PPO self-play** (`Learning`, `update_iteration` in
  `manabot.training.updates.ppo`): the control. Shuffled minibatches, advantage
  normalization and filtering, scalar value targets and optional reference
  and collection KL penalties. Torch prepares one host minibatch ahead while
  the current minibatch trains, with device work on the calling thread; the
  epoch joins its preparation worker even if an update fails. Use PPO as the
  baseline a new method must beat.

Both are written once and take the backend as their first argument: a
configured backend from `manabot.backends.resolve` or the module
`manabot.backends.torch`. Then come a
`Learner` (`manabot.backends.torch.learner`), which holds the model, its Adam
optimizer, the precision and a profiler, and a learning record
(`manabot.training.learning`). Call them directly to reproduce a recipe's
update exactly:

```python
import manabot.backends.torch as backend

learner = Learner(agent, "cpu", minibatch_rows=64)
update_damped_iteration(backend, learner, batch, move_learning(), iteration=1)
update_iteration(backend, learner, batch, Learning(), progress=0.0, rng=np.random.default_rng(0))
```

`move_learning()` (`manabot.recipes.builders`) is the current recipe's
`DampedPolicyLearning(gradient="damped_policy")`. Each returns the update's
diagnostics (`UpdateDiagnostics`, a typed dictionary); EMA and checkpoints
stay with the caller.

Recipes and the self-play loop pass `manabot.backends.torch`. Inside an update,
array math goes only through `backend.rl`, `backend.learner` and
`backend.arrays`, and observations come from `backend.rollout`; NumPy keeps
the host-side bookkeeping and diagnostics (which rows train, minibatch
indexes, action-type counts, selection groups). Running the same update in
another framework changes the import, and the learner comes from that
backend; the damped update runs on torch and on MLX. The
contract a backend meets is in
[framework backends](../manabot/backends/README.md#algorithms-are-written-once).

## The player's pieces

These read the player's observations, legal actions and `ActorCritic` heads.
Shapes use `T` for time, `E` for trajectories (columns), `B` for a flat batch
of rows and `A` for padded legal offers.

### Play

| Piece | What it does |
| --- | --- |
| `open_collector` (`manabot.training.rollout.collection`) | Opens native game streams as a `Collector`; the required `numerics` session (`backend.collection.CollectionSession(device=..., seed=..., precision=...)`) owns the device, precision and private sampling RNG. `collect(collector, agent, decisions)` plays with `agent` and returns a `RolloutBatch`; `collect_pair(collector, agent, decisions)` returns one batch per model when `opponent_mode="learning"` and `opponent_agent` is a second learner. `close_collector(collector)` stops the streams. |
| `RolloutBatch` | One update's rows, `[T,E]`. Complete games are packed into one column separated by end flags. `rewards`, `dones` (real game terminals only), `values` (critic at collection), `actions`, `logprobs` (behavior log likelihood of the chosen action), `next_obs` for bootstrapping unfinished tails, and `outcome_probabilities` `[T,E,3]` for a categorical critic. Optional `trajectories` (`TrajectorySpan` table) and aligned provenance fields (`episode_source`, `snapshot_id`, `branch_group_id`, `is_root_decision`, `sample_weight`) separate horizon cuts from terminals and carry snapshot metadata without mutating the batch. `observations(indexes)` materializes flat rows as NumPy observation buffers and the full behavior distribution. |
| `AdmittedRollout(agent, batch)` (`manabot.backends.torch.rollout`) | Checks a batch's policy input against `agent` once, so the model's forward does no validation. `admit_rows(indexes, chunk_rows)` admits the rows an update will train; `rows(indexes, device)` prepares learner rows on the host, independently of autograd mode, and returns the `Prepared` batch on `device` with an `ActionRows` whose `log_probabilities` `[B,A]` (NumPy) are the behavior log likelihoods a KL or ratio term reads, and raises `ValueError` for a row not admitted. `legal(prepared)` is those rows' legal-action input `[B,A]`. `minibatches(order, size, device)` returns the same learner preparation with one host batch prefetched. `next_observation(device)` returns inference-prepared bootstrap rows — one per column for legacy batches, or the indexed frozen viewer observations when `trajectories` is set (`None` when there are no bootstrap observations) — independently of autograd mode. The batch must not change afterwards. |

The opponent is `opponent_mode`: `"self"` (one model, both seats learn by
default), `"learning"` (two models), `"frozen"` (with `opponent_agent`, only
the learner's seat learns) or `"random"`. `batch_closure="complete-games"` is
the default; `"exact-quota"` pauses every stream at exactly `decisions` rows,
carries the game into the next batch and needs `learning_seats="one"` and a
critic bootstrap from `next_obs`
([batch boundaries](learning.md#can-a-batch-contain-unfinished-games)).
Collection never changes weights. A native game failure raises unless the
collector is given a `GameErrorPolicy`
([game failures](training.md#game-failures)).

### The policy and its terms

The policy reads a `Prepared` batch (`manabot.policy.prepare`): NumPy
observation buffers that the policy's `prepare` has checked and indexed on
the host, moved to the model's device. `AdmittedRollout.rows` returns one for
learner rows; elsewhere build it with `on_device(agent.prepare(obs), device)`
(`manabot.backends.torch.arrays`). `ActorCritic.forward(prepared)` returns
logits and a signed expected value `[B]`; `forward_distribution(prepared)`
returns the raw critic output instead: `[B,3]` logits for
`value_kind="categorical_wdl"`, `[B,1]` for scalar. An observation dictionary
passed to either raises `TypeError`. For NumPy in and out without gradients,
`agent.evaluate(agent.prepare(obs, inference=True))` returns logits and values
in the layout simulation uses
([backends](../manabot/backends/README.md#the-torch-backend)).

| Piece | Module | What it does |
| --- | --- | --- |
| `legal_mask(admitted.legal(prepared))` | `backend.rl` | The legal offers `[B,A]`, refusing a corrupt mask. |
| `policy_logs(logits, valid)` | `backend.rl` | Log likelihoods `[B,A]`, `-inf` off the legal mask; every policy term reads these. |
| `action_logs(logs, actions)` | `backend.rl` | Each row's entry at its sampled action `[B]`. |
| `reference_distribution(obs, kind)` | `manabot.policy.prepare` | The magnet a reverse KL pulls toward, in NumPy `[B,A]` float32 from the `actions_valid` and `actions` buffers of the NumPy observation rows: `"uniform"` over legal offers or `"action_type_uniform"`. `reference_logs(reference, valid)` (`backend.rl`) is its log, zero off the legal mask. |
| `damped_policy_loss(rl, logits, actions, advantages, behavior_logs, reference, valid, *, clip, collection_kl, tau, reference_forward_kl=0.0)` | `manabot.training.updates.objective` | The damped policy objective per row: the clipped surrogate plus both reverse KL penalties, plus the forward-KL floor when `reference_forward_kl` is nonzero ([why](learning.md#why-the-floor-is-a-forward-kl)). |
| `forward_kl(logs, target_logs, valid)` | `backend.rl` | KL(target ‖ π) per row `[B]`, summed over legal offers and zero on padding; the floor reads it against the magnet. |

`backend.rl` is `manabot.backends.torch.rl` for the PyTorch model.
Reduced precision is a context around the forward only:
`with backend.learner.forward_precision(precision, device):` (or
`learner.forward_precision()`) runs `bf16` autocast, and you cast the outputs
with `backend.arrays.float32` (`.float()`) before any loss.
`bf16` runs on CUDA and MPS and `tf32` on CUDA only; losses and their float64
terms stay in full precision on every device.

### Health and metrics

None of these change the update; call them when you want the numbers.

| Piece | Module | Records |
| --- | --- | --- |
| `batch_health(action_data, advantages, selected, *, actor_only)` | `manabot.training.updates.health` | A `NumericalHealth` record for the batch: legal-logit range, underflow, forced rows, zero advantages, rows each head trains on. Pass `batch.action_data`. |
| `OptimizerHealth(health, iteration, profiler=None, *, backend)` | `manabot.training.updates.health` | The player's `StepRecorder`. Pass as `learner.step(loss, max_grad_norm, recorder=observer)` (or to torch's `gradient_step`) to count completed and rejected steps. Both Torch and MLX learners supply gradient norms, including the critic readout's and optional `value_token`, tensor counts and parameter displacement on the first step of iteration 1 and every 25th iteration. With a profiler, backward and the optimizer step are timed. |
| `clip_fraction(log_ratio, clip)` | `backend.rl` | The fraction of rows whose likelihood ratio left `1 ± clip`, the record's `clip_fraction`. |
| `selection_diagnostics(advantages, residuals, selected, ends, action_types)` | `manabot.training.updates.selection` | Advantage and residual quantiles by action type and distance to the game's end ([reading a run](learning.md#reading-a-run)). Every argument is `[T,E]` host rows (NumPy, or CPU tensors): `ends` is `batch.dones` and `action_types` is `batch.action_data.selected_types(batch.actions, ACTION_TYPE_DIM)` (`manabot.env.observation`). |
| `combat_probe(policy, batch, *, limit=256, execution_rows=None)` | `manabot.training.combat_probe` | Attack and block declare/decline log-odds (`gap`) and board-edit `shift` metrics on evenly spaced two-offer decisions from `batch`, through `policy.prepare` and `policy.evaluate` ([combat probe](experiment-metrics.md#combat-probe)). |

## Standard methods, kept local

The rest is standard reinforcement learning, kept as small local functions in
the backend's `rl.py` and `learner.py` rather than taken from an RL library.
The torch versions below read only tensors, an `nn.Module`, an optimizer and
small typed records, and import nothing else from manabot
(`tests/test_package_boundaries.py` enforces it), so another role reuses them
unchanged. Another [backend](../manabot/backends/README.md) has the ones
its updates call, with the same names and arguments over its own arrays.
Each returns one value per row `[B]` where it is a
loss, so you choose the reduction, mask and weight.

| Piece | Module | What it is |
| --- | --- | --- |
| `transition_gae(rewards, values, ends, next_value, gamma, lam)` | `manabot.backends.torch.rl` | GAE (Schulman et al. 2016): advantages and lambda returns `[T,E]`. Each column is one agent's trajectory; an end cuts bootstrap and trace. `next_value` `[E]` is read only where a column's last row is not an end. Our updates call it once per lambda (policy and value). |
| `standardized_advantages(advantages, over=None)` | `manabot.backends.torch.rl` | Every row shifted and scaled by the mean and population std of the rows `over` selects (all rows when omitted); the std is clamped at 1e-8. Our PPO update takes the statistics over the rows it trains on. |
| `categorical_lambda_returns(probabilities, rewards, ends, bootstrap, trace_lambda)` | `manabot.backends.torch.rl` | Undiscounted loss/draw/win lambda-return targets `[T,E,3]`; a terminal row is the one-hot outcome (`rewards` -1, 0 or 1, nonzero only at an end). `bootstrap` `[E,3]` is the outcome distribution past each column's last row (read only where it is not an end). |
| `expected_outcome(probabilities)` | `manabot.backends.torch.rl` | The signed expected outcome, P(win) − P(loss), of loss/draw/win mixtures `[...,3]`. |
| `clipped_surrogate(log_ratio, advantages, clip)` | `manabot.backends.torch.rl` | Negative PPO clipped surrogate (Schulman et al. 2017), with float64 ratios. |
| `reverse_kl(logs, target_logs, valid)` | `manabot.backends.torch.rl` | KL(current ‖ frozen target) over valid entries. |
| `squared_value_loss(values, targets)`, `categorical_value_loss(value_logits, targets)` | `manabot.backends.torch.rl` | Scalar squared error; cross entropy of `[B,3]` loss/draw/win logits against target mixtures. |
| `entropy(logs, valid)` | `manabot.backends.torch.rl` | Entropy over valid entries. |
| `selection_mask(advantages, kind, fraction, minimum)`, `selected_mean(values, selected)` | `manabot.backends.torch.rl` | Keep a top count or inclusive quantile of advantage magnitude; average a per-row term over the mask (zero gradient when empty). |
| `gradient_step(loss, model, optimizer, max_grad_norm, *, recorder=None)` | `manabot.backends.torch.learner` | Backward, `clip_grad_norm_`, step; returns the pre-clip norm. |
| `Learner.loss(objective)`, `Learner.step(loss, max_grad_norm, *, recorder=None)` | `backend.learner` | The same step on a `Learner`'s policy and optimizer, in every backend: `objective` is a closure over the forward whose first result is the loss. |
| `update_ema(averaged, model, rate)` | `manabot.backends.torch.learner` | Moves `averaged`'s parameters toward `model`'s by `1 - rate` and copies buffers. Start from `deepcopy(model)`. |

`gradient_step` raises `NumericalError` (`manabot.backends.torch.rl`) instead of
stepping on a detached or nonfinite loss, a loss with no gradients, or
nonfinite gradients, and after a step that leaves a nonfinite parameter. Its
`invariant` names the check and its `tensors` hold the offending values. Any
object implementing `StepRecorder` (`manabot.training.updates.health`) observes
the step; `OptimizerHealth` is the player's. Both Torch and MLX learners check
`sampling()` and deliver their `StepSample` measurements through
`record_sample(sample)`. Gradient measurements arrive after clipping, so they
remain available if Adam produces a nonfinite parameter. Both learners call
`record_step()` after the updated parameters pass the finite check.
Set the learning rate with `learner.set_learning_rate(rate)`
(or on `optimizer.param_groups`) before stepping to schedule it.

### What a deck builder reuses

The deck builder's design, as it stands: free-order picks until a stop, a
win/draw/loss outcome head at every prefix, entropy-to-go, and PPO clip 0.2
plus reverse KL to the generating policy. Against that design:

- **Reused unchanged:** `gradient_step` with a recorder of its own,
  `update_ema`, `clipped_surrogate` (clip 0.2), `reverse_kl` against the
  generating policy's logs, `categorical_value_loss` for the outcome head at
  each prefix, `categorical_lambda_returns` for its targets (one column per
  build, an end at the stop, `trace_lambda=1` for the plain final outcome),
  `transition_gae` for advantages over the same columns, `entropy` per pick,
  and `selection_mask`/`selected_mean` if it filters.
- **Not here:** entropy-to-go, the sum of the remaining picks' entropies,
  has no piece; it is a reverse cumulative sum per build over `entropy`.
  The masked distribution over the remaining pool plus a stop belongs to the
  deck builder, as `legal_mask` and `policy_logs` belong to the player; so do
  its collection (building decks and playing the games that score them) and
  its loop.

## Snapshot-aware self-play

A `SnapshotBank` stores decisions that self-play can revisit. Use both learning
seats, complete-game closure and a self-play behavior policy. Snapshot collection
and adaptive matchup sampling each own the episode schedule.

Save this recipe as `snapshot_recipe.py`. Plain and snapshot collection support
`1 <= workers <= streams`. Adaptive collection uses one worker. Single-worker
snapshot collection also accepts a buffered `SnapshotVector` or a factory taking
`(observation_space, match, num_envs, seed)`. The deprecated
`InMemorySnapshotVector.factory` alias will be removed in 0.4.0.

```python
from typing import Literal

from etude.train.tracking import track
from pydantic import Field

from manabot.recipes.baseline import BaselineConfig, settings
from manabot.training.loops.self_play import CLOCKS, self_play
from manabot.training.rollout.snapshots import CapturePolicy, SnapshotConfig


class SnapshotRecipeConfig(BaselineConfig):
    updates: int = Field(default=8, ge=1)
    device: Literal["cpu", "mps", "cuda"] = "cpu"
    workers: int = Field(default=2, ge=1)
    replay_fraction: float = Field(default=0.5, ge=0, le=1)
    fresh_floor: float = Field(default=0.25, ge=0, le=1)
    group_size: int = Field(default=4, ge=1)
    horizon: int | None = Field(default=8, ge=1)
    plr_perp: bool = False


def train(config: SnapshotRecipeConfig) -> None:
    snapshot = SnapshotConfig(
        capacity=256,
        replay_fraction=config.replay_fraction,
        fresh_floor=config.fresh_floor,
        group_size=config.group_size,
        horizon=config.horizon,
        fresh_weight=0.0 if config.plr_perp else 1.0,
        capture=CapturePolicy(every_n_decisions=8, max_captures_per_batch=16),
    )
    inputs = settings(config).model_copy(
        update={"snapshot": snapshot, "workers": config.workers}
    )
    with track(clocks=CLOCKS, config=config) as run:
        self_play(inputs, run, updates=config.updates)
```

```bash
uv run etude train snapshot_recipe.py --experiment snap-demo --until updates=4
uv run etude train snapshot_recipe.py --experiment snap-demo --resume --until updates=8
uv run etude train snapshot_recipe.py --experiment snap-plr --config '{"plr_perp": true}'
```

### Capture, horizon cuts and `K`-branch scheduling

`CapturePolicy` captures a root before the chosen action. Set
`every_n_decisions`, `probability`, or both. By default only fresh decisions
with at least two legal actions qualify. Use `fresh_only`, `min_legal_actions`
and `max_captures_per_batch` to change eligibility. An optional
`capture_predicate` on `self_play` or `open_collector` receives a
`CaptureDecision` with the slot, chooser, one-based decision number, native
legal action count and current branch.

The replay allowance is `min(replay_fraction, 1 - fresh_floor)` of episode
starts. An empty bank starts fresh games and accrues no credit. Once the
allowance funds a whole group, free streams draw its `group_size` siblings.
At the batch floor they finish outstanding siblings and retries. A retry spends
a replay start and can borrow against future fresh starts. Each group has up to
`max_group_retries` retries (default one). Failed episodes lose their rows and
captures. Exhausting a group's retries discards every sibling. [Game failures](training.md#game-failures)

`horizon=H` counts surfaced decisions across both seats of each continuation.
At a terminal, each seat that acted receives its own result. At a horizon cut,
each contributing seat receives a frozen viewer observation at the same game
revision. For example, choosers 0, 0, 1 yield two spans of lengths two and one.
Each span indexes its viewer in `next_obs`. Real terminals alone set `dones`.
Credit and the batch inspector follow these spans. `batch.trajectory_rows()`
also handles strided rows in rectangular batches.

For a direct start, use `SnapshotRecipe.from_replay_json(raw, chooser)` with a
`managym-snapshot-v1` export. It preserves semantic Commands and interventions.
A retained positional prefix also supports
`SnapshotRecipe.from_inputs(game, digest, chooser)`.

Open the collector with `snapshots=True` (a bank or custom vector implies it)
and call `collect(..., snapshot_starts=[SnapshotBranch(id, recipe, horizon=3)])`.
Slots consume starts in order. A `None` entry or an exhausted sequence starts
a fresh game. Siblings with the same cold snapshot ID share one replay per
allocation. A bank keeps that handle across allocations.

Branches restore exactly by default. To shuffle siblings with common random
numbers, pass the same `RestoreOptions.shuffle_libraries(seed=...)` to each
branch's `restore_options`. A direct start may force its first action. If that
action ends the game, the branch closes with zero sampled decisions. Grouped
schedules require the policy to choose the opening action.

### RLOO baselines, stratified filtering and `PLR⊥`

For `K >= 2` siblings of one snapshot and chooser, each root's actor advantage
is its own return minus the mean of the other siblings' returns. The root
critic target is the mean across all siblings. Scalar returns use the rule's
`gamma`, with a frozen value at a cut. Categorical returns use the terminal
outcome or the frozen WDL distribution, with win probability minus loss
probability for the actor. Descendants and singletons use the ordinary lambda
targets. `root_actor_weight` on either learning rule scales the actor's root
contribution (default 1). Each head scales its weights to mean one over the
rows it trains in the whole batch. A minibatch containing only fresh rows
therefore still reflects their weight relative to snapshot rows. The damped
policy update reports collection and reference KL as unweighted means over
the trained rows of its last minibatch.

The advantage filter selects eligible rows independently within each
`episode_source`. Diagnostics in `source_selection` report eligible and
retained counts per source, plus the threshold and retention rate for nonempty
sources. `fresh_weight=0` enables discovery-only fresh games. Zero-weight rows
still yield scoring signals. Only positive-weight rows participate in filtering,
PPO standardization and optimizer denominators. An all-zero batch skips Adam
and EMA and can still commit discoveries.

### Scoring signals and custom scorers

`SnapshotConfig.signal` sets the priority. `signal_scope` selects root rows
or whole episodes (default).

| Signal | Measurement |
| --- | --- |
| `critic_residual` (default) | Absolute difference between target and collection value |
| `positive_value_loss` | Positive part of that difference |
| `wdl_entropy` | Entropy of the collection critic's categorical outcome distribution |
| `policy_entropy` | Entropy of the behavior policy over legal offers |
| `action_value_spread` | Maximum minus minimum caller-supplied legal action value |
| `staleness` | Generations since the last visit, evaluated at draw time |

Stored scores halve every `score_half_life` generations. Draws visit new entries
first. Then every fifth group advances a coverage queue. Other draws choose
the oldest visit with probability 0.25 and sample by score rank otherwise.
A full bank evicts the lowest-priority entry, breaking ties by oldest visit
and then snapshot ID. Built-in staleness evicts by oldest visit and then ID,
so newly captured roots can enter the bank.

Pass a runtime `snapshot_scorer` to `self_play` (or `scorer` to `SnapshotBank`) for a custom priority.
It receives `EntrySignals(root_signals, episode_signals, age_since_visit)`
and returns a finite nonnegative number. Both slices contain read-only NumPy
arrays in packed row order.

```python
with bank.begin_batch() as plan:
    batch = collect(collector, behavior_agent, num_steps, snapshot_plan=plan)
    admitted = backend.rollout.AdmittedRollout(learner.agent, batch)
    targets = compute_batch_targets(backend, learner, admitted, learning, behavior_agent)
    assert plan.receipt is not None
    prepared = bank.prepare_commit(plan.receipt, targets.signals)
    update_damped_iteration(backend, learner, batch, learning, iteration, targets=targets)
    bank.publish(prepared)
```

Import `compute_batch_targets` from `manabot.training.updates.targets`. PPO
also accepts `targets=`. For action-value spread, supply `action_values=` to
the target computation in `batch.action_data.slots` order. Ordinary self-play
collects no per-action values. Use a custom scorer there to override
`action_value_spread`. Built-in `wdl_entropy` requires a categorical WDL
critic. Self-play checks these settings at startup and scores the batch before
the optimizer or EMA advances. Exiting the plan releases uncommitted captures.
Native `GameSnapshot` handles are detached from their vector, so bank and
vector close order does not matter; custom `SnapshotVector` implementations
with vector-owned handles still require closing the bank before the vector.

### Requirements, accounting and recovery

Fresh games supply the outcome and game-setting tables, recorded games
(`record_games`), statistics and the rates `decisions_per_game`,
`skipped_share` and `games_per_hour`. A replay-only update writes empty
tables and omits those rates. Total throughput includes both sources. Under
`snapshot/`, `starts` and `choices` include retries, `rows` and `captures`
count retained work, and `terminals` and `horizon_cuts` count each branch
ending once. `groups_completed`, `groups_discarded` and
`failed_branches` report group and retry outcomes.

Each successful update commits its scores, allocation counters and RNG together.
A stale or reused receipt is rejected. Live captures store detached native
`GameSnapshot` handles in `SnapshotBank` and fan them out across all producer
workers without replaying the prefix; with `decision_memory`, siblings restored
at the same boundary share their immutable pre-fork `MemoryStore` chunks while
diverging independently afterwards.

Etude saves the generation and cold recipes in `Counts.snapshot_bank` alongside
the learner under `out/state/save-*/state.json`. New captures store
`managym-snapshot-v1` (`replay_json`) as cold authority (preserving
interventions when `fresh_only=False`), while earlier exact recipes retain
their schema and `root_id`s. Resume restores them with an empty hot cache and
new fresh-game and policy sampling seeds; supply runtime predicates and scorers
again. On a cold miss, `SnapshotBank` reconstructs each distinct root once,
verifies its digest and chooser, and promotes the `GameSnapshot` into the hot
cache for all siblings and later draws.

Native vectors leave finished and failed slots parked until the collector
refills active streams via `load_slots_into_buffers`, avoiding unused
replacement games at batch boundaries. Adaptive collection assigns the next
batch's roots when the learner commits its target errors. The lethal-target
practice vector accepts fresh loads of its configured match. Each fresh slot
keeps its own seed sequence and recording. Replacing an untouched fresh root
with a branch rewinds its seed without writing an interrupted game record or
partial statistics. Saving a collector opens any fresh roots it needs for
replay. A later snapshot allocation can reuse or replace those openings.
When a branch fails, its `GameAttempt` and `RecordedVectorEnv` failure record
materialize branch `interventions` lazily so `manabot.training.state.game_errors`
and `managym.replay` reproduce the failure cold after hot handles are closed.
[Recovery](training.md#recovery)

## Write your own loop

A new method is the pieces in a different arrangement. Write it as a loop of
your own to try it quickly; the loop below already saves, resumes and
publishes checkpoints through Etude. When it should run in our recipes, add a
learning record for it in `manabot.training.learning` and an
update in `manabot/training/updates/` beside the two above; our recipes then
run it as they run those. Write that update as they are written: take the
backend first, do array math only through `backend.rl`, `backend.learner`
and `backend.arrays`, and hand the minibatch loss to `learner.loss` as a
closure, so it runs unchanged on every backend whose `rl` has the functions
it calls.

### A complete self-play loop

`manabot/training/loops/minimal.py` trains one model against itself with PPO
on complete games, on CPU. It writes its update inline, inside the loop, so
the whole method reads on one page, and it survives losing its machine: seven
of its lines are Etude's (`track`, `run.seed`, `run.resume`, `run.steps`,
`run.log`, `run.save`, `run.checkpoint`). It is a PyTorch loop: it builds the
`ActorCritic` and its optimizer directly and takes its terms from the torch
backend's `rl` and `learner`:

```python
"""A resumable self-play PPO loop written from the pieces alone."""

from copy import deepcopy

from etude.train.tracking import track
import numpy as np
from pydantic import BaseModel, Field
import torch

from manabot.backends.torch import learner, rl
from manabot.backends.torch.collection import CollectionSession
from manabot.backends.torch.policy import ActorCritic
from manabot.backends.torch.rollout import AdmittedRollout
from manabot.env.match import Match, Reward
from manabot.env.observation import ObservationSpace
from manabot.infra.config import ModelConfig, RewardConfig
from manabot.training.corpus import log_corpus_outcomes, log_corpus_setup
from manabot.training.rollout.collection import (
    close_collector,
    collect,
    open_collector,
)


class Progress(BaseModel):
    updates: int = 0


class MinimalConfig(BaseModel):
    updates: int = Field(default=3, ge=1)
    streams: int = Field(default=4, ge=2)
    decisions: int = Field(default=64, ge=1)
    model: ModelConfig = ModelConfig(hidden_dim=64)


def train(config: MinimalConfig) -> None:
    """Train `config.updates` collect/update rounds and publish the EMA policy."""
    with track(clocks=["updates"], config=config) as run:
        torch.manual_seed(run.seed)
        agent = ActorCritic(ObservationSpace(), config.model)
        averaged = deepcopy(agent)
        optimizer = torch.optim.Adam(agent.parameters(), lr=2.5e-4, eps=1e-5)
        # Loads the last save into all three; a fresh run starts at update 0.
        progress = run.resume(
            agent=agent, averaged=averaged, optimizer=optimizer, extra=Progress()
        )
        start = progress.updates
        # Only the registered objects carry on: an attempt plays fresh games.
        torch.manual_seed(run.seed + start)
        rng = np.random.default_rng(run.seed + start)
        collector = open_collector(
            agent.observation_space,
            Match(),
            Reward(RewardConfig()),
            num_envs=config.streams,
            seed=run.seed + start,
            opponent_mode="self",  # one model plays and learns from both seats
            # Torch numerics on the CPU; this session samples both seats' actions.
            numerics=CollectionSession(device="cpu", seed=run.seed + start),
        )
        try:
            if start == 0:
                log_corpus_setup(
                    run,
                    collector,
                    behavior="current-self",
                    curriculum="fixed",
                    updates=0,
                )
            for update in run.steps("updates", to=config.updates):
                # Whole games, at least `decisions` learner rows per stream.
                batch = collect(collector, agent, config.decisions)
                ends = torch.as_tensor(batch.dones)
                values = torch.as_tensor(batch.values)
                # Complete games end every trajectory, so the tail value is never read.
                tail = torch.zeros(ends.shape[1])
                advantages, returns = rl.transition_gae(
                    torch.as_tensor(batch.rewards),
                    values,
                    ends,
                    tail,
                    gamma=1.0,
                    lam=0.95,
                )
                # Time-major [T,E] rows, the order `batch.observations` indexes.
                advantages, returns = advantages.flatten(), returns.flatten()
                advantages = rl.standardized_advantages(advantages)
                actions = torch.as_tensor(batch.actions).flatten()
                behavior = torch.as_tensor(batch.logprobs).flatten()

                # Admission checks each row's policy input once, before any epoch.
                admitted = AdmittedRollout(agent, batch)
                admitted.admit_rows(np.arange(len(actions)), 256)
                agent.train()
                for _epoch in range(2):
                    order = torch.as_tensor(rng.permutation(len(actions)))
                    for rows in order.split(256):
                        obs, _ = admitted.rows(rows.numpy(), torch.device("cpu"))
                        valid = admitted.legal(obs)
                        logits, value = agent(obs)
                        logs = rl.policy_logs(logits, valid)
                        chosen = rl.action_logs(logs, actions[rows])
                        loss = (
                            rl.clipped_surrogate(
                                chosen - behavior[rows], advantages[rows], 0.2
                            ).mean()
                            + 0.5 * rl.squared_value_loss(value, returns[rows]).mean()
                            - 0.01 * rl.entropy(logs, valid).mean()
                        )
                        learner.gradient_step(loss, agent, optimizer, max_grad_norm=0.5)
                learner.update_ema(averaged, agent, rate=0.99)
                run.log({"loss": float(loss.detach())})  # the last minibatch's
                log_corpus_outcomes(run, collector, updates=update)
                progress.updates = update
                run.save()
        finally:
            close_collector(collector)
        # Reached only at the target: a stopped loop raises `Stopped` instead.
        run.checkpoint(policy=averaged)
```

```bash
uv run etude train manabot/training/loops/minimal.py
uv run etude train manabot/training/loops/minimal.py --experiment minimal-long --config '{"updates": 200}' --seconds 60
uv run etude train manabot/training/loops/minimal.py --experiment minimal-long --resume
```

The first command trains three updates in a few seconds on a laptop CPU and
prints the run's id; they teach the policy almost nothing. The second trains until its target or deadline; at the deadline it saves
and exits 75; the third carries that run on to update
200. The first run's folder holds everything in ordinary formats:

```text
~/.etude/track/minimal/out/
  state/current.json                     {"save": "save-00000001"}: the save to load
  state/save-00000001/state.json         {"at": {"updates": 3}, "seed": 0, "files": {"agent": "agent.pt", ...}}
  state/save-00000001/agent.pt  averaged.pt  optimizer.pt
  checkpoints/updates-00000003/checkpoint.json
  checkpoints/updates-00000003/policy.pt   torch.save(averaged.state_dict())
```

To watch the games, give the collector a `games_path` and open one of its
games in Academy ([watch a game from your training loop](academy.md#watch-a-game-from-your-training-loop)).

`run.checkpoint(policy=averaged)` publishes plain weights. To publish a policy
bound to the decks it played, as a recipe's are bound to their setups, write
it with `save_checkpoint(policy, policy.observation_space, "policy.safetensors",
player_configs=Match().to_rust())` from `manabot.backends.torch.checkpoint`
and record the file with `run.checkpoint(path)`;
`load_checkpoint(path)` returns the policy and its observation space. The file
is the [shared container](../manabot/backends/README.md#policy-checkpoints)
every backend loads; `load_checkpoint` also reads `.pt` policies saved by
earlier revisions.

The entropy bonus is there to show that a term is one line. manabot's own
updates use none; they regularize with KL penalties instead
([the loss](learning.md#4-update)).
