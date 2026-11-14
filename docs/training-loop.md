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
- Write your own loop: [a complete self-play loop](#a-complete-self-play-loop).

## The updates we maintain

A recipe's loop ([training](training.md#regimes-and-stages)) runs one update
each iteration and adds checkpoints, tracking, recovery and evaluation around
it. Its learning record selects the update; both live in
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

Both are written once and take the backend as their first argument: the
module `manabot.backends.torch`, not an object to configure. Then come a
`Learner` (`manabot.backends.torch.learner`), which holds the model, its Adam
optimizer, the precision and a profiler, and a learning record
(`manabot.training.regime`). Call them directly to reproduce a recipe's
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

Recipes and the executor pass `manabot.backends.torch`. Inside an update,
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
| `SeatRoutedCollector` (`manabot.training.rollout.collection`) | Owns native game streams. `collect(agent, decisions)` plays with `agent` and returns a `RolloutBatch`; `collect_pair(agent, decisions)` returns one batch per model when `opponent_mode="learning"` and `opponent_agent` is a second learner. `close()` stops the streams. |
| `RolloutBatch` | One update's rows, `[T,E]`. Complete games are packed into one column separated by end flags. `rewards`, `dones` (an end belongs to that row), `values` (critic at collection), `actions`, `logprobs` (behavior log likelihood of the chosen action), `next_obs` for bootstrapping an unfinished tail, and `outcome_probabilities` `[T,E,3]` for a categorical critic. `observations(indexes)` materializes flat rows as NumPy observation buffers and the full behavior distribution. |
| `AdmittedRollout(agent, batch)` (`manabot.backends.torch.rollout`) | Checks a batch's policy input against `agent` once, so the model's forward does no validation. `admit_rows(indexes, chunk_rows)` admits the rows an update will train; `rows(indexes, device)` prepares learner rows on the host, independently of autograd mode, and returns the `Prepared` batch on `device` with an `ActionRows` whose `log_probabilities` `[B,A]` (NumPy) are the behavior log likelihoods a KL or ratio term reads, and raises `ValueError` for a row not admitted. `legal(prepared)` is those rows' legal-action input `[B,A]`. `minibatches(order, size, device)` returns the same learner preparation with one host batch prefetched. `next_observation(device)` returns inference-prepared bootstrap rows, one per column, independently of autograd mode. The batch must not change afterwards. |

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
| `damped_policy_loss(logits, actions, advantages, behavior_logs, reference, valid, *, clip, collection_kl, tau)` | `backend.rl` | The damped policy objective per row: the clipped surrogate plus both reverse KL penalties. |

`backend.rl` is `manabot.backends.torch.rl` for the PyTorch model.
Reduced precision is a context around the forward only:
`with backend.learner.forward_precision(precision, device):` (or
`learner.forward_precision()`) runs `bf16` autocast, and you cast the outputs
with `backend.arrays.float32` (`.float()`) before any loss.
`bf16` and `tf32` are CUDA-only; losses and their float64 terms stay in full
precision on every device.

### Health and metrics

None of these change the update; call them when you want the numbers.

| Piece | Module | Records |
| --- | --- | --- |
| `batch_health(action_data, advantages, selected, *, actor_only)` | `manabot.training.updates.health` | A `NumericalHealth` record for the batch: legal-logit range, underflow, forced rows, zero advantages, rows each head trains on. Pass `batch.action_data`. |
| `OptimizerHealth(health, iteration, profiler=None, *, backend=...)` | `manabot.training.updates.health` | The player's step recorder. Pass as `learner.step(loss, max_grad_norm, recorder=observer)` (or to torch's `gradient_step`) to count steps and rejections into the `batch_health` record and, on iteration 1 and every 25th, sample gradient norms (including the critic readout's) and the parameter change. With a profiler, backward and the optimizer step are timed. |
| `clip_fraction(log_ratio, clip)` | `backend.rl` | The fraction of rows whose likelihood ratio left `1 ± clip`, the record's `clip_fraction`. |
| `selection_diagnostics(advantages, residuals, selected, ends, action_types)` | `manabot.training.updates.selection` | Advantage and residual quantiles by action type and distance to the game's end ([reading a run](learning.md#reading-a-run)). Every argument is `[T,E]` host rows (NumPy, or CPU tensors): `ends` is `batch.dones` and `action_types` is `batch.action_data.selected_types(batch.actions, ACTION_TYPE_DIM)` (`manabot.env.observation`). |

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
object with the `StepRecorder` methods observes the step; `OptimizerHealth`
is the player's. Set the learning rate with `learner.set_learning_rate(rate)`
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

## Write your own loop

A new method is the pieces in a different arrangement. Write it as a loop of
your own to try it quickly. When it needs checkpoints, tracking and Etude
launches, add a learning record for it in `manabot.training.regime` and an
update in `manabot/training/updates/` beside the two above; our recipes then
run it as they run those. Write that update as they are written: take the
backend first, do array math only through `backend.rl`, `backend.learner`
and `backend.arrays`, and hand the minibatch loss to `learner.loss` as a
closure, so it runs unchanged on every backend whose `rl` has the functions
it calls.

### A complete self-play loop

`manabot/training/loops/minimal.py` trains one model against itself with PPO
on complete games, on CPU. It writes its update inline, inside the loop, so
the whole method reads on one page. It is a PyTorch loop: it builds the
`ActorCritic` and its optimizer directly and takes its terms from the torch
backend's `rl` and `learner`:

```python
"""A self-play PPO loop written from the pieces alone, with no recipe or executor."""

from copy import deepcopy

import numpy as np
import torch

from manabot.backends.torch import learner, rl
from manabot.backends.torch.policy import ActorCritic
from manabot.backends.torch.rollout import AdmittedRollout
from manabot.env.match import Match, Reward
from manabot.env.observation import ObservationSpace
from manabot.infra.config import ModelConfig, RewardConfig
from manabot.training.rollout.collection import SeatRoutedCollector


def train(
    iterations: int,
    *,
    streams: int = 4,
    decisions: int = 64,
    seed: int = 0,
    model: ModelConfig | None = None,
) -> ActorCritic:
    """Return the EMA of a policy trained `iterations` collect/update rounds."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    agent = ActorCritic(ObservationSpace(), model or ModelConfig(hidden_dim=64))
    averaged = deepcopy(agent)
    optimizer = torch.optim.Adam(agent.parameters(), lr=2.5e-4, eps=1e-5)
    collector = SeatRoutedCollector(
        agent.observation_space,
        Match(),
        Reward(RewardConfig()),
        num_envs=streams,
        seed=seed,
        opponent_mode="self",  # one model plays and learns from both seats
    )
    try:
        for _ in range(iterations):
            # Whole games, at least `decisions` learner rows per stream.
            batch = collector.collect(agent, decisions)
            ends = torch.as_tensor(batch.dones)
            values = torch.as_tensor(batch.values)
            # Complete games end every trajectory, so the tail value is never read.
            tail = torch.zeros(ends.shape[1])
            advantages, returns = rl.transition_gae(
                torch.as_tensor(batch.rewards), values, ends, tail, gamma=1.0, lam=0.95
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
    finally:
        collector.close()
    return averaged
```

```bash
uv run python -c "from manabot.training.loops.minimal import train; train(3)"
```

Three iterations take a few seconds on a laptop CPU and return a policy;
they teach it almost nothing. Keep a run with `save_checkpoint(policy,
policy.observation_space, "policy.safetensors",
player_configs=Match().to_rust())`
from `manabot.backends.torch.checkpoint`, which binds it to the decks it
played, as a recipe's checkpoints are bound to their setups;
`load_checkpoint("policy.safetensors")` returns the policy and its
observation space. The file is the
[shared container](../manabot/backends/README.md#policy-checkpoints) every
backend loads; `load_checkpoint` also reads `.pt` policies saved by earlier
revisions. `tests/training/test_minimal_loop.py`
runs the loop on two streams and a tiny model through the `model` argument,
then saves and reloads its checkpoint.

The entropy bonus is there to show that a term is one line. manabot's own
updates use none; they regularize with KL penalties instead
([the loss](learning.md#4-update)).
