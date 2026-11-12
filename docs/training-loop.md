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
  and collection KL penalties. Use it as the baseline a new method must beat.

Both take a `Learner` (`manabot.training.updates.learner`), which holds the
model, its Adam optimizer, the precision and a profiler, and a learning
record (`manabot.training.regime`). Call them directly to reproduce a
recipe's update exactly:

```python
learner = Learner(agent, "cpu", minibatch_rows=64)
update_damped_iteration(learner, batch, move_learning(), iteration=1)
update_iteration(learner, batch, Learning(), progress=0.0, rng=np.random.default_rng(0))
```

`move_learning()` (`manabot.recipes.builders`) is the current recipe's
`DampedPolicyLearning(gradient="damped_policy")`. Each returns the update's
diagnostics as a dictionary; EMA and checkpoints stay with the caller.

## The player's pieces

These read the player's observations, legal actions and `ActorCritic` heads.
Shapes use `T` for time, `E` for trajectories (columns), `B` for a flat batch
of rows and `A` for padded legal offers.

### Play

| Piece | What it does |
| --- | --- |
| `SeatRoutedCollector` (`manabot.training.rollout.collection`) | Owns native game streams. `collect(agent, decisions)` plays with `agent` and returns a `RolloutBatch`; `collect_pair(agent, decisions)` returns one batch per model when `opponent_mode="learning"` and `opponent_agent` is a second learner. `close()` stops the streams. |
| `RolloutBatch` | One update's rows, `[T,E]`. Complete games are packed into one column separated by end flags. `rewards`, `dones` (an end belongs to that row), `values` (critic at collection), `actions`, `logprobs` (behavior log likelihood of the chosen action), `next_obs` for bootstrapping an unfinished tail, and `outcome_probabilities` `[T,E,3]` for a categorical critic. `observations(indexes)` materializes flat rows as an observation dict and the full behavior distribution. |
| `AdmittedRollout(agent, batch)` (`manabot.training.updates.learner`) | Checks a batch's policy input against `agent` once, so the model's forward does no validation. `admit_rows(indexes, chunk_rows)` admits the rows an update will train; `rows(indexes, device)` returns their observations on `device` and an `ActionRows` whose `log_probabilities` `[B,A]` (NumPy) are the behavior log likelihoods a KL or ratio term reads, and raises `ValueError` for a row not admitted. `next_observation(device)` is the bootstrap row of each column. The batch must not change afterwards. |

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

`ActorCritic.forward(obs)` returns logits and a signed expected value `[B]`;
`forward_distribution(obs)` returns the raw critic output instead: `[B,3]`
logits for `value_kind="categorical_wdl"`, `[B,1]` for scalar.

| Piece | Module | What it does |
| --- | --- | --- |
| `legal_mask(obs["actions_valid"])` | `manabot.model.policy_distribution` | The legal offers `[B,A]`, refusing a corrupt mask. |
| `policy_logs(logits, valid)` | `manabot.model.policy_distribution` | Log likelihoods `[B,A]`, `-inf` off the legal mask; every policy term reads these. |
| `reference_distribution(obs, kind)` | `manabot.training.updates.references` | The magnet a reverse KL pulls toward: `"uniform"` over legal offers or `"action_type_uniform"`. |
| `damped_policy_loss(logits, actions, advantages, behavior_logs, reference, valid, *, clip, collection_kl, tau)` | `manabot.training.updates.damped_policy` | The damped policy objective per row: the clipped surrogate plus both reverse KL penalties. |

Reduced precision is a context around the forward only:
`with forward_precision(precision, device):` (`manabot.training.precision`)
runs `bf16` autocast, and you cast the outputs with `.float()` before any loss.
`bf16` and `tf32` are CUDA-only; losses and their float64 terms stay in full
precision on every device.

### Health and metrics

None of these change the update; call them when you want the numbers.

| Piece | Module | Records |
| --- | --- | --- |
| `batch_health(action_data, advantages, selected, *, actor_only)` | `manabot.training.updates.health` | A `NumericalHealth` record for the batch: legal-logit range, underflow, forced rows, zero advantages, rows each head trains on. Pass `batch.action_data`. |
| `OptimizerHealth(health, iteration, profiler=None)` | `manabot.training.updates.health` | The player's step recorder. Pass as `gradient_step(..., recorder=observer)` to count steps and rejections into the `batch_health` record and, on iteration 1 and every 25th, sample gradient norms (including the critic readout's) and the parameter change. With a profiler, backward and the optimizer step are timed. |
| `clip_fraction(log_ratio, clip)` | `manabot.training.updates.health` | The fraction of rows whose likelihood ratio left `1 ± clip`, the record's `clip_fraction`. |
| `selection_diagnostics(advantages, residuals, selected, ends, action_types)` | `manabot.training.updates.selection` | Advantage and residual quantiles by action type and distance to the game's end ([reading a run](learning.md#reading-a-run)). |

## Standard methods, kept local

The rest is standard reinforcement learning, kept as small local functions in
`manabot.rl` rather than taken from an RL library. They read only tensors, an
`nn.Module`, an optimizer and small typed records, and import nothing else
from manabot (`tests/test_package_boundaries.py` enforces it), so another
role reuses them unchanged. Each returns one value per row `[B]` where it is a
loss, so you choose the reduction, mask and weight.

| Piece | Module | What it is |
| --- | --- | --- |
| `transition_gae(rewards, values, ends, next_value, gamma, lam)` | `manabot.rl.returns` | GAE (Schulman et al. 2016): advantages and lambda returns `[T,E]`. Each column is one agent's trajectory; an end cuts bootstrap and trace. `next_value` `[E]` is read only where a column's last row is not an end. Our updates call it once per lambda (policy and value). |
| `standardized_advantages(advantages, over=None)` | `manabot.rl.returns` | Every row shifted and scaled by the mean and population std of the rows `over` selects (all rows when omitted); the std is clamped at 1e-8. Our PPO update takes the statistics over the rows it trains on. |
| `categorical_lambda_returns(probabilities, rewards, ends, bootstrap, trace_lambda)` | `manabot.rl.returns` | Undiscounted loss/draw/win lambda-return targets `[T,E,3]`; a terminal row is the one-hot outcome (`rewards` -1, 0 or 1, nonzero only at an end). `bootstrap` `[E,3]` is the outcome distribution past each column's last row (read only where it is not an end). |
| `clipped_surrogate(log_ratio, advantages, clip)` | `manabot.rl.losses` | Negative PPO clipped surrogate (Schulman et al. 2017), with float64 ratios. |
| `reverse_kl(logs, target_logs, valid)` | `manabot.rl.losses` | KL(current ‖ frozen target) over valid entries. |
| `squared_value_loss(values, targets)`, `categorical_value_loss(value_logits, targets)` | `manabot.rl.losses` | Scalar squared error; cross entropy of `[B,3]` loss/draw/win logits against target mixtures. |
| `entropy(logs, valid)` | `manabot.rl.losses` | Entropy over valid entries. |
| `selection_mask(advantages, kind, fraction, minimum)`, `selected_mean(values, selected)` | `manabot.rl.selection` | Keep a top count or inclusive quantile of advantage magnitude; average a per-row term over the mask (zero gradient when empty). |
| `gradient_step(loss, model, optimizer, max_grad_norm, *, recorder=None)` | `manabot.rl.step` | Backward, `clip_grad_norm_`, step; returns the pre-clip norm. |
| `update_ema(averaged, model, rate)` | `manabot.rl.step` | Moves `averaged`'s parameters toward `model`'s by `1 - rate` and copies buffers. Start from `deepcopy(model)`. |

`gradient_step` raises `NumericalError` (`manabot.rl.numerics`) instead of
stepping on a detached or nonfinite loss, a loss with no gradients, or
nonfinite gradients, and after a step that leaves a nonfinite parameter. Its
`invariant` names the check and its `tensors` hold the offending values. Any
object with the `StepRecorder` methods observes the step; `OptimizerHealth`
is the player's. Set the learning rate on `optimizer.param_groups` before
stepping to schedule it.

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
run it as they run those.

### A complete self-play loop

`manabot/training/loops/minimal.py` trains one model against itself with PPO
on complete games, on CPU. It writes its update inline, inside the loop, so
the whole method reads on one page:

```python
"""A self-play PPO loop written from the pieces alone, with no recipe or executor."""

from copy import deepcopy

import numpy as np
import torch

from manabot.env.match import Match, Reward
from manabot.env.observation import ObservationSpace
from manabot.infra.config import ModelConfig, RewardConfig
from manabot.model.actor_critic import ActorCritic
from manabot.model.policy_distribution import legal_mask, policy_logs
from manabot.rl.losses import clipped_surrogate, entropy, squared_value_loss
from manabot.rl.returns import standardized_advantages, transition_gae
from manabot.rl.step import gradient_step, update_ema
from manabot.training.rollout.collection import SeatRoutedCollector
from manabot.training.updates.learner import AdmittedRollout


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
            advantages, returns = transition_gae(
                torch.as_tensor(batch.rewards), values, ends, tail, gamma=1.0, lam=0.95
            )
            # Time-major [T,E] rows, the order `batch.observations` indexes.
            advantages, returns = advantages.flatten(), returns.flatten()
            advantages = standardized_advantages(advantages)
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
                    valid = legal_mask(obs["actions_valid"])
                    logits, value = agent(obs)
                    logs = policy_logs(logits, valid)
                    chosen = logs.gather(-1, actions[rows, None]).squeeze(-1)
                    loss = (
                        clipped_surrogate(
                            chosen - behavior[rows], advantages[rows], 0.2
                        ).mean()
                        + 0.5 * squared_value_loss(value, returns[rows]).mean()
                        - 0.01 * entropy(logs, valid).mean()
                    )
                    gradient_step(loss, agent, optimizer, max_grad_norm=0.5)
            update_ema(averaged, agent, rate=0.99)
    finally:
        collector.close()
    return averaged
```

```bash
uv run python -c "from manabot.training.loops.minimal import train; train(3)"
```

Three iterations take a few seconds on a laptop CPU and return a policy;
they teach it almost nothing. Keep a run with `save_checkpoint(policy,
policy.observation_space, "policy.pt", player_configs=Match().to_rust())` from
`manabot.model.checkpoint`, which binds it to the decks it played, as a
recipe's checkpoints are bound to their setups; `load_checkpoint("policy.pt")`
returns the policy and its observation space. `tests/training/test_minimal_loop.py`
runs the loop on two streams and a tiny model through the `model` argument,
then saves and reloads its checkpoint.

The entropy bonus is there to show that a term is one line. manabot's own
updates use none; they regularize with KL penalties instead
([the loss](learning.md#4-update)).
