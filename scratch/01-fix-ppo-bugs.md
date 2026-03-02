# Fix PPO Correctness Bugs

## Problem

The PPO training loop has never produced clean convergence. Seven confirmed
bugs corrupt the learning signal, ranging from a completely inverted reward
for half the agents to silent numerical instability. Until these are fixed,
no amount of architecture work will produce a learning agent.

This is the gate for the first-light wave. Nothing else matters until the
training loop matches CleanRL's core PPO correctness details within the
current multi-agent architecture.

## Approach

Fix all seven bugs in one atomic change. They're all small, they all touch
the same ~4 files, and several interact (inverted reward + missing advantage
normalization compound each other). Testing them in isolation would require
artificial scaffolding. Ship them together, verify together.

Reference implementation: CleanRL's `ppo.py`.

### 1. Fix inverted reward for player 0

`manabot/env/match.py:94`

```python
# Current (broken):
if new_obs.won == last_obs.agent.player_index:

# Fix:
if new_obs.won:
```

`new_obs.won` is already "the observing player won" — comparing it to
`player_index` exploits Python's `True == 1 / False == 0` identity and
inverts the reward for player 0. Half the training signal has been
backwards.

### 2. Handle truncation in rollout step

`manabot/model/train.py:393`

```python
# Current:
new_obs, reward, done, _, info = self.env.step(action)

# Fix:
new_obs, reward, terminated, truncated, info = self.env.step(action)
done = terminated | truncated
```

The env returns `(obs, reward, terminated, truncated, info)` per
Gymnasium spec. Currently `truncated` is silently discarded.

Full truncation bootstrapping (computing `V(s_final)` and adjusting
the return) requires the final pre-reset observation, which the
VectorEnv doesn't currently provide via `info["final_observation"]`.
For first-light this is acceptable — truncation is rare with simple
decks and short games. The fix ensures truncated episodes at least
end the rollout rather than leaking stale state.

Log truncation events so we know if this assumption breaks:

```python
if truncated.any():
    n = truncated.sum().item()
    self.logger.warning(f"Truncation in {n}/{self.hypers.num_envs} envs (no value bootstrap)")
```

### 3. Apply advantage normalization

`manabot/model/train.py` — in the minibatch loop, after slicing
`mb_advantages` (line 335) and before calling `_optimize_step`:

```python
if self.hypers.norm_adv:
    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
```

Per-minibatch normalization, matching CleanRL. The hyperparameter
`norm_adv` already exists (default `True`) but is never checked.

### 4. Use actual batch size for indexing

`manabot/model/train.py:324,329`

```python
# Current:
inds = np.arange(batch_size)
for start in range(0, batch_size, minibatch_size):

# Fix:
obs, logprobs, actions, advantages, returns, values = self.multi_buffer.get_flattened()
actual_batch_size = logprobs.shape[0]
if actual_batch_size < hypers.num_minibatches:
    self.logger.warning(
        f"Skipping update: actual_batch_size={actual_batch_size} < num_minibatches={hypers.num_minibatches}"
    )
    continue
minibatch_size = max(1, actual_batch_size // hypers.num_minibatches)
inds = np.arange(actual_batch_size)
for start in range(0, actual_batch_size, minibatch_size):
```

The multi-agent buffer flattens transitions from all (env, player)
pairs. The total should equal `num_envs * num_steps` under normal
conditions, but can differ if rollout steps are skipped on error.
Using actual data length is strictly better.

Keep the config-derived `batch_size` for SPS calculation and update
counting.

Guard against pathological small batches when rollout errors skip enough
steps. Without this guard, `minibatch_size` can become 0 and crash the
optimizer loop.

### 5. Fix policy head initialization

`manabot/model/agent.py:72`

```python
# Current:
layer_init(nn.Linear(embed_dim, 1))  # defaults to gain=1

# Fix:
layer_init(nn.Linear(embed_dim, 1), gain=0.01)
```

With `gain=1`, initial logits can be large enough to make the policy
deterministic at step 0, killing exploration. `gain=0.01` produces
near-uniform initial action probabilities.

### 6. Use Categorical(logits=) directly

`manabot/model/agent.py:200-202`

```python
# Current:
probs = torch.softmax(logits, dim=-1)
self.logger.debug(f"Action probabilities: mean={probs.mean().item():.4f}, std={probs.std().item():.4f}")
dist = torch.distributions.Categorical(probs=probs)

# Fix:
dist = torch.distributions.Categorical(logits=logits)
```

`Categorical(logits=)` uses log-softmax internally, which is more
numerically stable than computing softmax then passing probs. The
debug log line goes away — it was noise.

### 7. Remove weight decay

`manabot/model/train.py:232`

```python
# Current:
weight_decay=0.01

# Fix:
remove the kwarg entirely (defaults to 0)
```

L2 regularization on all parameters is not standard for PPO and
fights the policy gradient signal. CleanRL uses no weight decay.

### 8. Add action truncation warning

`manabot/env/observation.py:288`

```python
if len(obs.action_space.actions) > self.max_actions:
    log.warning(
        f"Action space truncated: {len(obs.action_space.actions)} -> {self.max_actions}"
    )
```

Not a correctness fix. Visibility for when the action space exceeds
`max_actions` (10). Shouldn't trigger with simple decks.

### 9. Add rollout-health counters (one more big push)

Instrumentation-only additions to make failures visible during stage 01:

- `rollout/skipped_steps` (count per update + cumulative)
- `rollout/truncated_episodes` (count per update + cumulative)
- `rollout/action_space_truncations` (count per update + cumulative)

These should be logged to both logger and wandb (when enabled). Stage 01
is still architecture-stable; this is observability to de-risk stage 02.

## Alternatives considered

| Approach | Tradeoff | Why not |
|----------|----------|---------|
| Fix one bug per PR | Easy to bisect | Bugs interact (reward + adv norm). 7 PRs for ~200 LOC of fixes creates more overhead than value. |
| Rewrite training loop from CleanRL | Guaranteed correct | Loses multi-agent buffer, observation handling, reward abstraction. Too much scope for this wave. |
| Full truncation bootstrapping now | Proper returns on truncation | Requires VectorEnv to provide `final_observation` in info dict. Truncation is rare with simple decks. Fix the env later. |
| Make weight_decay configurable | Flexibility | No. It shouldn't exist for PPO. Don't add knobs for wrong defaults. |

## Key decisions

**All seven fixes ship together.** They're all small corrections to the
same training loop. The reward inversion alone invalidates all prior
training runs, so there's no backwards compatibility concern.

**Per-minibatch advantage normalization**, not per-batch. Reduces variance
within each gradient step. This is the CleanRL pattern.

**Truncation: log-and-move-on for now.** Proper bootstrapping needs
`final_observation` from the env, which is a separate change. Logging
tells us if this matters in practice.

**Remove the probs debug log.** It fires on every forward pass and the
information is redundant with entropy logging.

**Keep the action truncation warning in stage 01.** It's a tiny
instrumentation-only change that immediately surfaces silent action loss
risk during correctness validation.

**Package selection: one more big push.** Stage 01 includes observability
for skipped steps / episode truncation / action truncation so we can
distinguish algorithm bugs from data-quality issues during bring-up.

## Scope

- In scope: all 7 correctness fixes, the action truncation warning, tests
  for each fix, and rollout-health counters for skipped steps/truncations
- Out of scope: truncation value bootstrapping (needs env changes),
  observation space changes, architecture changes, anything that requires
  a training run to validate

## Done when

Advancing first-light goals:
- *"Fix every confirmed PPO bug so the training loop matches CleanRL's core PPO implementation details (with truncation bootstrap explicitly deferred)"*

```bash
pytest tests/ -v
# All existing tests pass, plus new tests:
# - test_reward_correct_both_players: reward is win_reward when won=True for both player indices
# - test_truncated_episode_marks_done: truncated envs end rollout and emit truncation warning
# - test_advantage_normalization: mb_advantages are normalized when norm_adv=True, untouched when False
# - test_actual_batch_size_used: indexing uses flattened buffer length, not hardcoded batch_size
# - test_small_actual_batch_skips_update: no divide-by-zero or range-step crash when actual_batch_size < num_minibatches
# - test_policy_head_initial_logits: initial logit std is small (gain=0.01 effect)
# - test_categorical_logits: distribution constructed with logits= not probs=
# - test_no_weight_decay: optimizer has weight_decay=0
# - test_rollout_health_counters: skipped-step/truncation/action-truncation counters increment and log
```
