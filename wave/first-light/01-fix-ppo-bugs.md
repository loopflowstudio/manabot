# 01: Fix PPO Correctness Bugs

Fix every confirmed bug in the training loop so the PPO implementation
matches CleanRL's known-correct reference.

## Finish line

All bugs fixed. Existing tests pass. New tests confirm each fix.

## Changes

### 1. Fix inverted reward for player 0 (`match.py:Reward.compute`)

**CRITICAL.** `new_obs.won` is `bool` (True = observing player won).
`last_obs.agent.player_index` is `int` (0 or 1). The comparison
`won == player_index` uses Python's `True == 1` / `False == 0`:

- Player 0 wins: `True == 0` → False → **gets lose_reward (inverted)**
- Player 1 wins: `True == 1` → True → gets win_reward (correct)

Half the training signal is backwards. Fix:

```python
# Replace:
if new_obs.won == last_obs.agent.player_index:
# With:
if new_obs.won:
```

`won` already represents "the observing player won" — no comparison needed.

### 2. Handle truncation in rollout step (`train.py:_rollout_step`)

Currently truncation is discarded:
```python
new_obs, reward, done, _, info = self.env.step(action)
```

When a game hits max steps (truncated but not terminated), the value
should be bootstrapped, not treated as terminal. Fix:

```python
new_obs, reward, terminated, truncated, info = self.env.step(action)
done = terminated  # for buffer storage (episode ended)
# Bootstrap value for truncated episodes in GAE computation
```

Alternatively, combine as `done = terminated | truncated` for buffer
storage but pass `truncated` separately to GAE so it bootstraps correctly.

### 3. Apply advantage normalization (`train.py:_optimize_step`)

The `norm_adv` hyperparameter exists but is never checked. Add:

```python
if self.hypers.norm_adv:
    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
```

After line 335, before the advantages are used in the policy loss.

### 2. Compute batch size from actual data (`train.py:train`)

Replace the hardcoded `batch_size = hypers.num_envs * hypers.num_steps`
for indexing with the actual flattened buffer length:

```python
obs, logprobs, actions, advantages, returns, values = self.multi_buffer.get_flattened()
actual_batch_size = logprobs.shape[0]
minibatch_size = actual_batch_size // hypers.num_minibatches
inds = np.arange(actual_batch_size)
```

Keep the config-derived `batch_size` for SPS calculation and update
counting, but use `actual_batch_size` for indexing.

### 3. Fix policy head initialization (`agent.py`)

Change the final layer of the policy head from `gain=1` to `gain=0.01`:

```python
self.policy_head = nn.Sequential(
    layer_init(nn.Linear(embed_dim, embed_dim)),
    nn.ReLU(),
    layer_init(nn.Linear(embed_dim, 1), gain=0.01)  # was gain=1
)
```

This produces near-uniform initial action probabilities, enabling
exploration.

### 4. Use Categorical(logits=) (`agent.py:get_action_and_value`)

Replace:
```python
probs = torch.softmax(logits, dim=-1)
dist = torch.distributions.Categorical(probs=probs)
```

With:
```python
dist = torch.distributions.Categorical(logits=logits)
```

Numerically more stable — avoids double softmax.

### 5. Remove weight_decay from optimizer (`train.py`)

```python
# Current:
self.optimizer = torch.optim.Adam(..., weight_decay=0.01)

# Fix:
self.optimizer = torch.optim.Adam(..., weight_decay=0)  # or just remove the kwarg
```

CleanRL uses no weight decay for PPO. L2 regularization on all parameters
fights the learning signal and is not standard for policy gradient methods.

### 8. Add action truncation warning (`observation.py`)

When legal actions exceed `max_actions` (default 10), they're silently
dropped. Add a warning/metric:

```python
if len(obs.action_space.actions) > self.max_actions:
    log.warning(f"Action space truncated: {len(obs.action_space.actions)} → {self.max_actions}")
```

Not a fix — just visibility. With simple decks this shouldn't trigger,
but we need to know if it does.

## Done when

```bash
pytest tests/ -v
# All existing tests pass, plus:
# - test that reward is correct for both player indices
# - test that advantage normalization is applied when norm_adv=True
# - test that actual_batch_size matches flattened buffer length
```
