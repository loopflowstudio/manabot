# 01: Fix PPO Correctness Bugs

Fix every confirmed bug in the training loop so the PPO implementation
matches CleanRL's known-correct reference.

## Finish line

All four bugs fixed. Existing tests pass. New test confirms advantage
normalization is applied when `norm_adv=True`.

## Changes

### 1. Apply advantage normalization (`train.py:_optimize_step`)

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

## Done when

```bash
pytest tests/ -v
# All existing tests pass, plus:
# - test that advantage normalization is applied when norm_adv=True
# - test that actual_batch_size matches flattened buffer length
```
