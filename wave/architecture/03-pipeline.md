# 03: Observation Pipeline Optimization

## Finish line

Observations are flat tensors from env boundary through training.
2x+ throughput improvement measured.

## Changes

### 1. Flatten at env boundary

The env wrapper produces a single flat tensor per step instead of a
dict of 14 tensors. The policy network unflattens (slices) internally.

PufferLib pattern:
```python
# Env returns:
obs = np.zeros(total_obs_dim, dtype=np.float32)

# Policy slices:
def encode(self, flat_obs):
    player = flat_obs[:, :player_dim]
    cards = flat_obs[:, player_dim:player_dim+cards_dim].reshape(-1, max_cards, card_dim)
    ...
```

### 2. Pre-allocated rollout buffer

Replace Python lists + torch.cat with pre-allocated tensors:

```python
obs_buf = torch.zeros((num_steps, num_envs, obs_dim), device=device)
# Direct write: obs_buf[step] = torch.from_numpy(flat_obs)
```

No per-step allocation. No dict comprehension. No cat at the end.

### 3. Encode/decode/value policy interface

Formalize the PufferLib-style split:
- `encode(obs)` → shared representation (called once)
- `decode(hidden)` → action logits + sampling
- `value(hidden)` → value estimate

Single encoder pass per step instead of running the full network
for both actions and values.

### 4. Move encoding to Rust (if ready)

If the Rust engine migration is far enough, have the engine return
pre-encoded flat tensors directly. This eliminates all Python-side
per-card/per-permanent encoding loops.

## Done when

- SPS improves 2x+ over dict-based pipeline
- No regression in win rate or explained variance
- Profiler confirms encoding and buffer management are no longer
  in the top-5 time consumers
