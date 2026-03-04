# 05: Auxiliary Prediction Heads

Add auxiliary prediction targets to provide dense training signal.
This is the highest-impact investment for training efficiency after
correctness fixes, based on KataGo's experience.

## Finish line

Agent has three auxiliary prediction heads. Training loss includes
auxiliary terms. Verification ladder step 3 (beat passive) converges
faster with auxiliary heads than without.

## Changes

### 1. Auxiliary head architecture

Three additional linear heads off the shared encoder output (same
place the value head reads from):

```python
self.life_diff_head = nn.Sequential(
    layer_init(nn.Linear(embed_dim, embed_dim)),
    nn.ReLU(),
    layer_init(nn.Linear(embed_dim, 1), gain=1.0)
)

self.game_length_head = nn.Sequential(
    layer_init(nn.Linear(embed_dim, embed_dim)),
    nn.ReLU(),
    layer_init(nn.Linear(embed_dim, 1), gain=1.0)
)

self.creature_diff_head = nn.Sequential(
    layer_init(nn.Linear(embed_dim, embed_dim)),
    nn.ReLU(),
    layer_init(nn.Linear(embed_dim, 1), gain=1.0)
)
```

Each head uses max-pooling over the object dimension (same as value
head) to produce a scalar prediction.

### 2. Auxiliary targets

Computed retroactively at the end of each episode:

- **Life differential:** `(hero_life - villain_life)` at game end,
  normalized by dividing by 20. Applied to every transition in the
  episode as a fixed target.
- **Game length:** Total steps in the episode, normalized by dividing
  by some reasonable maximum (200?). Applied to every transition.
- **Creature differential:** `(hero_creatures - villain_creatures)` at
  game end, normalized. Applied to every transition.

These are stored in the buffer alongside rewards and values.

### 3. Auxiliary losses

```python
aux_loss = (
    aux_coef * F.mse_loss(pred_life_diff, target_life_diff)
    + aux_coef * F.mse_loss(pred_game_length, target_game_length)
    + aux_coef * F.mse_loss(pred_creature_diff, target_creature_diff)
)

total_loss = policy_loss - ent_coef * entropy + vf_coef * value_loss + aux_loss
```

`aux_coef` is a new hyperparameter, default 0.5. Can be tuned.

The auxiliary losses provide gradient signal to the shared encoder
without distorting the policy gradient. They help the encoder learn
features that are useful for understanding the game state.

### 4. Logging

Track per-head prediction accuracy:
- `aux/life_diff_mse`
- `aux/game_length_mse`
- `aux/creature_diff_mse`
- `aux/life_diff_explained_var`

## Constraints

- Auxiliary targets must not distort the policy gradient — they only
  affect the encoder through their own loss terms
- The env (or wrapper) must expose the terminal state information
  needed to compute targets (life totals, creature counts, game length)
- Auxiliary heads should be removable via config (for A/B comparison)
- The buffer is now flat pre-allocated tensors inline in `train.py`
  (not a separate class). Auxiliary target tensors follow the same
  pattern: `torch.zeros((num_steps, num_envs))`.
- `final_observation` plumbing for truncation value bootstrap is
  deferred — keep this in mind when computing episode-level targets
  across truncation boundaries.

## Done when

```bash
# A/B comparison:
# Run step 3 (beat passive) with and without auxiliary heads
# Auxiliary heads version should converge faster (fewer steps to 90% win rate)
python -m manabot.verify.step3_beat_passive --aux-heads=on
python -m manabot.verify.step3_beat_passive --aux-heads=off
```
