# 05: Auxiliary Prediction Heads

**Finish line:** Agent has auxiliary prediction heads providing dense training
signal. A first-light run with aux heads converges faster than one without.

## Context

The verification harness (`run_first_light`, `report_first_light`) is now the
primary experiment interface. Reward shaping (land play, creature play, opponent
life loss) is implemented and required — terminal-only PPO produces pass-collapse
where the agent learns to stop playing lands, breaking the entire gameplay chain:

```
play land -> tap for mana -> cast creature -> declare attacker -> deal damage -> win
```

Aux heads are the next lever: provide gradient signal to the shared encoder
without distorting the policy gradient, helping it learn features that matter
for understanding game state. This is the highest-impact investment for training
efficiency after reward shaping, based on KataGo's experience.

## Changes

### 1. Auxiliary head architecture

Three additional linear heads off the shared encoder output (same place the
value head reads from). The value head uses `MeanPoolingLayer(dim=1)` to
reduce the object dimension — aux heads should follow the same pattern:

```python
self.life_diff_head = nn.Sequential(
    layer_init(nn.Linear(embed_dim, embed_dim)),
    nn.ReLU(),
    MeanPoolingLayer(dim=1),
    layer_init(nn.Linear(embed_dim, 1), gain=1.0),
)

self.game_length_head = nn.Sequential(
    layer_init(nn.Linear(embed_dim, embed_dim)),
    nn.ReLU(),
    MeanPoolingLayer(dim=1),
    layer_init(nn.Linear(embed_dim, 1), gain=1.0),
)

self.creature_diff_head = nn.Sequential(
    layer_init(nn.Linear(embed_dim, embed_dim)),
    nn.ReLU(),
    MeanPoolingLayer(dim=1),
    layer_init(nn.Linear(embed_dim, 1), gain=1.0),
)
```

These heads live in `manabot/model/agent.py`, gated by a config flag in
`AgentHypers` (e.g. `aux_heads: bool = False`).

### 2. Auxiliary targets

Computed retroactively at the end of each episode:

- **Life differential:** `(hero_life - villain_life)` at game end,
  normalized by dividing by 20. Applied to every transition in the
  episode as a fixed target.
- **Game length:** Total steps in the episode, normalized by dividing
  by 200 (the truncation limit). Applied to every transition.
- **Creature differential:** `(hero_creatures - villain_creatures)` at
  game end, normalized. Applied to every transition.

These are stored as flat pre-allocated tensors in the rollout buffer in
`train.py`, following the existing pattern: `torch.zeros((num_steps, num_envs))`.

Terminal state information is available via `managym.Observation` —
`raw_obs.agent.life`, `raw_obs.opponent.life`. See `sim.py:determine_outcome`
for the access pattern.

### 3. Auxiliary losses

```python
aux_loss = (
    aux_coef * F.mse_loss(pred_life_diff, target_life_diff)
    + aux_coef * F.mse_loss(pred_game_length, target_game_length)
    + aux_coef * F.mse_loss(pred_creature_diff, target_creature_diff)
)

total_loss = policy_loss - ent_coef * entropy + vf_coef * value_loss + aux_loss
```

`aux_coef` is a new hyperparameter, default 0.5.

### 4. Logging

Track per-head prediction accuracy:
- `aux/life_diff_mse`
- `aux/game_length_mse`
- `aux/creature_diff_mse`
- `aux/life_diff_explained_var`

## Constraints

- Auxiliary targets must not distort the policy gradient — they only
  affect the encoder through their own loss terms
- Auxiliary heads should be removable via config (for A/B comparison)
- `final_observation` plumbing for truncation value bootstrap is
  deferred — keep this in mind when computing episode-level targets
  across truncation boundaries

## Done when

A/B comparison using the first-light harness:

```bash
python -m manabot.verify.run_first_light --mode dev --seed 42 --label aux-on
# (with aux_heads=True in agent config)

python -m manabot.verify.run_first_light --mode dev --seed 42 --label aux-off
# (with aux_heads=False)

python -m manabot.verify.report_first_light --run-id <aux-on-id>
python -m manabot.verify.report_first_light --run-id <aux-off-id>
```

Aux-on run should show faster convergence on causal-chain metrics
(landed_when_able, win rate vs random) compared to aux-off.
