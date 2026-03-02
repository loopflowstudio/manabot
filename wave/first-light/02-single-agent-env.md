# 02: Single-Agent Environment Wrapper

Wrap the existing env so the opponent is handled internally. From the
training loop's perspective, this is a single-agent environment — no
multi-agent buffer, no actor_id routing, no contradictory gradients.

## Finish line

Training loop uses a flat PPO buffer (like CleanRL). The opponent is
a configurable policy (passive or random) that acts inside the env
wrapper. MultiAgentBuffer is removed from the training path.

## Changes

### 1. SingleAgentEnv wrapper (`manabot/env/single_agent_env.py`)

Wraps the existing `Env`. On each `step()`:

```python
def step(self, hero_action):
    obs, reward, done, truncated, info = self.env.step(hero_action)

    # Keep stepping while it's the opponent's turn
    while not done and self._is_opponent_turn(obs):
        opponent_action = self.opponent_policy(obs)
        obs, reward, done, truncated, info = self.env.step(opponent_action)

    return obs, reward, done, truncated, info
```

`opponent_policy` is one of:
- `PassivePolicy`: always picks "pass priority" from the action list
- `RandomPolicy`: picks a random valid action

`_is_opponent_turn` checks `agent_player` index vs the active player.

Key subtlety: reward should reflect the hero's perspective. If the game
ends during an opponent step, the reward is the hero's outcome.

### 2. Flat PPO buffer (`train.py`)

Replace `MultiAgentBuffer` with a standard flat buffer matching CleanRL:

```python
# Pre-allocated tensors
obs_buf = {k: torch.zeros((num_steps, num_envs, *shape)) for k, shape in obs_shapes}
actions_buf = torch.zeros((num_steps, num_envs))
logprobs_buf = torch.zeros((num_steps, num_envs))
rewards_buf = torch.zeros((num_steps, num_envs))
dones_buf = torch.zeros((num_steps, num_envs))
values_buf = torch.zeros((num_steps, num_envs))
```

No per-player routing. No Python loops over envs for storage.
Standard GAE over the flat buffer.

### 3. Update VectorEnv to use SingleAgentEnv

`VectorEnv` wraps N `SingleAgentEnv` instances instead of raw `Env`.
The opponent policy is configured via hypers.

### 4. Change reward to +1/-1

Replace `win_reward=100.0` / `lose_reward=-100.0` with `+1.0` / `-1.0`.

Large reward magnitudes cause the value loss to dominate the policy
gradient (value targets of ~100 vs policy loss of ~1). +1/-1 is standard
(CleanRL Atari clips to {-1, 0, +1}) and plays well with default
`vf_coef=0.5`.

### 5. Remove multi-agent training path

`MultiAgentBuffer` stays in the codebase (sim still uses it) but is
no longer used by the trainer. The trainer's rollout loop simplifies to
the standard CleanRL pattern.

## Constraints

- The `SingleAgentEnv` wrapper must handle the case where the game ends
  during an opponent step (done=True after opponent acts)
- Auto-reset must still work — when a game ends, the next `reset()` or
  auto-reset should start a new game with the same opponent policy
- The observation returned must always be from the hero's perspective
- The env now returns `(obs, reward, terminated, truncated, info)` per
  Gymnasium spec (fixed in stage 01). The wrapper's inner opponent-step
  loop must propagate both `terminated` and `truncated` correctly.
- Truncation value bootstrap is deferred (no `final_observation` plumbing
  yet). The wrapper should preserve `info["action_space_truncated"]`
  from inner steps so rollout health counters remain accurate.
- Reward is currently `win_reward=100.0` / `lose_reward=-100.0` — the
  +1/-1 change (section 4) interacts with `vf_coef` tuning.

## Done when

```bash
pytest tests/ -v
# - SingleAgentEnv correctly auto-steps opponent
# - PassivePolicy always passes priority
# - RandomPolicy picks valid random actions
# - Flat buffer rollout produces correct shapes
# - GAE matches reference implementation on known inputs
# - Training loop runs for 100 steps without error
```
