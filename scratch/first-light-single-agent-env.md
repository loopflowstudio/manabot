# Single-Agent Environment Wrapper

## Problem

The training loop uses `MultiAgentBuffer`, which routes transitions to
per-`(env, player)` buffers based on `actor_ids`. In first-light, this
creates four concrete problems:

1. **Objective mismatch.** First-light target is hero learning against a
   fixed opponent (passive/random). Multi-agent collection instead trains
   both seats with one policy, which is self-play behavior and a different
   objective.

2. **Bootstrap mismatch in current implementation.** We keep separate
   per-player buffers but pass one `next_value` per environment when
   computing GAE. One side is bootstrapped from the wrong acting
   perspective.

3. **Variable per-player trajectories.** Uneven action counts are a
   fundamental part of Magic priority flow. Splitting transitions into
   per-player queues turns that into unstable queue lengths and unstable
   update composition.

4. **Deviation from reference.** CleanRL's known-correct PPO uses a flat
   buffer with pre-allocated tensors. Every structural difference is a
   potential bug hiding spot.

The phase-1 fix: wrap the env so the opponent acts internally. From the
trainer's perspective, this becomes a standard single-agent Gymnasium
environment with deterministic batch size.

## Approach

Seven changes, one PR. Each builds on the previous.

### 1. SingleAgentEnv wrapper (`manabot/env/single_agent_env.py`)

New `gymnasium.Env` subclass wrapping the existing `Env`. The hero is
always player 0. On each `step()`, after the hero acts, the wrapper
auto-steps the opponent until it's the hero's turn again (or the game
ends).

```python
class SingleAgentEnv(gym.Env):
    def __init__(self, match, obs_space, reward, opponent_policy, seed=0):
        self.inner = Env(match, obs_space, reward, seed=seed, auto_reset=True)
        self.opponent_policy = opponent_policy
        self.hero_player_index = 0

    def reset(self, **kwargs):
        obs, info = self.inner.reset(**kwargs)
        return self._skip_opponent(obs, info)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.inner.step(action)

        if info.get("true_terminated") or info.get("true_truncated"):
            # Hero's action ended the game. Env auto-reset.
            done_info = dict(info)  # latch terminal event before skip
            obs, info = self._skip_opponent(obs, info)
            info["true_terminated"] = done_info.get("true_terminated", False)
            info["true_truncated"] = done_info.get("true_truncated", False)
            return obs, reward, terminated, truncated, info

        while self._is_opponent(obs):
            opp_action = self.opponent_policy(obs)
            obs, opp_r, terminated, truncated, info = self.inner.step(opp_action)

            if info.get("true_terminated") or info.get("true_truncated"):
                # Game ended on opponent's turn. Negate reward for hero.
                reward = -opp_r
                done_info = dict(info)  # latch terminal event before skip
                obs, info = self._skip_opponent(obs, info)
                info["true_terminated"] = done_info.get("true_terminated", False)
                info["true_truncated"] = done_info.get("true_truncated", False)
                return obs, reward, terminated, truncated, info

        return obs, reward, terminated, truncated, info

    def _is_opponent(self, obs):
        return int(obs["agent_player"][0, 0]) != self.hero_player_index

    def _skip_opponent(self, obs, info):
        """After reset or auto-reset, skip opponent turns until hero acts."""
        saw_action_space_truncation = bool(info.get("action_space_truncated", False))
        while self._is_opponent(obs):
            action = self.opponent_policy(obs)
            obs, _, _, _, info = self.inner.step(action)
            saw_action_space_truncation = saw_action_space_truncation or bool(
                info.get("action_space_truncated", False)
            )
        info["action_space_truncated"] = saw_action_space_truncation
        return obs, info
```

**Reward handling:** The inner `Env` computes reward from the perspective
of whoever just acted. When the hero acts, the reward is correct. When
the opponent acts and the game ends, the reward is from the opponent's
perspective, so we negate it. For +1/-1 terminal rewards, `-opp_r` gives
the hero's outcome. Non-terminal intermediate rewards are 0 and discarded.

**Auto-reset interaction:** The inner `Env` has `auto_reset=True`. When
a game ends, it internally resets and returns the post-reset observation
with `terminated=False, truncated=False, info["true_terminated"]=True`.
After auto-reset, `_skip_opponent` handles the case where the opponent
goes first in the new game.

**Detecting opponent's turn:** `obs["agent_player"][0, 0]` is the
`player_index` field from managym. The engine sets the observation's
"agent" to whoever currently has priority. If `player_index != 0`,
the opponent has priority.

### 2. Opponent policies (same file)

```python
class PassivePolicy:
    """Always passes priority. Never plays lands, casts, or attacks."""
    def __call__(self, obs):
        actions, valid = obs["actions"], obs["actions_valid"]
        for i in range(len(valid)):
            if valid[i] > 0 and actions[i, 2] > 0:  # PRIORITY_PASS_PRIORITY
                return i
        return next(i for i in range(len(valid)) if valid[i] > 0)

class RandomPolicy:
    """Picks a uniformly random valid action."""
    def __call__(self, obs):
        valid = obs["actions_valid"]
        return random.choice([i for i in range(len(valid)) if valid[i] > 0])
```

Stateless callables operating on raw numpy observations. No torch, no
observation encoding overhead.

### 3. Flat PPO buffer (inline in `train.py`)

Replace `MultiAgentBuffer` with pre-allocated tensors matching CleanRL:

```python
# Pre-allocate
obs_buf = {k: torch.zeros((num_steps, num_envs, *shape)) for k, shape in ...}
actions_buf = torch.zeros((num_steps, num_envs))
logprobs_buf = torch.zeros((num_steps, num_envs))
rewards_buf = torch.zeros((num_steps, num_envs))
dones_buf = torch.zeros((num_steps, num_envs))
values_buf = torch.zeros((num_steps, num_envs))

# Rollout: direct tensor indexing, no routing
for step in range(num_steps):
    obs_buf[key][step] = next_obs[key]
    ...

# GAE: standard 2D computation
for t in reversed(range(num_steps)):
    nextnonterminal = 1.0 - dones_buf[t + 1]
    nextvalues = values_buf[t + 1]
    delta = rewards_buf[t] + gamma * nextvalues * nextnonterminal - values_buf[t]
    advantages[t] = lastgaelam = delta + gamma * lam * nextnonterminal * lastgaelam

# Flatten: reshape (num_steps, num_envs, ...) → (batch_size, ...)
b_obs = {k: v.reshape(-1, *v.shape[2:]) for k, v in obs_buf.items()}
```

Batch size is deterministic: `num_steps * num_envs`. No per-player
routing. No Python for-loops over envs during storage.

### 4. VectorEnv wires in SingleAgentEnv

```python
class VectorEnv:
    def __init__(self, num_envs, match, obs_space, reward, device, opponent_policy=None):
        if opponent_policy is not None:
            make = lambda: SingleAgentEnv(match, obs_space, reward, opponent_policy)
        else:
            make = lambda: Env(match, obs_space, reward, auto_reset=True)
        self._env = gym.vector.AsyncVectorEnv([make] * num_envs)
```

When `opponent_policy` is provided, VectorEnv creates SingleAgentEnv
instances. When None, it falls back to raw Env (for sim or other uses).

### 5. Reward defaults to +1/-1

Change `RewardHypers` defaults:

```python
win_reward: float = 1.0    # was 100.0
lose_reward: float = -1.0  # was -100.0
```

Large reward magnitudes cause value loss to dominate policy gradient.
+1/-1 is standard and works with default `vf_coef=0.5`.

### 6. Remove dead buffer code

Delete `PPOBuffer` and `MultiAgentBuffer` from `train.py`. Neither is
used by sim (sim creates raw `Env` instances and alternates players
manually). Dead code is worse than missing code — git has the history.

Also remove from the training loop:
- `actor_ids` tracking (`get_agent_indices` calls)
- `multi_buffer` initialization and usage
- Buffer size logging (`self.multi_buffer.buffers.values()`)

### 7. Clarify phasing toward self-play

This PR is a stabilization scaffold, not the end-state multi-agent
architecture.

- **Now (first-light):** single-agent hero vs fixed opponent to validate PPO
  correctness and learning signal.
- **Later (self-play wave):** deliberate multi-agent training design,
  likely population/league style rather than naive symmetric self-play.

## Alternatives considered

| Approach | Tradeoff | Why not |
|----------|----------|---------|
| Keep MultiAgentBuffer, only train player 0 | Minimal code change | Preserves per-player routing complexity. Uneven batch sizes persist. Doesn't match CleanRL. |
| Keep two-player collection but use one flat actor-labeled buffer | Better than per-player queues | Still a larger objective shift (self-play semantics) than this wave's fixed-opponent goal. Better in self-play wave after PPO baseline is stable. |
| Handle opponent in managym (C++) | Faster (no Python overhead for opponent steps) | Couples opponent policy to the engine. Loses flexibility to swap opponents from Python. Larger blast radius. |
| Inline opponent stepping in VectorEnv | One fewer abstraction layer | Less testable. SingleAgentEnv is a clean Gymnasium wrapper that can be tested independently. |
| Gymnasium's AutoResetWrapper | More standard | Current custom auto_reset already works. Switching wrappers is orthogonal to this change. |

## Key decisions

**Negate opponent reward, don't recompute from game state.** At game end
during opponent's turn, `hero_reward = -opp_reward`. This is correct for
symmetric terminal rewards (+1/-1).

**Inner Env keeps auto_reset=True.** Existing auto-reset logic works.
SingleAgentEnv delegates reset and handles opponent turns after auto-reset.

**Latch terminal info before post-reset opponent skipping.** After an
episode-ending action, `inner.step()` returns post-reset observation but
`info["true_terminated"]`/`info["true_truncated"]` indicates the event.
We preserve those flags across `_skip_opponent()` so done signals remain
correct for PPO/GAE.

**Delete buffer classes, don't just stop using them.** PPOBuffer and
MultiAgentBuffer are only used by trainer code.

**Opponent policies operate on numpy obs, not tensors.** The opponent
acts inside SingleAgentEnv before batching into tensors.

**Hero is always player 0.** `Match.to_cpp()` returns
`[hero_config, villain_config]`; managym assigns indices in order.

## Scope

In scope:
- SingleAgentEnv wrapper with opponent auto-stepping
- PassivePolicy and RandomPolicy
- Flat pre-allocated buffer replacing MultiAgentBuffer
- VectorEnv plumbing for `opponent_policy`
- Reward default change to +1/-1
- Removal of dead buffer code and actor_id routing
- `opponent_policy` hyper in TrainHypers
- Tests for all of the above
- Explicit phase-1 framing toward future self-play work

Out of scope:
- `final_observation` plumbing for truncation value bootstrap (deferred)
- Self-play (comes after beating random)
- Observation space changes (separate wave item)
- Attention mechanism tuning (wave says: turn it off first)
- C++ engine changes

## Done when

```bash
pytest tests/ -v
```

Tests verify:
- SingleAgentEnv correctly auto-steps opponent and returns hero-perspective obs
- SingleAgentEnv handles game ending during opponent's turn (reward negation)
- SingleAgentEnv preserves terminal flags across post-reset opponent skipping
- SingleAgentEnv handles auto-reset + opponent-goes-first in new game
- PassivePolicy always picks pass_priority action
- RandomPolicy picks from valid actions only
- Flat buffer produces shapes `(num_steps * num_envs, ...)`
- GAE matches CleanRL reference on deterministic inputs
- Training loop runs 100 steps without error using SingleAgentEnv

Wave goals advanced:
> "Reduce the system to single-agent training against a passive opponent"
> "Fix every confirmed PPO bug so training matches CleanRL's known-correct implementation"
