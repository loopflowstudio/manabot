# 02: Single-Agent Environment Wrapper

Wrap the env so the opponent acts internally. From the trainer's
perspective, this becomes a standard single-agent Gymnasium environment
with deterministic batch size.

## Finish line

Training loop uses a flat pre-allocated buffer (matching CleanRL) with
`SingleAgentEnv` instances that auto-step the opponent. `MultiAgentBuffer`
and `PPOBuffer` are deleted. Tests pass.

## Problem

The training loop uses `MultiAgentBuffer`, which routes transitions to
per-`(env, player)` buffers based on `actor_ids`. This creates:

1. **Objective mismatch.** Hero-vs-fixed-opponent is the first-light
   target. Multi-agent collection trains both seats — that's self-play.
2. **Bootstrap mismatch.** Separate per-player buffers but one
   `next_value` per environment for GAE. One side bootstraps wrong.
3. **Variable trajectories.** Magic priority flow creates uneven
   per-player action counts, producing unstable queue lengths.
4. **Reference deviation.** CleanRL uses flat pre-allocated tensors.
   Every structural difference is a potential bug.

## Changes

Seven changes, one PR.

### 1. SingleAgentEnv wrapper (`manabot/env/single_agent_env.py`)

`gymnasium.Env` subclass wrapping `Env`. Hero is always player 0. On
`step()`, after the hero acts, the wrapper auto-steps the opponent until
it's the hero's turn (or game ends).

Key mechanics:
- **Opponent detection:** `obs["agent_player"][0, 0]` is the
  `player_index` from managym. If != 0, opponent has priority.
- **Reward on opponent's terminal action:** Negate `opp_reward` for
  hero perspective. Correct for symmetric +1/-1 rewards.
- **Auto-reset interaction:** Inner `Env` has `auto_reset=True`. After
  episode end, `_skip_opponent()` handles opponent-goes-first in new game.
- **Terminal flag latching:** Preserve `info["true_terminated"]` /
  `info["true_truncated"]` across post-reset opponent skipping so done
  signals remain correct for GAE.
- **Action space truncation:** Accumulate `action_space_truncated` across
  all opponent steps within a single hero step.

### 2. Opponent policies (same file)

`PassivePolicy`: Always picks `PRIORITY_PASS_PRIORITY` action.
`RandomPolicy`: Uniform random over valid actions.

Stateless callables on raw numpy obs. No torch, no encoding overhead.

### 3. Flat PPO buffer (inline in `train.py`)

Replace `MultiAgentBuffer` with pre-allocated tensors:

```python
obs_buf = {k: torch.zeros((num_steps, num_envs, *shape)) for k, shape in ...}
actions_buf = torch.zeros((num_steps, num_envs))
# ... logprobs, rewards, dones, values

# GAE: standard 2D computation over (num_steps, num_envs)
# Flatten: reshape (num_steps, num_envs, ...) -> (batch_size, ...)
```

Batch size is deterministic: `num_steps * num_envs`.

### 4. VectorEnv wires in SingleAgentEnv

Add `opponent_policy` param to `VectorEnv.__init__()`. When provided,
create `SingleAgentEnv` instances. When None, fall back to raw `Env`.

### 5. Reward defaults to +1/-1

Change `RewardHypers` defaults from 100/-100 to 1/-1. Large magnitudes
cause value loss to dominate policy gradient.

### 6. Remove dead buffer code

Delete `PPOBuffer` and `MultiAgentBuffer` from `train.py`. Remove
`actor_ids` tracking, `multi_buffer` init/usage, buffer size logging.

### 7. `opponent_policy` hyper in TrainHypers

Config option to select opponent: `"passive"` or `"random"`.

## Key decisions

- **Negate opponent reward, don't recompute from game state.** Correct
  for symmetric terminal rewards.
- **Inner Env keeps auto_reset=True.** Existing logic works; wrapper
  handles opponent turns after auto-reset.
- **Delete buffer classes, don't just stop using them.** Only used by
  trainer. Git has history.
- **Hero is always player 0.** `Match.to_cpp()` returns
  `[hero_config, villain_config]`; managym assigns indices in order.

## Alternatives considered

| Approach | Why not |
|----------|---------|
| Keep MultiAgentBuffer, only train player 0 | Preserves routing complexity, uneven batch sizes. |
| Handle opponent in managym (C++) | Couples policy to engine, loses Python flexibility. |
| Inline opponent stepping in VectorEnv | Less testable than standalone wrapper. |

## Constraints

- `final_observation` plumbing for truncation value bootstrap is deferred.
- Observation space changes are separate (item 03).
- No C++ engine changes.

## Done when

```bash
pytest tests/ -v
```

Tests verify:
- SingleAgentEnv auto-steps opponent, returns hero-perspective obs
- Game ending during opponent's turn produces negated reward
- Terminal flags preserved across post-reset opponent skipping
- Auto-reset + opponent-goes-first in new game works
- PassivePolicy always picks pass_priority
- RandomPolicy picks from valid actions only
- Flat buffer produces shapes `(num_steps * num_envs, ...)`
- GAE matches CleanRL reference on deterministic inputs
- Training loop runs 100 steps without error using SingleAgentEnv
