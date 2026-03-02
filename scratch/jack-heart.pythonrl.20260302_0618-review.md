# PPO correctness + observability polish review

## What was implemented

- Fixed all planned PPO correctness issues in the training/env/agent path:
  - terminal reward inversion in `Reward.compute`
  - truncation-aware done handling in rollout (`done = terminated | truncated`)
  - per-minibatch advantage normalization (wired to `norm_adv`)
  - minibatch indexing based on actual flattened buffer length
  - policy head final layer init gain set to `0.01`
  - `Categorical(logits=...)` instead of `probs=...`
  - removed optimizer weight decay
- Added action-space truncation visibility:
  - warning in observation encoder when actions exceed `max_actions`
  - per-step `info["action_space_truncated"]` signal from env wrapper
- Added rollout health instrumentation in trainer:
  - per-update + cumulative counters for `skipped_steps`, `truncated_episodes`, `action_space_truncations`
  - structured logger output and wandb metrics
- Added/updated tests for reward correctness, truncation handling, advantage normalization, minibatch planning, categorical construction mode, no weight decay, and rollout health counters.
- Polish pass improvements:
  - removed unused import (`Profiler`)
  - guarded rollout-step logging behind `self.wandb`
  - switched `raise e` to `raise` for traceback preservation
  - fixed `Saving artifact` log typo
  - hardened advantage normalization for singleton minibatches to avoid NaNs.

## Key choices

- **Ship all PPO fixes together**: the bugs interact and corrupt one another’s signal.
- **Truncation bootstrap deferred**: tracked explicitly via warning/counter instead of silent behavior.
- **Normalize advantages per minibatch**: matches intended CleanRL-style behavior.
- **Use rollout health counters** rather than ad hoc logs so regressions are visible in wandb and logs.
- **Singleton-minibatch normalization guard** added in polish to prevent NaN updates when batch fragmentation produces size-1 minibatches.

## How it fits together

`Env.step` now emits truncation-related info and action-space truncation flags. `Trainer._rollout_step` consumes those flags, updates rollout-health counters, and handles terminated/truncated episode completion correctly. After rollout flattening, PPO updates use actual batch length and optional normalized advantages, while `Agent.get_action_and_value` uses a numerically stable logits-based categorical distribution.

## Risks and bottlenecks

- **Truncation returns are still approximate** (no final-observation value bootstrap yet).
- **Action-space truncation warning can be noisy** if config/decks routinely exceed `max_actions`.
- **Counter accuracy depends on vector-info shapes/masks**; current handling supports masked arrays but should be watched in full training runs.
- **High log volume** remains in trainer (pre-existing); could impact long runs.

## What's not included

- No VectorEnv `final_observation` plumbing for truncation bootstrap.
- No architecture changes (attention/LSTM/self-play scope untouched).
- No checkpoint compatibility/migration work.
- No additional CI/pipeline changes.

## Wave alignment

This branch directly advances First Light Goal #1 (PPO correctness parity) and adds observability that supports Goal #4’s verification ladder.

Observable metrics introduced/expanded:
- `rollout/skipped_steps` (+ `_total`)
- `rollout/truncated_episodes` (+ `_total`)
- `rollout/action_space_truncations` (+ `_total`)

These metrics reduce ambiguity when diagnosing training instability in early-stage runs.
