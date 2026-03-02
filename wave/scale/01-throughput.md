# 01: Throughput — Decoupled Actor/Learner

## Finish line

Actor (rollout) and learner (PPO update) run asynchronously. GPU
is never idle. 5-10x SPS improvement.

## Changes

### 1. Decoupled architecture (IMPALA/SEED RL pattern)

Separate the training loop into:
- **Rollout workers**: step environments, run inference, fill trajectory
  buffers. Multiple processes, each managing a batch of envs.
- **Learner**: consumes completed trajectory buffers, runs PPO updates,
  publishes new weights.

Communication via shared memory ring buffers (Sample Factory pattern):
- Rollout workers write trajectories to shared memory slots
- Learner reads from slots, signals "done" back
- No pickle, no pipes for tensor data

### 2. Policy versioning + lag metrics

Track which policy version each rollout was collected with. Log the
lag (how many updates old the policy is). APPO literature shows 1-2
updates of lag is fine; more than that degrades learning.

Optional: V-trace importance sampling correction for off-policy data.

### 3. Multi-GPU support

- Learner uses DataParallel or DistributedDataParallel for the PPO update
- Rollout workers can use a separate GPU for inference (or CPU if env
  stepping is the bottleneck)

### 4. Shared memory observation transport

Replace AsyncVectorEnv pickle-based IPC with shared memory buffers.
Pre-allocate observation tensors in shared memory. Env workers write
directly; learner reads without copy.

## Done when

- SPS > 5x first-light baseline
- GPU utilization > 80% during training
- Policy lag stays under 2 updates
- No regression in learning quality
