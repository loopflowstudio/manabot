# SPS Closeout

Date: March 6, 2026

## Conclusion

SPS work is done for now.

The active Rust vector env is fast enough that end-to-end training is no longer
env-gated. On the closeout rerun, the Rust path sustained **183k SPS** at 16
envs in the env-only benchmark, versus **24k SPS** for the pre-SPS baseline
commit (`4c55c4e501e729c9552859cdcd29d9e57d3e4aa9`). In the training-shaped
benchmark with inference enabled, throughput fell to **2.0k SPS** and torch
inference consumed **97%** of step time.

That is the headline: **training/inference is now the bottleneck, not env
stepping**.

## Scope

- hard-wire training onto `RustVectorEnv`
- keep the legacy `AsyncVectorEnv` wrapper only as benchmark-only compatibility
- trim reusable benchmark tooling down to current-path throughput + breakdown
- capture a quick historical baseline from the last pre-SPS commit
- migrate the SPS story out of `wave/` into this durable report

## Measurement environment

- Machine: Apple M4 Max
- OS: macOS 26.0.1
- Date run: March 6, 2026

## Commands used

### Current code

```bash
. .venv/bin/activate

python scripts/bench_vector_env.py --backend rust --num-envs 1 --steps 2048 --rounds 1
python scripts/bench_vector_env.py --backend rust --num-envs 16 --steps 2048 --rounds 3
python scripts/bench_vector_env.py --backend rust --num-envs 64 --steps 2048 --rounds 1
python scripts/bench_vector_env.py --backend rust --num-envs 128 --steps 2048 --rounds 1

python scripts/bench_breakdown.py --num-envs 16 --steps 1024
python scripts/bench_breakdown.py --num-envs 16 --steps 1024 --with-inference
```

### Historical baseline

Checked out commit `4c55c4e501e729c9552859cdcd29d9e57d3e4aa9` in a temporary
worktree and ran an equivalent 16-env passive-opponent loop against the legacy
`VectorEnv` for 2048 steps across 3 rounds.

## Results

### Env-only throughput

| Scenario | SPS |
|---|---:|
| Baseline commit `4c55c4e`, 16 envs, legacy vector env | 24,204 ± 405 |
| Current Rust vector env, 1 env | 90,898 |
| Current Rust vector env, 16 envs | 183,350 ± 660 |
| Current Rust vector env, 64 envs | 191,916 |
| Current Rust vector env, 128 envs | 191,474 |

At 16 envs, the current path is **7.6x** faster than the reproduced pre-SPS
baseline.

### Breakdown at 16 envs

#### Env-only (`python scripts/bench_breakdown.py --num-envs 16 --steps 1024`)

- Total throughput: **145,810 SPS**
- `step_into_buffers`: **86.6%**
- `apply_reward_policy`: **4.7%**
- `sync_tensors`: **3.6%**

#### With inference (`python scripts/bench_breakdown.py --num-envs 16 --steps 1024 --with-inference`)

- Total throughput: **2,035 SPS**
- `inference`: **97.2%**
- `step_into_buffers`: **2.2%**
- `apply_reward_policy`: **0.2%**
- `sync_tensors`: **0.1%**

Inference-on throughput is about **72x lower** than the env-only run, and the
env itself is only a small fraction of the step budget.

## What produced the gains

The big gains came from moving the hot path into Rust:

- Rust-owned vector stepping
- Rust-side observation encoding
- direct writes into preallocated Python-owned buffers
- tensor reuse instead of per-step Python stacking/tensorization

The relevant runtime API is the `step_into_buffers()` /
`reset_all_into_buffers()` path behind `RustVectorEnv`.

## What did not help

Rayon and manual threading were investigated and are not worth carrying today.

At current per-env step costs, thread scheduling overhead outweighed the work
being parallelized. The result was performance-neutral to slightly negative, so
the shipping implementation stays single-threaded.

## Why SPS stops here

Further env micro-optimization is no longer the best place to spend time.

The closeout benchmark shows:

- env-only throughput is already far above the historical baseline
- the active env path scales well enough through 128 envs
- once inference is enabled, torch dominates the profile and env stepping nearly disappears

That means future throughput work should focus on model/inference/training
architecture first.

## When to revisit env perf work

Revisit env parallelism or another SPS wave only if one of these becomes true:

- `step_into_buffers` becomes a material share of training-step time again
- per-env step cost grows substantially as the card pool / rules engine gets heavier
- training moves off the current inference bottleneck and env time becomes visible again

The prior investigation suggests threading becomes interesting only once
per-env work is much heavier than it is today (roughly an order of magnitude
above the current microsecond-scale step cost).

## Codebase closeout

This pass also changes the codebase shape:

- training builds `VectorEnv` (the single active vector env implementation)
- `LegacyVectorEnv` (AsyncVectorEnv wrapper) deleted
- reusable benchmark: `scripts/bench_breakdown.py`
- SPS wave docs under `wave/sps/` were deleted after migrating the durable story here
