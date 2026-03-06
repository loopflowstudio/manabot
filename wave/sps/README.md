# SPS

## Vision

Eliminate the Python↔Rust boundary as a training bottleneck. Python
observation encoding consumes ~77% of env step time. The game engine
itself is fast (~7% of step time); the data path around it is not.

This wave moves the hot path entirely into Rust: batch stepping, observation
encoding, and buffer writes. Python allocates memory once and reads results.

Prerequisite for the scale wave. Decoupled actor/learner and multi-GPU
don't help if each actor is still bottlenecked on Python encoding.

### Not here

- Decoupled actor/learner (scale wave)
- Multi-GPU training (scale wave)
- Game engine optimizations (the engine is already fast)
- New card implementations or rule changes

## Baseline (M4 Max, 16 envs, passive opponent)

Measured with `scripts/bench_breakdown.py` and `scripts/bench_ab.py`.

```
Env-only SPS (no inference):     ~19,000
Training SPS (with inference):   ~3,300

Time breakdown (env-only):
  encode (Python):    77%
  tensorize:          12%
  rust_step:           7%
  unpack:              3%

Time breakdown (with inference):
  inference (torch):  81%
  encode (Python):    14%
  tensorize:           2%
  rust_step:           2%
  unpack:              1%
```

The bottleneck depends on context. Without inference, Python encoding
dominates. With inference, torch dominates. Sprints 01-01 shipped the
in-process Rust vector env with zero-copy buffers. Sprint 02 adds Rayon
parallelism. Sprint 03 removes dead code and publishes final numbers.

## Goals

1. Rust VectorEnv that owns N games and steps them in one call (done)
2. Rust-side observation encoding that matches Python's output exactly (done)
3. Zero-copy buffer writes — Rust writes directly into Python numpy arrays (done)
4. Rayon parallelism for stepping games within a single Rust call
5. 10x env-only SPS improvement on same hardware

## Risks

- **Observation encoding parity.** The Rust encoder must produce
  bit-identical output to the Python encoder. Any mismatch silently
  corrupts training. Mitigation: automated comparison test on random
  game states.
- **Buffer lifecycle.** Python owns the numpy arrays, Rust holds raw
  pointers. If Python GCs the array while Rust is writing, crash.
  Mitigation: VectorEnv holds `Py<PyArray>` references that prevent GC.
- **Rayon + PyO3 interaction.** GIL must be released before spawning
  Rayon threads. Well-trodden path (pyo3 docs cover this), but needs
  care.

## Measurement protocol

Every sprint follows the same measurement cycle:

1. **Before**: Run the full benchmark suite and save results. This
   captures the baseline before any changes.

2. **After**: Run the same suite. The optimized phase should show a
   measurable reduction.

3. **A/B**: Run `scripts/bench_ab.py --a <new> --b <old> --rounds 5`
   to confirm the improvement is statistically significant (p < 0.05).

4. **Save**: Append results to the Results Tracker table below.

5. **Gate**: The sprint's SPS target must be met. Verification ladder
   must still pass.

```bash
# Breakdown (env-only, multiple scales)
for n in 1 16 64; do
    python scripts/bench_breakdown.py --num-envs $n --steps 2048
done

# Breakdown (with inference, matches training)
python scripts/bench_breakdown.py --num-envs 16 --steps 2048 --with-inference

# A/B comparison
python scripts/bench_ab.py --a rust --b async --num-envs 16 --rounds 5

# Verification
python -m manabot.verify.step0_env_sanity
```

## Results tracker

Record measurements after each sprint. All numbers on M4 Max unless
noted otherwise. Fill in after each sprint ships.

### Env-only SPS (no inference)

| num_envs | Baseline | S01 (Rust env + zero-copy) | S02 (Rayon) | S03 (final) |
|----------|----------|---------------------------|-------------|-------------|
| 1        |          |                           |             |             |
| 16       | ~19,000  |                           |             |             |
| 64       |          |                           |             |             |
| 128      |          |                           |             |             |

### Training SPS (with inference)

| num_envs | Baseline | S01 | S02 | S03 |
|----------|----------|-----|-----|-----|
| 16       | ~3,300   |     |     |     |
| 64       |          |     |     |     |

### Time breakdown (env-only, 16 envs)

| Phase      | Baseline | S01   | S02   | S03   |
|------------|----------|-------|-------|-------|
| encode     | 77%      |       |       |       |
| tensorize  | 12%      |       |       |       |
| rust_step  | 7%       |       |       |       |
| unpack     | 3%       |       |       |       |

### A/B comparisons

| Sprint | A config | B config | A SPS | B SPS | Delta | p-value |
|--------|----------|----------|-------|-------|-------|---------|
|        |          |          |       |       |       |         |

## Ablation protocol (sprint 03)

At wave end, measure each optimization in isolation to attribute SPS
gains. The ablation uses `bench_ab.py` to compare configurations
pairwise:

```bash
# 1. Rust env vs legacy AsyncVectorEnv (full wave impact)
python scripts/bench_ab.py --a rust --b async --num-envs 16 --rounds 5

# 2. Rust env with zero-copy vs Rust env without (sprint 03 impact)
#    Requires a "rust-python-encode" mode that uses Rust stepping
#    but Python-side encoding (sprint 02 behavior).
python scripts/bench_ab.py --a rust --b rust-python-encode --num-envs 16 --rounds 5

# 3. Rayon on vs off (sprint 02 impact)
#    Requires a "rust-single-thread" mode or RAYON_NUM_THREADS=1.
RAYON_NUM_THREADS=1 python scripts/bench_ab.py --a rust --b rust --num-envs 64 --rounds 5

# 4. Scaling curve: measure SPS at 1, 4, 16, 64, 128 envs
for n in 1 4 16 64 128; do
    python scripts/bench_ab.py --a rust --b async --num-envs $n --rounds 3
done
```

The ablation requires two things built during the wave:
- A `rust-python-encode` bench mode (Python encode path kept for parity
  testing — wire it into `bench_ab.py` as a mode)
- `RAYON_NUM_THREADS=1` support (Rayon respects this env var natively)

Record ablation results in a final table:

| Configuration           | SPS (16 envs) | SPS (64 envs) | vs baseline |
|-------------------------|---------------|---------------|-------------|
| Baseline (AsyncVectorEnv) |             |               | 1.0x        |
| S01: Rust env + zero-copy |             |               |             |
| S02: + Rayon              |             |               |             |
| S03: final (cleaned)      |             |               |             |

## Metrics

- Env-only SPS (target: >100,000 on M4 Max)
- Training SPS (target: >15,000 on CPU)
- Env step time breakdown (encode, rust_step, tensorize, unpack)
- Scaling curve (SPS vs num_envs)
- Verification ladder still passes
