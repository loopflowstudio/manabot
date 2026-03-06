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
dominates. With inference, torch dominates. Sprints 02-03 target the
encoding path. Sprint 04 targets the Rust step path. Sprint 05 removes
dead code and publishes final numbers.

## Goals

1. Rust VectorEnv that owns N games and steps them in one call (done)
2. Rust-side observation encoding that matches Python's output exactly
3. Zero-copy buffer writes — Rust writes directly into Python numpy arrays
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

1. **Before**: Run `scripts/bench_breakdown.py` and record the time
   breakdown. This identifies the phase being optimized and confirms
   it's actually on the hot path.

2. **After**: Run `scripts/bench_breakdown.py` again. The optimized
   phase should show a measurable reduction.

3. **A/B**: Run `scripts/bench_ab.py --a <new> --b <old> --rounds 5`
   to confirm the improvement is statistically significant (p < 0.05).

4. **Gate**: The sprint's SPS target must be met. Verification ladder
   must still pass.

```bash
# Breakdown (env-only)
python scripts/bench_breakdown.py --num-envs 16 --steps 2048

# Breakdown (with inference, matches training)
python scripts/bench_breakdown.py --num-envs 16 --steps 2048 --with-inference

# A/B comparison
python scripts/bench_ab.py --a rust --b async --num-envs 16 --rounds 5

# Verification
python -m manabot.verify.step0_env_sanity
```

## Metrics

- Env-only SPS (target: >100,000 on M4 Max)
- Training SPS (target: >15,000 on CPU)
- Env step time breakdown (encode, rust_step, tensorize, unpack)
- Verification ladder still passes
