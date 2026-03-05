# SPS

## Vision

Eliminate the Python↔Rust boundary as a training bottleneck. Today the
env is the ceiling — ~1,600 SPS with per-step PyO3 round-trips, Python
object allocation, and numpy conversion. The game engine itself is fast;
the data path around it is not.

This wave moves the hot path entirely into Rust: batch stepping, observation
encoding, and buffer writes. Python allocates memory once and reads results.
The boundary crossing drops from O(envs × steps) to O(steps) per update.

Prerequisite for the scale wave. Decoupled actor/learner and multi-GPU
don't help if each actor is still bottlenecked on single-env PyO3 calls.

### Not here

- Decoupled actor/learner (scale wave)
- Multi-GPU training (scale wave)
- Game engine optimizations (the engine is already fast)
- New card implementations or rule changes

## Goals

1. Rust VectorEnv that owns N games and steps them in one call
2. Rust-side observation encoding that matches Python's output exactly
3. Zero-copy buffer writes — Rust writes directly into Python numpy arrays
4. Rayon parallelism for stepping games within a single Rust call
5. 10x SPS improvement on same hardware

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

## Metrics

- Training SPS (target: >15,000 on CPU, >50,000 with GPU inference)
- Env step latency p50/p99
- Observation encoding time (Rust vs Python)
- Verification ladder still passes
