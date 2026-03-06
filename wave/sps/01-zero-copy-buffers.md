# 01: Zero-Copy Buffers

**Finish line:** Rust writes observations directly into pre-allocated numpy arrays via raw pointer access. No Python `__setitem__` calls on the hot path. No per-step memory allocation.

## Current state

The buffer protocol is in place: Python allocates numpy arrays from `obs_layout()`, passes them to `reset_into`/`step_into`, and Rust writes into them. However, Rust currently writes via Python `__setitem__` calls (PyO3 → Python interpreter → numpy), not via raw `*mut f32` pointers. This is correctness-complete but not SPS-optimized.

The `numpy` crate is not in `Cargo.lock`. Raw pointer access requires either adding `numpy` (pyo3 bindings) or extracting buffer pointers via `PyBuffer` protocol manually.

Key files from shipped sprints:
- `managym/src/python/vector_env.rs` — `PyVectorEnv`, `ObservationEncoder`, buffer write logic
- `manabot/env/env.py` — Python `VectorEnv` wrapper using `managym.VectorEnv`
- `managym/__init__.pyi` — type stubs for `VectorEnv`, `RewardConfig`, `ObservationConfig`, etc.

## Changes needed

### 1. Add `numpy` crate dependency

Add `numpy = "0.24"` (or compatible version matching pyo3) to `managym/Cargo.toml`. This provides `PyArray` types with safe raw pointer access.

### 2. Buffer registration with raw pointers

At `reset_into`/`step_into` call boundary:
- Extract `*mut f32` / `*mut bool` / `*mut i32` pointers from numpy arrays
- Validate shapes, dtypes, and contiguity
- Write directly into memory without GIL round-trips

Each env writes to a disjoint slice — no synchronization needed.

### 3. Keep `Py<PyArray>` references

Rust stores `Py<PyArray>` references to prevent Python GC from collecting buffers while Rust holds raw pointers. Pointers are refreshed at each call boundary (not cached across calls).

## Verification

- Parity test: buffer values match Python encoder output on 200+ random games
- SPS measured before/after the raw pointer switch
- No per-step Python allocations in steady state
- Verification ladder passes
