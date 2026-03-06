# SPS: In-Process Vectorized Environment — Reviewer Guide

## What shipped

Rust `PyVectorEnv` that owns N `Env` instances, steps them sequentially, encodes
observations in Rust, and writes into pre-allocated NumPy buffers. Python `VectorEnv`
is a thin wrapper that allocates buffers once and returns `torch.from_numpy()` tensors.

Key files:
- `managym/src/python/vector_env.rs` — `PyVectorEnv`, `ObservationEncoder`, buffer writes
- `manabot/env/env.py` — Python `VectorEnv` wrapper
- `managym/__init__.pyi` — type stubs
- `tests/env/test_env.py` — updated tests

## How to verify

1. `cargo test` — Rust unit tests including vector env
2. `pip install -e managym && pytest tests/env/test_env.py -v` — Python env tests
3. Check observation encoding parity: Rust encoder output should match Python `ObservationSpace.encode()` on random game states

## Known limitations

- Buffer writes use Python `__setitem__` calls, not raw pointers (correctness-focused, not SPS-optimized)
- Stepping is sequential (`num_threads` accepted but unused)
- `reset(seed=..., options=...)` ignores arguments, delegates to Rust `reset_into`
