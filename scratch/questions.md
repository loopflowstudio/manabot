# Open questions / blockers

## Stage 03 verification blockers in this sandbox

1. **Network-disabled environment prevents Python dependency install**
   - `python3 -m pip install pytest maturin` fails with DNS/network errors.
   - `pytest`, `torch`, `numpy`, `pandas`, `gymnasium`, `hydra-core`, `wandb` are not present locally.

2. **`uv venv` is not usable in this sandbox**
   - With default cache path: permission error opening `~/.cache/uv/...`.
   - With `UV_CACHE_DIR=/tmp/uv-cache`: uv panics in system-configuration initialization.

3. **Consequence**
   - Could not run the exact done-when commands:
     - `maturin develop --features python`
     - `pytest tests/env/ tests/agent/`

4. **What was validated instead**
   - Rust gates: `cargo fmt --all -- --check`, `cargo clippy --all-targets --all-features -- -D warnings`, `cargo test`.
   - Python extension smoke (local workaround build via `cargo rustc ... cdylib` + module copy).
   - Manual Python API smoke covering reset/step, enum parity, dict-backed collections, invalid action error message, and profiler methods.
