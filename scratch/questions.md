# Open Questions / Assumptions

1. **PyO3 numpy interop path**
   - Assumption: using `pyo3` + runtime `numpy` module calls (`np.array`, `np.copyto`) is acceptable for sprint 02, even though the design doc mentioned the Rust `numpy` crate.
   - Reason: this environment is offline and adding new Rust crates was risky; this path preserves behavior/contract while keeping zero-copy prep API semantics.

2. **Python verification environment**
   - Blocker: full `pytest` verification could not be executed here because `pytest` and `numpy` are not installed in the available Python environments and package installation from network is blocked.
   - What was validated instead: `cargo test`, `cargo check --features python`, and a Python smoke reset/step check via a local `maturin develop` install.
