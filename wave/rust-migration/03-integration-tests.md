# 03: Python Integration Tests Green

Get the existing Python tests passing against the Rust/PyO3 backend.

## Finish line

`pip install -e managym && pytest tests/env/ tests/agent/` passes with zero
test modifications.

## Starting point

02-bindings shipped the full PyO3 binding surface:
- 10 data classes, 5 enums, AgentError exception (876-line `bindings.rs`)
- Env returns structured PyObservation (not JSON strings)
- Maturin pyproject.toml, Cargo.toml crate-type fix
- Profiler parity (export_profile_baseline, compare_profile)
- Updated `__init__.pyi` stubs (ZoneEnum STACK=4/EXILE=5 fix, List not Dict)
- `manabot/env/observation.py` updated for list-backed card/permanent collections
- `cargo test` passes (9 tests)

### Known environment blockers (from 02-bindings)

- Local `pip` (Python 3.9) does not support editable installs for non-setuptools
  `pyproject.toml` projects. Need maturin + compatible pip, or use
  `maturin develop` directly.
- `pytest` not installed in current environment.
- Smoke-tested via manual `cargo rustc --features python --crate-type cdylib`
  and symlinking the `.so` — API behavior and enum parity verified manually.

## Work needed

1. **Environment setup.** Get a working Python environment with maturin, pytest,
   and compatible pip. Either upgrade pip or use `maturin develop` as the install
   path.

2. **Build and install.** `maturin develop` (or `pip install -e managym`) to
   produce a working `managym._managym` importable module.

3. **Run tests.** `pytest tests/env/ tests/agent/` — fix any failures. Likely
   failure modes:
   - Field name mismatches in observation encoding
   - Enum value mismatches (stubs are fixed but runtime behavior may differ)
   - InfoDict conversion edge cases
   - Reward value or sign differences
   - Game termination timing (turn limits, empty library)

4. **Profiler parity verification.** Run the profiler smoke test from the
   02-bindings done-when section to confirm end-to-end.

## Done when

```bash
pip install -e managym   # or: maturin develop
pytest tests/env/ tests/agent/
python -c "import managym; assert int(managym.ZoneEnum.STACK) == 4"
```

All pass, zero test modifications.
