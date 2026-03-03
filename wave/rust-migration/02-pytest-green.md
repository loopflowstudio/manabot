# 02: Python Integration Tests Green

Get `pytest tests/env/ tests/agent/` passing against the Rust/PyO3 backend with
zero test modifications.

## Finish line

```bash
maturin develop --features python
pytest tests/env/ tests/agent/
python -c "import managym; assert int(managym.ZoneEnum.STACK) == 4"
```

All pass, zero test modifications. CI enforces this on every PR.

## Starting point

The branch has the full binding surface and build system:
- 797-line `bindings.rs`: 10 data classes, 5 enums, AgentError exception
- Maturin `pyproject.toml` (replaces scikit-build/pybind11)
- `__init__.pyi` stubs fixed (ZoneEnum STACK=4, EXILE=5)
- CI workflow (`.github/workflows/rust-python-integration.yml`): OS matrix,
  cargo fmt/clippy/test, maturin develop, pytest, enum smoke
- `cargo test` passes (9 tests)
- Manual Python API smoke verified: reset/step, enum parity, dict-backed
  collections, invalid action error message, profiler methods

**Not yet verified:** `pytest tests/env/ tests/agent/` has not been run.
The binding code was smoke-tested via `cargo rustc --features python --crate-type cdylib`
+ module copy, not through a full maturin install + pytest run.

## Work needed

1. **Environment setup.** Use `uv` to create a working Python environment:
   ```bash
   uv venv .venv --python 3.12
   source .venv/bin/activate
   uv pip install maturin pytest torch numpy pandas gymnasium hydra-core wandb
   uv pip install -e .          # manabot
   maturin develop --features python  # managym
   ```

2. **Run tests, fix failures.** `pytest tests/env/ tests/agent/` — likely failure
   modes:
   - Field name mismatches in observation encoding
   - Enum value mismatches (stubs are fixed but runtime behavior may differ)
   - InfoDict conversion edge cases
   - Reward value or sign differences
   - Game termination timing (turn limits, empty library)

3. **Profiler parity.** Run the profiler smoke test to confirm
   `export_profile_baseline` and `compare_profile` work end-to-end through
   the Python API.

## Key decisions (from design)

- **Dict, not List, for card/permanent collections.** `observation.py:216` does
  `sorted(cards.keys())[:N]` and `test_observation.py:186` does
  `next(iter(observation.agent_cards.keys()))`. Changing these would require test
  modifications. The Rust `Vec<CardData>` converts to `Dict[int, Card]` keyed by
  `card.id` in the PyO3 layer.

- **PyO3 enums with `eq_int`, not Python IntEnum.** `#[pyclass(eq, eq_int)]`
  gives `__int__` conversion and integer equality. Satisfies both
  `int(managym.ZoneEnum.STACK) == 4` and `zone_counts[managym.ZoneEnum.BATTLEFIELD]`.

- **ZoneEnum values follow Rust.** STACK=4, EXILE=5. Stubs were wrong (inverted),
  now fixed. `observation.py` was already correct.

## Risks

- Tests may depend on observation field ordering or exact dict key sets that
  differ between C++ and Rust implementations.
- `manabot/env/observation.py` was updated for list-backed collections in an
  earlier iteration — verify this matches current dict-backed binding behavior.
