# 02: Python Integration Tests Green

Get `pytest tests/env/ tests/agent/` passing against the Rust/PyO3 backend with
zero test modifications.

## Finish line

```bash
source .venv/bin/activate
maturin develop --features python
pytest tests/env/ tests/agent/ -v
python -c "import managym; assert int(managym.ZoneEnum.STACK) == 4"
```

All pass, zero test modifications. CI enforces this on every PR.

## Current state

The binding surface and build system are in place. Two predicted failure modes
have been fixed on the branch:

**Fixed:**
- Negative action index handling — `Env.step()` accepts `i64`, bounds-checks
  in Rust, raises `AgentError("Action index -1 out of bounds: N")`.
  (`env.rs`, `bindings.rs`)
- Action space preservation after invalid step — `Game::step` no longer
  consumes the action space on out-of-range actions. Two new engine tests
  verify this. (`engine_tests.rs`)
- CI workflow — `tests-result` rollup job, branch filtering, `workflow_dispatch`.

**Not yet verified:** `pytest tests/env/ tests/agent/` has not been run against
the Rust backend. The fixes above were derived from code review, not from
running the tests.

Codebase facts:
- `bindings.rs`: 877 lines — 10 data classes, 5 enums, AgentError exception
- `cargo test`: 11 tests pass (3 unit, 8 integration)
- Maturin `pyproject.toml` replaces scikit-build/pybind11
- `__init__.pyi` stubs fixed (ZoneEnum STACK=4, EXILE=5)

## Work remaining

1. **Environment setup.** Use `uv` to create a working Python environment:
   ```bash
   uv venv .venv --python 3.12
   source .venv/bin/activate
   uv pip install maturin pytest torch numpy pandas gymnasium hydra-core wandb
   uv pip install -e .          # manabot
   maturin develop --features python  # managym
   ```

2. **Run tests, fix remaining failures.** `pytest tests/env/ tests/agent/ -v`
   The two most likely failures are already fixed (see above). Remaining
   predicted failure modes, ranked:
   - Game behavior differences — Rust engine produces different game states
     than C++ (card ordering, timing). Fix in engine Rust code.
   - `test_agent_turns_distribution` — `player_index` encoding differs
     between engines. Fix in `observation.rs` or `bindings.rs`.
   - Field name mismatches in observation encoding.
   - Build/import failures — module path, missing `__init__.py` re-exports.

3. **Profiler parity.** Run the profiler smoke test to confirm
   `export_profile_baseline` and `compare_profile` work end-to-end through
   the Python API.

## Key decisions

### Collections are List, not Dict

The earlier design doc claimed dict-backed collections were needed. This is
stale. Current code agrees on lists:
- `observation.py:216` does `cards[:self.cards_per_player]` (list slice)
- `test_observation.py:188` does `observation.agent_cards[0].id` (list index)
- `bindings.rs` exposes `Vec<PyCard>` as Python `list`

### PyO3 enums with `eq_int`, not Python IntEnum

`#[pyclass(eq, eq_int)]` gives `__int__` conversion, `__index__`, and integer
equality. Satisfies both `int(managym.ZoneEnum.STACK) == 4` and
`zone_counts[managym.ZoneEnum.BATTLEFIELD]` (Python lists call `__index__` for
non-int subscripts). Tests only use these operations.

### ZoneEnum values follow Rust

STACK=4, EXILE=5. Stubs were wrong (inverted), now fixed. `observation.py` was
already correct.

### InfoDict is empty on reset, sparse on step

- Rust `reset()` returns `empty_info_dict()` → empty dict → `all()` on empty
  iterable is True ✓
- Rust `step()` returns `winner_name` (string) only on game end
- Python `Env` wrapper adds `true_terminated`, `true_truncated`,
  `action_space_truncated` (all bool) — tests only check
  `isinstance(step_info, dict)` ✓

### Reward pipeline passes through

- Rust engine returns `1.0`/`-1.0` for win/loss, `0.0` otherwise
- Tests assert `−1.0 ≤ final_reward ≤ 1.0` and `isinstance(reward, float)` —
  both satisfied by Rust `f64` → Python `float`

## Risks

- Tests may depend on observation field ordering or exact key sets that differ
  between C++ and Rust implementations.
- Game behavior differences (card ordering, combat timing) may cause subtle
  test failures that only appear at runtime.
