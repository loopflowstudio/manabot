# 03: C++ Removal and Cutover

Remove the C++ backend and make Rust the only maintained runtime path.

## Finish line

All four cutover gate items from the wave README are satisfied:
1. Rust engine + binding tests green (cargo test, pytest)
2. Throughput at least parity with C++ baseline
3. Rust/PyO3 is the default and only managym runtime path
4. C++ runtime/backend path and CMake-based Python module build removed

## Starting point

Depends on 02-pytest-green completing. At that point:
- Rust engine is functionally complete (game loop, priority, combat, SBAs)
- PyO3 binding surface matches C++ pybind.cpp exactly
- Python tests pass against Rust backend
- Profiler supports baseline/compare workflows

## Work needed

1. **Throughput benchmarking.** Run the agreed benchmark against both C++ and
   Rust backends. Document results. Rust must be at least parity.

2. **C++ code removal.**
   - Remove C++ source: `managym/agent/*.{cpp,h}`, `managym/state/*.{cpp,h}`,
     `managym/flow/*.{cpp,h}`, `managym/infra/*.{cpp,h}`,
     `managym/cardsets/**/*.{cpp,h}`, `managym/main.cpp` (46 files total)
   - Remove C++ test files: `tests/agent/*.cpp`, `tests/flow/*.cpp`,
     `tests/infra/*.cpp`, `tests/state/*.cpp`, `tests/main.cpp`,
     `tests/managym_test.{cpp,h}` (10 files)
   - Remove root `CMakeLists.txt`, cmake config, pybind11 dependency
   - Clean up any C++-only CI/build scripts

3. **Build system cleanup.**
   - Maturin is the only build path for the Python module
   - `uv pip install -e managym` is the canonical install command
   - Remove scikit-build references if any remain

4. **Documentation update.** README references to C++ build commands
   (`mkdir -p build && cd build && cmake ..`) need updating. Remove
   C++ test commands from README testing section.

5. **Final verification.**
   - `cargo test` still green
   - `uv pip install -e managym && pytest tests/env/ tests/agent/`
   - Training smoke test (short PPO run)

## Risks

- Throughput regression — Rust may be slower on specific workloads (GIL
  overhead, allocation patterns). Profile before removing C++.
- Missed C++ files — grep for `.cpp`, `.h`, `cmake` patterns after removal.
  Note: C++ source is co-located alongside Rust `src/` — careful not to
  remove the Rust code in `managym/src/`.
- Third-party code depending on C++ headers (unlikely but check).

## Done when

```bash
# No C++ source remains
find managym/ -name "*.cpp" -o -name "*.h" -o -name "CMakeLists.txt" | wc -l
# => 0

# Rust tests
cargo test

# Python tests
uv pip install -e managym && pytest tests/env/ tests/agent/

# Throughput parity documented
```

This should be done also, but make sure that we didnt leave any parts ehind:

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

The binding surface and build system are in place. Several fixes derived from
code review are on the branch but **untested against pytest**:

**Fixed (code review, not pytest-verified):**
- Negative action index handling — `Env.step()` accepts `i64`, bounds-checks
  in Rust, raises `AgentError("Action index -1 out of bounds: N")`.
  (`env.rs`, `bindings.rs`)
- Action space preservation after invalid step — `Game::step` no longer
  consumes the action space on out-of-range actions. (`engine_tests.rs`)
- Observation validation + player alternation — Rust integration tests added
  for `observation_stays_valid_through_game` and
  `agent_player_index_alternates`. Both pass. (`engine_tests.rs`)
- CI workflow — `tests-result` rollup job, branch filtering, `workflow_dispatch`.

**Not yet verified:** `pytest tests/env/ tests/agent/` has not been run.
The fixes above were derived from code review. Running the tests is the only
way to know if they're correct and sufficient.

Codebase facts:
- `bindings.rs`: 877 lines — 10 data classes, 5 enums, AgentError exception
- `cargo test`: 10 tests pass (10 integration in `engine_tests.rs`)
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

2. **Triage run.** `pytest tests/env/ tests/agent/ -v --tb=short 2>&1 | tee scratch/triage.log`
   Run all tests before fixing anything. Classify failures into buckets:

   | Bucket | Example | Fix location |
   |--------|---------|-------------|
   | Import/build | `ModuleNotFoundError` | `pyproject.toml`, module layout |
   | API mismatch | Missing attribute, wrong type | `bindings.rs` |
   | Encoding mismatch | Wrong shape, NaN, wrong values | `observation.rs`, `bindings.rs` |
   | Engine behavior | Wrong game state, validation failure | `game.rs`, `flow/`, `state/` |

   Fix in dependency order: imports → API → encoding → engine behavior.
   **Fix Rust code only, never test code** (wave constraint: zero test modifications).

3. **Predicted failure surface** (ranked, from code review):

   1. **Engine validation (HIGH).** `test_observation_validation` calls
      `obs.validate()` every step — checks `owner_id` on cards, `controller_id`
      on permanents, `is_agent` exactly one player. If `populate_cards` or
      `populate_permanents` partition incorrectly (e.g., by controller instead
      of owner for cards), validation returns `false`. Rust integration test
      `observation_stays_valid_through_game` passes — but the Python validation
      checks additional fields the Rust test may not cover.
      Fix: `observation.rs` — `populate_cards`, `populate_permanents`.

   2. **Player alternation (MEDIUM).** `test_agent_turns_distribution` needs
      `agent.player_index` to take ≥2 distinct values. Rust integration test
      `agent_player_index_alternates` passes. If the Python test still fails,
      it's a binding conversion issue.
      Fix: `env.rs` or `bindings.rs`.

   3. **Play land / cast spell (MEDIUM).** `test_play_land_and_cast_spell`
      searches for `PRIORITY_PLAY_LAND` and `PRIORITY_CAST_SPELL` actions.
      **Warning:** assertions are inside `if land_idx is not None` — test
      passes vacuously if action types aren't generated. Watch for false green.
      Fix: `flow/priority.rs`.

   4. **Game termination (LOW-MEDIUM).** Tests loop until `terminated`. If
      engine has infinite priority cycling or combat damage bugs, tests hang at
      step limit. Rust `full_game_loop_completes` passes, so likely OK.
      Fix: `flow/` — combat damage, SBAs, player death.

   5. **Info dict type check (LOW).** `test_initialization` checks value types.
      Reset returns empty dict → `all()` on empty is True. Likely fine.

   6. **AsyncVectorEnv subprocess import (LOW).** `test_vectorenv_tensor_outputs`
      spawns 7 subprocess workers. If maturin module doesn't survive `fork()`,
      fails with obscure errors. Test in isolation: `pytest -k test_vectorenv -s`
      and check subprocess stderr.

4. **Profiler parity.** Run the profiler smoke test to confirm
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
- Previous sandbox attempts to run pytest were blocked by missing network
  access (`uv` panics, `uv pip install` fails DNS). This step requires a
  network-enabled environment with `uv`, `maturin`, and Python 3.12+.
