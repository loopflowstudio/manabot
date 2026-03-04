# Rust Migration: Pytest Green + C++ Cutover

## Problem

The Rust engine is functionally complete — 13/13 Rust tests pass, 877 lines of
PyO3 bindings, full CI pipeline. But the Python integration tests (`pytest
tests/env/ tests/agent/`) have never been run against the Rust backend. Until
they pass, C++ can't be removed and the migration stays incomplete.

Two wave steps remain (02-pytest-green, 03-cutover). They're sequential
dependencies with no meaningful boundary between them: once pytest is green,
cutover is mechanical. Treating them as one milestone avoids the dual-backend
limbo where both C++ and Rust exist but only one is tested.

## Approach

**Single milestone: pytest green → benchmark → remove C++ → ship.**

### Phase 1: Triage (no fixes)

```bash
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install maturin pytest torch numpy pandas gymnasium hydra-core wandb
uv pip install -e .
maturin develop --features python
pytest tests/env/ tests/agent/ -v --tb=short 2>&1 | tee scratch/triage.log
```

Run everything. Classify failures into buckets:

| Bucket | Fix location |
|--------|-------------|
| Import/build | `pyproject.toml`, module layout |
| API mismatch | `bindings.rs` |
| Encoding mismatch | `observation.rs`, `bindings.rs` |
| Engine behavior | `game.rs`, `flow/`, `state/` |

### Phase 2: Fix (Rust only, never tests)

Fix in dependency order: imports → API → encoding → engine.

**Predicted failure surface** (from code review, ranked):

1. **Observation validation (HIGH).** `test_observation_validation` calls
   `obs.validate()` every step. The `populate_cards` and `populate_permanents`
   methods in `observation.rs` must partition by `owner_id` (cards) and
   `controller_id` (permanents). If the Python-side validation checks fields
   the Rust integration test doesn't cover, this fails.

2. **Player alternation (MEDIUM).** `test_agent_turns_distribution` needs
   `agent.player_index` to take ≥2 distinct values. Rust test passes — if
   Python still fails, it's a binding conversion issue in `env.rs` or
   `bindings.rs`.

3. **Play land / cast spell (MEDIUM).** `test_play_land_and_cast_spell`
   searches for `PRIORITY_PLAY_LAND` and `PRIORITY_CAST_SPELL`. Assertions
   are inside `if land_idx is not None` — passes vacuously if action types
   aren't generated. Watch for false green. Check `flow/priority.rs`.

4. **AsyncVectorEnv subprocess (LOW).** 7 subprocess workers — if PyO3
   module doesn't survive fork(), obscure crash. Test in isolation first.

The observation encoder in `manabot/env/observation.py` does list slicing
(`cards[:self.cards_per_player]`) and list indexing
(`observation.agent_cards[0].id`). The Rust bindings expose `Vec<PyCard>` as
Python lists. This is already aligned.

### Phase 3: Benchmark

Record C++ throughput baseline before removing it:
```bash
python -c "
import managym
env = managym.Env(enable_profiler=True)
# ... run N games
baseline = env.export_profile_baseline()
print(baseline)
"
```

Run same benchmark against Rust. Rust must be ≥ C++ throughput.
Document results in `scratch/benchmark.md`.

### Phase 4: C++ removal

Mechanical deletion:
- 46 C++ source files (`managym/**/*.{cpp,h}`)
- 10 C++ test files (`tests/**/*.cpp`, `tests/managym_test.*`)
- Root `CMakeLists.txt`, cmake config
- pybind11 references

Verify nothing was missed:
```bash
find managym/ -name "*.cpp" -o -name "*.h" -o -name "CMakeLists.txt" | wc -l
# => 0
```

### Phase 5: Final verification

```bash
cargo test
uv pip install -e managym && pytest tests/env/ tests/agent/ -v
python -c "import managym; assert int(managym.ZoneEnum.STACK) == 4"
```

Update README: remove C++ build/test commands, update install instructions.

## Alternatives considered

| Approach | Tradeoff | Why not |
|----------|----------|---------|
| Fix one test at a time | Safest, slowest | Unnecessary caution — 13 Rust tests pass, binding surface is smoke-tested, failure modes are well-predicted |
| Split 02 and 03 into separate PRs | Cleaner git history | Creates dual-backend limbo. Once pytest is green, cutover is mechanical — no reason to wait |
| Add Python-side compatibility shim | Absorbs differences in Python | Violates "fix Rust only, never tests" constraint. Also: shims accumulate debt |
| Skip benchmarking, remove C++ immediately | Faster | Irreversible. Can't benchmark C++ after it's gone |

## Key decisions

**One milestone, not two.** 02-pytest-green and 03-cutover are sequential with
no interesting boundary. Splitting them means maintaining two backends longer
than necessary. The only gate between them is the throughput benchmark — take
it, then cut.

**Benchmark before deleting.** C++ throughput can't be measured after removal.
Record baseline first, even if it delays cutover by minutes.

**Fix Rust, never tests.** Wave constraint. If a test fails, the Rust code is
wrong. The C++ backend defined correct behavior; the Rust backend must match it.

**Trust the predicted failure surface.** Code review identified 6 failure
buckets ranked by probability. Don't explore blindly — run triage, match
failures to predictions, fix in dependency order.

**False greens are worse than failures.** The `test_play_land_and_cast_spell`
test passes vacuously if action types aren't generated (assertions inside
`if land_idx is not None`). After all tests pass, manually verify that
PRIORITY_PLAY_LAND and PRIORITY_CAST_SPELL actions actually appear in a game.

## Scope

- In scope: pytest green, throughput benchmark, C++ removal, README update
- Out of scope: new Rust features, additional card implementations, training
  runs beyond smoke test, CI changes beyond removing C++ steps

## Done when

```bash
# No C++ source remains
find managym/ -name "*.cpp" -o -name "*.h" -o -name "CMakeLists.txt" | wc -l
# => 0

# Rust tests
cargo test

# Python tests
source .venv/bin/activate
maturin develop --features python
pytest tests/env/ tests/agent/ -v

# Enum smoke
python -c "import managym; assert int(managym.ZoneEnum.STACK) == 4"

# Throughput parity documented
cat scratch/benchmark.md
```

Wave goals advanced:
- **"Rust engine + binding tests green"** — cargo test + pytest pass
- **"Throughput at least parity with C++ baseline"** — benchmark documented
- **"Rust/PyO3 is the default and only managym runtime path"** — C++ removed
- **"C++ runtime/backend path and CMake-based Python module build removed"** — deleted
