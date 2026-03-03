# Stage 03: Python Integration Tests Green

## Problem

The Rust game engine works (`cargo test` passes, 9 tests), but Python can't use
it. The PyO3 bindings are skeletal — 116 lines that return JSON strings instead
of structured objects. Every Python consumer (`env.py`, `observation.py`,
`match.py`, all tests) expects `obs.agent.life`, `obs.action_space.actions`,
`card.card_types.is_creature`, etc.

The build system still points at the old C++ toolchain (scikit-build + pybind11).
No Python environment exists with maturin or pytest installed.

Until this works, the Rust engine is dead weight.

## Approach

One PR, four layers: build system, structured bindings, environment bootstrap, CI automation.

### 1. Build system — maturin

Replace `managym/pyproject.toml` with a maturin-based config. The Cargo.toml
already has `crate-type = ["cdylib"]` and the `python` feature with PyO3. Just
needs the right pyproject.toml to drive it.

### 2. Full PyO3 binding surface (~700 lines in bindings.rs)

Wrap every Rust type that Python touches. No JSON serialization, no
intermediate layer — direct `#[pyclass]` structs with `#[pyo3(get)]` fields.

**Enums (5):**

| Python name | Rust source | Variants |
|-------------|-------------|----------|
| `ZoneEnum` | `ZoneType` | LIBRARY=0 HAND=1 BATTLEFIELD=2 GRAVEYARD=3 STACK=4 EXILE=5 COMMAND=6 |
| `PhaseEnum` | `PhaseKind` | BEGINNING=0 through ENDING=4 |
| `StepEnum` | `StepKind` | BEGINNING_UNTAP=0 through ENDING_CLEANUP=11 |
| `ActionEnum` | `ActionType` | PRIORITY_PLAY_LAND=0 through DECLARE_BLOCKER=4 |
| `ActionSpaceEnum` | `ActionSpaceKind` | GAME_OVER=0 through DECLARE_BLOCKER=3 |

All use `#[pyclass(eq, eq_int)]` for `int()` conversion and int comparison.

**Data classes (10):**

| Python name | Rust source | Key fields |
|-------------|-------------|------------|
| `ManaCost` | `ManaCost` | `cost: List[int]`, `mana_value: int` |
| `CardTypes` | `CardTypeData` | 12 bool flags (`is_creature`, `is_land`, etc.) |
| `Player` | `PlayerData` | `player_index`, `id`, `life`, `is_active`, `is_agent`, `zone_counts: List[int]` |
| `Card` | `CardData` | `zone: ZoneEnum`, `owner_id`, `id`, `power`, `toughness`, `card_types: CardTypes`, `mana_cost: ManaCost` |
| `Permanent` | `PermanentData` | `id`, `controller_id`, `tapped`, `damage`, `is_summoning_sick` |
| `Turn` | `TurnData` | `turn_number`, `phase: PhaseEnum`, `step: StepEnum`, `active_player_id`, `agent_player_id` |
| `Action` | `ActionOption` | `action_type: ActionEnum`, `focus: List[int]` |
| `ActionSpace` | `ActionSpaceData` | `action_space_type: ActionSpaceEnum`, `actions: List[Action]`, `focus: List[int]` |
| `Observation` | `Observation` | `game_over`, `won`, `turn`, `action_space`, `agent`, `opponent`, `agent_cards: Dict[int, Card]`, etc. |
| `PlayerConfig` | (already exists) | `name`, `decklist` |

**Collection types:** `agent_cards` and `agent_permanents` (and opponent
equivalents) return `Dict[int, T]` keyed by object id. The Rust side stores
`Vec<CardData>` — conversion happens in the `From` impl, building the dict
from `card.id`. Both `observation.py` and `test_observation.py` call `.keys()`,
so dict is non-negotiable.

**Observation methods:**
- `validate() -> bool` — delegates to Rust `Observation::validate()`
- `toJSON() -> str` — delegates to Rust `Observation::to_json()`

**Env returns structured types:**
- `reset()` → `(PyObservation, PyDict)` (not `(String, PyDict)`)
- `step()` → `(PyObservation, f64, bool, bool, PyDict)` (not `(String, ...)`)
- `info()` → `PyDict` (new — exposes profiler/behavior data)

**Exception:**
- `AgentError(RuntimeError)` — mapped from Rust `AgentError`

### 3. Environment bootstrap

Use `uv` (already installed at 0.9.22) to create a venv and install everything:
```bash
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install maturin pytest torch numpy pandas gymnasium hydra-core wandb
uv pip install -e .          # manabot
maturin develop --features python  # managym
```

### 4. Fix stubs

`__init__.pyi` has ZoneEnum EXILE=4/STACK=5 — backwards from Rust. Fix to
match Rust (STACK=4, EXILE=5). `observation.py` already has the correct
values.

### 5. Add GitHub Actions CI workflow

Add a workflow that runs on pull requests and pushes to the branch to enforce
the stage-03 gate automatically.

**Do-it-all package selected**

Implement one workflow with:

- **OS matrix**: `ubuntu-latest` and `macos-latest`
- **Rust quality gates**:
  - `cargo fmt --all -- --check`
  - `cargo clippy --all-targets --all-features -- -D warnings`
  - `cargo test` (in `managym/`)
- **Python integration gates**:
  - Python 3.12 + `uv` + `maturin`
  - `uv pip install -e .` (manabot)
  - `maturin develop --features python` (managym extension)
  - `pytest tests/env/ tests/agent/`
  - enum smoke: `python -c "import managym; assert int(managym.ZoneEnum.STACK) == 4"`
- **Caching**:
  - Rust: use `Swatinem/rust-cache@v2` to cache Cargo registry, git index, and
    target artifacts per OS/toolchain/lockfile.
  - Python: use `actions/setup-python@v5` cache for pip/uv wheels keyed by
    Python version + `pyproject.toml`/`uv.lock`.
  - Maturin/Rust extension reuse: keep `managym/target` in the Rust cache so
    repeated `maturin develop` runs avoid full rebuilds.

**Why this belongs in this branch:**
- Stage 03 is explicitly about Python integration being green end-to-end.
- Without CI, regressions in binding shape or install path can silently return.
- This makes the migration cutover gate observable for every PR.

**Known cost/risks:**
- macOS matrix increases CI runtime and flake surface.
- Clippy with `-D warnings` may fail on generated/macro-heavy paths; we keep
  per-file targeted allows where needed (already used in bindings.rs).

## Alternatives considered

| Approach | Tradeoff | Why not |
|----------|----------|---------|
| Python deserialization layer (keep JSON, deserialize in `__init__.py`) | Less Rust code, but adds serialization overhead per step and creates a parallel type system to maintain | Performance cost in the hot loop; fragile coupling between JSON schema and Python classes |
| Hybrid (PyO3 for Observation, Python dicts for nested types) | Fewer `#[pyclass]` definitions | `observation.py` calls `card.card_types.is_creature` — needs real objects, not dicts |
| Split into multiple PRs (build system first, then bindings, then tests) | Smaller reviews | Each piece alone doesn't deliver value; the build system PR has no way to verify it works without bindings, bindings can't be tested without the env. One coherent change. |

## Key decisions

**Dict, not List, for card/permanent collections.** The step doc mentions "List
not Dict" as a prior fix, but `observation.py:216` does
`sorted(cards.keys())[:N]` and `test_observation.py:186` does
`next(iter(observation.agent_cards.keys()))`. Changing these would require test
modifications, violating the "zero test modifications" constraint. The Rust
`Vec<CardData>` gets converted to `Dict[int, Card]` keyed by `card.id` in the
PyO3 layer.

**PyO3 enums with `eq_int`, not Python IntEnum.** PyO3's `#[pyclass(eq, eq_int)]`
gives `__int__` conversion and integer equality. This satisfies both
`int(managym.ZoneEnum.STACK) == 4` and `zone_counts[managym.ZoneEnum.BATTLEFIELD]`
(indexing by int-convertible value). No need for a Python-side IntEnum wrapper.

**uv, not conda or system pip.** System Python is 3.9.6 (Xcode). uv can
provision a modern Python (3.12) and handle maturin + all deps cleanly.

**ZoneEnum values follow Rust.** Rust has STACK=4, EXILE=5. Stubs are wrong
(inverted). observation.py is correct. Fix the stubs.

## Scope

- **In scope:** pyproject.toml for maturin, full bindings.rs expansion, stubs
  fix, env bootstrap, `pytest tests/env/ tests/agent/` green, GitHub Actions CI
  for Rust + Python integration checks
- **Out of scope:** New tests, observation.py changes, training pipeline, profiler
  deep-dive, C++ removal

## Done when

```bash
source .venv/bin/activate
maturin develop --features python
pytest tests/env/ tests/agent/
python -c "import managym; assert int(managym.ZoneEnum.STACK) == 4"
```

All pass, zero test modifications.

And CI runs the same gate checks on every PR for this branch.
