# PyO3 Full Binding Surface

## Problem

Stage 01 shipped a complete Rust engine behind a minimal PyO3 seam. The seam
returns observations as JSON strings and only exposes `PyPlayerConfig` and
`PyEnv`. The existing Python tests and observation encoder need structured
objects — `obs.agent.life`, `obs.turn.phase`,
`obs.action_space.actions[i].action_type` — not JSON.

Until the binding surface matches C++ pybind.cpp exactly, the Python tests can't
run against the Rust backend, and the migration can't cut over.

## Approach

Expand `managym/src/python/bindings.rs` with PyO3 wrapper types that convert from
Rust domain types. One `#[pyclass]` per C++ data class. Each reset/step builds
the wrapper tree from the Rust `Observation` struct.

Ship maturin build config so `pip install -e managym` builds against Rust.

### Binding surface

```rust
#[pymodule]
fn _managym(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Exception
    m.add("AgentError", py.get_type::<PyAgentError>())?;

    // Enums (integer values match C++ exactly)
    m.add_class::<ZoneEnum>()?;         // 0-6
    m.add_class::<PhaseEnum>()?;        // 0-4
    m.add_class::<StepEnum>()?;         // 0-11
    m.add_class::<ActionEnum>()?;       // 0-4
    m.add_class::<ActionSpaceEnum>()?;  // 0-3

    // Data classes
    m.add_class::<PyPlayerConfig>()?;   // exists
    m.add_class::<PyObservation>()?;
    m.add_class::<PyPlayer>()?;
    m.add_class::<PyTurn>()?;
    m.add_class::<PyCard>()?;
    m.add_class::<PyCardTypes>()?;
    m.add_class::<PyManaCost>()?;
    m.add_class::<PyPermanent>()?;
    m.add_class::<PyAction>()?;
    m.add_class::<PyActionSpace>()?;

    // Main API
    m.add_class::<PyEnv>()?;            // exists, needs structured obs return
    Ok(())
}
```

### Data class field specs

Each must match C++ pybind.cpp field names and types exactly.

**PyObservation** — `#[pyclass(name = "Observation")]`
- `game_over: bool`, `won: bool`
- `turn: PyTurn`
- `action_space: PyActionSpace`
- `agent: PyPlayer`, `opponent: PyPlayer`
- `agent_cards: Vec<PyCard>`, `opponent_cards: Vec<PyCard>`
- `agent_permanents: Vec<PyPermanent>`, `opponent_permanents: Vec<PyPermanent>`
- Methods: `validate() -> bool`, `toJSON() -> String`

**PyPlayer** — `#[pyclass(name = "Player")]`
- `player_index: i32`, `id: i32`
- `is_agent: bool`, `is_active: bool`
- `life: i32`
- `zone_counts: Vec<i32>` (7 elements, indexed by ZoneEnum int value)

**PyTurn** — `#[pyclass(name = "Turn")]`
- `turn_number: i32`
- `phase: PhaseEnum`, `step: StepEnum`
- `active_player_id: i32`, `agent_player_id: i32`

**PyCard** — `#[pyclass(name = "Card")]`
- `zone: ZoneEnum`
- `owner_id: i32`, `id: i32`, `registry_key: i32`
- `power: i32`, `toughness: i32`
- `card_types: PyCardTypes`, `mana_cost: PyManaCost`

**PyCardTypes** — `#[pyclass(name = "CardTypes")]`
- 12 bool fields: `is_castable`, `is_permanent`, `is_non_land_permanent`,
  `is_non_creature_permanent`, `is_spell`, `is_creature`, `is_land`,
  `is_planeswalker`, `is_enchantment`, `is_artifact`, `is_kindred`, `is_battle`

**PyManaCost** — `#[pyclass(name = "ManaCost")]`
- `cost: Vec<i32>` — **6 elements** [W, U, B, R, G, C], matching C++
  (Rust internal has 7 with Generic; drop slot 6 in conversion)
- `mana_value: i32`

**PyPermanent** — `#[pyclass(name = "Permanent")]`
- `id: i32`, `controller_id: i32`
- `tapped: bool`, `damage: i32`, `is_summoning_sick: bool`
- Note: C++ omits `card_id`. So do we.

**PyAction** — `#[pyclass(name = "Action")]`
- `action_type: ActionEnum`
- `focus: Vec<i32>`

**PyActionSpace** — `#[pyclass(name = "ActionSpace")]`
- `action_space_type: ActionSpaceEnum`
- `actions: Vec<PyAction>`
- `focus: Vec<i32>`

### Enum specs

All use `#[pyclass(eq, eq_int)]` for integer comparison from Python.

| Python name | Rust source | Values |
|-------------|-------------|--------|
| `ZoneEnum` | `ZoneType` | LIBRARY=0, HAND=1, BATTLEFIELD=2, GRAVEYARD=3, STACK=4, EXILE=5, COMMAND=6 |
| `PhaseEnum` | `PhaseKind` | BEGINNING=0, PRECOMBAT_MAIN=1, COMBAT=2, POSTCOMBAT_MAIN=3, ENDING=4 |
| `StepEnum` | `StepKind` | BEGINNING_UNTAP=0 through ENDING_CLEANUP=11 |
| `ActionEnum` | `ActionType` | PRIORITY_PLAY_LAND=0 through DECLARE_BLOCKER=4 |
| `ActionSpaceEnum` | `ActionSpaceKind` | GAME_OVER=0, PRIORITY=1, DECLARE_ATTACKER=2, DECLARE_BLOCKER=3 |

Note: ZoneEnum STACK=4 / EXILE=5 matches C++ runtime values. The existing
`__init__.pyi` stubs have these swapped (EXILE=4 / STACK=5) — that's a
documentation bug. Fix stubs as part of this work.

### InfoDict conversion

C++ `convertInfoValue` handles 4 variant types: string, dict, int, float. The
current Rust `InfoValue` only has String and Map. Add Int and Float variants:

```rust
pub enum InfoValue {
    String(String),
    Map(BTreeMap<String, InfoValue>),
    Int(i64),
    Float(f64),
}
```

Update `info_dict_to_pydict` in `convert.rs` to handle all 4 types recursively.

### Build system

Replace the scikit-build pyproject.toml with maturin:

```toml
[build-system]
requires = ["maturin>=1.0"]
build-backend = "maturin"

[project]
name = "managym"
version = "0.2.0"
requires-python = ">=3.8"

[tool.maturin]
features = ["python"]
module-name = "managym._managym"
manifest-path = "Cargo.toml"
```

### cargo test link fix

Remove `cdylib` from `Cargo.toml` crate-type. Maturin adds cdylib
automatically when building the Python module. Without it, `cargo test` links
normally:

```toml
[lib]
name = "managym"
# No crate-type — defaults to rlib. Maturin overrides for Python builds.
```

### Env API changes

`PyEnv.reset()` and `PyEnv.step()` currently return observation as a JSON
string. Change to return `PyObservation` (structured object). Add conversion
function `Observation → PyObservation` that builds the full wrapper tree.

`PyEnv.info()` method — expose, returning a Python dict.

Keep profiling API parity on `Env`:
- `export_profile_baseline() -> str`
- `compare_profile(baseline: str) -> str`
- If profiler is disabled, mirror C++ behavior (`""` baseline export and
  `"Profiler not enabled"` compare response).

Error handling: wrap `AgentError` as a Python `RuntimeError` subclass, matching
C++.

`Observation.toJSON()` must return structured, stable JSON-like output (parity
with C++ intent), not a debug `Debug`/`repr` dump.

## Alternatives considered

| Approach | Tradeoff | Why not |
|----------|----------|---------|
| `#[pyclass]` on domain types directly | Less code, no conversion | PyO3 constraints bleed into domain design. Can't handle ManaCost 7→6 mapping or PermanentData card_id omission. |
| Return Python dicts instead of typed classes | No pyclass boilerplate | Tests expect `obs.agent.life` attribute access. Would require test modifications (violates zero-change constraint). |
| Dual module naming (`_managym_rust`) during bring-up | Can run C++ and Rust side by side | Unnecessary complexity. The migration branch is the migration. If you need C++, build from main. |
| C++ trace recording for parity | Catches subtle engine drift | Over-engineering. The Python tests ARE the parity gate. Subtle differences will surface in training, not traces. |

## Key decisions

1. **Wrapper types, not derives.** PyO3 wrappers convert from domain types. This
   handles ManaCost 7→6, PermanentData card_id omission, and keeps domain types
   clean.

2. **ManaCost exposes 6 elements.** C++ has `std::array<int, 6>` [W,U,B,R,G,C].
   Rust internal has `[u8; 7]` with Generic. Python sees 6 elements — drop
   Generic in conversion, cast u8→i32.

3. **agent_cards is list, not dict.** C++ pybind.cpp uses `def_readwrite` on
   `std::vector<CardData>`. With `pybind11/stl.h`, this becomes a Python list.
   The `__init__.pyi` stubs incorrectly say `Dict[int, Card]` — fix stubs to
   `List[Card]`. Update `manabot` observation encoding to consume list-backed
   collections as the canonical API (with temporary support for both list and
   dict inputs during migration if needed).

4. **No dual module naming.** Build as `_managym` from the start. No parallel
   C++ module.

5. **No trace recording.** Skip Tier 1. Python tests are the gate.

6. **Maturin replaces scikit-build.** Single pyproject.toml change. `pip install
   -e managym` builds Rust.

7. **ID fields are object IDs, never indices.** Python-facing `*_id` fields
   (`owner_id`, `controller_id`, `active_player_id`, `agent_player_id`, etc.)
   must carry stable object-id semantics. Do not source these from arena/player
   index values even if they currently coincide in 2-player setups.

8. **IDs are opaque references; determinism is for reproducibility, not semantics.**
   Keep deterministic ID assignment to make tests/debug/profiling reproducible:
   - reset initializes a fresh ID generator,
   - players are allocated first (stable player IDs),
   - deck/card materialization order is deterministic.
   But model-facing code should not treat absolute ID values as meaningful game
   features; IDs exist to reference objects (e.g., action focus), not encode
   strategy-relevant semantics.

9. **Profiler parity now (full).** Ship full profiler capability in this step,
   not placeholder methods:
   - meaningful scoped stats in `Env.info()["profiler"]`
   - `export_profile_baseline()` with reusable baseline payload
   - `compare_profile(baseline)` with actionable diff output
   This is required for Rust vs previous-runtime profiling and regression
   diagnosis during migration.

## Scope

In scope:
- All PyO3 data classes matching C++ pybind.cpp (10 types)
- All PyO3 enums (5 enums)
- AgentError exception
- InfoDict full variant conversion (add int/float)
- Env returning structured PyObservation
- validate() and toJSON() methods
- Env profiling API parity (`export_profile_baseline`, `compare_profile`)
- Profiler implementation parity sufficient for real baseline+compare workflows
- pyproject.toml maturin config
- Cargo.toml crate-type fix
- Fix `__init__.pyi` stubs (ZoneEnum ordering, List vs Dict)
- manabot observation encoder update for list-backed card/permanent collections

Out of scope:
- C++ code removal (separate cutover step after tests pass)
- Performance optimization
- New card implementations

### Scope exception (wave boundary)

Wave README says Python manabot code changes are generally out of scope for this
wave. We are taking a narrow exception here: update only the observation encoder
surface needed to consume the canonical list-backed card/permanent collections.
Rationale: avoid baking a Rust-only dict compatibility shape into the binding
API and keep a single coherent runtime contract.

## Done when

```bash
# Rust engine tests (no link errors)
cd managym && cargo test

# Python module builds and installs
pip install -e managym

# Python integration tests pass against Rust backend
pytest tests/agent/test_managym.py tests/env/

# Enum parity holds
python -c "import managym; assert int(managym.ZoneEnum.STACK) == 4"

# Profiler parity holds
python - <<'PY'
import managym
pc=[managym.PlayerConfig("A", {"Forest":20}), managym.PlayerConfig("B", {"Forest":20})]
e=managym.Env(enable_profiler=True)
e.reset(pc)
for _ in range(10): e.step(0)
b=e.export_profile_baseline()
assert isinstance(b, str)
assert isinstance(e.compare_profile(b), str)
PY
```

Advances wave goals:
> "Rust engine + binding tests are green (cargo test, pytest tests/env/ tests/agent/)"
> "Rust/PyO3 backend is the default managym runtime path"
