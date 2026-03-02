# 02: Full Python Parity + Cutover

Expand the stage-01 PyO3 seam to full Python API parity, then cut over to Rust
as the only maintained backend.

## Finish line

The existing Python tests in `tests/env/` and `tests/agent/` pass against the
Rust backend with zero modifications, and the C++ backend/runtime path is
removed.

## Starting point

Stage 01 shipped a complete Rust engine (`managym/src/`) with:
- Full game loop: turn/phase/step progression, priority, combat, SBAs
- Typed index arenas (`CardId`, `PermanentId`, `PlayerId`) with flat `GameState`
- Card registry: 5 basic lands, Llanowar Elves, Grey Ogre
- `Env.reset/step` API in Rust, observation building, behavior tracking
- Deterministic RNG (ChaCha8, seeded) — same seed + actions = same trace
- 6 passing tests: setup, full game, empty-library loss, determinism, enum contract, mana

A minimal PyO3 seam already exists:
- `src/python/mod.rs`, `bindings.rs`, `convert.rs` — behind `python` Cargo feature
- Exposes `PyPlayerConfig` and `PyEnv` (reset/step)
- `Cargo.toml` already has `pyo3 = "0.22"` as optional dependency, dual crate-type `["cdylib", "rlib"]`
- **Known issue:** `cargo test --all-features` fails at link time in some environments
  (missing `Py*` symbols). The seam compiles but hasn't been linked end-to-end yet.

`Observation::to_json()` currently returns a debug snapshot string, not
schema-validated canonical JSON. This needs to be replaced with proper field-level
serialization for the Python binding surface.

## Architecture

### Expand existing PyO3 surface

The module structure already exists. Work needed:

```
managym/src/python/
├── mod.rs        # feature gate — exists, needs full module registration
├── bindings.rs   # PyPlayerConfig, PyEnv — exists, needs all data classes
└── convert.rs    # InfoDict → PyDict — exists, needs full recursive conversion
```

Add `pyproject.toml` for maturin build (replaces CMake for Python module).

### Binding surface

Expand from the current minimal seam to mirror C++ pybind.cpp exactly:

```rust
#[pymodule]
fn _managym(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Exceptions
    m.add("AgentError", py.get_type::<AgentError>())?;

    // Enums (same integer values as C++)
    m.add_class::<ZoneEnum>()?;      // 0-6
    m.add_class::<PhaseEnum>()?;     // 0-4
    m.add_class::<StepEnum>()?;      // 0-11
    m.add_class::<ActionEnum>()?;    // 0-4
    m.add_class::<ActionSpaceEnum>()?; // 0-3

    // Data classes
    m.add_class::<PyPlayerConfig>()?;  // exists
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
    m.add_class::<PyEnv>()?;  // exists, needs full observation return

    Ok(())
}
```

### Key binding details

**Observation fields** — must match C++ field names exactly:
- `game_over`, `won` (bool)
- `turn` (Turn struct with turn_number, phase, step, active_player_id, agent_player_id)
- `action_space` (ActionSpace with action_space_type, actions, focus)
- `agent`, `opponent` (Player with player_index, id, is_agent, is_active, life, zone_counts)
- `agent_cards`, `opponent_cards` (Vec<Card>)
- `agent_permanents`, `opponent_permanents` (Vec<Permanent>)

**Env API** — identical signatures:
```python
env = managym.Env(seed=0, skip_trivial=True, enable_profiler=False, enable_behavior_tracking=False)
obs, info = env.reset([PlayerConfig("Alice", {"Forest": 17, "Llanowar Elves": 13}), ...])
obs, reward, terminated, truncated, info = env.step(action_index)
```

**Enum value parity** — make integer values explicit in Rust with
`#[repr(i32)]` and fixed discriminants matching C++.

**InfoDict conversion** — recursive Rust HashMap → Python dict, matching
`convertInfoDict` / `convertInfoValue` from C++.

**validate() and toJSON()** — Observation methods preserved for debugging.
Replace the current debug-string `to_json()` with proper field-level JSON
serialization.

### Build system

Add maturin config (replaces CMake Python module build):

```toml
# pyproject.toml
[build-system]
requires = ["maturin>=1.0"]
build-backend = "maturin"

[tool.maturin]
module-name = "managym._managym"
```

### Parity testing strategy

Use two tiers:
- **Tier 1 (migration bring-up, optional):** C++ trace replay to catch engine drift quickly
- **Tier 2 (long-term):** semantic/API compatibility tests for Rust as source of truth

**1. Game trace recording (C++ side):**

Before removing C++ code, record reference traces:
- Pick N seeds (e.g., 100 games)
- For each seed, play with a deterministic policy (e.g., always pick action 0)
- At each step, serialize: observation fields, reward, terminated, action taken
- Save as JSON files in `tests/parity/traces/`

**2. Trace replay (Rust side):**

For each recorded trace:
- Create Env with same seed and player configs
- Step through the same action sequence
- At each step, compare every observation field against the recorded value
- Any mismatch = test failure with detailed diff

**3. Existing Python test reuse:**

The tests in `tests/env/` and `tests/agent/` already test:
- Env initialization and stepping
- Observation consistency
- Tensor shapes and dtypes
- Enum value parity
- Game completion

These should pass unmodified against the Rust backend.

After migration cutover, keep Tier 1 traces as optional regression fixtures (or
retire them) and rely on Tier 2 as the required gate.

### Migration path

1. Resolve PyO3 link environment issue — get `cargo test --all-features` passing
2. Build Rust module alongside C++ module using a temporary name (e.g.
   `_managym_rust`) to avoid import collision during parity bring-up
3. Add `pyproject.toml`, install via maturin: `pip install -e managym`
4. Run existing Python tests — they import `managym` and should work
5. Run parity traces — validate identical behavior
6. Once all green, remove C++ source and CMakeLists.txt

## Risks

- **PyO3 link environment gap:** The existing seam fails to link in some
  environments (missing `Py*` symbols). First task is resolving this.
- **Observation JSON fidelity:** Current `to_json()` is debug-string based.
  Need proper field-level serialization matching C++ output.
- **Combat/priority complexity:** Step/priority transitions are the highest-risk
  area for subtle rules drift as the Python surface exposes more engine state.
- **PyO3 GIL overhead:** Need to ensure the Python↔Rust boundary isn't slower
  than pybind11.

## Done when

```bash
# Rust engine tests
cargo test

# Python integration tests (against Rust backend)
pip install -e managym && pytest tests/env/ tests/agent/

# Parity check (optional tier 1)
pytest tests/parity/

# Cutover completion
# - Rust/PyO3 path is the default and only maintained backend
# - C++ managym runtime path removed
# - CMakeLists.txt/C++ Python module build path removed
```
