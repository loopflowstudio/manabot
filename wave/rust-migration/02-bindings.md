# 02: PyO3 Bindings + Parity

Add Python bindings via PyO3 so `import managym` works identically to the C++
pybind11 version. Validate with parity tests and existing Python test suite.

## Finish line

The existing Python tests in `tests/env/` and `tests/agent/` pass against the
Rust backend with zero modifications to the Python test code.

## Architecture

### PyO3 module structure

```
managym/
├── src/
│   └── python/        # PyO3 binding layer
│       ├── mod.rs
│       └── convert.rs # InfoDict → PyDict conversion
├── Cargo.toml         # add pyo3 dependency
└── pyproject.toml     # maturin build config (replaces CMake for Python module)
```

### Binding surface

Mirror the C++ pybind.cpp exactly:

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
    m.add_class::<PyPlayerConfig>()?;
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
    m.add_class::<PyEnv>()?;

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

**InfoDict conversion** — recursive Rust HashMap → Python dict, matching
`convertInfoDict` / `convertInfoValue` from C++.

**validate() and toJSON()** — Observation methods preserved for debugging.

### Build system

Replace CMake Python module build with maturin:

```toml
# Cargo.toml additions
[lib]
name = "managym"
crate-type = ["cdylib", "rlib"]

[dependencies]
pyo3 = { version = "0.22", features = ["extension-module"] }
```

```toml
# pyproject.toml
[build-system]
requires = ["maturin>=1.0"]
build-backend = "maturin"

[tool.maturin]
module-name = "managym._managym"
```

### Parity testing strategy

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

### Migration path

1. Build Rust `_managym` module alongside C++ `_managym`
2. Install Rust version: `pip install -e managym` (maturin replaces CMake)
3. Run existing Python tests — they import `managym` and should work
4. Run parity traces — validate identical behavior
5. Once all green, remove C++ source and CMakeLists.txt

### Done when

```bash
# Rust engine tests
cargo test

# Python integration tests (against Rust backend)
pip install -e managym && pytest tests/env/ tests/agent/

# Parity check
pytest tests/parity/
```
