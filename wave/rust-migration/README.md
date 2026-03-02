# rust-migration

Rewrite the managym C++ game engine in Rust. Keep identical observable behavior
for the manabot Python training framework.

## Vision

A Rust managym that is faster, safer, and more extensible than the C++ version.
The architecture shifts from pointer-based OOP to index-based data-oriented design:
arenas instead of `unique_ptr` webs, enums instead of virtual dispatch, flat
`GameState` instead of nested ownership trees.

The Python-facing API (`Env.reset()`, `Env.step()`) stays identical. The
observation struct, action types, reward semantics, and enum values are all
preserved. manabot training code should work with zero changes after cutover.

### Not here

- New card implementations beyond what C++ already has (5 basic lands, Llanowar
  Elves, Grey Ogre)
- Gameplay features the C++ doesn't have (instants, triggered abilities, etc.)
- Changes to the Python manabot code (env wrapper, observation encoder, agent)
- Performance optimization beyond "at least as fast as C++"
- Long-term dual-backend maintenance (C++ + Rust)

## Principles

- **Arenas over pointers.** All game objects live in flat `Vec`s, referenced by
  typed indices (`CardId`, `PermanentId`, `PlayerId`). No `Box<dyn T>`, no
  pointer graphs, no borrow checker fights.
- **Enums over vtables.** Steps, phases, actions, zones — all enums with
  exhaustive `match`. Compiler catches missing cases. No virtual dispatch.
- **Flat state over nested ownership.** One `GameState` struct owns everything.
  No `unique_ptr` chains, no back-pointers. Functions take `&mut GameState`
  explicitly.
- **Data, not objects.** Turn/phase/step progression is state + match, not an
  object hierarchy with virtual `tick()` methods.
- **Semantic correctness as the required gate.** C++ trace replay is an optional
  migration debugging aid, not the long-term source of truth. Rust tests assert
  intended semantics.
- **Single-backend destination.** Rust is the only maintained backend by wave
  end. C++ runtime/backend path is removed once cutover gates are green.

## Risks

- **Borrow checker friction in the game loop.** The tick/priority/step cycle
  mutates game state from many call sites. Index-based design mitigates this but
  requires discipline.
- **Observation field ordering.** Python encoder depends on exact field names and
  types. Any mismatch silently breaks training.
- **Mana payment determinism.** The greedy left-to-right tapping must be stable
  and reproducible under seeded runs.
- **PyO3 GIL overhead.** Need to ensure the Python↔Rust boundary isn't slower
  than pybind11.

## Cutover gate (must be true before removing C++)

1. Rust engine + binding tests are green (`cargo test`, `pytest tests/env/ tests/agent/`)
2. Throughput is at least parity with current C++ baseline on agreed benchmark
3. Rust/PyO3 backend is the default `managym` runtime path
4. C++ runtime/backend path and CMake-based Python module build are removed
