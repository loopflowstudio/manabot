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
preserved. manabot training code should work with zero changes — just swap the
import.

### Not here

- New card implementations beyond what C++ already has (5 basic lands, Llanowar
  Elves, Grey Ogre)
- Gameplay features the C++ doesn't have (instants, triggered abilities, etc.)
- Changes to the Python manabot code (env wrapper, observation encoder, agent)
- Performance optimization beyond "at least as fast as C++"

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

## Goals

1. **Exact behavioral parity** with C++ managym — same seeds produce same game
   traces
2. **PyO3 bindings** exposing identical `Env` API to Python
3. **Clean Rust architecture** using arenas, enums, and flat state
4. **Comprehensive tests** — unit tests for each module + parity tests against
   C++ engine
5. **Drop-in replacement** — `import managym` works with either backend

## Risks

- **Borrow checker friction in the game loop.** The tick/priority/step cycle
  mutates game state from many call sites. Index-based design mitigates this but
  requires discipline.
- **Observation field ordering.** Python encoder depends on exact field names and
  types. Any mismatch silently breaks training.
- **Mana payment determinism.** The greedy left-to-right tapping must produce
  identical results to C++ for same-seed parity.
- **PyO3 GIL overhead.** Need to ensure the Python↔Rust boundary isn't slower
  than pybind11.

## Metrics

- Parity test pass rate: 100% (same seed, same actions → same observations)
- Rust test coverage: every module has unit tests
- Throughput: ≥ C++ steps/second on same hardware
- Python integration: existing `tests/env/` and `tests/agent/` pass unmodified
