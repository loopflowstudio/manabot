# rust-migration: design exercise

Architectural analysis for migrating managym from C++ to Rust.
See `wave/rust-migration/` for the staged implementation plan.

## Analysis summary

### Keep from C++
- Dependency DAG: agent → flow → state → infra
- Env as thin API (reset/step)
- Skip-trivial tick optimization
- Fixed-size observation with validity masks
- Hierarchical profiling with RAII/Drop scopes
- Turn/phase/step hierarchy (structure, not mechanism)

### MtG rules that dictate architecture
- Priority is a protocol (yield at arbitrary points within steps)
- Stack creates interleaving (state interrupted mid-resolution)
- Zone transitions are the primary event bus
- SBAs run at priority check, can cascade (fixpoint loop)
- Summoning sickness is per-permanent (Permanent wraps Card)
- Mana is typed and fleeting (empties between steps)
- Information asymmetry (hand is private)

### Rust mappings
- `unique_ptr<T>` → owned values or `Box<T>`
- `T*` raw pointers → typed indices (`CardId(usize)`)
- Virtual dispatch → enums with `match`
- `std::optional<T>` → `Option<T>`
- Exceptions → `Result<T, E>`
- `card_to_zone` map → field on Card or `HashMap<CardId, ZoneType>`
- RAII profiler scopes → `Drop` trait

### What changes from C++
- Raw pointer web → index-based arenas
- Virtual Step/Phase/Action → flat enums
- Mutable aliasing through pointers → `&mut GameState` passed explicitly
- `GameObject` base class → just an `id` field
- Nested ownership (`Game` → `Zones` → `Battlefield` → `Permanent`) → flat `GameState`

### v2 improvements
- Flat serializable/cloneable `GameState` (enables MCTS, replay, undo)
- Separate "what happened" from "what to do" (Action = data, execute = pure fn)
- Deterministic RNG from the start
- Compile-time card registration
