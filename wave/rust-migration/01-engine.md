# 01: Rust Engine

Port the entire game engine to Rust: state, flow, and infra layers. No Python
bindings yet — this stage is pure Rust with Rust tests.

## Finish line

`cargo test` passes with comprehensive tests covering the full game loop:
two-player games with basic lands + creatures, combat, priority passing, and
win/loss conditions.

## Architecture

### Crate structure

Single crate `managym` with modules mirroring the C++ dependency DAG:

```
managym/
├── Cargo.toml
├── src/
│   ├── lib.rs
│   ├── infra/          # logging, profiler, info_dict
│   │   ├── mod.rs
│   │   ├── log.rs
│   │   └── profiler.rs
│   ├── state/          # cards, mana, zones, players, permanents
│   │   ├── mod.rs
│   │   ├── card.rs
│   │   ├── mana.rs
│   │   ├── zone.rs
│   │   ├── player.rs
│   │   ├── permanent.rs
│   │   └── game_object.rs
│   ├── flow/           # turn system, priority, combat, game loop
│   │   ├── mod.rs
│   │   ├── game.rs
│   │   ├── turn.rs
│   │   ├── priority.rs
│   │   └── combat.rs
│   ├── agent/          # observation, action, action_space, env, behavior_tracker
│   │   ├── mod.rs
│   │   ├── observation.rs
│   │   ├── action.rs
│   │   ├── env.rs
│   │   └── behavior_tracker.rs
│   └── cardsets/       # card registry, basic lands, alpha creatures
│       ├── mod.rs
│       └── alpha.rs
```

### Core design decisions

**Index-based arena pattern.** All game objects identified by typed indices:

```rust
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct CardId(usize);

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct PermanentId(usize);

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct PlayerId(usize);
```

`GameState` owns all data in flat `Vec`s. No `Box<dyn T>`, no pointer graphs.

**Flat GameState:**

```rust
struct GameState {
    // Object storage
    cards: Vec<Card>,
    permanents: Vec<Permanent>,
    players: [Player; 2],

    // Zone tracking
    zones: ZoneManager,  // tracks CardId → ZoneType, per-player ordered lists
    stack: Vec<CardId>,

    // Turn state
    turn: TurnState,
    priority: PriorityState,
    combat: Option<CombatState>,

    // Caches
    mana_cache: [Option<Mana>; 2],

    // Infrastructure
    rng: StdRng,
    id_gen: IdGenerator,
}
```

**Enums for all variants:**

```rust
enum StepKind {
    Untap, Upkeep, Draw, Main,
    BeginningOfCombat, DeclareAttackers, DeclareBlockers,
    CombatDamage, EndOfCombat,
    End, Cleanup,
}

enum Action {
    PlayLand { card: CardId },
    CastSpell { card: CardId },
    PassPriority,
    DeclareAttacker { permanent: PermanentId, attack: bool },
    DeclareBlocker { blocker: PermanentId, attacker: Option<PermanentId> },
}

enum ActionSpaceKind {
    GameOver,
    Priority,
    DeclareAttacker,
    DeclareBlocker,
}
```

**Turn state as data, not objects:**

```rust
struct TurnState {
    active_player: PlayerId,
    turn_number: u32,
    lands_played: u32,
    current_phase: usize,    // index into PHASE_ORDER
    current_step: usize,     // index into current phase's steps
    step_initialized: bool,
    turn_based_actions_complete: bool,
}
```

Phase/step progression driven by `match` on `StepKind`, not virtual methods.

**Game loop:**

```rust
impl Game {
    fn step(&mut self, action: usize) -> StepResult { ... }

    fn tick(&mut self) -> bool {
        loop {
            let action_space = self.turn_tick();
            if self.is_game_over() { return true; }
            if let Some(space) = action_space {
                if !self.skip_trivial || space.actions.len() > 1 {
                    self.current_action_space = Some(space);
                    return false;
                }
                self.execute_action(&space.actions[0]);
            }
        }
    }
}
```

### What to port from C++

| C++ file | Rust module | Notes |
|---|---|---|
| `state/mana.h/cpp` | `state/mana.rs` | ManaCost, Mana, Color enum, parsing |
| `state/card.h/cpp` | `state/card.rs` | Card, CardTypes, CardType enum |
| `state/game_object.h` | `state/game_object.rs` | ObjectId, IdGenerator |
| `state/player.h/cpp` | `state/player.rs` | Player struct |
| `state/battlefield.h/cpp` | `state/permanent.rs` | Permanent struct (zone tracking in zone.rs) |
| `state/zones.h/cpp`, `zone.h` | `state/zone.rs` | ZoneManager, ZoneType enum |
| `state/stack.h` | folded into `zone.rs` | Stack is just a `Vec<CardId>` |
| `flow/game.h/cpp` | `flow/game.rs` | Game struct, tick loop, step execution |
| `flow/turn.h/cpp` | `flow/turn.rs` | TurnState, phase/step progression |
| `flow/priority.h/cpp` | `flow/priority.rs` | PriorityState, SBA, action generation |
| `flow/combat.h/cpp` | `flow/combat.rs` | CombatState, damage calculation |
| `agent/action.h/cpp` | `agent/action.rs` | Action enum, ActionSpace |
| `agent/observation.h/cpp` | `agent/observation.rs` | Observation struct, building |
| `agent/env.h/cpp` | `agent/env.rs` | Env (reset/step), no Python yet |
| `agent/behavior_tracker.h/cpp` | `agent/behavior_tracker.rs` | BehaviorTracker |
| `cardsets/*` | `cardsets/alpha.rs` | Card registry, basic lands, creatures |
| `infra/profiler.h/cpp` | `infra/profiler.rs` | Hierarchical profiler |
| `infra/log.h/cpp` | `infra/log.rs` | Thin wrapper over `tracing` or `log` |
| `infra/info_dict.h/cpp` | `infra/profiler.rs` | InfoDict folded into profiler |

### Tests

- **state tests:** Card creation, mana parsing/payment, zone transitions, permanent state
- **flow tests:** Full game loop with known seed, turn progression, priority passing, combat resolution
- **agent tests:** Observation building, action space generation, env reset/step
- **parity tests:** Record C++ game traces (seed + action sequence → observations), replay in Rust, compare field-by-field

### Dependencies

```toml
[dependencies]
rand = "0.8"
rand_chacha = "0.3"  # deterministic RNG

[dev-dependencies]
serde = { version = "1", features = ["derive"] }
serde_json = "1"
```

Minimal dependencies. No frameworks. `serde` only in dev for parity test serialization.

### Done when

```bash
cargo test
# All tests pass, including:
# - Full game simulation with basic lands + creatures
# - Combat with attackers/blockers/damage
# - Win by damage, win by library empty
# - Deterministic: same seed → same game
```
