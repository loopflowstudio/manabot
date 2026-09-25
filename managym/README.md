# managym

The deterministic Magic: The Gathering rules engine and search environment
behind Etude Fantasia. Rust, with PyO3 bindings consumed by `manabot` (the
agent) and `etude` (the experience server).

## Architecture

managym is Etude Fantasia's authoritative world: match execution, semantic
Commands, viewer Observations, deterministic replay, exact forks, and the
meaning and materialization of possible worlds. The cross-package contracts
and convergence status are in [docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md).

1. **`src/agent/`**: RL-facing API (`Env`, action spaces, observations)
2. **`src/flow/`**: Game progression (turns, priority, combat)
3. **`src/state/`**: Core game state (cards, players, zones, mana)
4. **`src/cardsets/`**: Card implementations
5. **`src/infra/`**: Logging and profiler infrastructure
6. **`src/python/`**: PyO3 bindings and Rust→Python conversions

Dependencies flow: python → agent → flow → state/infra.

The semantic kernel direction — typed card programs, exact object identity,
structured offers, proposed events, fork/rollback — is documented in
[docs/research/semantic-kernel.md](../docs/research/semantic-kernel.md), with
conformance fixtures under [conformance/](../conformance/).

Authored setups come from `SemanticPack::player_config` (Python:
`managym.authored_deck_setup(pack_key, deck_key)`). Carry that complete config
when swapping seats or rebuilding roots: deck and sideboard together identify
an authored setup. `PlayerConfig::new` intentionally gives custom setups an
empty sideboard; it does not infer one from card names or a matching main deck.
Sideboard rosters are fixed at setup, and remaining copies are derived from
absence from game zones. They are not part of the library or a new game zone.

Viewer observations carry remaining owner copies in `agent_sideboard`, using
private `candidate_id` values that join Learn action focus. These are outside-game
descriptors, not `CardData` with a fabricated zone or in-game `ObjectRef`.
Both players expose definition-only initial and remaining sideboard counts;
only the owner gets copy candidates. Python observation JSON and validation
use the native Rust implementations so new visibility fields cannot drift.

Public hand knowledge is a minimum multiset of `CardDefId` counts on `Player`,
not a set of revealed physical cards. Its per-definition changes have dedicated
undo entries; `journal_player` still captures only scalars. Exercise nested
choice rollback through normal Commands/branch-driver `apply`: transition
queues are journaled at that boundary, not by direct calls to resolution helpers.

## Build and test

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test
```

**CI runs `cargo test` in debug — validate in debug before landing.** The
engine guards its invariants with `debug_assert!`, which compiles out of
release entirely, so a test can pass green in `--release` and still fail CI.

After changing Rust, rebuild the Python extension into the uv-managed venv:

```bash
uv run --python 3.12 --extra play maturin develop --release \
  --manifest-path managym/Cargo.toml --features python
```

## Style

```rust
// filename.rs
// One-line purpose of file

use crate::flow::game::Game;
use crate::state::player::PlayerId;
```

Prefer explicit types and focused modules. Keep game behavior in enums +
`match` expressions instead of inheritance-like abstractions.

Bounded observation encoders reserve card rows after the owner's visible cards
for outside candidates. Their seven zone bits stay zero and the explicit outside
bit is set; action focus uses the same padded object table as the model. The
current default has 64 action rows. Card, permanent, action and focus capacity
excesses raise encoding errors; they must not be treated as a random-policy
fallback. Complete outside program bindings and public known-hand definition
counts still require the semantic input path during the Learn migration.
