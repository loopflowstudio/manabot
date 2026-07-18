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
