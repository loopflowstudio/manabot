# managym

The deterministic Magic: The Gathering rules engine and training environment
behind Manabot. Rust, with PyO3 bindings consumed by `manabot` (the agent)
and `academy` (the optional play UI). The Python API is `manabot.managym`,
bundled with the `manabot` distribution; `import managym` remains a compatibility
alias. Rust sources stay in this directory.

## Architecture

managym is Manabot's authoritative world: match execution, semantic
Commands, viewer Observations, deterministic replay, exact forks, and the
meaning and materialization of possible worlds. The cross-package contracts
and convergence status are in [docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md).

1. **`src/agent/`**: RL-facing API (`Env`, action spaces, observations)
2. **`src/flow/`**: Game progression (turns, priority, combat)
3. **`src/state/`**: Core game state (cards, players, zones, mana)
4. **`src/cardsets/`**: Card set files (one per printed set); **`src/cards/`**: `ContentPack`, rules fixtures, authored setups, pool support and learning tokens
5. **`src/infra/`**: Logging and profiler infrastructure
6. **`src/python/`**: PyO3 bindings and Rust→Python conversions

Each card is a function in the file of its first printing's set under
`src/cardsets/`, returning the engine's own `CardDefinition` in its rules types
(`Effect`, `Ability`, `TargetSpec`, …). The engine runs that value directly, and
the learning catalog (`src/cards/tokens.rs`, `managym.learning_catalog`)
derives policy tokens from it. `src/world.rs` owns `WORLD_VERSION`, the card/deck
support check (defined and not marked incomplete) and the set filter. Keep
decklist data separate from card behavior; see
[the content contract](../content/semantic/README.md).

Dependencies flow: python → agent → flow → state/infra.

The semantic kernel direction — typed card programs, exact object identity,
structured offers, proposed events, fork/rollback — is documented in
[docs/research/semantic-kernel.md](../docs/research/semantic-kernel.md), with
conformance fixtures under [conformance/](../conformance/).

Authored setups come from `AuthoredPack::player_config` (Python:
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

Spell-level modes are announced before targets and payment. Independent target
clauses use `Effect::TargetClause`; effects within a clause share its targets.
Repeated specs in separate clauses may choose the same object. A resolving
frame retains its active clause and each chosen slot's resolution-start eligibility
across suspended decisions (CR 608.2b). Effects use those retained results; live
presence checks separately prevent following departed objects (CR 400.7).
Illegal slots remain positional, and zero optional targets still resolve.
Graveyard targets
carry an exact `ObjectRef`, so continuations cannot follow a physical card into
a later incarnation. Journal permanent mutations at their event authority;
source-relative effects require the original battlefield object.

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

New registered real cards also need their Scryfall shell and Oracle text in
`tests/fixtures/scryfall_cards.json`; `cargo test --test conformance_tests`
checks every registration against that snapshot. For an accepted rules-world
change, record a new versioned conformance corpus and update the CI/test root
without rewriting earlier receipts. The current corpus is
[semantic-kernel-w18-v1](../conformance/semantic-kernel-w18-v1/).

After changing Rust, rebuild the Python extension into the uv-managed venv:

```bash
uv run --python 3.12 --extra play maturin develop --release \
  --manifest-path managym/Cargo.toml --features python
```

The [pinned Comprehensive Rules audit](../docs/rules/audit-2026-10/README.md)
inventories every numbered entry, with explicit unknowns and scoped evidence.
Card-shell checks and replay agreement do not independently prove rules
correctness. Prefer directed tests with independently specified outcomes and
small bounded combinations before adding longer random-game cohorts.

Check the committed inventory offline from the repository root with
`uv run python scripts/audit_rules.py`. Supplying `--source <downloaded-txt>`
also checks the pinned byte digest and exact source inventory. Do not upgrade
child rules merely because their family has a test or a matching citation.
The Prism/Treasure audit probe now follows explicit nested payment decisions
and passes its final mana, sacrifice and empty-stack assertions. Its original
failure remains historical evidence; this bounded pass does not certify all
payment rules. See the audit's
[ownership review](../docs/rules/audit-2026-10/unification.md#refreshed-ownership-and-delivery-boundary)
before using its historical Task attribution.

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

The optional [policy-history v1 projection](../docs/recent-events.md) is a bounded
derived suffix updated at the native event append. Observation reads share it;
clone/undo and diagnostic injection must preserve its documented lifecycle.

`PlayerConfig.content_pack` can explicitly select the compiled authored runtime
for repeated-deck matches. Both seats must name the same pack and each complete
deck/sideboard must be one of its authored setups; unknown packs or partial
bindings fail reset. Omitting it preserves exact-pair automatic selection.
Python `MatchHypers.authored` carries this binding for same-deck matches so replay
cannot silently use generic card definitions.

Mana feasibility and production share typed costs and pools. `Mana.lesson`
retains Herbalist's restriction; `effective_spell_cost` supplies the Lesson
casting context, while resolving costs and activations remain unrestricted
contexts. Chosen-color mana abilities pay their activation costs before their
immediate output decision. `activated_abilities_for` derives battlefield grants;
those granted mana abilities never become activated stack objects. Waterbend
taps are cost payment, not tapping for mana.

A resolving spell remains on the stack until the zone-move authority commits
its departure. That authority applies flashback exile replacement and captures
cast provenance before removing the stack object. Do not remove spells early in
countering or finalization. Copied tokens detach immutable characteristics from
the source definition, retain its semantic identity, and expose their actual
legendary status in observations; canonical hashing includes their copied data.

Resolving optional costs use `Effect::OptionalPayment` and `PaymentProgress`.
Their [ownership and authoring contract](../docs/rules/optional-payments.md)
covers both ordinary mana and waterbend without a card-specific decision.

Mana offers, waterbend affordability and automatic production use executable
witnesses from `flow/mana.rs`. A mana source is reserved while its printed
upfront cost is pending; `Decision::ManaUpfront` retains the parent decision
and frame. Only after that obligation finishes do its tap/life/sacrifice costs
commit and its output become a visible color decision. Mana abilities do not
create stack objects. Fixed production keeps source order; a nested automatic
prefix considers concrete pool remainders and the parent's completion.

Every registered chosen-output mana mode consumes a tap (sometimes also a
sacrifice), so each source has one use during a payment. The registry contract
requires a model extension before introducing separate tap/sacrifice mana uses,
repeatable free sources, or additional mana-ability cost methods.
Search retains identities rather than grouping equal outputs. Its total and
spend-purpose bounds only reject impossible branches; every accepted payment
still requires a complete witness. This is finite search, without a general
large-board performance guarantee.

`Env.capture_decision(viewer, history_capacity=0)` supplies atomic decision capture
v4: the schema-7 semantic Observation, shared native continuation frames, bound public card
characteristics, visible object joins, world/content identity and optional public
history v1 (0 disables it; 1..32 admits a chronological suffix). Its SHA-256 covers
all fields except the digest itself using recursively key-sorted compact JSON.
`DECISION_CAPTURE_VERSION` owns this aggregate's inclusion/order/digest contract;
`SEMANTIC_DECISION_VERSION` owns embedded schema 7. Historical captures retain
their original schema; w14 deliberately changes the current action contract.

The opaque native capture binds an Env and reset generation. Use
`execute_captured_command_json(capture, command_json)` for atomic origin and
current-digest validation plus ordinary semantic execution. Portable capture JSON
is evidence, not execution authority; clones need their own live capture.


Fresh policy input v5 keeps v4's trigger-placement action joins and retained
resolution-start eligibility in native current stack/continuation relations, and
widens player and card rows for cast permissions, the hand-only lock, restricted
mana by spending subtype and announced X. Its
layout, legacy checkpoint admission and measurement commands are documented in
[the observation input contract](../docs/recent-events.md#current-continuation-input).
The presentation context and text capture derive from the same typed frames;
portable schema-7 viewer JSON is validated by the native observation types.
