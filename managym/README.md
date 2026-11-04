# managym

The deterministic Magic: The Gathering rules engine and training environment
behind Manabot. Rust, with PyO3 bindings consumed by `manabot` (the agent)
and `academy` (the optional play UI). The Python API is `manabot.managym`,
bundled with the `manabot` distribution; `import managym` remains a compatibility
alias. Rust sources stay in this directory.

[Engine development principles](PRINCIPLES.md) records the lessons from
Allies–Lessons, Elemental and TLA: shared rule ownership, preserved choices and
timing, trustworthy learning evidence, and measured runtime/search performance.

## Architecture

managym is Manabot's authoritative world: match execution, semantic
Commands, viewer Observations, deterministic replay, exact forks, and the
meaning and materialization of possible worlds. The cross-package contracts
and convergence status are in [docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md).

1. **`src/agent/`**: RL-facing API (`Env`, action spaces, observations)
2. **`src/flow/`**: Game progression (turns, priority, combat)
3. **`state/`**: the `managym-state` crate — rules vocabulary and game-state
   types (cards, zones, mana, effects), re-exported here as `managym::state`;
   **`src/state/`** adds players, decks and the canonical match-state hash
4. **`cards/`**: the `managym-cards` crate — card set modules (one per printed set)
   and their tests, reached by the engine only through `managym_cards::catalog()`;
   **`src/cards/`**: `ContentPack`, rules fixtures, authored setups, pool support and learning tokens
5. **`src/infra/`**: Logging and profiler infrastructure
6. **`src/python/`**: PyO3 bindings and Rust→Python conversions

Each card is a function in the file of its first printing's set under
`cards/src/`, returning the engine's own `CardDefinition` in its rules types
(`Effect`, `Ability`, `TargetSpec`, …). The engine runs that value directly, and
the learning catalog (`src/cards/tokens.rs`, `managym.learning_catalog`)
derives policy tokens from it. `src/world.rs` owns `WORLD_VERSION`, the card/deck
support check (defined non-token cards, excluding rules fixtures) and the set
filter. Known rules bugs are tracked as Tasks without revoking playability. Keep
decklist data separate from card behavior; see
[the content contract](../content/semantic/README.md).

Dependencies flow: python → agent → flow → state/infra, and across crates
`managym` → `managym-cards` → `managym-state`. `cargo test` in this directory
runs all three (the workspace's default members).

[docs/research/semantic-kernel.md](../docs/research/semantic-kernel.md) is the
dated 2026-07-15 research record on typed card programs, exact object identity,
structured offers and proposed events; conformance fixtures live under
[conformance/](../conformance/). Its undo, journal and page-COW sections were
superseded by full-clone forking ([docs/ARCHITECTURE.md §10](../docs/ARCHITECTURE.md)
and the w2-182/198/199 benchmark reports under `experiments/`).

Authored setups come from `AuthoredSetup::player_config` (Python:
`managym.authored_deck_setup(setup, deck_key)`, naming the setup by preset or
by the `pack_key` frozen evidence records). Carry that complete config
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

Input v7 factors resolved `EffectRecords` through `agent/shared_effects.rs`
before admission: complete equal payloads share one header and typed detail
slice, with one explicit membership per original record. Controller, exact
source incarnation, timestamps, fixed affected sets and duplicate edges survive.
Public object IDs are joins; private card storage IDs do not enter this view.
Native scalar/vector and Python encoders mirror bounded arrays of 128 × 18,
512 × 3 and 2048 × 2; every capacity fails explicitly. V1–v6 still use their
frozen per-holder fact projection, while v7 local facts exclude active effects.
Projection never mutates or semantically coalesces the engine’s records.

Spell-level modes are announced before targets and payment. Independent target
clauses use `Effect::TargetClause`; effects within a clause share its targets.
Repeated specs in separate clauses may choose the same object. A resolving
frame retains its active clause and each chosen slot's resolution-start eligibility
across suspended decisions (CR 608.2b). Effects use those retained results; live
presence checks separately prevent following departed objects (CR 400.7).
Illegal slots remain positional, and zero optional targets still resolve.
Graveyard targets
carry an exact `ObjectRef`, so continuations cannot follow a physical card into
a later incarnation. Mutate permanents at their event authority;
source-relative effects require the original battlefield object.

Delayed abilities compose event matching, frequency, expiry, captured subjects
and effects. `CreateDelayedTrigger.event` only matches the event;
`DelayedFrequency::Once` consumes its watcher on the first match, even if the
occurrence is countered or later does nothing. `EachMatch` keeps waiting until
`DelayedExpiry::ThisTurn` removes the watcher at cleanup, or `Never` supplies no
stated expiry. Exact-object death watchers retire when the watched incarnation
leaves, regardless of frequency. Expiry removes waiting watchers, never queued
or stacked occurrences. No printed trigger vocabulary is migrated.
`CreateDelayedTrigger.remember` resolves an ordinary `ObjectSelector` when the
ability is created. Its exact incarnations travel through placement, stack and
suspended resolution; `ObjectSelector::Remembered` never follows a departed
object into a new incarnation. Event context (such as the spell just cast) stays
separate. End-step timing can wait for either player's step or the controller's,
independently of what was captured. Viewer-safe v5/v6 relations and text expose
the remembered references, expiry and frequency. Relation row kind 9 carries
frequency (column 28: 1 once, 2 each match); column 48 keeps its expiry meaning.
Missing historical frequency is unknown. Widths and existing meanings stay fixed; earlier input versions retain their exclusions.

Public hand knowledge is a minimum multiset of `CardDefId` counts on `Player`,
not a set of revealed physical cards. Search forks by full clone: clone the
`Game` before a normal Command `apply` and compare `search_state::witness` to
check that the source is unchanged. Exercise nested choices through Commands,
not by direct calls to resolution helpers.

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
current default has 512 action rows. Card, permanent, action and focus capacity
excesses raise encoding errors by default; they must not be treated as a random-policy
fallback. An explicit `action_overflow="first-legal"` policy retains the first
`max_actions` legal actions with their original indices and Commands. The TLA
pilot selects 64 slots and records replay-linked cutoff counts; the native and
Academy legal action sets remain complete. Other capacity excesses still fail.
Complete outside program bindings and public known-hand definition
counts still require the semantic input path during the Learn migration.

The optional [policy-history v1 projection](../docs/training.md#policy-inputs) is a bounded
derived suffix updated at the native event append. Observation reads share it;
clone and diagnostic injection must preserve its documented lifecycle.

Every match, authored or not, runs on the world's one catalog
(`cards::content_pack::default_content_pack`): every set-file definition in name
order, so authored setups are decklists over the world pool, not packs.
`PlayerConfig.submitted` (Python: `world="world"`, `managym.WORLD_SETUP`, set by
`Deck.player()`) submits a deck to the installed world: it must pass
`development-v1` admission against the world's supported cards (see [decklist
import](../docs/decklist-import.md)) or reset fails. Fixed setups and tests leave
it unset and run as given without submitted-deck construction checks.
Any other `world` value is rejected: authored content packs are retired.

Mana feasibility and production share typed costs and pools. `ManaRestriction`
contains positive casting/activation subtype predicates over an open vocabulary;
`SpendingContext` records the payment purpose and characteristic snapshot, plus
an activation's exact source and zone. Resolving payments do not inherit casting
permissions. Restricted classes retain their identity through nested payment:
the witness enumerates class remainders as well as colors, preserving a parent's
completion. The historical `lesson_mana` field remains an aggregate projection.
Chosen-output abilities commit their costs before producing mana. Waterbend and
selected-object tap costs tap without the mana event. The view is the one reader of a
battlefield permanent's characteristics, legendary and ward included: the
legend rule reads its `is_legendary`, and every `CardData` field that states a
characteristic reads it there (`tests/cards/observability.rs` classifies each
field as layered or printed by definition).

A resolving spell remains on the stack until the zone-move authority commits
its departure. That authority applies flashback exile replacement and captures
cast provenance before removing the stack object. Do not remove spells early in
countering or finalization. Copied tokens detach immutable characteristics from
the source definition, retain its semantic identity, and expose their actual
legendary status in observations; canonical hashing includes their copied data.

Resolving optional costs use `Effect::OptionalPayment` and `PaymentProgress`.
Their [ownership and authoring contract](../docs/rules/optional-payments.md)
covers both ordinary mana and waterbend without a card-specific decision.
`OptionalManaPaymentX` first chooses an affordable X, then uses the same payment
progress. Its `WhenYouDo` branch creates a distinct occurrence retaining X;
`TargetRequirement.total_mana_value_x` bounds the whole chosen clause, including
retargeting. Named searches retain native permission to inspect the searched
zones only during the instruction; semantic capture uses that permission too.

Mana offers, waterbend affordability and automatic production use executable
witnesses from `flow/mana.rs`. A mana source is reserved while its printed
upfront cost is pending; `Decision::ManaUpfront` retains the parent decision
and frame. Only after that obligation finishes do its tap/life/sacrifice costs
commit and its output become a visible color decision. Mana abilities do not
create stack objects. Fixed production keeps source order; a nested automatic
prefix considers concrete pool remainders and the parent's completion.

The resource model separates tapping, source departure, pending activation,
once-turn/exhaust usage, life, and chosen distinct tap-cost objects. Repeatable
abilities must consume bounded resources: selected-object taps, life, or strictly
more total mana than they produce. Source identity is retained across repeats;
a tapped source may still use a non-tapping ability. There is no iteration cap
or infinite-loop solver. The registry guard rejects definitions requiring an
unmodeled mana-ability cost (selected sacrifice/discard, crew, waterbend, or
activation outside the battlefield) until its resource transition is added.
This guard documents implementation coverage; it never removes playable cards.

Variable outputs stay as rules effects in `Characteristics`; payment evaluates
them after costs. Hypothetical resource changes use a fork and the ordinary
layered view for these outputs. Ordinary fixed sources do not clone the game.
Activation and selected-tap offer probes share one source inventory per
enumeration, with independent resource states. There is no cache across decisions:
mana, life, taps, usage, turn and pending payment all matter.
Performance remains subject to the same-runner Engine Budget.

Explicit input v8 preserves v7 object widths and shared effects. Relations have
50 columns (column 49 is the remaining selected-object mana tap cost); facts
add kinds 11/12/13 for restriction amounts, purpose/subtype predicates and their
pool/output location. Class handles preserve alternatives within each class;
output locations retain frame and mode. Native scalar/vector and Python buffers
select v8 by its relation width. V1–v7 remain frozen and reject new subtype or
pending tap-cost states they cannot represent. Fresh defaults remain v6.

`Env.capture_decision(viewer, history_capacity=0)` supplies atomic decision capture
v5: the schema-7 semantic Observation, shared native continuation frames, bound public card
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


Fresh policy input v6 has configurable fact capacity (512 by default), with
v5 frozen at 128 rows and identical column meanings. It keeps v4's trigger-placement action joins and retained
resolution-start eligibility in native current stack/continuation relations, and
widens player and card rows for cast permissions, the hand-only lock, restricted
mana by spending subtype and announced X. Its
layout, legacy checkpoint admission and measurement commands are documented in
[the observation input contract](../docs/training.md#policy-inputs).
The presentation context and text capture derive from the same typed frames;
portable schema-7 viewer JSON is validated by the native observation types.

`Game::card_power` reads contextual power in every zone: the layered view on
the battlefield, and characteristic-defining abilities with the spell's
controller or card's owner elsewhere (including outside the game). Zone
filters use `card_matches_predicate`; printed-only matching belongs to
structural definition checks and private layer stages. `LookAndSelect` names
whether the remainder is random; chosen order suspends on native
`OrderLibraryBottom`, placing cards nearest the top first through ordinary
SelectCard Commands. Only the chooser sees the remaining inspected cards.

### Bounded mana attribution

`cargo run --release --features benchmark --example bench_mana` measures ten
payment and offer samples on each declared small position. Opt-in counters name
explored, deduplicated and pruned states, pending obligation depth, concrete
remainders and dynamic output clones. Normal builds contain no counter writes.
`bench_steps [games]` and `bench_search [decisions-per-seed]` accept smaller
attribution samples; omitted arguments preserve the CI Engine Budget workloads.
See `research/mana-risk-2026-11/` for the before/after receipts. These bounded
samples do not replace the full same-runner CI budget.
