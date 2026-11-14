# managym

The deterministic Magic: The Gathering rules engine and training environment
behind Manabot. Rust, with PyO3 bindings consumed by `manabot` (the agent).
Install the `managym` distribution for `import managym` without the training
stack. `manabot` depends on the exact same version and keeps `manabot.managym`
as a compatibility alias. Rust sources stay in this directory. Start with the
[Python installation example](https://github.com/loopflowstudio/manabot/blob/main/docs/engine-python.md).

## Play in the browser

The distribution includes Academy, a browser table for playing a game against
the engine's built-in opponents and stepping through recorded games:

```bash
uv venv --python 3.13
uv pip install --python .venv/bin/python --only-binary :all: 'managym[academy]'
uv run --no-project managym academy
```

This starts a local server and opens `http://127.0.0.1:8000` with a game
ready; Ctrl-C stops it. It needs CPython 3.12 or 3.13 and nothing else: no
manabot, Torch, Node or Rust. Finished and unfinished games are listed under
**Games** and open in **Replay**. `uv run --no-project python -m managym.academy` is the same
command, and `uv run --no-project managym academy --help` lists the port, host and storage
options. Installing `manabot` in the same environment adds trained opponents.
The [Academy guide](https://github.com/loopflowstudio/manabot/blob/main/docs/academy.md#running-academy)
covers playing, replays, trained bots and launch troubleshooting.

### Start a game with your chosen decks

After launching Academy, open **Choose decks**. Browse a bundled deck and
select **Use for me** or **Use for opponent** to assign each seat, choose an
opponent such as **Search 64**, then click **New Game**.

For your own lists, expand **Import decklist files** and check **Use imported
decks for the next game**. Upload or paste both decks, one `count card name`
per line, with optional `Main` and `Sideboard` headings. Click **Check decklists**,
fix any reported issues, and select **New Game**. Imported decks use the
development format and supported cards; choose a built-in opponent, since
checkpoint opponents do not accept imported decks. See the
[play guide](https://github.com/loopflowstudio/manabot/blob/main/docs/academy.md#play-a-game).

### Watch a replay from an evaluation

Evaluation trace import needs the optional manabot integration, installed in
the same environment and at the same release version as managym:

```bash
uv pip install --python .venv/bin/python --only-binary :all: --torch-backend cpu manabot
uv run --no-project managym academy
```

Open **Watch** at `http://127.0.0.1:8000/replay?watch=1`. Under **Open arena /
Tournament game**, select the evaluation's retained command trace, usually
`traces/<cell-id>.commands.jsonl.gz` in its output directory. Download the
artifact first if the evaluation ran remotely. Select a **Recorded game** and
**Open selected game** when the file contains more than one game, then use
**Step decision**, **Previous decision** or **Play**. Import verifies the
recorded actions and state digests without running the original players or
loading their checkpoints. Incompatible historical worlds need their original
revision. The [evaluation replay guide](https://github.com/loopflowstudio/manabot/blob/main/docs/academy.md#watch-a-game-from-an-evaluation)
covers supported files and troubleshooting.

For building a deck before play, see [Limited deck building](../docs/deck-building.md):
`CardPool` holds the cards available, `format.Limited` holds the rules, and a
build started from both reports the copies still available and when stopping
is legal. The same rules read and write NumPy arrays over the card axis for
batched builders.
Format rules live in `src/format/` (`format::limited`); they build on the
definitions, decklists and world admission in `src/cards/`.

[Engine development principles](PRINCIPLES.md) records the lessons from
Allies–Lessons, Elemental and TLA: shared rule ownership, preserved choices and
timing, trustworthy learning evidence, and measured runtime/search performance.

## Architecture

managym is Manabot's authoritative world: match execution, semantic
Commands, viewer Observations, deterministic replay, exact forks, and the
meaning and materialization of possible worlds. The cross-package contracts
are in the [README](../README.md#contracts-the-code-enforces).

1. **`src/agent/`**: RL-facing API (`Env`, action spaces, observations)
2. **`src/flow/`**: Game progression (turns, priority, combat)
3. **`state/`**: the `managym-state` crate — rules vocabulary and game-state
   types (cards, zones, mana, effects), re-exported here as `managym::state`;
   **`src/state/`** adds players, decks and the canonical match-state hash
4. **`cards/`**: the `managym-cards` crate — card set modules (one per printed set)
   and their tests, reached by the engine only through `managym_cards::catalog()` and `token_catalog()`;
   **`src/cards/`**: `ContentPack`, rules fixtures, authored setups, pool support and learning tokens
5. **`src/format/`**: Format construction rules and card selection;
   `limited.rs` adds Limited pool counts and deck size rules to world admission
6. **`src/infra/`**: Logging and profiler infrastructure
7. **`src/python/`**: PyO3 bindings and Rust→Python conversions

Each card is a function in the file of its first printing's set under
`cards/src/`, returning the engine's own `CardDefinition` (key, Scryfall id and
`PrintedCharacteristics`) in its rules types (`Effect`, `Ability`, `TargetSpec`,
…). Tokens are not cards: each is a `TokenDefinition` in `cards/src/tokens.rs`
with the same `PrintedCharacteristics`, keyed `token.*` and created by that
key, with a Scryfall id unless it has no printing. The engine registers both
as `ObjectDefinition` rows with token status. In a game every object has a kind
(`ObjectKind::Card`, `Token` or `SpellCopy`): a token is never a card, and
each zone holds only the objects the Comprehensive Rules allow it
([objects and zones](../docs/engine.md#objects-and-zones)). The engine runs these values
directly, and
the learning catalog (`src/cards/tokens.rs`, `managym.catalog_tokens`)
derives policy tokens from it. `src/world.rs` owns `WORLD_VERSION`, the card/deck
support check (defined cards, excluding tokens and rules fixtures) and the set
filter. Known rules bugs are tracked as Tasks without revoking playability. Keep
decklist data separate from card behavior; see
[the content contract](../content/semantic/README.md).

Dependencies flow: python → agent → flow → state/infra, and across crates
`managym` → `managym-cards` → `managym-state`. `cargo test` in this directory
runs all three (the workspace's default members).

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

The policy input factors resolved `EffectRecords` through `agent/shared_effects.rs`
before admission: complete equal payloads share one header and typed detail
slice, with one explicit membership per original record. Controller, exact
source incarnation, timestamps, fixed affected sets and duplicate edges survive.
Public object IDs are joins; private object storage IDs (`ObjectId`) do not enter this view.
Native scalar/vector and Python encoders mirror bounded arrays of 128 × 18,
4096 × 3 and 2048 × 2; every capacity fails explicitly. Local facts exclude
active effects.
Projection never mutates or semantically coalesces the engine’s records.

Catalog naming uses `Decision::NameCard` over a shared immutable `NameVocabulary`.
The catalog builds sorted real names once, including back faces and excluding
tokens and rules fixtures. Decisions/forks share that allocation; canonical state
records its digest. A decision holds a `NameScope`; a scope over
`NAME_BRANCHING` (64) names offers a deterministic balanced partition into
contiguous groups, and choosing one only narrows the scope. Selecting a name
constructs one `ExileNamed` effect, then enters the existing private search
continuation. Public names and optional learning definition references are not
in-game card objects or deck admission. The policy input carries name joins and
group spans in `name_candidates[A,4]` at ordinary action capacity.
See [policy inputs](../docs/observation-and-model.md#catalog-naming-input).

Spell-level modes are announced before targets and payment. Independent target
clauses use `Effect::TargetClause`; effects within a clause share its targets.
Repeated specs in separate clauses may choose the same object. A resolving
frame retains its active clause and each chosen slot's resolution-start eligibility
across suspended decisions (CR 608.2b). Effects use those retained results; live
presence checks separately prevent following departed objects (CR 400.7).
Illegal slots remain positional, and zero optional targets still resolve.
`ObjectSelector::Targets` applies one verb to all eligible slots of that clause
in one event batch. `ForEachTarget` instead runs a separate child program per
slot, retaining its position across choices. Leaves from the Vine's simultaneous
counter placement uses the former.
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
independently of what was captured. Viewer-safe relations and text expose
the remembered references, expiry and frequency. Relation row kind 9 carries
frequency (column 28: 1 once, 2 each match); column 48 keeps its expiry meaning.
Missing historical frequency is unknown.

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
[semantic-kernel-w17-v2](../conformance/semantic-kernel-w17-v2/).

After changing Rust, rebuild the Python extension into the uv-managed venv:

```bash
uv run --extra play maturin develop --release \
  --manifest-path managym/Cargo.toml --features python
```

Run this from the repository root with its CPython 3.13 development environment.
For a 3.12 compatibility check, use a separate environment and rebuild for that
interpreter; a `cp313` extension cannot be reused by `cp312`. Follow
[AGENTS.md](../AGENTS.md#commands) when replacing an already loaded extension.

The [pinned Comprehensive Rules audit](https://github.com/loopflowstudio/manabot/blob/cac6fa27d/docs/rules/audit-2026-10/README.md)
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
[ownership review](https://github.com/loopflowstudio/manabot/blob/cac6fa27d/docs/rules/audit-2026-10/unification.md)
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

The optional [policy-history v1 projection](../docs/observation-and-model.md#policy-inputs) is a bounded
derived suffix updated at the native event append. Observation reads share it;
clone and diagnostic injection must preserve its documented lifecycle.

Every match, authored or not, runs on the world's one catalog
(`cards::content_pack::default_content_pack`): every set-file definition in name
order, so authored setups are decklists over the world pool, not packs.
`PlayerConfig.submitted` (Python: `world="world"`, `managym.WORLD_SETUP`, set by
`Deck.player()`) submits a deck to the installed world: it must pass
`development-v1` admission against the world's supported cards (see [decklist
import](../docs/engine.md#decks)) or reset fails. Fixed setups and tests leave
it unset and run as given without submitted-deck construction checks.
Any other `world` value is rejected: authored content packs are retired.

Mana feasibility and production share typed costs and pools. `ManaClass`
retains each class's spending restriction and spend effect; `SpendingContext`
distinguishes casting, activation and resolving costs. A spell's payment records
`ManaSpent`, carried to resolution and the entering permanent's own triggers.
Spending is a native unit/class decision whenever receipts or remaining pools
differ, including generic colors, restrictions, riders and combat expiry. Each
offer can complete every saved payment. Partial spending, nested activation
fees and their saved caller survive exact forks and Command replay; the spell's
receipt excludes those fees. Academy and policy inputs expose the same choices.
Chosen-color mana abilities pay their activation costs before their
immediate output decision. `Game::characteristics` derives battlefield grants;
those granted mana abilities never become activated stack objects. Waterbend
taps are cost payment, not tapping for mana. The view is the one reader of a
battlefield permanent's characteristics, legendary and ward included: the
legend rule reads its `is_legendary`, and every `CardData` field that states a
characteristic reads it there (`tests/cards/observability.rs` classifies each
field as layered or printed by definition).

The cached raw layer board tracks discovered reads of mana pools and tapped
state, including nested predicates and granted statics. Independent pool/tap
writes retain the board and fixed activations; dynamic activation outputs and
costs refresh separately. Trigger kinds follow raw layer dependencies. Immutable
scans use `layer_views` (or `characteristics_many` when activation finalization
is needed), preserving caller order and validating the shared board once in
debug. Singular reads retain their cache checks. Opt-in `benchmark`
instrumentation attributes board, clone and dependency work without adding
counters to ordinary wheels.

A resolving spell remains on the stack until the zone-move authority commits
its departure. That authority applies flashback exile replacement and captures
cast provenance before removing the stack object. Do not remove spells early in
countering or finalization. Copied tokens detach immutable characteristics from
the source definition, retain its semantic identity, and expose their actual
legendary status in observations; canonical hashing includes their copied data.

Resolving optional costs use `Effect::OptionalPayment` and `PaymentProgress`.
Their [ownership and authoring contract](../docs/engine.md#resolution-contracts)
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

Most payment questions never reach the search. With four untapped Islands and
an empty pool, "can this player pay `{2}{U}`?" is answered by adding: the pool
plus every source's output either covers the cost or does not, and the first
witness taps sources in source order until the pool pays, useful or not. That
direct answer applies when the question is one mana cost and no available
source has a mana ability beyond tapping itself for one fixed output: no color
choice, upfront cost, life payment, counted tap, sacrifice or limited use.
Restricted fixed outputs (CR 106.6) stay in the class, counted only toward the
purposes they may pay. A waterbend cost stays in it too: every source worth
mana is tapped for mana, artifacts and creatures worth none cover the generic
that taps may pay, and each candidate tap of one waterbend decision is judged
against one model instead of a new search. Automatic production never
activates abilities, so its single cost is direct on any board.

Legal actions computed for one decision build each player's sources once
(`PaymentScan`). Those sources are a function of the game alone, which cannot
change while the computation borrows it; the pending payment's reservations, a
sacrifice under consideration and payable life are applied per question. Debug
builds rebuild the sources on every reuse and assert they match.

The direct answer is exact or it is not used. Modal lands, filters, mana
abilities with other costs, counted taps, nested activations and the automatic
prefix of a nested activation go to the search described below, which stays
the one definition of a legal payment: debug builds run it beside every direct
answer and assert the same affordability and the same first witness, so
automatic tapping order cannot differ between the two. Adding a kind of mana
source needs no change here unless it should join the direct class, which
requires a test comparing both paths on that source.

Payment search tracks an object's untapped resource, presence and limited
ability uses separately. Counted tap costs select distinct untapped matching
permanents through ordinary SelectCard decisions; The Cabbage Merchant can
activate twice by tapping four distinct Foods without tapping itself. Each
selected tap commits before the next selection and is unavailable to nested
costs or waterbending. Pending tap
counts use the continuation mode field; source and ability retain their usual
joins. Single-object costs keep their serialized definition shape.
Every registered mana ability must consume a finite resource. Search bounds
count repeatable outputs optimistically; every accepted payment still requires
an identity-preserving witness with explicit tap choices. Every saved counted-tap
cost must retain enough present, untapped candidates throughout the search;
production and spending cannot restore a consumed resource. The whole-payment
bound counts only outputs a pending cost could spend and takes the maximum of
competing uses of one tap; the color-set bound applies the same supply to every
set of owed colors, so one "U or R" source cannot cover `{U}{R}`. Search
memoization shares feasibility across interchangeable sources (including
membership in every counted-tap set), while retaining source order and exact
identities in the first witness. Repeated lands therefore contribute resource
counts instead of all identity subsets. A search checks the color-set bound at
its root and keeps identity keys until it has expanded 16 states; only then
does it classify the sources and apply both to every state, so ordinary small
payments pay nothing for them. Distinct
source behaviors, overlapping tap sets, nested obligations and concrete pool
remainders can still make exact search combinatorial; there is no general
large-board work limit or truncation of legal payments. The
[stress states](../docs/development.md#stress-states) measure payment and every
other engine layer on large duplicative and large diverse boards; run them when
changing this search.

Spending restrictions (CR 106.6) are typed: `ManaRestriction::Only` holds
`SpendPredicate`s naming any subtype a spell must have or an activation's
source must have, and `ExceptNonartifactSpells` is a Powerstone's exclusion.
Every cost carries a `SpendingContext`: `Cast` with the spell's announced
subtypes and types, `Activate` with the source's zone and subtypes, or
`Resolving`, which only exclusions admit. Observation data names each
restriction whole. The policy input's restricted-mana columns name exactly
Lesson and Shrine spells and Ally spells or sources (and, in v10, Powerstone
mana); every encoder refuses other restricted mana (`unrepresentable_state`)
rather than drop it.

`Env.capture_decision(viewer, history_capacity=0)` supplies atomic decision capture
v5: the schema-8 semantic Observation, shared native continuation frames, bound public card
characteristics, visible object joins, world/content identity and optional public
history v1 (0 disables it; 1..32 admits a chronological suffix). Its SHA-256 covers
all fields except the digest itself using recursively key-sorted compact JSON.
`DECISION_CAPTURE_VERSION` owns this aggregate's inclusion/order/digest contract;
`SEMANTIC_DECISION_VERSION` owns embedded schema 8. Historical captures retain
their original schema; w14 deliberately changes the current action contract.

The opaque native capture binds an Env and reset generation. Use
`execute_captured_command_json(capture, command_json)` for atomic origin and
current-digest validation plus ordinary semantic execution. Portable capture JSON
is evidence, not execution authority; clones need their own live capture.

Frozen viewer JSON omits continuation frames. To reconstruct a complete
stateless policy observation, use
`Observation.from_json(viewer_json, continuation_json=frames_json)`; the frames
pass through the same canonical validator as retained decision captures.
Neither input creates an executable decision binding.


The policy input has configurable fact capacity (512 by default). It keeps
trigger-placement action joins and retained
resolution-start eligibility in native current stack/continuation relations, and
widens player and card rows for cast permissions, the hand-only lock, restricted
mana by spending subtype and announced X. Its
layout, checkpoint admission and measurement commands are documented in
[the observation input contract](../docs/observation-and-model.md#policy-inputs).
The presentation context and text capture derive from the same typed frames;
portable schema-8 viewer JSON is validated by the native observation types.

`Game::object_power` reads contextual power in every zone: the layered view on
the battlefield, and characteristic-defining abilities with the spell's
controller or card's owner elsewhere (including outside the game). Zone
filters use `object_matches_predicate`; printed-only matching belongs to
structural definition checks and private layer stages. `LookAndSelect` names
whether the remainder is random; chosen order suspends on native
`OrderLibrary`, placing cards first-to-draw first through ordinary SelectCard
Commands. Bottom ordering exposes the remaining inspected cards; Surveil uses
the same owner to order kept cards on top, retaining the chosen prefix privately
until ordering finishes. Top ordering has obligation code 27 (mode is the
committed prefix length); code 23 retains its bottom-order meaning.

`EffectFrame::completion` owns the return destination: the interrupted scope,
a retained caller decision, a completed-operation event, root spell cleanup,
or root priority. Completion events fire before the caller's suffix; Scry and
Surveil wait for every partition and ordering choice. `continue_resolution` runs
any pending batch before exposing the next decision. An `EventBatch` on the active
frame retains every proposal, applied replacement identity, pre-departure facts
and result until replacement choices and consequent linked returns finish. The
affected player chooses each replacement; applicability is recomputed after every
transformation. No member
commits until the complete batch is ready. Native obligation code 29 names this
choice, with source-specific labels and focus.

Mana activation sacrifice effects retain their output and parent payment there
after upfront mana, tap and life costs commit. Returning from a linked-entry
replacement restores the saved decision without paying costs again or opening
priority. Combat, SBA departures, spell finalization, activated costs, token
entry and Surveil movement use this owner. SBA progress distinguishes stable,
changed and pending; a pending turn action cannot enter priority stabilization.
Immediate fixture/arithmetic helpers require completion without a question and
fail explicitly if used for a choice-bearing event. Sacrifice facts match their
pre-move subjects after replacement choices and before any batch member leaves;
trigger placement still waits for resolution. The policy input projects the
retained program, results, completion and proposals through fact-kind 14 as bare
name/value facts, with viewer-safe reference slots in relation kind 10; a frame
that retains nothing emits no occurrence. Academy's pending-resolution summary
and LLM text are rendered by the same walk only through `describe_effect_execution`
(decision context and capture), never by a policy observation.

`SelectCards` snapshots eligible exact objects from its selector, retains min/max
cardinality across choices, and runs the chosen effects in a child scope with
`ObjectSelector::Selected`. `CardsThisWay` names the scope's latest mill/discard
result independently of a later `Move`; the caller keeps those results while
selection effects run. Mill, optional discard/draw and basic/tutor searches use
ordinary `MillCards`, `If`, `SelectCards`, `Move` and `ShuffleLibrary` composition.
`ObjectSelector::Zone` records source-zone/predicate provenance; both hidden-world
materializers refresh unknown selected subsets throughout saved callers and child
scopes, preserving cardinality and exact viewer-known choices. Restricted partial
selections exclude incompatible count worlds. Foretold assignments use the same
subset constraints: exact weights count only compatible ordered fills, and the
materializer samples that conditional measure. Commitments are captured before
zone reassignment, so a sampled face-down slot cannot erase an earlier selection.
Frozen count-only belief samplers reject these additional constraints before
inference. `ObjectBinding` carries exact incarnation, captured zone/predicate and
per-seat exact knowledge through waiting, queued, placing, stack and resolving
trigger lifetimes. Both samplers use one assignment across those captures and
active selections; known exact captures remain fixed, and departed incarnations
never follow a later object. Capturing `Moved` uses its new zone. This binding
provenance is canonical state, independent of viewer-safe reference projection. Library inspection exposes membership in stable definition/object order,
never draw order. Prospective battlefield entrants are public during replacement
choices and remain pinned in both samplers.

`Effect::Move.library_end` states top/bottom placement without a shuffle; its
placement commits before linked returns can pause. `put_in_library` authors that
operation, and `ShuffleLibrary` can follow after all child work. Ordinary `move_to`
retains shuffle-into-library behavior. Shared selection has obligation code 28;
retired codes 5, 6 and 7 keep their historical meanings.

`SuspendedResolution::continuation` walks retained frames and decisions root first;
observation projection and hidden-world sampling share that chain for caller
ancestry and inspection permissions.

### Offline payment-shape measurement

After `Env.reset`, `capture_payment_shapes(cap)` enables an offline diagnostic
outside policy inputs and rules state. Replay the game, then call
`take_payment_shapes_json()` once to drain its rows. A reset disables capture.
The cap (2–65,536) bounds source-subset and receipt traversal work per payment;
`cap_hit` marks incomplete lower bounds. The diagnostic enumerates fixed-source
payments using the existing payment model and native unit spending. It records
untapped-source capabilities, floating pools, receipts, exact source identities
and today's offered spend steps separately. It also counts capability and exact
identity outcomes over non-redundant source sets: no selected source's whole
output can be removed while keeping the current cost pooled and saved callers
feasible on the post-tap board (the over-tap counter's criterion). Unavoidable
surplus from an indivisible source remains. All counts share the same traversal
and cap; a cap hit makes every count a lower bound, including non-redundant
counts. Saved payment obligations must stay
feasible. Activated mana abilities are counted as present but are not enumerated.

`MANAGYM_EXPERIMENT_AUTOSPEND=1` is MTG-347's temporary comparison arm. It selects
the first feasible pooled unit and preserves automatic source order, saved costs
and receipts. The experiment's job command records the switch; replay and
heuristic evaluation must use the same value. Ordinary runs leave it unset.
This switch is experiment scaffolding, pending the report's design review.

`MANAGYM_EXPERIMENT_SOURCE_TAPS=1` is the per-source arm of
`experiments/source_taps.py`. A cast or activation mana payment then offers
each fixed source whose output the cost can still use as a `TapForMana` step
(action type `TapForCost`) beside `PayCost`, which finishes automatically.
Nothing is offered when the pool covers the cost or every offered source is
needed. Waterbend, resolution payments, upfront fees and foretell stay
automatic. Replay and evaluation must use the same value.
