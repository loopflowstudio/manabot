# MTG-320: runtime objects and checked card ids

## Outcome and accepted scope

Tokens and spell copies are objects, never cards. Runtime design and examples:
`docs/engine.md#objects-and-zones`. Preserve observation/Python/PyO3 names,
serialized numeric ids, zone order, seeded deals, and frozen evidence. Stop at
review; the task steer says "Stop at review. Do not enable auto-merge."

The latest scope is steer f81d37a0 (Jack Heart, 2026-11-12), replacing both
"finish the job" and "land as is": "Typed card ids. Add a `CardId` that can only
name an object whose kind is Card. Converting an ObjectId to a CardId is checked once; converting a
CardId to an ObjectId is free. Library, hand, decklists/sideboards and effects
that say \"card\" hold or take CardId"; "`card`/`card_id`/`cards` always holds a
CardId (or a collection of them), and `object`/`object_id`/`objects` always holds
an ObjectId"; "`game.state.objects[*card]` must not survive unless `card` is a
CardId". Definitions, published observation fields and card-name strings are
not runtime-id bindings. The same steer says: "Not in #491 (I am filing a
separate follow-up Task): the physical Card store, a new object per zone change,
CR 400.7 carry-over records and their cache-invalidation measurements."

## Representation and boundaries

- `state/src/game_object.rs::CardId` has a private row index.
  `ObjectVec::card_id` checks kind once; conversion back is free and indexing
  asserts kind in debug. `GameState::card_id` exposes that boundary.
- `state/src/zone.rs::ZoneManager` stores library/hand as one ordered sequence
  of `CardZoneEntry::Card(CardId)` or `Transient(ObjectId)`. Every real card
  has a checked id; visitors retain their position until SBA. Both variants
  serialize as the original numeric id. No second membership/order authority.
- `ZoneObjects` borrows either entry or mixed-object slices. `iter()` projects
  all occupants through standard iterators; `cards()` reads checked entries or
  narrows mixed zones. `GameState::zone_cards` delegates to it. `reposition`
  and `swap` preserve entry kinds. Full-sequence shuffles preserve RNG calls.
  Setup and hidden reassignment certify entries against the object store.
- Deck/sideboard rosters and `ActivationSource::Card` carry checked ids.
  Card-only candidates, play, costs, graveyard targets and card-type counts
  narrow at their boundary; `flow/action.rs::chosen_card` owns command rejection.
- Generic Actions, selection transport and exact `ObjectRef`s stay object-based.
  Observations include visitors; card-only choices skip them. Preserve the
  existing tucked-token draw behavior. Predicate token filters match only
  `ObjectKind::Token`; observation `is_token` continues to mean non-card.
- Private projection bindings in `populate_cards`, `effect_execution`,
  continuation, history and `decision_context` name runtime ids as objects.
  Published CardData fields, text/learning labels and projection order stay.

The typed sequence replaces the earlier ordered-ObjectId fallback. Iteration
feedback: "The current Vec<ObjectId> fallback is not an accepted replacement
for that requirement; duplication concerns do not establish impossibility."

## Concept review (2026-11-12, headless)

No implement direction. Read against the steer: `CardId` construction is
private and checked; sampled `objects[card]` sites in `flow/` and `agent/` bind
`CardId`; a grep for `card: ObjectId` / `object: CardId` style bindings in
`managym/src` and `managym/state/src` found none. `docs/engine.md` claimed every
zone list "is typed for what can be in it"; that sentence now says what is built.

Kept on purpose, for the reviewer: graveyard and exile hold the same contents
as library and hand (cards plus transient non-cards) but store plain
`ObjectId`, so `ZoneObjects` has two backing shapes and `zone_cards` narrows
graveyard/exile per read. The Task brief quotes "plenty of places that should
hold cards ... like decks, graveyards" and "basically anything but stack and
battlefield"; steer f81d37a0, the latest scope, names "Library, hand,
decklists/sideboards and effects that say \"card\"". This change follows the
steer. Typing graveyard/exile entries would make the four card zones one shape
and remove `mixed_objects_mut`'s unreachable arm, at twice the entry width in
state that search clones; nobody has said to do it here.

## Delete — do not maintain

Completed: runtime `Card`/`CardVec`, definition-owned runtime token flags,
unchecked card-only consumers, library/hand `Vec<ObjectId>`, raw
`zone_objects_mut` mutation, and private mixed-id `card` projection bindings.
The compression pass also deletes the hand-written zone iterator, the
card-zone-only reader that panicked on mixed zones, duplicate scry top-ordering,
and the second reorder pass after hidden-world placement. No remaining deletion
target in this slice. Physical-card storage and new zone identity are excluded.

## Compression rationale

A zone view is a borrowed collection, not an iterator cursor. Standard slice
iterators preserve forward/reverse order and remove custom cursor state.
Card narrowing belongs to that view, which knows the stored representation;
`GameState` no longer branches through optional iterator chains. Hidden-world
inspection restoration places objects once, in reverse order, because placement
appends. Scry reuses `put_on_top_of_library`. Card-name witnesses and definitions
keep card names rather than acquiring runtime-object terminology.

## Validation and remaining work

Compression passed 31 focused debug tests, workspace Clippy, formatting,
whitespace checks, and the matching extension rebuild/import before the latest
main merge (receipts below).
Tests cover numeric zone bytes, checked scans, visitors, draw ordering, ordinary and
canonical shuffle RNG, retained inspections, scry/surveil, projections and
card-definition observability. The first compile found one multiline `for`
caller in `agent/env.rs` missing `.iter()` after the view cutover; it is fixed.

The typed-zone and projection work is checkpointed in c76f01f02. Head 96d3aa14e
then merged b531853a5, including main's 6ce618b8a payment fast paths and shared
`PaymentScan` sources (`flow/mana.rs`, `flow/action.rs`, `flow/decision.rs`).
Realignment found one newly merged test fixture still constructing
`ManaSource { card: CardId(index), .. }`; it now uses `object: ObjectId(index)`.
The earlier four fixture repairs remain intact. No production mismatch was
found in the inspected merge or projection/zone boundaries.

Gate still owns current-head affected suites/conformance, seeded before/after
QA, Engine Budget and authored reviews. The paired reference is main 6ce618b8a
(the remote main tip read during realignment), which includes both payment
optimizations; the MTG-298 comparison alone cannot establish compatibility.
The merged `.lf/skills/qa.md` adds authored setups plus TLA pool human builds
for waterbend/Treasure/any-color coverage, explicit extension-path confirmation
and matching retained errors. Keep both-viewer observations and full state
digests in this task's comparison. A clean Academy game to completion remains. The
historical table stall's cause is unknown; the payment-search merge is not proof
of a fix. Card-only filtering deliberately rejects transient tokens in
constructed choices; seeded comparisons must establish observation/state-byte
compatibility in real games.

Current-head CI 43375879983 failed its Rust job at `flow/mana.rs:2268` with
E0560/E0425 for that fixture (job 131722179161); semantic conformance, Python
unit tests, Academy integration and Clean-machine Play passed. Other jobs and
package run 43375879939 were still running at readback. They cover 96d3aa14e,
not the subsequent repair. Record final-head Engine Budget search/step numbers
and unchanged digests in the PR body.

Delivery: publish the fixture repair and refreshed PR draft through the caller's
delivery step, preserving its Task context block. Stop at review with auto-merge disabled.
No Wave is bound. No evidence was regenerated for the rename.

## Run receipts before this compression pass

These are dated observations, not validation of later edits. Full earlier
history: `df58f2bff:scratch/mtg-320-plan.md`; typed ids were checkpointed in
6baafcba3. Uncommitted notes before this curation were moved to
`/tmp/mtg320-compress-20261112/` (including complete commands).

- Before main merge: 53 focused debug tests passed in 1.732 s, JUnit
  `823626c6-c439-42d4-b354-a4b6b4911a47`. Workspace Clippy passed; cp313/arm64
  release build passed (4m), extension installed atomically, fresh import passed.
- Merge head 8e5e8e62b includes main ef28f3a58, payment-search pruning/memoization
  (#504) and package 0.2.3. Realign (2026-11-12) read PR #491: open, no recorded
  GitHub review/comment, auto-merge null. MTG-298/#476 merged at 458cf59ec on
  2026-11-13 00:05 UTC. CI 43369147820 failed compilation in four merged payment
  fixtures still using `card: CardId(...)`; production was unaffected. Semantic
  Kernel Conformance, Python Unit Tests, Academy Integration and Clean-machine
  Play passed. At realignment readback, Engine Budget and all Integration groups
  had passed; package run 43369147921 completed successfully. These remain
  evidence for 8e5e8e62b, not the later typed-zone work or payment merge.
- Fixture repair: `CARGO_BUILD_JOBS=2 cargo nextest run --manifest-path
  managym/Cargo.toml --lib --profile ci --test-threads 2 -E
  'test(flow::mana::tests)'`: 15 passed, 214 skipped, zero failures (4.430 s;
  build 5m 30s), JUnit `6f4f5a90-5edb-4068-9a31-f247968336ee`.
  All-target/all-feature Clippy passed (1m 58s); cp313 release build passed
  (7m 30s), matching extension installed atomically, fresh import passed.
- Typed-zone implementation: `CARGO_BUILD_JOBS=2 cargo nextest run
  --manifest-path managym/Cargo.toml --lib --test rules_tests --test cards
  --test possible_worlds_tests --profile ci --test-threads 2 -E
  'test(cr_400_identity) | test(effect_execution_inputs) | test(surveil_kept_order)
  | test(surveil_commits) | test(hidden_selection_library) | test(distinct_worlds_yield)
  | test(volcanic_torrent) | test(learn_retrieves_each)'`: 17 passed, zero failures
  (0.464 s; build 7m 10s), JUnit `4aa058b1-4678-4325-8eeb-a6052740c051`.
  Workspace Clippy with all targets/features and warnings denied passed
  (52.77 s). CPython 3.13/arm64 release wheel built (3m 21s), installed atomically,
  fresh `uv run --no-sync python` import passed. Rustfmt/whitespace checks passed.
  PR readback still open at remote 8e5e8e62b, auto-merge null; changes local.
- Historical CI/Engine Budget and seeded/browser QA are summarized with exact
  numbers in `scratch/pr-491-body.md`; the browser receipts are `scratch/qa.html`.
  The earlier Python failure attribution and host-load attribution were
  inferences, not reproduced causes.

## Compression validation (2026-11-12)

- `CARGO_BUILD_JOBS=2 cargo nextest run --manifest-path managym/Cargo.toml
  --lib --test rules_tests --test cards --test possible_worlds_tests --profile ci
  --test-threads 2 -E 'test(cr_400_identity) | test(effect_execution_inputs)
  | test(surveil_kept_order) | test(surveil_commits) | test(retained_inspection)
  | test(hidden_selection_library) | test(distinct_worlds_yield)
  | test(volcanic_torrent) | test(learn_retrieves_each) | test(scry)
  | test(every_static_and_cost_variant)'`: 31 passed, zero failures (0.381 s;
  build 4m 31s), JUnit `340c9cb6-ce05-49db-aac3-59ff3e17fde9`.
- `CARGO_BUILD_JOBS=2 PYO3_PYTHON="$PWD/.venv/bin/python" cargo clippy
  --manifest-path managym/Cargo.toml --workspace --all-targets --all-features
  -- -D warnings`: passed (31.72 s).
- `cargo fmt --manifest-path managym/Cargo.toml --all -- --check` and
  `git diff --check`: passed. Scratch is below both configured context limits.
- `CARGO_BUILD_JOBS=2 uv run --no-sync --extra play maturin build --release
  -i .venv/bin/python -m managym/Cargo.toml -o managym/target/wheels`: passed
  (7m 40s), cp313/arm64 0.2.3 wheel. Extracted the interpreter's exact
  `EXT_SUFFIX` member, installed with `os.replace`, and verified
  `uv run --no-sync python -c 'import managym._managym'` in a fresh process.
  These changes were subsequently checkpointed in c76f01f02 and published with
  merge head 96d3aa14e. PR #491 readback during realignment: open at that head,
  no GitHub reviews/comments, auto-merge null.

## Realignment validation (2026-11-12)

- The existing 31-test JUnit receipt was confirmed before running new tests.
  No zone/projection suite was repeated for this documentation reconciliation.
- `CARGO_BUILD_JOBS=2 cargo nextest run --manifest-path managym/Cargo.toml
  --lib --profile ci --test-threads 2 -E 'test(flow::mana::tests)'`: 19 passed,
  214 skipped, zero failures (1.791 s; build 2m 52s), JUnit
  `67eccb69-a1b8-4bda-bf3a-0924779ef581`. This includes the newly merged direct
  payment/search equivalence fixture and shared-source scan lifecycle test.
- Workspace all-target/all-feature Clippy with warnings denied passed (1m 05s),
  using the command above. Rustfmt and `git diff --check` passed.
- The release build command above rebuilt the cp313/arm64 0.2.3 wheel (1m 59s).
  Its exact `EXT_SUFFIX` member was installed with `os.replace`; a fresh
  `uv run --no-sync python -c 'import managym._managym'` succeeded.
  This rebuild includes the merged payment implementation.
- Scratch remains below both configured limits. The fixture, documentation and
  context edits are local; this pass did not publish or change auto-merge.

## Final-head QA (2026-11-13)

Seeded before/after comparison against main 6ce618b8a found a regression:
`Effect::SelectCards` kept only `ObjectKind::Card` candidates for every
selector, so Kyoshi Battle Fan never attached to its `CreatedThisWay` Ally
token (33 of 120 TLA pool games diverged; authored and Azula games matched).
Fixed in `flow/resolution.rs`: card-only narrowing applies to `Zone`,
`TargetPlayersZone` and `CardsThisWay` selectors; other selectors match any
object, as on main. The two existing Battle Fan card tests fail without the fix.
After it, 538 seeded games (108 authored, 30 Azula, 400 TLA pools; 118,781
decisions) are byte-identical, including 4 matching `PyAgentError` endings.
Two Academy games reached Game Over. Debug suite: 1195/1199 passed; the four
`vector_env` timeouts were load contention and pass alone. Clippy, fmt pass.
Receipts: `scratch/qa.html`. The fix and release extension are local, not
committed or published.

## Conversion cleanup (steer ced7607f, 2026-11-13)

Jack Heart, 2026-11-13: "yeah lets do that fairly exhaustively in 320. if there
are any super expensive cases we can skip them". Full steer:
`.lf/prompts/20261113-081526-LxK62h-steer.md`. Implemented locally.

Docs written first, describing intended behavior: `docs/engine.md`, the opening
example of [Objects and zones] and `#### Names` (parameter rule: any object
takes `impl Into<ObjectId>`; cards alone take
`CardId`; plain `ObjectId` stays for trait-object/fn-pointer signatures,
serialized fields, PyO3 bindings and measured hot paths). The example names
`zones.zone_of`, `Game::current_object_ref` and `play_land`; the first two
now accept either runtime id directly.

Implementation owes the PR body: explicit-conversion count before/after
(steer says about 81 sites on #491's head) and each skipped site with its
reason. Engine Budget digests unchanged; observations and evidence
byte-identical. Stop at review; no auto-merge.


### Conversion implementation (2026-11-13)

Object APIs, zone membership included, accept `impl Into<ObjectId>` and
convert where they first use the id. Definition counting accepts an
iterator of either id. Removed redundant caller conversions and widened locals
in land play, casting, foretell, card abilities, discard/exile costs and Lesson
eligibility. Card-only entry points and checked construction stay unchanged.
Delete — do not maintain: the concrete-only object parameter signatures and
redundant call-site widening are replaced; no alternate API or compatibility
path is added. Wire fields and mixed collections retain concrete ids.

An audited inventory, including inline and integration tests, counts 88 explicit
CardId-to-ObjectId expressions before this pass (66e8b7ae5) and 41 after.
It includes iterator maps, aliases, mixed comparisons and identity assertions;
it excludes EntityId conversions and `possible_worlds.rs` card-name strings.
The PR draft lists all retained sites and why they remain. No skip is claimed
on unmeasured performance grounds. Gate owns final Engine Budget/digests,
conformance and replay-byte comparison; earlier QA does not validate this pass.


### Focused validation of conversion cleanup

- `CARGO_BUILD_JOBS=2 cargo nextest run --manifest-path managym/Cargo.toml
  --lib --test rules_tests --test cards --test learn_setup_tests --profile ci
  --test-threads 2 -E 'test(cr_400_identity) | test(learn_retrieves_each) |
  test(learn_consumes_each) | test(foretell) | test(pop_quiz_learn_retrieves) |
  test(battle_fan)'`: 13 passed, 1088 skipped (0.637 s; build 10m 03s),
  run `ef20f456-65e7-4164-b089-69ffa963a143`.
- Compile iteration removed ambiguous `.into()` at a roster call and fixed
  the destination type of an EntityId local after object lookups became generic.
  Clippy found four redundant borrows in zone membership (two internal, two
  test callers); all are removed.

- `CARGO_BUILD_JOBS=2 PYO3_PYTHON="$PWD/.venv/bin/python" cargo clippy
  --manifest-path managym/Cargo.toml --workspace --all-targets --all-features
  -- -D warnings`: passed after those borrow fixes (17.88 s final invocation).
- `cargo fmt --manifest-path managym/Cargo.toml --all -- --check` and
  `git diff --check`: passed. Scratch stays under both configured limits.

- `CARGO_BUILD_JOBS=2 uv run --extra play maturin build --release
  -i .venv/bin/python -m managym/Cargo.toml -o managym/target/wheels`:
  cp313/arm64 0.2.3 wheel built successfully (6m 27s). The wheel's exact
  interpreter `EXT_SUFFIX` member was installed with `os.replace`.
  `uv run --no-sync python -c 'import managym._managym'`: passed in a fresh
  process, loading this checkout's `_managym.cpython-313-darwin.so`.

Implementation and documentation are local. No publication, Graph change,
auto-merge or frozen-evidence update was performed. Gate still owes final-head
conformance, paired observation/state-byte QA and Engine Budget with unchanged
digests, followed by the selected review steps.

### Compression of the conversion cleanup (2026-11-13)

One parameter convention: `ZoneObjects::contains` takes `impl Into<ObjectId>`
like every other object API, so the reflexive `AsRef<ObjectId> for ObjectId`
is deleted and the docs rule loses its borrow case (three test callers
dereference). `grants_naming` converts for `is_foretold`, `exile_grant` and
`exile_grant_on`; wrappers that only delegate (`Game::move_object`,
`move_object_with_entry`, `GameState::place_object`, `Game::object_reference`,
`Projection::object_id`, `can_pay_cast`, `latest_object_lki`) no longer convert
first. The retained-site count stays 41; the PR table's line numbers are current.

- Focused nextest command above plus `test(learn_tests) | test(engine_events)
  | test(azula_has_the_opponent) | test(dft::)`: 44 passed, 1057 skipped
  (1.242 s). Workspace all-target/all-feature Clippy with warnings denied,
  rustfmt and `git diff --check` passed.
- Release wheel rebuilt (cp313/arm64 0.2.3, 2m 33s), its `EXT_SUFFIX` member
  installed with `os.replace`; fresh `import managym._managym` loaded it.
