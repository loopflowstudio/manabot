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
whitespace checks, and the matching extension rebuild/import (receipts below).
Tests cover numeric zone bytes, checked scans, visitors, draw ordering, ordinary and
canonical shuffle RNG, retained inspections, scry/surveil, projections and
card-definition observability. The first compile found one multiline `for`
caller in `agent/env.rs` missing `.iter()` after the view cutover; it is fixed.

Gate still owns affected suites/conformance, seeded final-head before/after QA,
Engine Budget and authored reviews. Paired QA must put both engines on the same
upstream payment implementation; the MTG-298 comparison alone cannot establish
compatibility after the merge. A clean Academy game to completion remains. The
historical table stall's cause is unknown; the payment-search merge is not proof
of a fix. Card-only filtering deliberately rejects transient tokens in
constructed choices; seeded comparisons must establish observation/state-byte
compatibility in real games.

Delivery: refresh CI and the PR draft, preserve its Task context block, publish
through the caller's delivery step, and stop at review with auto-merge disabled.
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
  Play passed. Engine Budget and part of package run 43369147921 were pending.
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
  Changes remain local; no publication, merge or auto-merge action was taken.
