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
steer. Typing graveyard/exile entries would make the four card zones one shape,
at twice the entry width in state that search clones; nobody has said to do it
here. `ZoneManager::store_mut` returns the zone's backing list as
`ZoneStore::Cards` or `Objects`, the one place that maps a zone to its shape
for mutation; it replaced `mixed_objects_mut` and its unreachable arm.

Re-read 2026-11-13 at 9c9124627 (after the 22ed4a733 merge), headless: same
result. Greps over `managym/src`, `managym/state/src`, tests and examples found
no `card`-named binding, field or `fn` return typed `ObjectId`, no
`object`-named `CardId`, and no `CardId(` construction outside
`state/src/game_object.rs`; the merged mana-spending code follows the naming
rule. No implement direction and no new open choice.

## Delete — do not maintain

Completed: runtime `Card`/`CardVec`, definition-owned runtime token flags,
unchecked card-only consumers, library/hand `Vec<ObjectId>`, raw
`zone_objects_mut` mutation, and private mixed-id `card` projection bindings.
Compression also deleted the hand-written zone iterator, the
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

## Remaining work

No implementation or deletion target remains in the accepted slice. Main is
merged through 1d7b5fde9 (#513 only; its `managym/` tree equals 22ed4a733 and
2c47f59ae), so the post-merge QA at 9c9124627 compared against the current
engine. CI run 43482226694 on merge c52d81b16 passed. The widened-local
cleanup in 4ef4a0829 (`flow/action.rs`, `flow/play.rs`, `state/src/zone.rs`)
followed that QA; CI run 43489082846 on it was in progress at the 2026-11-13
realign, and covers Clippy, which was not rerun locally after the `zone.rs`
edit. Gate owns, on the final head: that CI result, Engine Budget numbers with
unchanged digests in the PR body, and the selected reviews; a fresh seeded
comparison is needed only if the final head changes behavior beyond 4ef4a0829's
local-binding edits. Delivery refreshes the PR body from
`scratch/pr-491-body.md`, keeping its Task context block, and stops at review
with auto-merge disabled. No Wave is bound. No evidence was regenerated.

Open: the historical Academy table stall's cause is unknown; the payment-search
merge is not proof of a fix. The earlier Python failure attribution and
host-load attribution were inferences, not reproduced causes.

## What runs showed

Dated observations, each valid only for the code of its time. Full text of
every entry, with complete commands: `29423d3fdf:scratch/mtg-320-plan.md`
(earlier history: `df58f2bff:scratch/mtg-320-plan.md`).

- **CI on merge heads (2026-11-12).** 8e5e8e62b (main ef28f3a58, #504, package
  0.2.3): run 43369147820 failed compilation in four merged payment fixtures
  still using `card: CardId(...)`; conformance, Python unit tests, Academy
  integration, Clean-machine Play, Engine Budget and Integration groups passed.
  96d3aa14e (main b531853a5 with 6ce618b8a's payment fast paths): run
  43375879983 failed Rust at `flow/mana.rs:2268` (E0560/E0425, job
  131722179161) on one more such fixture, now `object: ObjectId(index)`.
  Production code was unaffected both times. PR #491 at that readback: open, no
  reviews or comments, auto-merge null.
- **Final-head QA (2026-11-13), before the conversion cleanup.** Seeded
  comparison against main 6ce618b8a found a regression: `Effect::SelectCards`
  kept only `ObjectKind::Card` candidates for every selector, so Kyoshi Battle
  Fan never attached to its `CreatedThisWay` Ally token (33 of 120 TLA pool
  games diverged; authored and Azula games matched). Fixed in
  `flow/resolution.rs`: only `Zone`, `TargetPlayersZone` and `CardsThisWay`
  (mill/discard results, not reveals) narrow to cards. The two Battle Fan card
  tests fail without the fix. After it, 538 seeded games (108 authored, 30
  Azula, 400 TLA pools; 118,781 decisions) were byte-identical, including 4
  matching `PyAgentError` endings; two Academy games reached Game Over. Debug
  suite 1195/1199; the four `vector_env` timeouts passed alone. Receipts:
  `scratch/qa.html`.
- **Conversion cleanup (steer ced7607f).** Jack Heart, 2026-11-13: "yeah lets
  do that fairly exhaustively in 320. if there are any super expensive cases we
  can skip them". Full steer: `.lf/prompts/20261113-081526-LxK62h-steer.md`.
  Object APIs, zone membership included, take `impl Into<ObjectId>` and convert
  where they first use the id; delegating wrappers pass it through; card-only
  entry points keep `CardId`; wire fields and mixed collections keep concrete
  ids (`docs/engine.md`, `#### Names`). Audited explicit CardId-to-ObjectId
  expressions: 88 at 66e8b7ae5, 41 after, each retained site listed with its
  reason in the PR draft. No skip is claimed on performance grounds.
- **Latest focused debug receipts (2026-11-13).** Conversion cleanup: 44 passed,
  1057 skipped (1.242 s), run `c94d50c5-a56e-4e7d-99d5-5fe63f105d06`. Zone-store
  compression (`9564d30e41`): 31 passed, 1075 skipped (0.831 s), run
  `02f73ae2-48c8-4d17-b836-80783ecce7cd`, including both Battle Fan tests and
  typed-zone identity/order tests. Both used `CARGO_BUILD_JOBS=2 cargo nextest
  run --manifest-path managym/Cargo.toml --profile ci --test-threads 2` with
  test-name filters. Workspace all-target/all-feature Clippy with warnings
  denied, rustfmt and `git diff --check` passed after each. The cp313/arm64
  0.2.3 release wheel was rebuilt after the last Rust change, its `EXT_SUFFIX`
  member installed with `os.replace`, and a fresh `import managym._managym`
  loaded it. Debug builds here take 3-10 minutes.
- **Compression review (2026-11-13).** Read `state/src/zone.rs`, the `CardId`
  boundary in `state/src/game_object.rs` and the uncommitted docs edit: nothing
  further to reduce in code. No Rust changed, so nothing was rebuilt or rerun.

- **Sync conflict resolution (2026-11-13).** Integrated pinned main
  `22ed4a733484d27db2fd51274d1b4e23245a6741` through the owned `lf sync`
  sequencer (merge `c653e5dc0f`, compatibility repair `91ba66c3a4`). Preserved
  the runtime object boundary and main's mana-spending behavior and fixtures.
  Focused debug nextest run `cd2a8bf7-a9a0-482e-a19a-a19476dbe152` passed
  both `every_decision_code_is_within_the_validated_bound` and
  `lita_creates_a_zeppelin_that_crews_until_end_of_turn` (2 passed, 1238
  skipped; build 7m, tests 0.029s). Command: `CARGO_BUILD_JOBS=2 cargo nextest
  run --manifest-path managym/Cargo.toml --profile ci --test-threads 2 -E
  'test(lita_creates_a_zeppelin_that_crews_until_end_of_turn) or
  test(every_decision_code_is_within_the_validated_bound)'`. No release rebuild
  or wider gate checks in this conflict-recovery run; nothing pushed.
- **Post-merge QA at 9c9124627 (2026-11-13).** Release cp313 engines for main
  22ed4a733 and this head: 568 seeded games (108 authored, 30 Azula, 30
  Heartbeat/Lita/Release to Memory mirrors, 400 TLA pools), 137,518 decisions,
  observation+state hashes byte-identical. One `PyAgentError: source cannot pay
  tap cost` (pool 84736ef2 vs 56f0ddb0) occurs identically on main: an existing
  bug, not this change. Two Academy games reached Game Over with no page
  errors. Receipts: `scratch/qa.html`.

## Widened-local cleanup for steer fedebf6b (2026-11-13)

`docs/engine.md` `#### Names`, third bullet: returning a card as an `ObjectId`
is an explicit widening, and a function does not widen a card into an
`ObjectId` local only to call object functions. No other doc binds this.

- Widened locals removed from `priority_card_ability_actions`,
  `priority_action_for_card`, `can_cast_with` and
  `prepare_card_activated_ability`; Action, Payment and stack fields widen at
  assignment. Compression then removed the same pattern from
  `ZoneManager::contains` and `top` (`state/src/zone.rs`). The remaining
  `let x: ObjectId = x.into()` lines are `impl Into<ObjectId>` functions that
  use the id themselves.
- The steer regex counts 76→43 matching lines from 66e8b7ae5; 17 are card-name
  strings on both sides, so 59→26 CardId lines (an earlier 18/24 split was
  wrong). The PR draft lists each retained match and its reason.
- Compression review of the rest (candidate helpers in `flow/decision.rs`,
  `chosen_card`, `ZoneStore`): nothing further to cut. The paired
  `*_objects`/`*_candidates` helpers stay because observations show visitors
  while choices are card-only.
- Validation on the final tree: `CARGO_BUILD_JOBS=2 cargo nextest run
  --manifest-path managym/Cargo.toml --profile ci --test-threads 2 -E
  'test(card_ids_narrow_kind) or test(card_zone_readers_preserve) or
  test(typed_card_zones_preserve) or test(landcycl) or
  test(kyoshi_battle_fan)'`: 6 passed, 1234 skipped (0.034 s). An earlier run
  of the action/play edits also passed Force of Negation's alternative-cost
  test and workspace Clippy with warnings denied (run
  `79f5c7ad-fff8-458b-b116-6fbb792a5b2e`); Clippy was not rerun after the
  two-line `zone.rs` edit.
- Gate and delivery own Engine Budget reporting and publication; stop at review.
