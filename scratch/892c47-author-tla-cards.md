# TLA cards — direct Rust cards and one world

Task 892c47 updates PR394 and stops at demo; merge, auto-merge, paid compute,
training and follow-up launches are not authorized. D1/f871a8 keeps parser,
construction, `admit_deck` and consumer input; the `content_pool`/`pool_content`/
`Eligibility` API shape stays as D1 uses it. History of the superseded JSON
pipeline: `544ae312c:scratch/892c47-author-tla-cards.md`.

## Accepted direction

- 2026-10-28, Jack Heart — direct Rust cards, one current world
  ([tla-rust-authoring-direction.md](tla-rust-authoring-direction.md)).
- 2026-10-29, Jack Heart — **every card lives in the set file of its first
  printing.** Basics move from `tla.*` to `lea.plains` etc. (TLA decks still admit
  them). The native-only cards in `cards/native.rs` move to their first-printing
  set files, so `cardsets/` is the one place a card is filed; `native.rs` keeps
  only `CardDefId` slot ordering if still needed. Keep the rename-equivalence
  proof and the `content_pool` API that D1/f871a8 uses.
- 2026-10-29, Jack Heart (corrects the same steer's item 3, which had limited
  the pool to TLA + basics) — **the world's pool is every card a set file
  defines, all supported.** Restricting to TLA is a deck generation/validation
  filter selected by set (key prefix such as `tla.*`); Jack may train on a wider
  pool. Remove per-pack membership or eligibility lists that only encode "which
  set". Keep a simple way to ask for the cards of one or several sets.
- 2026-10-29, Jack Heart — **"simplify simplify simplify"; replaces the
  qualification model.** Remove `qualifications.json`, closure hashes, generated
  eligibility and the review-record gate. A card is supported in the current
  world when its set file defines it; the world's support check answers from
  that alone. An incomplete TLA card is either not defined yet (its T2–T7 batch
  authors it) or carries one explicit incomplete marker beside its Rust
  definition when a fixed setup needs it — whichever leaves less machinery. Rules
  tests stay ordinary tests; coverage notes are plain docs under `docs/rules/tla/`
  or beside tests, never engine data. Pool eligibility derives from the support
  check. Delete code, docs and tests that existed only for the gate.

- 2026-10-29, Jack Heart — **card tests get a folder mirroring `cardsets/`**:
  one test file per set (e.g. `managym/tests/cards/tla.rs`, `lea.rs`, `stx.rs`,
  via one `tests/cards.rs` entry), per-card rules tests moved there from
  `managym/tests/rules/`; `tests/rules` keeps engine-rule tests. The folder may
  hold a coverage README and a report script (which cards have tests). Tests and
  coverage never qualify or gate a card.
- 2026-10-29, Jack Heart (hold lifted; **supersedes** the same day's steers to
  convert native cards to typed `Card` and to lower `Card` directly) — **one card
  representation**: each card is Rust in the engine's own types
  (`CardDefinition`/`Ability`/`Effect`/`TargetSpec`) with small constructor
  helpers, one function per card in its first-printing set file. Delete
  `cards/authoring.rs` (`Card`/`Op`/`Program` and lowering), `semantic/compile.rs`
  and SemanticPack IR parsing for current content; no JSON. Convert the 140+
  typed cards (Op→Effect is nearly 1:1, typed replacements for symbolic target
  names); native cards stay and gain tokens. Learning tokens derive in Rust from
  `Effect`/`Ability`. Content identity is `ContentPack::content_digest`. New world
  **w17** (WORLDS.md). Frozen IR/evidence stay readable through existing readers,
  off the current path. Oracle-text parsing only as a separate code-generator
  Task. New behavior is a new `Effect` variant, never one-off card logic.
- 2026-10-29, Jack Heart — **every mechanic is observable in both the
  observation tensor (`agent/observation_encoder.rs`) and the learning tokens.**
  Exhaustive (no wildcard) guard over `Effect`, `Ability`, keywords, counters and
  status fields naming tensor columns and tokens; a directed test per mechanic;
  viewer-safety tests (no hidden information in either). Tests live beside the
  card tests (e.g. `tests/cards/observability.rs`) plus Python encoder tests.
  Known gap: T3 stun/fire counters, `attached_to`, granted keywords never reach
  the encoder. Width changes recorded in WORLDS.md.
- 2026-10-29, Jack Heart — **every definition (tokens included) carries a
  required `scryfall_id`**: the Scryfall card-object UUID of the printing in its
  set file. Offline test: present, well-formed, unique. A network maintenance
  script beside the card tests checks name and set (never in CI or play).
  `text_box` stays for offline play.

## Implemented

1. In-process compile of typed set definitions to IR (`semantic/compile.rs`);
   keys `<set>.<card>` in first-printing set files; one default pack.
2. Support by definition (2026-10-29 direction): `qualifications.json`,
   `inventory.json`, `t1-clause-review.json`, `Qualification`/`qualified()`/
   `check_qualifications`/`inventory()` and their tests are gone.
   `world::supports_card` = typed definition not in `tla::INCOMPLETE` (14 cards,
   each with its missing behavior). Pool rows: tokens and incomplete cards
   unsupported; the world pool also lists native set-file cards as unsupported
   (no IR, not in `pool_content`). `world::set_cards(&["tla","lea"])` /
   `managym.set_cards` filter by key prefix. `tla-pool-v1` key kept for D1.
3. Basics are `lea.plains`…`lea.forest`. Proof (`scratch/prove_basic_rename.py`
   over dumps of all 10 packs before/after): every IR equivalent apart from the
   rename; default-pack digest and `CardDefId` manifest unchanged.
4. `cards/native.rs` deleted. Native cards live in first-printing set files'
   `native_cards()` (lea, por, isd, ori, sth, rtr, grn, ice, vis, stx, mkm; sets
   checked on Scryfall); Crossroads of Destiny in `cards/fixtures.rs` (still in
   the default pack, which tests and `manabot/verify` build games from);
   `LEGACY_ORDER`/`TRAILING_ORDER` in `content_pack.rs`.
5. Docs rewritten: `content/semantic/README.md`, `docs/rules/tla/README.md`,
   `managym/README.md`, AGENTS.md typed-card section, WORLDS.md.

6. Card tests: `tests/cards.rs` includes one module per set
   (`tests/cards/<set>.rs`, 14 sets, 225 tests); 17 engine-rule tests (injected
   abilities, tokens, Crossroads modal) moved to `tests/rules/engine_events.rs`.
   Each test was filed by the card it is about (mapping:
   `scratch/cardtests/assign.json`, overrides for ties). Shared helpers live in
   `tests/support/` (used by both binaries; one module per helper family, which
   also names the inline test groups). `tests/cards/README.md` and
   `tests/cards/coverage.py` (120/150 cards have a test in their set file) never
   gate support. The semantic coverage evidence validator
   (`manabot/semantic/coverage.py`) accepts card-set test files; evidence and
   `coverage-gaps.json` regenerated; docs and the public-commitment source
   manifest point at the new paths.
7. `scripts/refresh_card_fixture.py` reads names from `managym.content_pool`
   (150 real cards = fixture; tokens skipped by `token.*` key).

## Remaining

1. One card representation (above): convert typed cards to engine types, derive
   learning tokens in Rust, delete the authoring layer and `compile.rs`, w17.
2. Observability guard, per-mechanic and viewer-safety tests (above).
3. `scryfall_id` on every definition, offline test, network check script.
4. Gate, PR394 update, demo.

Check (2026-10-29, after card tests): debug `cargo test --no-fail-fast` — only
the known w15 semantic-conformance pair fails (`cards` 225 pass, `rules_tests` 131);
clippy `--all-targets -D warnings` and fmt clean. Focused pytest (semantic,
academy, eval admission/models, monitor, choice separability, policy catalog,
package boundaries): 42 failures, the same count as before; coverage gate 13/13.

Compress (2026-10-29): `world::unsupported_cards` reads the pool once and
`supports_card` delegates to it; unused `world::card` removed (pool rows carry
the reason). `cargo test --lib world::` 2/2, clippy clean; gate owns full suites.

Realign (2026-10-29): plan and code agree; no upstream main changes since the
sync. Stale qualification wording removed from `delivery-plan.json` and the
historical scratch notes. Native set-file cards being unsupported for deck entry
is an operator choice (questions.md) against Jack Heart's "all supported"
wording; review at demo. Check: `delivery-plan.json` parses; no test reads it.
