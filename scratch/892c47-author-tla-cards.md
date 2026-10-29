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

- 2026-10-29, Jack Heart — **tests/support by purpose**: one small scenario API
  (`support/game.rs` build, `support/play.rs` act, `support/inspect.rs` read),
  each primitive defined once; helpers.rs's one-card decks become one
  deck-of-card helper; card-specific setups move next to their tests; flat
  `cards/<set>.rs` in card order, no helper-family modules; behavior identical.

## Implemented

1. **One card representation (w17).** Set files return engine `CardDefinition`s
   (generated once by `scratch/conversion/generate.py` from w16's lowering; all
   161 proven JSON-identical). `cards/authoring.rs`, `semantic/compile.rs`,
   `semantic/content.rs`, the IR parser/interpreter and their exclusive tests
   (`semantic_ir_tests.rs`, `semantic_interpreter_tests.rs`) are deleted.
   `semantic/` now holds `setups.rs` (authored decklists → packs with token
   closure), `pool.rs` and `tokens.rs`; `IrError` became `ContentError`.
2. **Keys and Scryfall ids.** Every definition has `key` and `scryfall_id`
   (`#[serde(skip)]`, so content digests equal w16's). A first-printing audit
   moved 5 cards: `roe.prophetic_prism`, `mh2.tireless_provisioner`,
   `ptk.corrupt_court_official`, `eld.epic_downfall`, `ths.lightning_strike`.
   `tests/cards/scryfall.rs` (offline) and `tests/cards/check_scryfall.py`
   (network, maintenance only). Pool rows add `scryfall_id`.
3. **Learning tokens in Rust.** `semantic::tokens` serializes definitions with a
   serde `Serializer` (schema 9: kinds structure/field/variant/integer/boolean/
   text/definition_ref/null/program_kind; values are blake3 name hashes via
   `managym.learning_token_value`). `managym.learning_catalog(pack_key)`;
   `BoundSemanticPack` binds by digest + registry name; Python IR token builder,
   `current_ir_document` and frozen-schema loading are gone (the offline Python
   compiler stays only for frozen-evidence reproduction). Manifests carry
   `pack_key` instead of `compiled_semantics`; `ContentPool` drops it.
4. **Observability.** `tests/cards/observability.rs`: exhaustive guards over
   `Keywords`, `Permanent`, `Player`, `Effect` (63 witnesses), `TriggerCondition`,
   `Ability`; directed column/token tests; viewer-safety (hidden opponent cards,
   no names/text in tokens). New `attacking` permanent column (width 29), Python
   encoder and LLM text; serialized only when true so retained captures stay
   canonical.
5. Academy authored-match receipt names programs from the catalog
   (`conformance/authored-match-w17-v1`). Docs: AGENTS.md card section,
   WORLDS.md w17, content/semantic and managym READMEs, TLA notes.
6. Earlier this Task (before 2026-10-29 rewrite): support by definition, set
   filter, basics in `lea.rs`, card tests by set (`tests/cards.rs`), fixture
   refresh reads the pool.

## Remaining

1. tests/support reorganization (steer above); move tests for the moved cards
   (Corrupt Court Official, Epic Downfall, Lightning Strike) out of
   `tests/cards/tla.rs` into `ptk.rs`/`eld.rs`/`ths.rs` as part of it.
2. Gate: full debug `cargo test`, clippy, focused pytest; regenerate or retire
   world-bound receipts (parity w11, learn demo/trigger-order w15, w13 increment)
   — they already mismatched at w16. PR394 update, demo.

Check (2026-10-29, after rewrite): debug `cargo test --no-fail-fast` — only the
known w15 semantic-conformance pair fails; `cards` 235 incl. observability 8.
pytest semantic+model 341 pass; academy 28 fail, all frozen-world receipts/
fixtures (w11 parity source hashes, w13 asserts, w15 demos).
