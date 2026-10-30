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
6. tests/support by purpose: `game.rs`/`play.rs`/`inspect.rs`, each primitive
   once (renames listed in the commit); card-specific setups beside their tests;
   flat `cards/<set>.rs` in card order (new `cards/mkm.rs`). Counts unchanged
   (cards 235, rules 131 + 2 ignored).
7. Earlier this Task (before 2026-10-29 rewrite): support by definition, set
   filter, basics in `lea.rs`, card tests by set (`tests/cards.rs`), fixture
   refresh reads the pool.

8. **Concept review (2026-10-29) implemented.** The world pool `"world"`
   (was `tla-pool-v1`) describes `default_content_pack()` itself: same
   `CardDefId`s, digest and `Arc`; the Crossroads fixture is an unsupported row
   with no key ("A rules fixture, not a printed card"), tokens stay unsupported.
   Pool rows' `semantic_key` is `key` (Rust, PyO3 dict, `.pyi`, Python readers,
   Academy receipt programs). `semantic/{pool,setups,tokens}.rs` moved to
   `cards/`; `ContentError` (a message newtype) lives in `cards/mod.rs`; the
   `semantic` Rust module is gone. Pool rows' `card_def_id` is now required
   (`CardDefId`, Python `int`), since every row describes a runtime definition.
   Renames were posted to f871a8 (2026-10-29, at 67b1fad34). Kept deliberately: authored setup packs; v1 Python coverage gate
   (its evidence identifiers were updated for the flattened card tests).
9. **World-bound evidence for w17.** New `conformance/semantic-kernel-w17-v1`
   (w15 cases re-recorded; test + CI root). `authored-match-parity-w17-v1`
   (142 Commands; dead `HISTORICAL_*` constants removed). Academy
   `learn/precision/precision-mana-demo-w17.json`, `choice-navigation-w17.json`,
   `trigger-order-w17.json` re-recorded by offer label (same decision revisions;
   tools in `scratch/tools/`). `test_ur_lessons_increment` reads the current
   world instead of pinning w13/28.

## Remaining

1. **Public-commitment parity bound by prefix (2026-10-29).** Rebound to the
   w17 tape (curated sideboard configs; Island discard at 30 with 484 worlds,
   `decline_learn` at 102). Viewer 0's tracker stops before Command 68 (South
   Pole Voyager revealed from library into hand: no transport fact); viewer 1
   tracks all 142. `TRACKED_COMMANDS` in `academy/public_commitment_parity.py`;
   Task 3be183 owns the transport. Receipt
   `conformance/public-commitment-parity-w17-v1`.
2. CI fixes: rules-audit evidence paths follow the moved card tests
   (`tests/cards/<set>.rs`, `cardsets/tla.rs`); the choice-navigation
   Playwright spec reads `choice-navigation-w17.json`.
3. f871a8 told `card_def_id` is required (2026-10-29). Demo; PR394 stays draft.

Check (2026-10-29, after concept-review pass): debug `cargo test --no-fail-fast`
0 failures; clippy `-D warnings` (with and without `python`) and fmt clean.
pytest: semantic + model 334 pass/1 skip; current-world arena/monitor/choice
66 pass; academy 257 pass, 2 fail (public-commitment pair, above); CI
coverage/compile `--check` steps pass. `rg 'tla-pool-v1|semantic_key'` hits only
frozen v1/w11 evidence and readers and WORLDS.md history. After 93a542677:
clippy `-D warnings` clean, `cargo test --test cards` 235 pass,
`tests/semantic/test_current_content.py` 17 pass.
