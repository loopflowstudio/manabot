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

## Implemented (PR394, summary)

Full list: `2c50c8793:scratch/892c47-author-tla-cards.md`. Set files return
engine `CardDefinition`s (w17; authoring language, compiler, IR deleted); keys +
Scryfall ids (`#[serde(skip)]`; five cards moved to first printings); learning
tokens from a serde serializer (schema 9, `managym.learning_catalog`);
exhaustive observability guards; `world` pool = default pack; card modules
under `cards/`; w17 conformance, authored-match, parity and Academy fixtures
(tools in `scratch/tools/`); public-commitment parity bound by prefix
(`TRACKED_COMMANDS`, Task 3be183 owns removing it). Renames posted to f871a8
(2026-10-29). Stacked Task afbd6a later deleted the v1 coverage gate and
reshaped statics/tokens (see `afbd6a-delete-old-world.md`).

## Concept review, second pass (2026-10-29, headless)

No design change. The implemented concepts match the accepted direction: set
files are the only card representation; `world` pool = default pack;
`supports_card`/`unsupported_cards` answer from definitions plus
`tla::INCOMPLETE`; `set_cards` is the set filter; authored setup packs remain
decklists only (frozen setups and f871a8's exact-preset selection need their
identities). The v1 Python coverage gate checks only the frozen v1 corpus, off
the current path, so it is not the review-record gate Jack Heart removed.

Counterexample to show at demo (settled by Jack Heart's "key prefix" wording,
not an open choice for this PR): Corrupt Court Official, Epic Downfall and
Lightning Strike are TLA reprints filed under their first printings (`ptk`,
`eld`, `ths`), so `set_cards(["tla"])` omits them. No authored TLA deck uses
them today. A printing-aware "TLA-legal" filter would be a new product concept
(printings per card), for f871a8 or a later Task if Jack wants it.

## Remaining

1. PR394 CI at head 07fcdb6aa (b1579b887's run was cancelled, so the
   regenerated authored-match parity receipt has no CI result yet).
2. Demo; PR394 stays draft.

Check (2026-10-29): academy pytest 259 pass; current-world arena/monitor 66
pass; CI green at 38c883f08, pending at 07fcdb6aa.
