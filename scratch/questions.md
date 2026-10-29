# TLA direct-Rust decisions

Reversible operator choices made headless (2026-10-28 unless dated); each is
noted in the PR. Jack Heart's 2026-10-29 direction superseded the qualification
and pool-membership choices that were here (history: `be2b7586b:scratch/questions.md`).

- Old worlds run from their recorded git revision; no in-process rules shims.
- Superseded 2026-10-29 by Jack Heart's one-representation decision (w17, no
  JSON, tokens from `Effect`): the earlier choices to stay in w16, keep the IR's
  JSON shape and build tokens in `learning.py`.
- 2026-10-29: the 14 setup-used incomplete TLA cards keep their definitions and
  are listed in one `tla::INCOMPLETE` const (key, missing behavior) beside
  `cards()`, rather than a field on all 269 `Card` literals. Native cards sit in
  set files as native `CardDefinition`s and are unsupported (no IR, so not in
  `pool_content`). Crossroads of Destiny moved to `cards/fixtures.rs` but stays
  in the default pack, because Rust integration tests and `manabot/verify` build
  games from it. D1 keeps the `tla-pool-v1` key; `set_cards` is the set filter.
  Undefined TLA cards are no longer listed as unsupported pool rows.
- Not rewritten (dated evidence): the Academy curated pack `local_identity` values
  keep pre-rename keys.
- Full TLA completion remains the eight-PR plan; no merge or new Task authorized.

Main's notes at the syncs live at `6d069d6e5:scratch/questions.md` and `origin/main@633dca159:scratch/questions.md`.

- 2026-10-29 (one-representation rewrite): five cards were filed under a later
  printing and moved to their first printing's set (see plan). Three are TLA
  reprints (Corrupt Court Official, Epic Downfall, Lightning Strike), so
  `set_cards(["tla"])` no longer lists them; a TLA-legal filter would need
  printings, not first-printing key prefixes. Followed Jack Heart's
  first-printing rule; flag for f871a8 deck validation.
- 2026-10-29: `key` and `scryfall_id` are `#[serde(skip)]` — identity metadata,
  not rules — so content and state digests stay equal to w16's and digest-bound
  fixtures keep validating.
- 2026-10-29: learning token values are stable blake3 hashes of Rust/printed
  names (no vocabulary file); the policy builds its vocabulary from observed
  (kind, value) pairs as before.
- 2026-10-29: D1/f871a8 interface change: `ContentPool.compiled_semantics`
  (IR/source hashes) is gone; manifests carry `pack_key`; rows add
  `scryfall_id`; `IrError` is `ContentError`. `content_pool`/`pool_content`/
  `Eligibility` otherwise unchanged.
- 2026-10-29: the Python offline compiler (`manabot/semantic/compiler.py`) and its
  frozen-source tests stay as the reader/reproducer of frozen v1–w11 IR; nothing
  current reads it.
- 2026-10-29 (support reorg): tests that exercise several cards stay where
  they were (Corrupt Court Official with Ozai's Cruelty in `cards/tla.rs`, Epic
  Downfall with Power Word Kill in `cards/afr.rs`); Lightning Strike has no test
  of its own, so no `ptk.rs`/`ths.rs` test files exist yet.

Main's MTG-193 notes at the sync live at `6d069d6e5:scratch/questions.md`.

## T2 (87e012) — 2026-10-29 operator choices

- Printed flashback plus Iroh's grant: the grant's cost is used.
- Synced onto T1's w17 (`8790821da`): T2's 51 cards were regenerated in engine
  types by T1's generator (adapted for X costs and restricted mana) and proven
  Debug-identical to T2's prior lowering for all 206 definitions; T2's
  authoring ops, IR opcodes 66–79, schema-8 growth and Python token-builder
  edits are gone with the layer. T2 is world **w18**.
- Card abilities use `ActivateCardAbility` (sharing `PriorityActivateAbility`);
  a discard-cost ability refers to its card's graveyard object after costs.
- `lesson_mana` stays the total of any restriction; per-class detail is
  `restricted_mana`. `PlayerVitals` ≤ 40 bytes, `UndoEntry` ≤ 72.
- Avatar's Wrath's hand-only lock ends at its caster's next untap and blocks
  flashback too; mass airbend exiles one at a time.
- Kicked Aang's Journey runs two searches (no TLA Shrine is defined yet).
  Dai Li's revealed hand is public `known_hand`. Elemental Teachings with fewer
  than two lands: the opponent bins all.
- X is offered as `ChooseMode { mode: X }`, 0 through the largest payable
  value. Spirit Water Revival's "waterbend {6}" is a waterbend kicker.
- Free casts during resolution (Yue, Solstice) are a nested
  `Decision::CastWithoutPaying`; valid because the free cast is each card's
  last instruction.
- Hama/Azula: `Effect::ExileLinked` creates the source-linked grant; "any type
  of mana" prices the cast as generic equal to mana value.
- Foretell is `PriorityActivateAbility` (Academy labels it "Foretell X"); a
  face-down move is hidden from history and events for both players.
- v5 tensor: restricted mana per spending subtype × color (a multi-subtype
  class counts under each); announced X on the stack spell's card row; a grant
  whose source has left is not shown. The w17 guard surfaced four T2 status
  fields with no tensor column; v5 now carries them (permanent 29 → 32: used
  exhaust abilities, used once-each-turn abilities, granted firebending;
  player 59 → 60: no maximum hand size), with JSON, text and Academy types.
- T2's `mod casting` tests were merged into T1's flat `tests/cards/tla.rs` in
  card order; T2-only primitives joined `support/` (`cast_card`,
  `activate_card`, `move_from_library`, `can_activate[_card]`,
  `offered_targets`, `offered_cards`, `zone_of`), each defined once.
- Gaps: possible-worlds ignore foretold cards; granted land plays of another
  player's card would enter under its owner; Academy's card face does not yet
  badge exhausted / once-each-turn / granted firebending (types carry them).
