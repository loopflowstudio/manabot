# MTG-216 (decklist disclosure) — 2026-10-30, implementation (headless)

Jack Heart accepted one open/closed option, default open (supersedes the
kickoff's four-way assumption). Reversible choices made while implementing:

- No player-row "known" flag: an open main deck is never empty, so all-zero
  `decklists` rows mean closed. The v5 player width stays 60 and the native
  width-binding rule is unchanged.
- No separate Python `DecklistHidden` class: refusals are `AgentError`
  ("decklist closed: …"). `Game::determinize` and
  `HiddenHandConstraints::for_viewer` assert; `Env` entry points check first.
- Search refusal at match build lists observation-only kinds
  (`checkpoint`, `heuristic`, `random`, `scripted_greedy`); every other kind is
  refused in closed matches. Academy refuses its search villain in closed games.
- Decklist pooling is semantic-pack only, like `known_hand`; generic-model
  policies ignore the `decklists` buffer. Adding `decklist_projection` changes
  the v5 semantic `ActorCritic` parameters, so v5 semantic checkpoints saved
  before this change fail strict load (open world; none frozen).
- Disclosure is not part of `world_binding` or deck identity: a checkpoint can
  play both modes, and in a closed game it sees zero opponent rows.
- `canonical_replay`/`study` privacy validators do not check decklist maps:
  frames carry no disclosure, and the engine projection is the authority.

# TLA direct-Rust decisions

Reversible operator choices made headless (2026-10-28 unless dated); each is
noted in the PR. Jack Heart's 2026-10-29 direction superseded the qualification
and pool-membership choices that were here (history: `be2b7586b:scratch/questions.md`).

- Old worlds run from their recorded git revision; no in-process rules shims.
- 2026-10-29: the 14 setup-used incomplete TLA cards keep their definitions and
  are listed in one `tla::INCOMPLETE` const (key, missing behavior) beside
  `cards()`, rather than a field on every definition. Since the one-representation
  rewrite every non-token definition outside `INCOMPLETE` is eligible. Crossroads of Destiny moved to `cards/fixtures.rs` but stays
  in the default pack, because Rust integration tests and `manabot/verify` build
  games from it. `set_cards` is the set filter (pool key since renamed `"world"`).
  Undefined TLA cards are no longer listed as unsupported pool rows.
- Not rewritten (dated evidence): the Academy curated pack `local_identity` values
  keep pre-rename keys.

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
- 2026-10-29 (support reorg): tests that exercise several cards stay where
  they were (Corrupt Court Official with Ozai's Cruelty in `cards/tla.rs`, Epic
  Downfall with Power Word Kill in `cards/afr.rs`); Lightning Strike has no test
  of its own, so no `ptk.rs`/`ths.rs` test files exist yet.
- 2026-10-29 (concept review): world pool key becomes `"world"` (not per-world
  `"w17"`), describing the default pack; row `semantic_key` becomes `key`.
  Coordinate with f871a8 in one comment. (The v1 coverage gate was deleted by
  Task afbd6a.)

- 2026-10-29 (assumption): superseded corpora now without consumers
  (`semantic-kernel-w15-v1`, `authored-match-w11-v1`, `authored-match-parity-w11-v1`,
  `public-commitment-parity-v1`) stay in the tree like w12–w14; archiving them
  to S3 is a separate publish step, not done here.
  Superseded 2026-10-30 by Jack Heart's steer: afbd6a archived them (and w12–w14,
  v1, learn-lesson-w4-v3) to `experiments/data/archive/conformance/`.
- 2026-10-29 (public-commitment parity, decision): bound to w17 by prefix. Only
  viewer 0's tracker fails (Command 68, South Pole Voyager revealed from library
  into hand); viewer 1 consumes the hero's Island discard (30) and
  `decline_learn` (102) over all 142 Commands. `TRACKED_COMMANDS = {0: 68,
  1: 142}`; identity-stream parity still covers the whole tape. Task 3be183
  (engine wave) owns the transport and removing the bound. No xfail.

Main's MTG-193 notes at the sync live at `6d069d6e5:scratch/questions.md`.

## T2 (87e012) — 2026-10-29 operator choices

- Printed flashback plus Iroh's grant: the grant's cost is used.
- Synced onto T1's w17 (`8790821da`): T2's 51 cards were regenerated in engine
  types by T1's generator (adapted for X costs and restricted mana) and proven
  Debug-identical to T2's prior lowering for all 206 definitions; T2's
  authoring ops, IR opcodes 66–79, schema-8 growth and Python token-builder
  edits are gone with the layer. T2 extends the open world **w17** in place
  (Jack Heart, 2026-10-30; it was w18 until then).
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
- Gaps: granted land plays of another
  player's card would enter under its owner (unreachable: no T2 grant lets a player play another player's land). (Academy's card face now prints
  and names exhausted / once-each-turn / granted firebending, text-5 wording.)
- Possible worlds v3 (2026-10-29): foretold slots are ordered (each has its
  own foretell turn), so the fill factor is `P(m, F)`, not `C(m, F)`; the
  ratio is constant across worlds. Materialization picks the slot set
  uniformly but keeps an already-foretold chosen card in place, so slot
  assignment is not uniform over orderings. Not yet foretold-aware (follow-up
  for belief owners): `PhysicalHandSampler`'s hypergeometric prior and
  `BeliefState.draw_unknown`/hidden-draw convolution in `belief/range.py`
  still treat every pool card as hand-or-library, as does the library
  count row in `belief/encoding.py` (pool − hand).
  2026-10-30 (concept review 4, 1b): the tracker's accounting now counts the
  slot; which hand card was foretold is assumed uniform by count (MTG-215).
- `rules_tests::structured_offer_rejects_fabricated_and_stale_ids_without_mutation`
  failed 2 of 9 full `rules_tests` runs (Debug output of the root changed
  after a rejected stale submission); passes alone and in 7 later runs. It
  touches no possible-world code; likely order/timing-dependent shared state.

## afbd6a (delete old world), 2026-10-30 — headless choices

- Current-world checks in `test_temur/test_sacrifice/test_elemental/test_optional_payments`
  stay in those files (they already run on `setups.json`); only compiler assertions
  were deleted.
- Phase-matrix rule-family closure against `v1/coverage.evidence.json` dropped; the
  matrix keeps its other structural checks. `matchup_source_digest` left receipts;
  conformance decks come from `cards::setups`.
- `TRAILING_ORDER` removed with `LEGACY_ORDER`: default pack is name order.
- Statics live in `CardDefinition.statics: Vec<StaticAbility>` (not `Ability::Static`)
  so trigger indices into `abilities` are unchanged. Ward is listed as a static per
  Jack Heart's direction though CR 702.21 makes it a triggered keyword; registration
  still expands it to its trigger. `blocking_restriction`/`block_restriction` merged
  as `StaticAbility::BlockRestriction(CantBlock | CantBeBlockedBy)`.
- Academy curated pack manifests keep `source_uri` provenance strings naming
  `content/semantic/v1/...` and `cardsets/tla.rs` (frozen pack assets; not rewritten).
- Token first printings (Scryfall oracle-id earliest print, 2026-10-30): Ballistic
  Boulder tltr, Bear tktk, Clue tsoi, Food teld, Monk tdmu, Human Soldier tsoi,
  Treasure txln; Ally, Dragon, Soldier, Spirit first printed in ttla.

## afbd6a sync onto T2 (c7979d310), 2026-10-30

- T2's new `CardDefinition` fields became variants: `StaticAbility::{Flashback,
  Foretell, FlashIf, CostsLessPerGraveyardCard}` (the old
  `cost_reduction_in_graveyard` flag on `cost_reduction_per` is its own
  variant) and `AdditionalCost::Waterbend` (was `kicker` + `kicker_waterbend`).
  `kicker()` returns either optional cost, so the kicker card column and the
  kicked cast stay as T2 built them. Each new variant has a guard row, witness,
  token test and real-card test in `observability.rs`.
- Fire Nation Attacks now produces Soldier, so it left `TOKENS_AWAITING_PRODUCER`.
- AGENTS.md T2 note on `Cost.zone`/`Program::landcycling` named the removed IR;
  rewritten as `ActivatedAbilityDefinition.zone` (`ActivationZone`).
- `manabot/sim/heuristic.py` reads T2's activated-ability `effects` list and
  skips hand/graveyard abilities (landcycling, foretell) when picking prompt,
  spare-mana and pump abilities. T2's nine new targeted spells are pinned in
  `test_heuristic.py`; the seven it cannot rate (airbend, linked exile, …) get
  polarity `None`, the documented neutral fallback.
