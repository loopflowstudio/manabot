# TLA direct-Rust decisions

Reversible operator choices made headless (2026-10-28 unless dated); each is
noted in the PR. Jack Heart's 2026-10-29 direction superseded the qualification
and pool-membership choices that were here (history: `be2b7586b:scratch/questions.md`).

- Old worlds run from their recorded git revision; no in-process rules shims.
- Superseded 2026-10-29 by Jack Heart's one-representation decision (w17, no
  JSON, tokens from `Effect`): the earlier choices to stay in w16, keep the IR's
  JSON shape and build tokens in `learning.py`.
- 2026-10-29: card tests are grouped inside each set file by helper family
  (`mod stage2 { … }`), since families define same-named helpers with
  different signatures; merging the helpers is a later cleanup.
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

Main's MTG-193 notes at the sync live at `6d069d6e5:scratch/questions.md`.

## T2 (87e012) — 2026-10-29 operator choices

- Printed flashback plus Iroh's grant: the grant's cost is used (dominant for
  every TLA card).
- No schema 9: main is schema 7, so unreleased schema 8 grows append-only
  (opcodes 66–79, fields incl. `mana_x`/`chooser`/`grant`/`below`, quantity
  `x`, enum `grant`, role `each:creature_you_control`). Landcycling reuses
  `search_basic_land`; restricted mana reuses opcode 56 with `spend_only`.
  World stays w16 while T2 only adds cards.
- Card abilities use `ActivateCardAbility` (sharing `PriorityActivateAbility`);
  a discard-cost ability refers to its card's graveyard object after costs.
- `lesson_mana`/`lesson` stay the total of any restriction; per-class detail
  is `restricted_mana` (concept review 3). `PlayerVitals` ≤ 40 bytes,
  `UndoEntry` ≤ 72.
- Avatar's Wrath's hand-only restriction ends at its caster's next untap and
  blocks flashback too; mass airbend exiles one at a time. `another` is
  enforced for triggers (activations never offer their source).
- Kicked Aang's Journey runs two searches (no TLA Shrine is defined yet).
  Sokka's trigger uses `any_of`. Dai Li's revealed hand is public
  `known_hand`, one offer per definition. Elemental Teachings with fewer than
  two lands: the opponent bins all.
- T1's three frozen-schema pytest assertions (3/7/4 vs 8) are left for T1.
- X is offered as `ChooseMode { mode: X }` (Modal, stage `mode`), 0 through
  the largest payable value. Spirit Water Revival's "waterbend {6}" is
  `Cost { kicker, waterbend: true }` → `kicker_waterbend`; "if the additional
  cost was paid" is `kicked`.
- Free casts during resolution (Yue, Solstice) are a nested cast decision,
  `Decision::CastWithoutPaying` (hand spells derived at offer time, or the
  exiled card; plus Decline), not a one-shot grant. Choosing finishes the
  effect, then casts with `CastChoices.free`; valid because the free cast is
  each card's last instruction.
- Hama/Azula: one opcode 78 `exile_linked` creates the source-linked grant at
  exile (Azula's static folds into it); "any type of mana" prices the cast as
  generic equal to mana value. Control change is T2's, not T3's: only the
  battlefield holder and resolution's entry controller change.
- Foretell is `PriorityActivateAbility` (no new action type or offer verb;
  Academy labels it "Foretell X"). A face-down move is hidden from history
  and events for both players (they are not per viewer). Possible-worlds
  constraints still ignore foretold cards (not a leak; prior slightly off).
- Text observation admits stages `revealed_hand`, `split_revealed`,
  `free_cast`, `linked_exile`.
- v5 tensor (2026-10-29): restricted mana is encoded per spending subtype
  (Lesson/Shrine/Ally × color) as "may pay for that subtype", so a
  multi-subtype class counts under each and spell vs ability use is merged.
  Announced X joins the card row of a stack spell (`CardData.x`), not the
  relations, so v5 keeps v4's relation width. A grant whose source has left
  the battlefield is not shown (it can never be used); a card holds at most
  one live grant (every producer exiles a new object).
- The learning-token half of the observability steer, the Effect/Ability
  guard and Scryfall ids wait for T1's engine-type cards (w17) rather than
  being built on the authoring layer T1 is removing.
