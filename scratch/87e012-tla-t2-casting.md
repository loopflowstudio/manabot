# T2 — TLA casting, costs, zone permissions and private zones (Task 87e012)

Stacked on T1 (892c47, PR #394 at `be2b7586b`). Task directive lists the 51 cards
and obligations; allocation in `docs/rules/tla/allocation.jsonl` (`planned_pr: T2`).
Every merge needs Jack Heart's review. Card support = defined in the set file
(Jack Heart, 2026-10-29); T1 is removing the qualification gate in its own
worktree, so new T2 cards read "unsupported" in this base until T1 lands.

## Decisions (reversible operator choices: questions.md)

- 2026-10-29, Jack Heart: card tests live in `managym/tests/cards/<set>.rs`
  (one `tests/cards.rs` entry); existing per-card `tests/rules/` tests move
  there. Operator reading: those files predate T2 (T1's base), so T1 moves
  them and T2 reconciles on sync. T1 is now doing the move (see Remaining).
- Activated abilities carry an effect sequence (`ActivatedAbilityDefinition.effects`).
- Printed flashback is `Cost.flashback` → `CardDefinition.flashback`.
- Earthbend takes a `Count`; printed counts stay `EffectValue::Fixed` (IR unchanged).
- Learning schema 8 grows append-only (as T1's `1e0d3c998`).
- Card-zone activation: `ActivationSource` (permanent or card) in
  `Payment::Activation`/`PendingChoice::ActivatedTarget`; hand/graveyard
  abilities are `Action::ActivateCardAbility { card, ability_index }` (type
  `PriorityActivateAbility`, card focus), indexing printed abilities in program
  key order. Landcycling = opcode 24 with `destination: hand`; graveyard
  self-return = `return_to_hand` targeting `source`; Team Avatar uses
  `attacks_alone` + role `triggering_permanent`.
- Exhaust/once-each-turn: `Cost.exhaust`/`once_each_turn` → ability flags;
  `Permanent::{activation_used, record_activation}` own both limits
  (`exhausted` for the object's life, `activated_this_turn` cleared at cleanup),
  recorded at mana activation or stack push. Predicate `with_counter`.
- Restricted mana is keyed by `ManaUse` (state/mana.rs): subtype bit masks over
  `SPEND_SUBTYPES` (Lesson, Shrine, Ally) for spells and ability sources.
  `Mana.restricted` = sorted boxed classes; `ManaCost.purpose` is set from the
  cast spell's subtypes or, in `price_activation`, the source's; narrowest
  usable class spends first. Opcode 56 `add_lesson_mana` takes optional
  `spend_only` (omitted = Lesson spells). `cost_reduction_per` (Cost
  `affinity`) reduces generic activation cost. Viewers see restricted mana
  per class (concept review 3); `lesson_mana` keeps the total.
- Casting outside hand: `flow/cast_permission.rs` owns `GameState.cast_permissions`
  — exact-`ObjectRef` exile grants (stale once the card leaves exile) and
  `HandOnly` restrictions that override every non-hand permission until the
  restricter's next untap. `Game::zone_cast_cost` is the one non-hand cost
  authority (graveyard flashback, exile grants). Hash omits the field when empty.
- Airbend = opcodes 66 `airbend {target}`, 67 `airbend_all {except, predicate}`
  (inside a target clause; spares the chosen target), 68
  `opponents_cast_only_from_hand`. `Exile {target: "source"}` exiles the resolving
  spell; finalize leaves a spell that already left the stack. Targets take
  `another` ("other target", trigger source excluded); predicate `attacking`.
- Schema 8 also gained opcodes 70–73 (`sacrifice_source`,
  `put_into_library_second_or_bottom`, `search_lands_opponent_splits`,
  `reveal_hand_discard_chosen` with public `known_hand`), fields through
  `foretell` (101) and predicate `any_of`.
- Targeted waterbend activations carry their target in
  `PendingChoice::Waterbend`; a {T} ability can't waterbend-tap its source.
  "Discard a card" cost: `PendingChoice::DiscardPayment`, before mana.
- Conditional flash → `CardDefinition.flash_if`; `Game::has_instant_timing`.
- Cast-time announcements are one `CastChoices { kicked, x, free }` in
  `PendingChoice`/`Payment::Spell`; `can_cast_with`/`can_pay_cast` own
  castability for priority and free-cast offers.
- Exile grants carry `cost` (None = mana cost), `this_turn`, `source`,
  `own_turn_only`, `waterbend`, `flash`, `any_mana`, `after_turn`, `foretold`;
  granted lands play from exile (`may_play_from_zone`). Opcode 69 (Zuko).
- Foretell: `Cost.foretell` → `CardDefinition.foretell`; `Action::Foretell`
  (type `PriorityActivateAbility`, card focus) pays {2} by automatic mana and
  exiles face down off the stack. The `foretold` grant is the face-down
  authority (`Game::is_foretold`). `ZoneEntry` (was `BattlefieldEntry`)
  carries `face_down` onto `CardMoved`: observation events and policy history
  name no card, known-hand facts clear. Observations list it only for its
  owner (`CardData.face_down`); Academy's `redact_observation` drops an
  opponent's face-down exile (protocol `LegacyCardView.face_down`, schemas
  regenerated); `determinize` resamples it from unseen cards with foretell.
- Control: each player's battlefield lists permanents they control
  (`ZoneManager` holder = owner, or controller on the battlefield; undo
  records it). A resolving permanent spell enters under its controller
  (`ZoneEntry.controller`, CR 110.2); cast triggers read `spell_caster`;
  observations pair battlefield cards with their controller.

## Done (`648edee6b`…`2673c95cd`; foretell `1144d0f32`; control `431a31f65`)

All 51 T2 cards are defined, none in `tla::INCOMPLETE` for T2 behavior, each
with a Scryfall fixture entry and a witness in `tests/cards/tla.rs` (T2's in
`mod casting`, on `tests/support/casting.rs`; Toph and Dai Li Agents in T1's
`mod elemental`). Synced onto PR #394's head `e403b8a1f` (merge `e7933c6fd`)
and onto T1's local card-test move `f9732c12f`. WORLDS.md w16 records T2 (166 supported).
Earlier check (`35404b274`): debug `cargo test` passes except the
pre-existing semantic-conformance pair.

## Concept review 3 (Jack Heart, accepted 2026-10-29; page `scratch/concept-review.html`)

- 1b: exile cast permissions and the hand-only lock reach every viewer
  surface, the policy tensor included (new policy input v5; v1–v4 widths and
  checkpoints unchanged). Foretold grants reach only the owner.
- 3c: restricted mana split by class on every surface, in the same v5.
  `lesson_mana` stays for v1–v4.
- 2: T2 lands only after T1/PR #394; before landing, the pre-existing per-card
  tests (Toph, Dai Li Agents in `tests/rules/elemental_cards.rs`) must be in
  `tests/cards/<set>.rs` (T2 moves them on sync if T1 did not). Don't arm
  T2's merge while #394 is open.
- Validation: the CLAUDE.md arena admission/monitoring suite plus a redaction
  test (opponent's foretold grant absent from JSON, text and tensor).

### Implemented (2026-10-29)

`CardData.cast_permission` (`CastPermissionData`, from
`Game::exile_grant_on` + `grant_cast_cost`; any timing, dead-source grants
hidden) and `CardData.x` (a stack spell's announced X);
`PlayerData.restricted_mana` (`RestrictedManaData` per `ManaUse` class) and
`hand_only_until` (player id). Policy input v5 (default): player 40→59
(`PLAYER_RESTRICTED_INDEX` 18 subtype×color, `PLAYER_HAND_ONLY_INDEX`), card
41→55 (`CARD_PERMISSION_INDEX` 12 terms, `CARD_X_INDEX` 2, validity last);
native buffers tell v5 from v4 by player width. Python reference encoder,
PyO3 `CastPermission`/`RestrictedMana`, text-5 lines, Academy protocol
(`LegacyCastPermissionView`, `LegacyRestrictedManaView`; schemas regenerated)
and UI. Guard + directed + viewer-safety tests in
`managym/tests/cards/observability.rs`; Python text/tensor redaction test.

## Later steers (Jack Heart, 2026-10-29) — wait for the T1 sync

- Observability: every mechanic in both the tensor and the learning program
  tokens, enforced by an exhaustive guard (Effect/Ability, keywords, counters,
  status fields) plus per-mechanic and viewer-safety tests. Done here for
  observation status fields → tensor. Tokens and the Effect/Ability guard
  depend on the representation change.
- One card representation: cards in engine types (`CardDefinition`/`Ability`/
  `Effect`/`TargetSpec`) with constructor helpers; remove `cards/authoring.rs`
  and `semantic/compile.rs`; tokens derived in Rust; world w17. The steer says
  T1 removes the layer and T2 converts its 51 cards (no new authoring `Op`
  variants). Observed 2026-10-29 after the sync: T1's head `e403b8a1f` still
  has both files. Confirmed 2026-10-29 (T1 worktree plan, after `352faa968`):
  T1's Remaining lists, in order, one card representation (convert typed
  cards, Rust tokens, delete authoring/compile.rs, w17), the observability
  guard, `scryfall_id`, then gate and PR394 update. T1's implement step is live.
  T1 moved the per-card tests (225 tests in 14 set files, `tests/support/`,
  `tests/cards/README.md`, `coverage.py`; `f9732c12f`), so question 2 is
  settled on T1's side; T2's `tests/cards/` follows it since the
  `f9732c12f` merge.
- Scryfall object id on every definition (first printing, required field),
  offline uniqueness test and an online maintenance script beside the card
  tests. Lands with the engine-type definitions after the sync.

## Remaining

- Next T1 sync, after T1 publishes engine-type cards/w17 (PR #394 still at
  `e403b8a1f`; T1's worktree clean at `f9732c12f`, its Remaining 1–4 unstarted
  in commits; main gained unrelated `9a62e8b39` and `2fc0af676`, which T1's
  next main sync brings in; still so on a later 2026-10-29 check, T1 idle on PR
  copy): the three steers above, then the world gate.
- Gate (not run here): CLAUDE.md arena admission/monitoring suite; full debug
  `cargo test`; Academy check/build (local node can't load libllhttp; CI).
- Known gaps (questions.md): possible-worlds ignore foretold cards; granted
  land plays of another player's card would enter under its owner.

## Check

2026-10-29, after merging T1's `f9732c12f` (temur restricted-mana edits ported
to `cards/{arb,tla,lea}.rs`): debug `cargo test --test cards` 293,
`--test rules_tests` 131 (424 = the earlier 68 + 356), `--lib` 121, clippy
`--all-targets` clean; `semantic_conformance_tests` fails only the known pair.
Earlier v5 check: continuation-input parity v1–v5, text observation,
experience protocol, play modes, env observation and `tests/model` pytest pass;
`tests/eval/test_batched_eval.py` fails on T1's missing w16 `WorldVersion`.
Compress: one `priority_candidate_cards` feeds both priority scans; player
observations read one combined regular+combat pool. Debug `--lib`/`cards`/
`rules_tests` 121/293/131 pass, clippy clean. Authoring layer left for T1.
