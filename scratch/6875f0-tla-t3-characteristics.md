# T3 — characteristics, counters, replacements, attachments (Task 6875f0)

Brief: the Task directive (49 cards, obligations in `docs/rules/tla/allocation.jsonl`,
`planned_pr == "T3"`). Stacked on T1 (892c47, PR #394 at `be2b7586b`); build beside
T2 and reconcile shared costs before landing. Every merge needs Jack Heart's review.
Card tests live in `managym/tests/cards/tla.rs` (steer, Jack Heart 2026-10-29).

## Done (44 of 49)

Detail per slice is in each commit message (`git log 86ebe9ccf..`).

1. Locations: `ReplacementEffect::EntersTappedUnlessControls`, `Predicate::Supertype`,
   `Cost.sorcery_timing`; multi-instruction activations lower to `IfCondition{Always}`.
2. Static grants: `StaticKeywordGrant` via `effective_keywords`, one
   `static_scope_applies`. Static haste grants still miss `Permanent::can_tap`.
3. Counters: `CounterKind`, stun in `Permanent::untap`, `Cost.exhaust`,
   `Predicate::NoCounters`.
4. Base P/T 7b: `set_base_pt` (end of turn only).
5. Equipment: `Permanent.attached_to`, `attach`, `create_token.attach_source`, SBA 704.5n.
6. Auras: `Program::enchant` targets (303.4a), 704.5m via `permanent_matches_spec`,
   `StaticScope::Attached`, `static_restrictions`, attached object captured
   before costs, `cost_reduction_if_targets`.
7. Vehicles: `Permanent.becomes_creature` + `CardDefinition.creature_with_counters`
   → `Permanent::animated_creature`; `become_artifact_creature`; `Cost.crew` →
   `PendingChoice::Crew`; `Target.another`.
8. Mana: `current_mana_output` (CR 605.3a) in `activated_abilities_for`;
   `StaticCondition::ControlsMatching`; `add_mana_of_one_color`;
   `StaticRestriction.unless`.
9. Effective characteristics: `ScopeView` (Types → Keywords → PowerToughness →
   Rules); layer 4 `StaticTypeGrant` via `permanent_is_land`; `earthbend_value`,
   `cant_be_blocked_by_more_than`, `attacks_despite_defender`
   (`permanent_can_attack`), `end_step_begins`; `BecomeCreatureTarget`,
   `SetBasePtTarget`.
10. Mana remainder: `add_subtypes_controlled` (`temp_subtypes`, derived CR 305.6
   mana ability); `ReplacementEffect::UnspentManaBecomesRed` in
   `clear_mana_pools`/`clear_combat_mana_pools`; `StaticCondition::UnspentManaAtLeast`.
11. X costs: `ManaCost.x`/`with_x`; `PendingChoice::SpellX`/`ActivationX` as Modal
   `ChooseMode`; X on `Payment::Spell`, `SpellOnStack`, `ActivatedAbilityOnStack`,
   `PendingChoice::Waterbend`, `EffectFrame`; `x_min`; `enters_with_x_counters`
   (`BattlefieldEntry.x`), `opponent_damage_plus_counters`,
   `destroy_creatures_losing_abilities` (`Permanent.abilities_lost`),
   `SetBasePtControlled`.

12. Razor Rings, The Last Agni Kai, Lo and Li, Hakoda:
   - `Game::apply_creature_damage_measuring_excess` (CR 120.4a) sets
     `EffectFrame.excess_damage`, read by quantity 7 `excess_damage`; opcodes
     84 `fight` (CR 701.14b: any illegal/absent fighter → no damage), 85
     `gain_life_value`, 86 `add_mana_value`, 87 `keep_unspent_red_mana`
     (`Player.keeps_red_mana`, cleared at cleanup; red of combat mana kept);
     predicate 16 `attacking_or_blocking` (`CardPredicate.in_combat`).
   - 88 `search_library_to_hand` → `Decision::SearchLibraryToHand`
     (continuation kind 6, as the basic-land search).
     Roles 13 `each:noble_creature_you_control`, 14 `each:lesson_spell_you_control`
     (`StaticScope::SpellsYouControl`, read by `Game::spell_keywords` in
     `damage.rs::source_keywords`).
   - 89 `look_at_library_top`, 90 `cast_from_library_top`: one zone check,
     `Game::in_casting_zone`; `Game::library_top_card` is observed by its
     controller and pinned by `determinize`. `grant_keywords` EOT on
     `each:creature_you_control` → `Effect::GrantKeywordsControlled`.

13. Trusty Boomerang: `Program::granted_to("each:equipped_creature")` (field 97)
   lowers an activated program into `CardDefinition.equipped_grants`;
   `activated_abilities_for` lists them on the attached creature after its own
   (`AbilityGrant`), and every activation path reads abilities only through it.
   `ActivatedAbilityOnStack.granted_by` carries that exact grantor object
   and grant at activation; `EffectFrame.grantor` feeds `return_to_hand` on
   special role 15 `grantor` (`Effect::ReturnGrantorToHand`, CR 400.7).

Schema 8 (unreleased, T1 precedent) appended in place: fields 85-97, predicates
13-16, special roles 6-15, opcodes 66-90, enum `counter`, conditions 7-9,
quantities 5-7, trigger_event 13, subtypes 64-74.
`scratch/add_fixture.py` copies pinned evidence into the conformance fixture.

Check (slice 13, debug): `cargo test` all targets pass (cards 52, rules_tests
356, lib 120) except `semantic_conformance_tests` (w16 vs w15 and receipt digest
drift, as in PR #394 CI 40819903624); clippy, fmt clean; extension rebuilt;
`pytest tests/semantic` 158 pass, 4 fail on `schema_version == 3/4/7` asserts
from `be2b7586b`.

## Remaining (5 cards), suggested next slices

- Glider Staff (2026-10-29, deferred to the T2 sync): its entry trigger
  airbends, which T2 owns (`Op::Airbend`, exact-object cast permission,
  `0d124a66e`); a second airbend here would be a parallel authority. After the
  sync it needs only that op plus `equipped_pt` 1/1, an equipped flying grant
  (`StaticKeywordGrant` on `each:equipped_creature`) and `equip {2}`.

- Secret Tunnel (Cave; "this land can't be blocked" is `unblockable_if` Always;
  {T}: Add {C}): deferred from slice 10. Its "{4}, {T}: two target creatures you
  control that share a creature type" needs two-target activated abilities —
  `PendingChoice::ActivatedTarget` chooses one target, and its observation,
  continuation and stale-rejection paths assume one — plus a constraint that
  reads the first target, and partial legality at resolution (release notes:
  the survivor needs a type the departed creature had as it left).
- Honest Work (Aura; loses all abilities, becomes Citizen "Humble Merchant" 1/1
  with "{T}: Add {C}" — characteristic replacement, layers 1-7b).
- Crashing Wave (deferred from slice 11): after the T2 sync, build on T2's
  waterbend additional cost (`a1773e4e6`) and its X announcement; then "up to
  X targets" (requirement max read from X) and a resolution-time distribution
  of three stun counters among tapped opponents' creatures (repeated
  one-at-a-time choice, as Heartless Act).
- Fire Nation Palace (granted firebending; observation shape change).

## Reconcile at sync (T2 87e012 at `65544f0ff`, 2026-10-29)

- Both branches append to schema 8 from 86ebe9ccf with colliding numbers. T2:
  fields 85-96, predicates 13-15, subtypes 64-74, special role 6, opcodes 66-73,
  trigger_event 13, destination 2, zone 3. This branch: fields 85-97 (97
  `granted_to`), predicates 13-16, subtypes 64-74, special roles 6-15 (15
  `grantor`), opcodes 66-90, conditions 7-9,
  quantities 5-7, trigger_event 13 (`end_step_begins`), enum `counter`. T2 since
  added opcodes 74-79, fields 97-101 (101 `foretell`), quantity 5 `x`, enum
  `grant`, role `each:creature_you_control` 7. The later lander renumbers its own entries
  (append-only, contiguous).
- X has two implementations: T2 `PendingChoice::AnnounceX` + `CastChoices.x`
  (`69c23c358`; quantity 5 `x`, field 97 `mana_x`) and this branch's
  `SpellX`/`ActivationX` (quantity 6, field 95). Keep one spell announcement
  (T2's `CastChoices`), one `x` quantity and one `mana_x` field; keep
  `ActivationX` only if T2 has no activation X, and carry `x_min` and
  `BattlefieldEntry.x` onto the survivor.
- Keep one exhaust field and gate (T2 field 87, here 89) and one with-counters
  predicate reading every `CounterKind` (T2 `with_counter` 13; here `no_counters`
  14 complements it). Merge T2's `tests/cards.rs`, `support.rs`, `tla.rs` with
  these. Moving `tests/rules/tla_cards.rs` belongs to whichever PR lands it.
- Both branches added `state::is_zero` (T2 `2673c95cd`, here `2598ddacf`); keep
  one. T2's foretell (`1144d0f32`) adds face-down exile to observations and
  protocol schemas; Fire Nation Palace's observation change should build on it.
- T2 renamed `BattlefieldEntry` → `ZoneEntry` (`face_down`, `controller`);
  carry this branch's `x` onto it. T2 keys battlefield zones by controller
  (`431a31f65`) and lets `move_card_with_entry` alone invalidate the mana cache
  for moves (`35404b274`); keep this branch's invalidations for non-move
  changes (attach, counters, `abilities_lost`, `temp_subtypes`,
  `becomes_creature`), which alter mana output without a move.

## Delete — do not maintain

Qualification entries, `Permanent::local_power`/`local_toughness`,
`StaticScope::Equipped`, `Game::equipped_creature` and `Permanent::can_attack`
are gone; do not restore them.
