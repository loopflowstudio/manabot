# T3 — characteristics, counters, replacements, attachments (Task 6875f0)

Brief: the Task directive (49 cards, obligations in `docs/rules/tla/allocation.jsonl`,
`planned_pr == "T3"`). Stacked on T1 (892c47, PR #394 at `be2b7586b`); build beside
T2 and reconcile shared costs before landing. Every merge needs Jack Heart's review.
Card tests live in `managym/tests/cards/tla.rs` (steer, Jack Heart 2026-10-29).

## Done (43 of 49)

1. Locations (Abandoned Air Temple, Agna Qel'a, Ba Sing Se, Realm of Koh):
   `ReplacementEffect::EntersTappedUnlessControls`; `Predicate::Supertype`;
   `Cost.sorcery_timing`; multi-instruction activations lower to one `IfCondition{Always}`.
2. Static grants (Air Nomad Legacy, Badgermole, Diligent Zookeeper, Foggy Swamp
   Hunters, The Walls of Ba Sing Se): `StaticKeywordGrant` read by
   `Game::effective_keywords`; one `static_scope_applies`; `per_creature_type_max`.
   Static haste grants still miss `Permanent::can_tap`.
3. Typed counters (Rowdy Snowballers, Mai, Heartless Act): `CounterKind`, stun in
   `Permanent::untap`, `Cost.exhaust`, `Predicate::NoCounters`, opcode 66.
4. Base P/T, layer 7b (Flexible Waterbender, Turtle-Duck): opcode 67 `set_base_pt`
   (end of turn only).
5. Equipment (Kyoshi Battle Fan, Meteor Sword, Twin Blades): `Permanent.attached_to`,
   opcode 68 `attach`, `create_token.attach_source`, SBA 704.5n.
6. Auras (Path to Redemption, Watery Grasp, Swampsnare Trap): `Program::enchant`
   is the cast targeting (303.4a) and 704.5m legality via `permanent_matches_spec`;
   `StaticScope::Attached`; `static_restrictions` (opcodes 69-70); activated
   abilities capture the attached object before costs (opcode 71);
   `cost_reduction_if_targets`.
7. Vehicles (Fire Nation Warship, Invasion Submersible, Tundra Tank, War Balloon):
   - `Permanent.becomes_creature` (EndOfTurn < Incarnation) and
     `CardDefinition.creature_with_counters` feed `Permanent::animated_creature`,
     read by `is_creature`, predicates, trigger subjects, LKI and `is_animated`.
   - Opcode 72 `become_artifact_creature` (`end_of_turn`, absent = incarnation,
     or `static` under `source_counters_at_least`).
   - `Cost.crew` → `PendingChoice::Crew` taps other positive-power creatures
     (LookAndSelect) until power is covered. Vehicles enter summoning sick;
     `Target.another` excludes the source from trigger targets.

8. Mana (Great Divide Guide reviewed and its `INCOMPLETE` entry dropped; Raucous
   Audience, White Lotus Tile, The Lion-Turtle):
   - `Game::activated_abilities_for` fixes state-dependent mana outputs when
     offered (`current_mana_output`, CR 605.3a): an `IfCondition` (now with
     `otherwise`) between two immediate `AddMana`s, or `AddOneColorManaValue`.
   - Condition 8 `controls_matching` → `StaticCondition::ControlsMatching`;
     quantity 5 `greatest_shared_creature_type`; opcode 73 `add_mana_of_one_color`.
   - `StaticRestriction.unless`, authored as a static `branch` with empty `then`
     and `cant_attack_or_block` in `otherwise`.

9. Effective characteristics (Earthen Ally, Flopsie, The Boulder, Platypus-Bear,
   Toph the First Metalbender, The Mechanist):
   - `ScopeView` (Types → Keywords → PowerToughness → Rules) says what a static
     scope reads; `permanent_matches_predicate` is the Rules view. Restriction
     scopes now read effective power (Flopsie).
   - Layer 4 `StaticTypeGrant` (opcode 77 `add_card_types`) read through
     `Game::permanent_is_land` and predicate matching; predicate 15 `token`.
   - Field 93 `per_color_among`; opcodes 74 `earthbend_value`, 75
     `cant_be_blocked_by_more_than` (structured blocker offer goes sequential,
     as for menace), 76 `attacks_despite_defender` (read by
     `Game::permanent_can_attack`, which reads static defender too);
     trigger_event 13 `end_step_begins`.
   - `become_artifact_creature` and `set_base_pt` lower for targets too
     (`BecomeCreatureTarget`, `SetBasePtTarget`); earthbent permanents have
     base 0/0 even if they print P/T.

10. Mana remainder (Energybending, Ozai, the Phoenix King):
   - Opcode 78 `add_subtypes_controlled` (until-EOT `temp_subtypes` on matching
     permanents at resolution); `Game::activated_abilities_for` appends one
     derived "{T}: Add" choosing among gained basic land types' colors (CR 305.6).
     Basic land types never count as creature types.
   - Opcode 79 `unspent_mana_becomes_red` → `ReplacementEffect::UnspentManaBecomesRed`,
     read by `clear_mana_pools`/`clear_combat_mana_pools` (combat mana joins the
     regular pool as red); `Mana::become_red` keeps Lesson restrictions.
   - Condition 9 `unspent_mana_at_least` → `StaticCondition::UnspentManaAtLeast`
     (both pools).

11. X costs (Fated Firepower, Day of Black Sun, Katara, Water Tribe's Hope):
   - `ManaCost.x` counts {X} (0 off the stack, CR 202.3e); `with_x` pays it.
     `PendingChoice::SpellX` (after kicker, before modes) and `ActivationX`
     (before waterbend) offer payable values as Modal `ChooseMode { mode: X }`;
     X rides `Payment::Spell`, `SpellOnStack`, `ActivatedAbilityOnStack`,
     `PendingChoice::Waterbend` and `EffectFrame` (serialized only if nonzero).
     Quantity 6 `x`; field 95 `mana_x`; cost field 96 `x_min`. X activations
     must waterbend without targets (lowering rejects others).
   - Opcodes 81 `enters_with_x_counters` (`BattlefieldEntry.x`/`counters`) and
     82 `opponent_damage_plus_counters` (source-controller side, applied last).
   - Opcode 80 `destroy_creatures_losing_abilities` (field 94 `max_mana_value`):
     `Permanent.abilities_lost` (keywords, activated, static, replacement and
     triggered abilities, with `ObjectLki.abilities_lost`), one destroy batch.
   - Opcode 83 `set_base_pt_value` (team role) → `Effect::SetBasePtControlled`.

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
     `Game::in_casting_zone`; the top card is observed by its controller and
     pinned by `determinize`. `grant_keywords` EOT on
     `each:creature_you_control` → `Effect::GrantKeywordsControlled`.

Schema 8 (unreleased, T1 precedent) appended in place: fields 85-96, predicates
13-16, special roles 6-14, opcodes 66-90, enum `counter`, conditions 7-9,
quantities 5-7, trigger_event 13, subtypes 64-74.
`scratch/add_fixture.py` copies pinned evidence into the conformance fixture.

Check (slice 12, debug): `cargo test` all targets pass (cards 50, rules_tests
356, lib 120) except `semantic_conformance_tests` (w16 vs w15 and receipt digest
drift, as in PR #394 CI 40819903624); clippy, fmt clean; extension rebuilt;
`pytest tests/semantic` 158 pass, 4 fail on `schema_version == 3/4/7` asserts
from `be2b7586b`.

## Remaining (6 cards), suggested next slices

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
- Fire Nation Palace (granted firebending; observation shape change); Glider
  Staff (airbend), Trusty Boomerang (granted activated ability).

## Reconcile at sync (T2 87e012 at `83dd0cd46`, 2026-10-29)

- Both branches append to schema 8 from 86ebe9ccf with colliding numbers. T2:
  fields 85-96, predicates 13-15, subtypes 64-74, special role 6, opcodes 66-73,
  trigger_event 13, destination 2, zone 3. This branch: fields 85-96, predicates
  13-16, subtypes 64-74, special roles 6-14, opcodes 66-90, conditions 7-9,
  quantities 5-7, trigger_event 13 (`end_step_begins`), enum `counter`. T2 since
  added opcodes 74-79, fields 97-100, quantity 5 `x`, enum `grant`, role
  `each:creature_you_control` 7. The later lander renumbers its own entries
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

## Delete — do not maintain

Qualification entries, `Permanent::local_power`/`local_toughness`,
`StaticScope::Equipped`, `Game::equipped_creature` and `Permanent::can_attack`
are gone; do not restore them.
