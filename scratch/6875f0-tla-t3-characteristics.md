# T3 — characteristics, counters, replacements, attachments (Task 6875f0)

Brief: the Task directive (49 cards, obligations in `docs/rules/tla/allocation.jsonl`,
`planned_pr == "T3"`). Stacked on T1 (892c47, PR #394 head `86ebe9ccf`) and synced onto T2
(87e012, PR #398 head `16dfd3dbb`). Every merge needs Jack Heart's review.
Card tests live in `managym/tests/cards/tla.rs` (steer, Jack Heart 2026-10-29).

## Done (49 of 49)

Per-slice detail: `git log 86ebe9ccf..`.

1. Locations: `ReplacementEffect::EntersTappedUnlessControls`, `Predicate::Supertype`,
   `Cost.sorcery_timing`; multi-instruction activations lower to `IfCondition{Always}`.
2. Static grants: `StaticKeywordGrant` via `effective_keywords`, one
   `static_scope_applies`.
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
   lowers an activated program into `CardDefinition.attached_grants`;
   `activated_abilities_for` lists them on the attached creature after its own
   (`AbilityGrant`), and every activation path reads abilities only through it.
   `ActivatedAbilityOnStack.granted_by` carries that exact grantor object
   and grant at activation; `EffectFrame.grantor` feeds `return_to_hand` on
   special role 15 `grantor` (`Effect::ReturnGrantorToHand`, CR 400.7).
14. Honest Work: static opcode 91 `set_characteristics` (field 98 `name`, enum
   `name`) → `CardDefinition.attached_characteristics`; `Game::set_characteristics`
   feeds `has_lost_abilities` (every former `Permanent.abilities_lost` read:
   statics, keywords, triggers' LKI, replacements, combat fields, fixed and
   activated mana), `permanent_name` (legend rule), `permanent_has_subtype` /
   `creature_types` and 7b base P/T. `granted_to` also accepts
   `each:enchanted_creature`; only the setter's own grants survive the loss.
   Opcode 92 `remove_all_counters` and `tap` on the enchanted creature lower to
   `Effect::RemoveAllCountersFromAttached` / `TapAttached`, read through
   `frame_attached` (activation capture, else the source's attachment).
   Moving an attachment with grants or characteristics invalidates both
   players' mana caches.

15. One tap authority: `Game::permanent_can_tap` (statics.rs) over
   `Game::effective_keywords` and `permanent_is_creature` serves mana, activation,
   tap costs and attack offers; `Permanent::can_tap`, `producible_mana` and the
   `abilities_lost` check in `Permanent::effective_keywords` are gone. Tests:
   `honest_work_takes_haste_from_a_summoning_sick_merchant`,
   `a_static_haste_grant_lets_a_creature_tap_the_turn_it_enters`.
16. Secret Tunnel: target field 99 `shares_creature_type`
   (`TargetRequirement.shares_creature_type`, authoring `sharing_creature_type()`,
   `learning.py` optional target field); subtype 75 `Cave`.
   `PendingChoice::ActivatedTarget.chosen` (actions, continuation clause 0
   targets, decline once `min` is met) via `Game::activation_target_requirement`
   and `activated_ability_targets(.., chosen)`, which offers a first target
   only with a partner. `resolution_targets` applies
   `require_shared_creature_type` over `last_known_creature_types`; a departed
   target counts with `ObjectLki.creature_types`, found by `ObjectLki.permanent`
   through `departed_permanent_lki` (no stack shape change; `ObjectLki` is now Clone).

Schema 8 (unreleased, T1 precedent) appended in place: fields 85-99, predicates
13-16, special roles 6-15, opcodes 66-92, enums `counter` and `name`, conditions 7-9,
quantities 5-7, trigger_event 13, subtypes 64-75.
`scratch/add_fixture.py` copies pinned evidence into the conformance fixture.

17. T2 sync (2026-10-29): merged PR #398 head `16dfd3dbb` (T2 87e012). One X
   announcement for spells (T2's `CastChoices`/`AnnounceX`); `ActivationX`
   stays, its X on `Payment::Activation.x`. One exhaust gate
   (`Permanent::activation_used`), one `state::is_zero`, one `ZoneEntry` (with
   `x`, `counters`). `EarthbendValue` deleted for T2's `Earthbend { count: Count }`.
   Activated targets run on `ActivationSource` with `chosen`; library-top casting
   is a `zone_cast_cost` case. Schema 8 renumbered after T2's entries (details
   and test-helper names: `scratch/questions.md`). World w17 with its own entry.
   Compress after sync: spell and activation X share `payable_x_values` (no cap);
   Badgermole's scope is `YouControl` with `CardPredicate.with_counter` and Dai
   Li Agents' quantity counts +1/+1 counters only (a stun counter no longer
   qualifies).

18. Glider Staff and Crashing Wave (2026-10-29): Glider Staff is authored on
   T2's `Op::Airbend` (up to one target creature), `equipped_pt` 1/1, a static
   flying `GrantKeywords` on `each:equipped_creature` and `equip {2}`.
   Crashing Wave: authoring `Cost.additional` (with `waterbend`) lowers to
   `CardDefinition.waterbend_cost`, a required waterbend additional cost whose
   {X} is announced with the spell's (`begin_cast`), added in
   `effective_spell_cost` and waterbent through `cast_waterbend_generic`; the
   mana value is unchanged. `Target::up_to_x` (field `max_x`) →
   `TargetRequirement.max_x`/`max_for(x)` in target selection. Opcode 106
   `distribute_counters` on special role 17
   `each:tapped_creature_opponents_control` →
   `Effect::DistributeCountersToTappedOpponentCreatures` →
   `Decision::DistributeCounters` (continuation kind 19, stage
   `distribute_counters`), one `SelectCard` placement per counter. Schema 8
   gains fields 115 `additional`, 116 `max_x`.

Compress after item 18: Glider Staff uses `Target::up_to`; `counter_kind` takes
`Option<&str>`; the "up to X" bound check is `max < 1 && !max_x`; a counter
recipient resolves through `card_to_permanent`. IR unchanged.

Check (item 18 + compress, debug): `cargo test --no-fail-fast` passes on every target
(lib 122, cards 124, rules_tests 356) except `semantic_conformance_tests` (w15
receipts, as on PR #394/#398 CI); clippy and fmt clean; before the compress (IR unchanged by it) extension rebuilt and
`pytest tests/semantic` 160 pass, 3 fail on T1's frozen `schema_version` asserts. World-change
Python suite (CLAUDE.md) left to gate.

19. Fire Nation Palace and the observation change (2026-10-29): a fifth
   `basic_check_land`; `grant_firebending_until_eot` takes an optional `target`
   (Sozin's Comet's IR unchanged) → `Effect::GrantFirebendingToTarget`.
   Observation-only fields on T2's `face_down` precedent, omitted at defaults:
   `PermanentData.{stun,fire,double_strike}_counters`, `attached_to`, `name` and
   `creature_types` (set by an effect), `is_granted_land`, `firebending`;
   `StackObjectData.{x, granted_keywords}`; battlefield `CardData.is_ally` reads
   `permanent_has_subtype`. PyO3, `.pyi`, protocol views (`counters`,
   `attached_to`, `creature_types`, `is_granted_land`, `firebending`, effective
   `name`; stack `x`), Academy `types.ts` and the six protocol schemas follow.
   No tensor column or policy-input version.

Check (item 19, debug): fmt and clippy clean; `cargo test --no-fail-fast` 721 pass,
2 fail (`semantic_conformance_tests`: w15 receipts, known). Extension rebuilt.
Protocol schema tests and `pytest tests/semantic` pass but for 3 frozen
`schema_version` asserts. CLAUDE.md world suite: 56 pass, 10 fail, all on
`manabot/sim/registration.py`'s world literal (stops at w15) or that assert;
Academy `test_ur_lessons_increment` / study bench fail on w13 pins and the stale
study source closure — all world-release evidence, left to gate.

Compress after item 19 (`57742379c`): shared `Game::grant_firebending`; one
`set_characteristics` read in `add_permanent`.

21–22. Card-test layout (Jack Heart's 2026-10-29 steer; move-only): every
   single-card test from `tests/rules/` (including all 37 of the deleted
   `tla_content.rs`) is in `tests/cards/<set>.rs`, by the card it names, else its
   first `setup` card; TLA's own tokens stay in `tla.rs`. Shared drivers are in
   `cards/support.rs`, which `tests/rules/` also mounts; `rules/tla_mechanics.rs`
   keeps cross-card machinery. PR #394 (head `86ebe9ccf`) keeps the old layout.
   Compress (`d261acf7a`, then this pass): support helpers build on one another;
   `tla.rs` uses support's `resolve`, `mode`, `ready`, `hand_of`,
   `available_card`, `library_card` instead of local copies, and one
   `put(s, player, name, zone)`. Check (debug): fmt/clippy clean; cards 349 +
   rules_tests 132 = 481; one intermittent `structured_offers` failure
   (`scratch/questions.md`).

## Remaining

All 49 cards are done; next is gate. World release items (registration
literal, receipts, frozen asserts) belong to the stack's world release, not this
slice. Browser proof of the stunned and equipped creature is left to demo. Policy
tensor encoding of item 19's fields is undecided: Jack Heart's PR review chooses
between a policy-input version bump here and a later Task (`scratch/questions.md`).

## Accepted model (concept reviews, 2026-10-29)

A permanent's characteristics are what `Game` derives from them (`ScopeView`
layers, `has_lost_abilities`, `permanent_name`, `permanent_has_subtype`,
`activated_abilities_for`). `Permanent` and `Card` fields are inputs, never
answers. Observations stay complete (acceptance; Jack Heart's 2026-10-21
shared-decision-surface direction), so item 19 added counters by kind,
`attached_to`, effective name and creature types, granted land, stack X, spell
keyword grants and granted firebending, following T2's `CardData.face_down`
precedent. Frozen widths are unchanged.

Item 20 (`a1d67b416`) gives people and LLM bots the same fields as labels only:
`permanent-notes.ts` (counters by kind, "attached to X" / "X attached" across
seats via `tablePermanents`, set creature types, granted land, firebending) feeds
`Card`'s accessible name and margin chips, and the ordered stack shows "X = n".
`text_observation.py` appends the same fields only when they are present, so
existing text is byte-identical. `_characteristics` in the projection omits on
`None`, not falsiness. Compress: `tablePermanents` is required through
PlayerArea → Battlefield → BattlefieldZone (GameBoard always supplies it); only
`PermanentRow` defaults it.

Check (item 20 + compress): vitest 23 files/113 pass, `npm run check` 0 errors
(nvm node 24; Homebrew node 25.8 lacks llhttp); `pytest
tests/semantic/test_text_observation.py` 18 pass, `tests/academy/test_play_modes.py`
12 pass; ruff clean. No Playwright run.

## Delete — do not maintain

Qualification entries, `Permanent::local_power`/`local_toughness`,
`Permanent::can_tap`/`producible_mana`,
`StaticScope::Equipped`/`YouControlWithCounters`, play.rs `MAX_X`,
`Game::equipped_creature` and `Permanent::can_attack`
are gone; do not restore them.
