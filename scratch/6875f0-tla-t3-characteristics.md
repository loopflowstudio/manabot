# T3 — characteristics, counters, replacements, attachments (Task 6875f0)

Brief: the Task directive (49 cards, obligations in `docs/rules/tla/allocation.jsonl`,
`planned_pr == "T3"`). Stacked on T1 (892c47, PR #394 at `be2b7586b`); build beside
T2 and reconcile shared costs before landing. Every merge needs Jack Heart's review.
Card tests live in `managym/tests/cards/tla.rs` (steer, Jack Heart 2026-10-29).

## Done (20 of 49)

Slice 1, locations (Abandoned Air Temple, Agna Qel'a, Ba Sing Se, Realm of Koh):
`enters_tapped_unless` → `ReplacementEffect::EntersTappedUnlessControls` (owner's
battlefield); `Predicate::Supertype`; `Cost.sorcery_timing`; multi-instruction
activated programs lower to one `IfCondition{Always}`.

Slice 2, static grants (Air Nomad Legacy, Badgermole, Diligent Zookeeper, Foggy
Swamp Hunters, The Walls of Ba Sing Se): `StaticKeywordGrant` (layer 6) read by
`Game::effective_keywords`; scopes through one `static_scope_applies`;
`modify_pt.per_creature_type_max`; `Keyword::Defender`. Static haste/defender grants
still miss `Permanent::can_tap/can_attack` (no card needs them).

Slice 3, typed counters (Rowdy Snowballers, Mai, Jaded Edge, Heartless Act):
`CounterKind` {+1/+1, stun, fire, double strike} with one accessor; stun in
`Permanent::untap`; `Cost.exhaust`; `Predicate::NoCounters`; opcode 66
`remove_counters`.

Slice 4, base P/T (layer 7b; Flexible Waterbender, Turtle-Duck):
`Permanent.temp_base_power/_toughness`, cleared at cleanup; opcode 67 `set_base_pt`
(source, end of turn only). X costs deferred (choosing X is a new agent decision).

Slice 5, Equipment (Kyoshi Battle Fan, Meteor Sword, Twin Blades):
`Permanent.attached_to: Option<ObjectRef>`; opcode 68 `attach` →
`Effect::AttachSourceTo`; `Program::equip`/`equipped_pt`; `create_token.attach_source`;
SBA 704.5n unattaches.

Slice 6, Auras (Path to Redemption, Watery Grasp, Swampsnare Trap):
- An Aura's spell program (`Program::enchant`) is one `attach` instruction on its
  enchant target. `Card::enchant()` reads that spec; it is the cast targeting
  (303.4a) and the 704.5m legality (via `permanent_matches_spec`, which
  `target_is_legal` now wraps with presence/hexproof). The spell records the
  target in `FrameFinalize::Spell.attach_to` and enters attached (303.4f).
- `StaticScope::Equipped` → `StaticScope::Attached`, from roles
  `each:equipped_creature` and new `each:enchanted_creature`; `attached_permanent`
  / `attached_creature` replace `equipped_creature`.
- `CardDefinition.static_restrictions` (`Restriction::{CantAttackOrBlock,
  DoesntUntap}`, opcodes 69-70) read by `Game::has_restriction` in
  `combat_condition_met` and the untap step.
- `ActivatedAbilityOnStack.attached` captures the attached object before costs
  (sacrifice), becoming the frame's `context_target`; `exile`/opcode 71
  `shuffle_into_library` on `each:enchanted_creature` lower to
  `Effect::ExileAttached`/`ShuffleAttachedIntoLibrary`.
- `Cost.reduced_if_targets` → `cost_reduction_if_targets`; `effective_spell_cost`
  takes the chosen targets (None before choice: reduction if any legal target
  earns it); targets that leave the cost unpayable are not offered.

Schema 8 (unreleased, T1 precedent) appended in place: fields 85-91, predicates
13-14, special roles 6-12, opcodes 66-71, enum `counter`, subtypes 64-67.
`scratch/add_fixture.py` copies pinned evidence into the conformance fixture.

Check (slice 6, debug): `cargo test` all targets pass (cards 25, rules_tests 356,
lib 119) except `semantic_conformance_tests` (w16 vs w15; fails identically in
PR #394 CI run 40819903624); clippy, fmt clean; extension rebuilt; `pytest
tests/semantic` 158 pass, 4 fail on `schema_version == 3/4/7` asserts, inherited:
T1's head already has schema 8 (CI shows `8 == 7`) and this branch only appends.

## Remaining (29 cards), suggested next slices

- Honest Work (Aura; enchant creature an opponent controls; ETB tap + remove all
  counters; loses all abilities, becomes Citizen "Humble Merchant" base 1/1 with
  "{T}: Add {C}" — characteristic replacement, layers 1-7b).
- Equipment left: Glider Staff (needs airbend), Trusty Boomerang (granted
  activated ability).
- Vehicles/crew (new tap-creatures cost choice; type change to artifact
  creature): Fire Nation Warship, Invasion Submersible, Tundra Tank, War Balloon
  (fire counter ready; counter-threshold animation static).
- X costs: Fated Firepower (X entry fire counters; damage-addition replacement),
  Crashing Wave (waterbend {X} additional cost; up to X targets; distribute three
  stun counters), Katara, Water Tribe's Hope (waterbend {X}, X>0, own turn: team
  base X/X — needs a team role on `set_base_pt`), Day of Black Sun.
- Fire Nation Palace: granted firebending until EOT (permanent-local grant, attack
  trigger via inline effects, observation field — shape change).
- Lo and Li: Lesson-or-Noble tutor and lifelink on Lesson spells (stack-source
  keywords in `damage.rs::source_keywords`); the Noble grant is a role.
- Mana: Great Divide Guide (defined; summoning-sick grant check), Raucous Audience,
  White Lotus Tile, Energybending, Ozai, The Lion-Turtle, Secret Tunnel.
- Effective-characteristic reads: Earthen Ally, Flopsie (team counters reuse slice
  1's role; blocker restriction by effective power), The Boulder, Platypus-Bear,
  Toph (artifacts gain land type), The Mechanist (temporary animation).
- Excess damage: Razor Rings, The Last Agni Kai. Others: Hakoda.

## Reconcile at sync (observed 2026-10-28 in sibling worktrees, uncommitted)

- T1 (892c47) deleted `qualifications.json` and `cards/native.rs`; reconciled at
  sync 86ebe9ccf (entries dropped, TLA basics now `lea.*`).
- T2 (87e012, uncommitted on `be2b7586b`, re-read 2026-10-28) appends to schema 8
  with colliding numbers: fields 85-87, predicate 13 `with_counter`, subtypes
  64-70, special role 6. The later lander renumbers its own entries (append-only,
  contiguous); this branch now also holds opcodes 67-71, fields 90-91, special
  roles 11-12 and subtypes 66-67 (Turtle, Aura).
- T2 also implements exhaust and `Predicate::WithCounter` (+1/+1 only). Keep one
  exhaust cost field and gate, and one with-counters predicate reading every kind.
- T2 created its own `tests/cards.rs`, `support.rs` and `tla.rs`; merge into one of
  each. Moving `tests/rules/tla_cards.rs` belongs to T1 or whichever PR lands it.

## Delete — do not maintain

- Qualification entries (gone at the T1 sync).
- `Permanent::local_power`/`local_toughness` (removed in slice 4).
- `StaticScope::Equipped`, `Game::equipped_creature` (replaced in slice 6).
