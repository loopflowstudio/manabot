# T3 — characteristics, counters, replacements, attachments (Task 6875f0)

Brief: the Task directive (49 cards, obligations in `docs/rules/tla/allocation.jsonl`,
`planned_pr == "T3"`). Stacked on T1 (892c47, PR #394 at `be2b7586b`); build beside
T2 and reconcile shared costs before landing. Every merge needs Jack Heart's review.
Card tests live in `managym/tests/cards/tla.rs` (steer, Jack Heart 2026-10-29).

## Done (24 of 49)

1. Locations (Abandoned Air Temple, Agna Qel'a, Ba Sing Se, Realm of Koh):
   `ReplacementEffect::EntersTappedUnlessControls`; `Predicate::Supertype`;
   `Cost.sorcery_timing`; multi-instruction activations lower to one `IfCondition{Always}`.
2. Static grants (Air Nomad Legacy, Badgermole, Diligent Zookeeper, Foggy Swamp
   Hunters, The Walls of Ba Sing Se): `StaticKeywordGrant` read by
   `Game::effective_keywords`; one `static_scope_applies`; `per_creature_type_max`.
   Static haste/defender grants still miss `Permanent::can_tap/can_attack`.
3. Typed counters (Rowdy Snowballers, Mai, Heartless Act): `CounterKind`, stun in
   `Permanent::untap`, `Cost.exhaust`, `Predicate::NoCounters`, opcode 66.
4. Base P/T, layer 7b (Flexible Waterbender, Turtle-Duck): opcode 67 `set_base_pt`
   (source, end of turn only).
5. Equipment (Kyoshi Battle Fan, Meteor Sword, Twin Blades): `Permanent.attached_to`,
   opcode 68 `attach`, `create_token.attach_source`, SBA 704.5n.
6. Auras (Path to Redemption, Watery Grasp, Swampsnare Trap): `Program::enchant`
   is the cast targeting (303.4a) and 704.5m legality via `permanent_matches_spec`;
   `StaticScope::Attached`; `static_restrictions` (opcodes 69-70); activated
   abilities capture the attached object before costs (opcode 71);
   `cost_reduction_if_targets`.
7. Vehicles (Fire Nation Warship, Invasion Submersible, Tundra Tank, War Balloon):
   - `Permanent.becomes_creature: Option<BecomesCreature{EndOfTurn, Incarnation}>`
     (longer wins; cleanup clears EndOfTurn) and `CardDefinition.creature_with_counters`
     feed `Permanent::animated_creature`, which `is_creature`, predicate matching,
     trigger subjects, LKI and the observed `is_animated` all read.
   - Opcode 72 `become_artifact_creature` (source; `end_of_turn`, absent =
     incarnation, or `static` under a `source_counters_at_least` branch).
   - `Program::crew` / `Cost.crew` → `ActivatedAbilityDefinition.crew`;
     `PendingChoice::Crew` offers other untapped positive-power creatures as
     LookAndSelect `SelectCard`, tapping each at once until power is covered.
   - Vehicles enter summoning sick (CR 302.6); `Target.another` →
     `TargetRequirement.another` excludes the source from trigger targets.

Schema 8 (unreleased, T1 precedent) appended in place: fields 85-92, predicates
13-14, special roles 6-12, opcodes 66-72, enum `counter`, condition 7, subtypes 64-68.
`scratch/add_fixture.py` copies pinned evidence into the conformance fixture.

Check (slice 7, debug): `cargo test` all targets pass (cards 29, rules_tests 356,
lib 119) except `semantic_conformance_tests` (world w16 vs w15 and receipt digest
drift; fails in PR #394 CI run 40819903624); clippy, fmt clean; extension rebuilt;
`pytest tests/semantic` 158 pass, 4 fail on `schema_version == 3/4/7` asserts,
inherited from `be2b7586b` (test files unchanged since it).

## Remaining (25 cards), suggested next slices

- Effective-characteristic reads: Earthen Ally, Flopsie (team counters; blocker
  restriction by effective power), The Boulder, Platypus-Bear, Toph (artifacts gain
  land type), The Mechanist (temporary animation — `becomes_creature` may fit).
- Mana: Great Divide Guide, Raucous Audience, White Lotus Tile, Energybending,
  Ozai, The Lion-Turtle, Secret Tunnel.
- Honest Work (Aura; loses all abilities, becomes Citizen "Humble Merchant" 1/1
  with "{T}: Add {C}" — characteristic replacement, layers 1-7b).
- X costs: Fated Firepower, Crashing Wave, Katara Water Tribe's Hope (team role
  on `set_base_pt`), Day of Black Sun.
- Fire Nation Palace (granted firebending; observation shape change); Lo and Li
  (stack-source keywords in `damage.rs::source_keywords`); Glider Staff (airbend),
  Trusty Boomerang (granted activated ability); Razor Rings, The Last Agni Kai
  (excess damage); Hakoda.

## Reconcile at sync (observed 2026-10-28 in sibling worktrees, uncommitted)

- T1 (892c47) deletions reconciled at sync 86ebe9ccf.
- T2 (87e012) appends to schema 8 with colliding numbers: fields 85-87, predicate
  13 `with_counter`, subtypes 64-70, special role 6. This branch holds fields 85-92,
  predicates 13-14, subtypes 64-68, special roles 6-12, opcodes 66-72, condition 7.
  The later lander renumbers its own entries (append-only, contiguous).
- Keep one exhaust cost field and gate, and one with-counters predicate reading
  every `CounterKind`. Merge T2's `tests/cards.rs`, `support.rs`, `tla.rs` with
  these. Moving `tests/rules/tla_cards.rs` belongs to whichever PR lands it.

## Delete — do not maintain

Qualification entries, `Permanent::local_power`/`local_toughness`,
`StaticScope::Equipped` and `Game::equipped_creature` are gone; do not restore them.
