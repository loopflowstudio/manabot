# T3 — characteristics, counters, replacements, attachments (Task 6875f0)

Brief: the Task directive (49 cards, obligations in `docs/rules/tla/allocation.jsonl`,
`planned_pr == "T3"`). Stacked on T1 (892c47, PR #394 at `be2b7586b`); build beside
T2 and reconcile shared costs before landing. Every merge needs Jack Heart's review.
Card tests live in `managym/tests/cards/tla.rs` (steer, Jack Heart 2026-10-29).

## Done (17 of 49)

Slice 1, basic-check locations: Abandoned Air Temple, Agna Qel'a, Ba Sing Se,
Realm of Koh. `Characteristics.enters_tapped_unless` → `ReplacementEffect::
EntersTappedUnlessControls` (owner's battlefield); `Predicate::Supertype`;
`Cost.sorcery_timing`; multi-instruction activated programs lower to one
`IfCondition{Always}`; `each:creature_you_control` → `PutCountersOnEachMatching`.

Slice 2, static grants/anthems: Air Nomad Legacy, Badgermole, Diligent Zookeeper,
Foggy Swamp Hunters, The Walls of Ba Sing Se. `StaticKeywordGrant` (layer 6) read by
one `Game::effective_keywords`; `StaticScope::{YouControl, YouControlWithCounters}`
through one `static_scope_applies`; `modify_pt.per_creature_type_max` (CR 205.3m,
`NONCREATURE_SUBTYPES` excluded); roles 7-10; `Keyword::Defender`. Static haste/
defender grants still miss `Permanent::can_tap/can_attack` (no card needs them).

Slice 3, typed counters: Rowdy Snowballers, Mai, Jaded Edge, Heartless Act.
`CounterKind` {+1/+1, stun, fire, double strike} with one accessor
(`counter_count`/`change_counters`/`has_counters`); stun in `Permanent::untap`;
double strike counter feeds `effective_keywords`; `put_counters.counter`;
`Effect::TapTarget`; `Cost.exhaust` → `exhausted_abilities`; `Predicate::NoCounters`;
opcode 66 `remove_counters` (repeated resolution `Decision::Modal`).

Slice 4, base P/T setting (layer 7b): Flexible Waterbender, Turtle-Duck.
X costs were deferred: choosing X is a new agent decision (action-space and
observation change), larger than one slice.
- `Permanent.temp_base_power/_toughness: Option<i32>` (serialized only when set,
  cleared at cleanup; a later setter overwrites = timestamp order). `effective_pt`
  applies them after the 7a CDA/printed base, before 7c terms and counters.
- Opcode 67 `set_base_pt {target, duration, power?, toughness?}` lowers only for
  `source`/end of turn to `Effect::SetBasePtUntilEot`. `grant_keywords` on
  `source` until EOT lowers to `Effect::GrantKeywordsToSource`.
- Deleted `Permanent::local_power/local_toughness` (no callers; would ignore 7b).

Slice 5, Equipment: Kyoshi Battle Fan, Meteor Sword, Twin Blades.
`Permanent.attached_to: Option<ObjectRef>` (exact object; serialized only when
set); opcode 68 `attach` → `Effect::AttachSourceTo`; `Program::equip` (sorcery
timing, creature you control) and `Program::equipped_pt`; static scope role
`each:equipped_creature` → `StaticScope::Equipped` via `Game::equipped_creature`;
`create_token.attach_source` (authored as `Op::CreateTokenAttached`, one token);
SBA 704.5n clears `attached_to`; authoring `Keyword::DoubleStrike`.

Schema 8 (unreleased, T1 precedent) appended in place: fields 85-90, predicates
13-14, special roles 6-11, opcodes 66-68, enum `counter`, subtypes 64-66
(Ranger, Wall, Turtle). Card tests drive through `tests/cards/support.rs`;
`scratch/add_fixture.py` copies pinned evidence into the conformance fixture.

Check (slice 5, debug): `cargo test` all targets pass except known
`semantic_conformance_tests` (w16 vs w15, digest drift; from T1); `cards` 20/20,
lib 119, `rules_tests` 356; clippy, fmt clean; `pytest tests/semantic` 158 pass,
4 fail on inherited `schema_version == 3/4/7` asserts (T1's schema 8).

## Remaining (32 cards), suggested next slices

- Equipment left: Glider Staff (needs airbend), Trusty Boomerang (granted
  activated ability).
- Auras (cast targeting, enchant legality, 704.5m): Path to Redemption,
  Swampsnare Trap, Watery Grasp, Honest Work (ability loss, base 1/1 via 7b).
  `attached_to` and `StaticScope::Equipped` generalize to "enchanted".
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
  contiguous); this branch now also holds opcodes 67-68, field 90, special role 11 and subtype 66.
- T2 also implements exhaust and `Predicate::WithCounter` (+1/+1 only). Keep one
  exhaust cost field and gate, and one with-counters predicate reading every kind.
- T2 created its own `tests/cards.rs`, `support.rs` and `tla.rs`; merge into one of
  each. Moving `tests/rules/tla_cards.rs` belongs to T1 or whichever PR lands it.

## Delete — do not maintain

- Qualification entries (gone at the T1 sync).
- `Permanent::local_power`/`local_toughness` (removed in slice 4).
