# T3 — characteristics, counters, replacements, attachments (Task 6875f0)

Brief: the Task directive (49 cards, obligations in `docs/rules/tla/allocation.jsonl`,
`planned_pr == "T3"`). Stacked on T1 (892c47, PR #394 at `be2b7586b`); build beside
T2 and reconcile shared costs before landing. Every merge needs Jack Heart's review.
Card tests live in `managym/tests/cards/tla.rs` (steer, Jack Heart 2026-10-29).

## Done

Slice 1, the basic-check locations (4): Abandoned Air Temple, Agna Qel'a,
Ba Sing Se, Realm of Koh. Shared machinery, no new opcode:
- `Characteristics.enters_tapped_unless: Predicate` → `ReplacementEffect::
  EntersTappedUnlessControls`; evaluated in `collect_replacements` against the
  entering card's owner's battlefield.
- `Predicate::Supertype`; `Cost.sorcery_timing` → `ActivatedAbilityDefinition.
  sorcery_timing` (gated in legal actions and activation).
- Activated programs with several instructions lower to one `IfCondition{Always}`.
- `each:creature_you_control` role → `PutCountersOnEachMatching` (Flopsie reuses it).
- Learning schema 8 (unreleased, T1 precedent) gains fields 85/86, predicate 13,
  special role 6 (slice 3: fields 88/89, predicate 14, opcode 66, enum `counter`); `learning.py` maps supertype values and the new predicate field.

Slice 2, static grants/anthems (9 of 49): Air Nomad Legacy, Badgermole, Diligent
Zookeeper, Foggy Swamp Hunters, The Walls of Ba Sing Se.
- `StaticKeywordGrant` (layer 6) on `CardDefinition.static_keyword_grants`; one
  `Game::effective_keywords` adds them (combat vigilance/evasion now read it).
- `StaticScope` gains `YouControl`, `YouControlWithCounters`; one
  `static_scope_applies` serves P/T and keywords (P/T flying reads effective keywords;
  grants read non-static keywords, so no recursion). Scopes and
  `permanent_matches_predicate` share `permanent_matches_with` (temp subtypes,
  animation; caller supplies power and flying views).
- Static `modify_pt`/`grant_keywords` lower through `static_scope(role)`; a
  conditional `branch` wraps either. `modify_pt.per_creature_type_max` (field 87)
  scales by creature-type count (CR 205.3m; `NONCREATURE_SUBTYPES` excluded).
- Roles 7-10 (`each:creature_you_control_with_flying|_with_counters`,
  `each:non_human_creature_you_control`, `each:other_permanent_you_control`); `Keyword::Defender`.
- Conformance fixture: nine entries added from pinned `scratch/tla-evidence/cards.json`
  (no full refresh); type lines now render in printed order.
- Not covered: static haste/defender grants still miss `Permanent::can_tap/can_attack`
  (no current card grants them statically).

Slice 3, typed counters (12 of 49): Rowdy Snowballers, Mai, Jaded Edge, Heartless Act.
- `state::permanent::CounterKind` {+1/+1, stun, fire, double strike} replaces the
  crate-private proposal enum; `Permanent.counters` holds non-+1/+1 kinds;
  `counter_count`/`change_counters`/`has_counters` are the one accessor
  (`Game::change_counters` takes the kind; no +1/+1-only wrapper).
  `ProposedEvent::CounterChange` commits any kind.
- Stun (CR 122.1d) lives in `Permanent::untap`, so every untap path obeys it;
  a double strike counter feeds `Permanent::effective_keywords` (CR 122.1b).
  "With counters" scopes and quantities now read every kind.
- `put_counters.counter` (field 88, enum `counter`); `Effect::TapTarget`
  (`tap` with a target); `Cost.exhaust` (field 89) → `exhausted_abilities`
  per incarnation, gated in legal actions and activation (CR 702.177a).
- `Predicate::NoCounters` (predicate 14); opcode 66 `remove_counters` lowers to
  `Effect::RemoveCounters`, a repeated resolution `Decision::Modal` (one counter
  of a present kind, or stop).

Card tests drive through `tests/cards/support.rs` (`to_hand`, `play_land`, `cast`,
`counters_of`). `scratch/add_fixture.py` copies pinned evidence into the
conformance fixture.

Check (implement, slice 3): debug `cargo test` passes except the known
`semantic_conformance_tests` (w16 vs w15, inherited from T1); `cards` 15/15,
`rules_tests` 356; clippy clean; learning projection/compiler/current-content 41/41
after rebuilding the extension. Slice 2's Ranger and Wall subtypes lacked schema
entries (projection failed); appended as subtypes 64/65.
Check (compress, slice 3): debug `cards` 15/15, lib 119, `rules_tests` 356; clippy clean.

## Remaining (37 cards), suggested next slices

- Fire Nation Palace: granted firebending until EOT. Needs a permanent-local grant,
  an attack trigger via inline effects, and an observation field (shape change).
- Lo and Li: needs a Lesson-or-Noble library tutor and lifelink on Lesson spells
  (stack-source keywords in `damage.rs::source_keywords`); the Noble grant is a role.
- X costs (typed counters now exist): Fated Firepower (X entry fire counters;
  damage-addition replacement for sources you control, ordered with others),
  Crashing Wave (waterbend {X} additional cost; up to X targets; distribute three
  stun counters among tapped opposing creatures). Neither X nor distribution exists.
- Base P/T setting layers: Flexible Waterbender, Turtle-Duck, Katara, Honest Work.
- Attachments/Equipment/Vehicles/crew: Glider Staff, Kyoshi Battle Fan, Meteor Sword,
  Trusty Boomerang, Twin Blades, Path to Redemption, Swampsnare Trap, Watery Grasp,
  Fire Nation Warship, Invasion Submersible, Tundra Tank, War Balloon ({1}: fire
  counter is ready; needs crew and a counter-threshold animation static).
- Mana: Great Divide Guide (defined; summoning-sick grant check), Raucous Audience,
  White Lotus Tile, Energybending, Ozai, The Lion-Turtle, Secret Tunnel.
- Effective-characteristic reads: Earthen Ally (power by Ally colors), Flopsie
  (team counters reuse slice 1's role; blocker restriction by effective power),
  The Boulder (count four-power creatures), Platypus-Bear (attack despite
  defender), Toph (artifacts gain land type), The Mechanist (temporary animation).
- Excess damage: Razor Rings, The Last Agni Kai (mana until end of turn).
- Others: Day of Black Sun (X, ability removal before destroy), Hakoda (private
  top-card look, cast Ally from library, mass indestructible).

Static grants/anthems (feedback's first slice) are done except Lo and Li, which
needs tutor and stack-source keyword work above.

## Reconcile at sync (observed 2026-10-28 in sibling worktrees, uncommitted)

- T1 (892c47) deleted `qualifications.json` and `cards/native.rs`; reconciled at
  sync 86ebe9ccf (entries dropped, TLA basics now `lea.*`).
- T2 (87e012) also appends to schema 8 and took field 85 (`flashback`), which
  collides with `enters_tapped_unless: 85`; the later lander renumbers
  (keep 85..N contiguous, append-only). Same for opcode 66, predicate 14 and
  subtypes 64/65 if T2 appends them too.
- T2 also created `tests/cards.rs`, `tests/cards/support.rs` and `tla.rs`
  independently; merge into one `support.rs` (this one's `cast`/`to_hand`/
  `play_land` plus T2's helpers) and one `tla.rs`. T1 has no `tests/cards/` yet;
  moving `tests/rules/tla_cards.rs` there belongs to T1 or whichever PR lands it.

## Delete — do not maintain

Nothing replaced yet beyond the qualification entries above.
