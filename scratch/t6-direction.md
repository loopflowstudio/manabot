# T6 direction (Jack Heart, 2026-11-02)

- Develop T6 stacked on T5 (PR #405, Task a7234f / MTG-224). T5 is being synced
  onto main; when its synced head publishes, rebase or merge T6 onto that head
  rather than the pre-sync T5 commit (24b335fe0).
- Follow the TLA PR brief: one cohesive PR covering all 16 T6 cards, then stop
  for Jack Heart's review. No merge without that review.

## Concept review (with Jack Heart, 2026-11-02) — proposals, not yet decided

Page: `scratch/concept-review.html`. Evidence is from origin/main `4c8324a59`
(T1–T4) and the T5 notes in `../manabot.a7234f-tla-t5-permanent`.

- **Effect vocabulary grows linearly with cards.** `Effect` had 63 variants at
  113 definitions (before T1), 83 at 412 (T2), and 107 at 657 (T4). About 45
  of the 107 are used by at most one card. The object a verb acts on is fused
  into the verb: PutCounters ×7 (OnSource, OnTriggeringPermanent,
  OnCreatedToken, Value, OnEachMatching, …), Return ×6, Sacrifice ×4, Exile ×9.
  Each variant means its own resolution arm, observability rows and an opaque
  hashed learning token, which gives the policy no transfer between cards.
  Proposal: a verb takes an `ObjectSelector` (This, Target(i), Triggering,
  Attached, CreatedThisWay, Linked, Each(predicate)) plus an `EffectValue`.
  **Open: when (question 1).**
- **Performance.** Search-64 reply time is the only engine-speed measurement.
  The worst reply was 4.34 s before T3, 9.82 s with T3, and 4.97 s after
  #410's characteristics cache. Totals were 128.7 s, 305.5 s and 151.8 s. T1,
  T2, T4 and T5 were never measured alone, and nothing has measured
  steps/s or games/s since July. No CI budget exists; Clean-machine's wait was
  raised from 30 s to 120 s. Hot spots: the cache is cleared on any mutable
  borrow; static index and characteristics cost O(permanents × (sources +
  effects)); `effects_on` scans linearly; trigger scanning walks
  `object_lki`, which grows over the game. **Open: perf gate (question 2).**
- **T6 shape (proposed).** The back face is nested in the definition
  (`back_face: Option<Box<CardDefinition>>`), and the face is recorded on the
  permanent (it resets on zone change; "return transformed" enters back face
  up). `Game::characteristics` and copiable values start from the current face,
  so a copy of a transformed DFC copies the back face and cannot transform.
  Sagas: `CounterKind::Lore`, a chapter trigger on lore added, the 714.4
  sacrifice state-based action. Turn scheduling: an extra-turn queue and an
  extra-combat count on `TurnState`, reusing T4's delayed triggers.
  **Open: faces model (question 3).**

## Engine foundations guide (2026-11-02)

`scratch/engine-foundations.html` (published: https://claude.ai/artifact/3MQACwbNDkMb2RoJMhCJfd)
lays out, with status marks (main / T6 tree / proposed):
- Faces as catalog rows: `CardDefinition.back_face`, `ContentPack::{back_face,
  front_face}`, `Card::transform`, `Game::transform_permanent`; copies take the
  showing face's row, so a copy of a transformed DFC cannot transform.
- Sagas: `CounterKind::Lore`, `TriggerCondition::Chapter`, SBA 714.4; proposed
  generalization `CountersReach { kind, at_least }`.
- Turn scheduling: `TurnState.extra_turns`/`resume_with`, `Effect::TakeExtraTurn`;
  proposed `extra_combats` + `Effect::AddCombatPhase { restriction }`.
- Proposed effect vocabulary: `ObjectSelector` + `PutCounters` / `Sacrifice` /
  `Move` / `Tap` / `Untap`, replacing ~25 fused variants; constructor helpers keep
  card code unchanged; directed tests are the acceptance check.
- Proposed derived-state work: revisions split by what the layers read, effect
  records indexed by object, caches carried into search clones, trigger index by
  event kind, subtype interning; `bench_search` + steps/s budget in CI.
- Toward Standard: sparse counters, keyword enum, `managym-cards` crate, layers
  2/3/5, retire card-shaped predicates, the approved Oracle-text generator.
Order proposed: T6 → engine budget → vocabulary → revisions → card-set crate → T7.
None of the proposals is decided.

## Accepted into T6 scope (Jack Heart, 2026-11-02): one catalog, no pack multiplicity

Decision: the content-pack simplification is part of T6, not a separate Task.

- A world has one card list. `ContentPack` keeps its two essential jobs,
  stable `CardDefId` rows (name order, so w17's manifest and `registry_key`
  values do not move) and `content_digest`, and loses multiplicity: one
  process-wide catalog, built from `cardsets::catalog()`, which `WorldIdentity`
  reads directly.
- Remove the per-game pack plumbing that exists only for authored preset packs:
  `ContentPack::from_definitions` with a `pack_key`, `pack_key` on player
  configs, `admits(pack, player)`, the same-pack-for-both-players error, the
  `CardRegistry` alias, and `pools()` as a map of packs. Authored presets in
  `setups.rs` become decklists over the one pool. `content_pool("world")` and
  `WORLD_POOL` collapse onto the catalog (keep the `"world"` name where frozen
  evidence or the Python API reads it).
- `Card.definition` may become `&'static CardDefinition` once there is one
  catalog; `Arc<ContentPack>` on game state goes away. Back-face rows and
  `back_face`/`front_face` hang off the catalog.
- Ward lowering moves out of `register_card` into the definition constructor,
  or is documented as the catalog's one normalization.
- Invariants to prove: w17's `content_digest` is unchanged (it sorts by name
  and ignores pack structure); every directed card test and
  `tests/cards/observability.rs` pass; `world_identity_names_the_version_and_card_digest`
  still holds; `CARD_DEFINITION_COUNT` equals the catalog length.
