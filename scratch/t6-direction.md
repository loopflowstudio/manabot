# T6 direction (Jack Heart, 2026-11-02)

## Accepted review completion (Jack Heart, 2026-11-03)

Jack Heart selected 1a: complete Academy's native schedule projection in T6
before landing. Display ordered scheduled turns and current/pending extra
combats with their public attacker restrictions; per-player counts are
insufficient. Keep native authority, viewer safety, historical unknowns and
retained-session recovery. The concrete contract, counterexamples and proof
are in the interactive concept-review section of `mtg-225-tla-t6.md`.
Implement on the existing Task/PR #417, publish and return to review; no merge
or auto-merge is authorized.

- Original stacking decision: develop T6 on T5 (PR #405, Task a7234f / MTG-224),
  using its synced head rather than pre-sync `24b335fe0`. Fulfilled: T5 and the
  later MTG-235/238/239 work are integrated in T6 head `bf7631c6a` on main
  `3b122de90` (2026-11-03).
- Follow the TLA PR brief: one cohesive PR covering all 16 T6 cards, then stop
  for Jack Heart's review. No merge without that review.

## Delayed-trigger correction (Jack Heart, 2026-11-03)

Accepted review feedback removes token-specific delayed timing. Capture an
`ObjectSelector` at creation, retain exact incarnations separately from event
context, then execute ordinary effects when the event arrives. Visions keeps
controller-next-end-step timing; Trebuchet keeps any-next-end-step timing.
The working design records the cutover, observation contract and remaining gate
work. This is part of T6 on PR #417, with the same stop-for-review boundary.

## Concept review (with Jack Heart, 2026-11-02)

Page: `scratch/concept-review-preserved-untracked.html`. Its evidence is from
main `4c8324a59` and predates MTG-236. Its T6 shape was a proposal; the built
model is in [mtg-225-tla-t6.md](mtg-225-tla-t6.md) and differs in three ways: the
face is on `Card`, additional combats are a stack with an optional restriction,
and a copy of a transformed permanent can transform (CR 707.8a).

The vocabulary, performance and derived-state proposals there and in
`scratch/engine-foundations.html` (https://claude.ai/artifact/3MQACwbNDkMb2RoJMhCJfd)
became the follow-up Tasks below; T6 does not implement them.

## Accepted into T6 scope (Jack Heart, 2026-11-02): one catalog, no pack multiplicity

Decision: the content-pack simplification is part of T6, not a separate Task.

- A world has one card list. `ContentPack` keeps its two essential jobs,
  stable `CardDefId` rows (name order, so w17's manifest and `registry_key`
  values do not move) and `content_digest`, and loses multiplicity: one
  process-wide catalog, built from `managym_cards::catalog()` after MTG-238, which `WorldIdentity`
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

## Related work (reconciled 2026-11-03)

Jack Heart launched MTG-235 effect vocabulary (#412), MTG-236 derived state
(#411), MTG-237 engine budget (#413), and MTG-238 catalog-scale types (#414)
on 2026-11-02, targeting main. Their code is now integrated; MTG-238
precedence is fulfilled. MTG-239 (#415) also landed and supplies the corrected
admission, library ordering and Rallier traces. These integrations do not
settle the related Tasks' independent empirical acceptance. T6's PR #417
is open at `bf7631c6a`, including the audit/checkpoint repairs, schedule projection,
composable delayed capture and main's v6 fact-capacity integration. Mutable w17
receipt refresh and combined validation remain outstanding in the current plan;
old pending-CI and missing-recovery notes are superseded.

## Accepted delayed-lifetime composition (Jack Heart, 2026-11-03)

Jack Heart selected 1a in the subsequent delayed-ability concept review:
within T6, independently compose event matching, firing frequency and expiry
for the existing delayed abilities. Keep exact creation-time selector capture,
separate event context, existing card semantics and exact death-watch retirement.
Do not migrate all printed triggers or add a second DSL/hypothetical cards.
Use the accepted contract and proof cases in the working design. This resolves
the headless review's open choice. The composition and focused proofs are now
implemented; gate owns combined validation and mutable receipt refresh, followed
by publication of PR #417 and review, without merge authorization.
