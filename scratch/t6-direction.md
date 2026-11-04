# T6 direction (Jack Heart, 2026-11-02)

- Original stacking decision: develop T6 on T5 (PR #405, Task a7234f / MTG-224),
  using its synced head rather than pre-sync `24b335fe0`. Fulfilled: T5 and the
  later MTG-235/238/239 work are integrated in T6 head `64b79d895` on main
  `c823c9c8f` (2026-11-03).
- Follow the TLA PR brief: one cohesive PR covering all 16 T6 cards, then stop
  for Jack Heart's review. No merge without that review.

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

## Related work (reconciled 2026-11-03)

Jack Heart launched MTG-235 effect vocabulary (#412), MTG-236 derived state
(#411), MTG-237 engine budget (#413), and MTG-238 catalog-scale types (#414)
on 2026-11-02, targeting main. Their code is now integrated; MTG-238
precedence is fulfilled. MTG-239 (#415) also landed and supplies the corrected
admission, library ordering and Rallier traces. These integrations do not
settle the related Tasks' independent empirical acceptance. T6's PR #417
is open at `a82a963aa`, including the audit and checkpoint-fixture repairs and simplification;
its remaining validation and review are tracked in the current plan.
