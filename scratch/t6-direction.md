# T6 direction (Jack Heart, 2026-11-02)

- Develop T6 stacked on T5 (PR #405, Task a7234f / MTG-224). T5 is being synced
  onto main; when its synced head publishes, rebase or merge T6 onto that head
  rather than the pre-sync T5 commit (24b335fe0).
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

## Follow-up Tasks (launched by Jack Heart, 2026-11-02; state on 2026-11-02)

All target main, not T6 as first briefed: MTG-235 effect vocabulary (#412, open);
MTG-236 derived state (#411, merged, in HEAD; its Task stays open on acceptance
of absolute budgets); MTG-237 engine budget in CI (#413, merged, in HEAD);
MTG-238 catalog-scale types (#414, open, merges before T6).
