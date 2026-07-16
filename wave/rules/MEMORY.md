# Rules memory

## Durable evidence and scheduling

- Runtime foundation evidence resolved 2026-07-15: W2-208/PR #81 proves independently reset environments, exact root/sibling forks, and retained RolloutPool slots share one versioned `Arc<ContentPack>` while mutable branch facts remain isolated; the allocation gate retains 4,096 pack references at 0 allocations/0 bytes and shows immutable-definition bytes do not increase GameState/Game clone allocations. W2-216/PR #89 merged at d8a92aad729adabf82867c19b021510c8d473165 and corrected the receipt to the canonical measured revision f7d5878b6be4b2276ac5f42f50dc5f390a6982ad with passing thresholds. Runtime KRs 1 and 2 hold; KR3 remains open. For revision-bound evidence, verify `measurement_code_revision` after the final rebase and before landing; merged status alone is not proof of freshness.
- Semantic choice and learning-input evidence verified 2026-07-15: W2-181 and
  W2-188 make Semantic KRs 2 and 3 hold; W2-189/PR #88 at
  dd0032697bcffc0dd03d374910772879e8d923b6 makes KR4 hold with the
  structured-decoder benchmark. W2-215/PR #92 at
  f567f600b0ff256fda9cdb18b9bda70bc2b569b8 makes KR5 hold with a
  viewer-safe versioned token graph, exact ContentPack binding, ragged
  projection and deterministic padding, semantic-only identity ablation,
  fail-closed artifact provenance, unknown-schema/opcode rejection, and a
  4,096-state receipt with zero projection failures or unadmitted visible
  objects. Semantic KRs 1 and 6 remain open.
- Semantic learning order advanced 2026-07-15: W2-214 is unblocked and owns
  KR6's reproducible four-arm held-out transfer experiment; W2-213 follows its
  evidence. W2-223 owns KR1's complete two-deck offline-compiler admission and
  no-card-name-dispatch proof and remains separate from the learning run.
  Filed Tasks, PR publication, and training activity are not KR evidence.
- Metta observation-history research 2026-07-15: tokenization alone did not
  create robustness. Metta needed symbolic feature specs, restore-time runtime
  ID remapping, environment-owned normalization, explicit padding, measured
  token caps, compatibility shims, bundled architecture specs, and
  cross-language packing fixes. It remains robust to changing sets of known
  facts, not to genuinely novel semantic primitives. W2-215 implements the
  versioned schema, symbolic binding, explicit program structure/masks,
  identity ablation, unknown-op rejection, and uncapped ragged projection.
  W2-214 owns permutation/rebind controls and overflow/performance receipts
  within the four-arm transfer experiment; W2-223 owns compiler-side primitive
  admission.
- Identity and event evidence verified 2026-07-15: W2-180, W2-190, and W2-191 make all Identity and Event Semantics KRs hold. The admitted slice has incarnation-changing leave/re-enter with stale `ObjectRef` rejection and LKI, typed proposed-event replacement/prevention before commit for damage, life, destruction, counters, and zone movement, and deterministic trigger/SBA fixpoint tests with incarnation-safe public rules boundaries. Treat further breadth as gap-driven follow-up.
- Search verification state 2026-07-15: Search KRs 3 and 4 hold. W2-200/PR #94
  at ac76b782c15649f4d5924e660e15e43d2febf09a adds checked reference-versus-
  optimized terminal replays, property/metamorphic checks, bounded valid-action
  fuzzing, a pinned Phase overlap matrix, and blocking semantic-conformance CI.
  KRs 1 and 2 remain open. W2-197 owns the executable fork/rollback contract
  and its failing Rust CI repair; W2-207's runtime regression gate remains
  sequenced after the branching contract and matched design evidence.

## Decisions

- The destination remains the creator's curated cube/decks; comprehensive Magic or Commander coverage is not the objective.
- Steal Phase's proven invariants, not its repository shapes: exact object incarnation and LKI, typed card meaning, proposed-event replacement flow, explicit legal interaction, and viewer-safe projections.
- Use immutable, versioned `ContentPack` definitions plus compact dense mutable `MatchState` facts.
- Prefer offline compilation into checked-in typed IR over runtime natural-language parsing.
- Project typed IR as complete variable-length semantic programs rather than
  fixed-width feature vectors. Runtime numeric IDs are transport; checkpoints
  carry stable symbolic vocabulary, value/structure encoding, budgets, and
  ContentPack compatibility in a `SemanticInputSpec`.
- Treat unseen compositions of known primitives as the transfer target. Reject
  or explicitly migrate genuinely unknown primitives; do not map them to a
  generic unknown token while claiming semantic generalization.
- Silent semantic-program truncation is forbidden. Prove the admitted pack's
  maximum budget or reject it, and report token counts and overflow alongside
  every transfer/performance result.
- Replace flat enumerate-then-clone-and-validate action lists with structured offers that are legal by construction and admit structured policy decoders.
- The search contract likely combines safe snapshot forks outside a rollout with dense transactional execution and mark/rollback inside it.
- Benchmark compact full clone, compact clone plus undo, and dense page-COW fork plus undo at realistic worker x actor x rollout load. Decide on rollout throughput and peak RSS, not clone latency alone.
- Use a readable reference reducer and optimized executor as differential oracles. Explore Phase as a pinned conformance oracle and opponent pool, not as the primary training backend.
- Verification should become mechanical: conformance CI, gap-analysis worklists, property/metamorphic/differential/fuzz testing, and a content-change to kernel-change ratio that can trigger a redesign kill decision.

## Evidence

- `docs/research/semantic-kernel.md`
- `docs/research/etude-vs-phase.md`
- `docs/research/metta-observation-robustness.md`
- Phase comparison pinned to phase-rs commit 553b97bd
- Existing capability and card-pool audits under `wave/rules/`

## Open tensions

- Prove the acceptance-slice offline compiler boundary without card-name dispatch.
- Preserve dense-state rollout speed while making branching exact and safe.
- Keep human experience, learning observations, and search snapshots as separate viewer-safe projections of one authoritative match.
- Preserve executable program structure without making the semantic encoder so
  large or irregular that it destroys the rollout throughput needed for
  training and search.
- Keep revision-bound performance and allocation receipts aligned with the exact post-rebase source they measure.
