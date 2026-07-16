# Rules memory

## Durable evidence and scheduling

- Runtime receipt provenance correction 2026-07-15: W2-211/PR #84 merged at f7d5878b6be4b2276ac5f42f50dc5f390a6982ad with both W2-208 artifacts still recording measurement revision 4f49ad512f1af93316b82f1a8c3d00e0aa98c1b5 even though PR opening rebased onto 06acd1728ea641751c7905fafd8e78d2de5fb385. The live correction was accepted but did not incorporate before completion-triggered landing. Runtime KR1 remains open; W2-216 owns a canonical-main rerun and corrected KR1-only receipt. For revision-bound measurements, verify the recorded measurement_code_revision after the final rebase and before landing; a merged PR or completed Task is not proof of receipt freshness.
- Semantic learning portfolio expanded 2026-07-15: Semantic Programs and Choice ABI includes open KRs for a viewer-safe typed-program learning projection and a reproducible four-arm held-out transfer experiment. W2-215 and active W2-189 establish the projection and decoder inputs; W2-214 remains gated on both, and W2-213 remains gated on W2-214. These are open hypotheses; filed Tasks or training activity are not KR evidence.
- Identity and event evidence verified 2026-07-15: W2-180, W2-190, and W2-191 make all Identity and Event Semantics KRs hold. The admitted slice has incarnation-changing leave/re-enter with stale ObjectRef rejection and LKI, typed proposed-event replacement/prevention before commit for damage, life, destruction, counters, and zone movement, and deterministic trigger/SBA fixpoint tests with incarnation-safe public rules boundaries. Treat further breadth as gap-driven follow-up.
- ContentPack sharing evidence verified 2026-07-15: W2-208 merged PR #81. Focused Rust contracts prove independently reset environments, exact root/sibling forks, and retained RolloutPool slots share the admitted Arc<ContentPack> pointer, schema, and digest while mutable branch facts remain isolated. The release allocation gate retained 4,096 Arc references with 0 allocations/0 bytes and measured 1,024 GameState and Game clones with positive mutable allocation totals but exact 0 allocation/0 byte deltas after adding 4,364,032 serialized immutable-definition bytes. This is Runtime KR1-only semantic evidence, not clone latency, rollout, step, RSS, undo/page-COW, or branching-design evidence; final KR judgment still awaits W2-216's current-revision receipt.
- Semantic choice evidence verified 2026-07-15: combined W2-181 and W2-188 proof makes Semantic Programs and Choice ABI KR2 and KR3 hold. Priority, Lightning Bolt targeting, and declare attackers use revision/prompt-bound structured offers and atomic commands without the 32-choice cap, and the two-deck legacy adapter has differential legal-outcome and deterministic-trace coverage. The offline-compiler, structured-decoder benchmark, semantic projection, and transfer-experiment KRs remain open.

## Decisions

- The destination remains the creator's curated cube/decks; comprehensive Magic or Commander coverage is not the objective.
- Steal Phase's proven invariants, not its repository shapes: exact object incarnation and LKI, typed card meaning, proposed-event replacement flow, explicit legal interaction, and viewer-safe projections.
- Use immutable, versioned ContentPack definitions plus compact dense mutable MatchState facts.
- Prefer offline compilation into checked-in typed IR over runtime natural-language parsing.
- Replace flat enumerate-then-clone-and-validate action lists with structured offers that are legal by construction and admit structured policy decoders.
- The search contract likely combines safe snapshot forks outside a rollout with dense transactional execution and mark/rollback inside it.
- Benchmark compact full clone, compact clone plus undo, and dense page-COW fork plus undo at realistic worker x actor x rollout load. Decide on rollout throughput and peak RSS, not clone latency alone.
- Use a readable reference reducer and optimized executor as differential oracles. Explore Phase as a pinned conformance oracle and opponent pool, not as the primary training backend.
- Verification should become mechanical: conformance CI, gap-analysis worklists, property/metamorphic/differential/fuzz testing, and a content-change to kernel-change ratio that can trigger a redesign kill decision.

## Evidence

- docs/research/semantic-kernel.md
- docs/research/manabot-vs-phase.md
- Phase comparison pinned to phase-rs commit 553b97bd
- Existing capability and card-pool audits under wave/rules/

## Open tensions

- Prove the acceptance-slice offline compiler boundary without card-name dispatch.
- Preserve dense-state rollout speed while making branching exact and safe.
- Keep human experience, learning observations, and search snapshots as separate viewer-safe projections of one authoritative match.
- Keep revision-bound performance and allocation receipts aligned with the exact post-rebase source they measure.
