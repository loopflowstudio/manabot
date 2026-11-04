# TLA trophy pilot with configured native capacities

V6 follows the [executed v5 contract](tla-trophy-213-v5.md) with a fresh
initialization, current runtime world/content identity and current tensor widths.
The [protocol](tla-trophy-213-v6.json) binds all shapes and the hash of the
[regime](../regimes/tla-trophy-213-v6.json). The active experiment, canonical
evaluation callback and analysis use `experiments/tla_pilot_v6.py`; historical
v5 execution and analysis require source
`97a2cfb963d7e3a3b2816f31c45dda72f34cdddd`. Frozen contracts and results remain
unchanged and cannot fill any v6 evaluation slots.

V5 reached three GPU updates before 256 legal actions exceeded its 64-action
allocation. Its retained stream 103 / seed 32172 has a 76-action prefix.
Previously, native buffers ignored configured action, permanent, focus and
legacy event capacities. They now derive those axes, together with cards and
facts, from the supplied ObservationSpace. Companion buffers must agree before
writes; public-history bounds and version-fixed relation/effect layouts remain
unchanged. Choices and objects still fail on overflow rather than truncate.

This pilot explicitly chooses 512 actions under the existing enabling-fix
authorization. It keeps input v7, 128 cards per player, 40 permanents per player,
128 local facts, two focus objects and 32 legacy events, plus the fixed shared
effect shapes. Current input rows are 73 player / 56 card / 38 permanent columns.
These are bounded engineering allocations, not proof that every reachable state
fits. No new observation version or representation is introduced.

All 134 decks, independent seat draws, training/deal seeds, the learning recipe,
100-update target, milestones, eight-pair/four-leg comparisons, uncertainty
analysis and error accounting are unchanged. The complete expected evaluation
schedule remains 288 games. Current world differences from the historical
Elemental and v5 runs must remain visible in any learning/cost comparison.

The exact executable revision and source archive are recorded by Etude before
submission; `source_base` records this change's parent. A fresh provider inventory,
prior-bill reconciliation and quote must precede any rental. The cumulative
$10 ceiling and <=$1 allocation still apply. The 2026-11-04 17:33:51 settlement
recorded $0.32710677365251912 billed-plus-estimated and $3.9139916049382716
conservatively reserved; these dated amounts are not a fresh balance. The
published Etude 0.2.14 control client can select Vast under the existing GPU
direction; worker dependencies remain those pinned in the prospective source.
