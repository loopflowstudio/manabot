# MTG-177 shipping integration

Jack Heart authorized shipping the reviewed bounded increment on 2026-10-22.
The original whole-rules audit remains unfinished; keep MTG-177 open.

Gate passed before syncing newer main. Sync incorporated the separately owned
shared-payment repair from main and preserved the frozen w12 targeting corpus.
At head 749e9caec, both CI Rust Tests and Semantic Kernel Conformance fail on
the same witness: `temur-164`, checkpoint after command 10. The state hash
differs, while action-space digest, pending-choice digest, event boundary and
terminal flags match. Rust job 120032706254 in run 39838399655 confirms it.
This is one reproduced integration failure, not two independent defects.

Resolve the combined rules/world evidence correctly. Diagnose the semantic
change before deciding identity; do not overwrite frozen corpora or weaken
assertions. If combined mechanics require a fresh world, update selectors and
produce fresh compatible evidence with the existing bounded manifests and
acceptance budgets. Preserve both independently accepted rule repairs.

The initial landing process stopped before repair. Fresh Task/Session/process
reads confirmed no live landing or CI repair process. A fresh `lf sync` completed
without changes. CI additionally found a stale `compatible w11` expectation in
`tests/semantic/test_elemental_world.py:125`; runtime correctly rejects the old
world with `compatible w12`. Other completed jobs passed.

Resume `lf land --wait-and-fix` for publication, CI repair and merge waiting on
PR354. Do not start competing repair while its process is live. The whole-rules
audit remains open after this bounded increment lands.
