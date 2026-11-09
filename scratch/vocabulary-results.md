# Vocabulary preflight result — 2026-11-08

Read on source `bf408ec48c207fb034ab130341be96c61c544dad` (only the uncommitted
shared-effect batching candidate differs; no protocol or analysis edits).
This is old-world vocabulary evidence, not a scored frontier run. #433's later
prospective world must not be substituted when reproducing this reading.

`experiments.tla_frontier_vocabulary.read_check` returned `passed=true`, no
failures, both arms at update 150. Run prefixes include both original arms,
both failed extension starts, both successful extensions and both evaluation
passes. The scoped reader changes only listing/cache, not the analysis.

| Comparison | Mean | Paired 95% interval |
|---|---:|---:|
| catalog u150 vs catalog u0 | .953125 | [.90625, 1] |
| hashed u150 vs hashed u0 | .9375 | [.859375, 1] |
| catalog u150 vs hashed u150 | .53125 | [.4375, .640625] |

64 games / 16 pairs each. `score`, `worst`, `best` coincide: no failed legs.
Median value losses over updates 76–150: catalog .0483422913, hashed
.0519827195; ratio .929968493 (threshold 1.10). Both `learned=true`.
This passes the declared rule; it is not an equivalence/noninferiority proof,
a replication across training seeds, or a frontier strength result.

Descriptive u100: catalog vs u0 61/64, hashed vs u0 61/64, catalog vs hashed
36/64; all games valid. Other early comparisons and their intervals are in the
reading. The full JSON is `.runs/c5fba4-20261108/c5fba4-vocabulary-scoped.json`.
The endpoint pass `tla-frontier-c5fba4-vocabulary-v1-8eecfee66463` returned
96/96 matches; its allocation estimate is $.1106688784 and rental ended.

Timing, parameter correction, candidate repair, tests and outstanding rental
reconciliation are in `scratch/direction.md`. Same-L4 baseline `7446ef290152`
finished, rental ended, allocation estimate $.0324023344 (not whole-rental invoice).
Original catalog inference/learning cost is 1.42–1.50x hashed on fixed inputs.
Before/after `9754bb253795` has no worker result: ambiguous RunPod create and
rate-limited observation. Retain the one request and its watcher; no duplicate.
Candidate code remains uncommitted until its CUDA correctness/timing reading.

Remaining before freeze: measure/deliver the repair if useful; fact-contribution
reading; integrate #433 and recalibrate its new world; implement the existing
`checkpoint_hours` stub from Etude wall_ms; fit horizons and bind the protocol.
Long runs still require the execution review. MTG-267 remains active even though
its current recorded PR #450 is merged.

The slow read was waiting on the shared Etude cache's capacity lock. A separate
small cache plus memoized reads completed the unchanged criterion. Keep future
reads scoped and isolated; don't start several shared-cache readers to speed it up.
