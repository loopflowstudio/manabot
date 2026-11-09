# Vocabulary performance evidence — 2026-11-08

Jack Heart: "you might want to take note of c5fba4's finding that catalog is 50% slower than hashed for facts right nwo"

Jack Heart: "for now probably better to test with ashed, but also use this as a lesson to look for other places we could have similar unnecessary slow downs"

c5fba4's retained diagnostic `7446ef290152`, source
`bf408ec48c207fb034ab130341be96c61c544dad`, compared catalog and hashed on
identical sampled TLA inputs on one L4, ABBA order, fresh weights, no optimizer
steps. Its timing probe excludes native collection, transfer and optimizer work.
These are model-path timings, not a whole-update multiplier or a large-model result.

| Batch | Path | Catalog seconds | Hashed seconds | Ratio |
|---|---|---:|---:|---:|
| 64 | Cached inference | .0992876 | .0664239 | 1.49476 |
| 64 | Forward/backward | .214001 | .150998 | 1.41725 |
| 256 | Cached inference | .310062 | .214489 | 1.44559 |
| 256 | Forward/backward | .740119 | .492068 | 1.50410 |

The batch-256 catalog inference profile recorded 3,699
`cudaStreamSynchronize` calls, 1,452 `aten::nonzero` calls and 15,041 kernel
launches. c5fba4 is investigating batching catalog lookups in
`SharedEffectEncoder`; its candidate has no CUDA before/after result as of the
12:16 UTC source note. That task owns this repair.

Sources read: c5fba4's `scratch/direction.md`, `scratch/vocabulary-results.md`
and `scratch/vocabulary_timing_probe.py`. The relevant evidence is copied here
so this checkout does not depend on another checkout's scratch being loaded.

b015d5 inspection: `elemental_recipe` resolves to policy input v6, configured
`record_vocabulary=catalog`, effective vocabulary **hashed**. At the frozen
`cb722dc8c`, `ModelConfig.catalog_records` requires input >=8 and a model world
in addition to the catalog setting. Thus the approved size ladder already uses
hashed; the config default alone does not describe its execution path.

Apply the finding to interpretation: label the effective input/vocabulary in
the report and do not extrapolate this ladder's costs to TLA catalog models.
Keep the frozen runs consistent. Investigate avoidable overhead in the slow
phases: repeated per-record lookups, device synchronization, host/device transfers,
and small kernel launches. Measure candidates with fixed inputs and output/gradient
parity before claiming gains; separate kernel arithmetic from implementation
overhead. No new paid arm or silent recipe change follows from this note.
