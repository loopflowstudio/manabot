# GPU validation — current evidence and allocation

Jack Heart authorized **$50 total** GPU spend for this goal. It supersedes the
original no-paid boundary. No other experiments, Tasks or publication are selected.
Complete preregistration and earlier operational history are retained byte-for-byte
in `gpu-probes/eac79ab23e09900cabe7f6bf3e060e602c6e3fdfb56b572f70dcc9d73f07c4b0.protocol.txt`
and the committed versions of this note. Do not lose failed attempts or predictions.

## Evidence from the first completed measurement

Run `a7df3b8d1a77`, source `724da6e1ab87381c65d1e85ed54cb4176df25ed5`,
L4 23,661,248,512bytes, Torch2.10.0+cu128, Python3.12.3, float32/TF32off.
All53 manifest entries verified after fetch. Small raw records, failure logs,
summary and current capacity report are in `gpu-probes/l4-002`. Complete tensors,
trace and archive are retained under `.runs/gpu-validation/result-002`;
archive SHA `fa2b9efa8615831ec7b1a4baabb947421eaa6b4df97536883dc802a732c02e8d`.

| Small-model batch | Inference observations/s, median of3 windows |
|---|---:|
|1|314|
|4|1,314|
|16|5,151|
|64|15,071|
|256|26,316|
|1024|28,064|

Batch256 is1.75x batch64, meeting the preregistered>=20% prediction. In contrast,
width384/depth8 achieved942/934 observations/s at64/256. Batching is not always
beneficial. Actual attention length is202, versus203 in historical curves.

Repeated whole collection across three untrained seeds delivered972–1,210
**decisions**/s at16streams,2,269–2,707 at64,3,785–4,970 at256. All nine exact
same-seed rollout comparisons passed; no truncations. Every case was below50%
of best model-only throughput, meeting the second prediction. These are different
boundaries, not measured distributed efficiency. 256stream traces end after16
learner steps and only1–2 games: not mature-policy or steady traffic.

The single host profiler interval is1.389588s, with0.207562s kernel/copy union
(14.94%),68,114kernels and4,955copies. Three same-named GPU annotations are nested
duplicates, not repetitions. CPU GRU inclusive time0.2447s over119forwards;
launch runtime0.5246s overlaps those operations. Busy intervals are not SM occupancy.
CPU catalog dominance cannot be transferred quantitatively to GPU.

Both actual-learning cells failed before learning: Git archives lack`.git`.
The existing source owner supports a verified`.manabot-source.json` bundle;
use that contract, retain failed logs, and make no learning-throughput claim.

## Catalog experiment (completed)

Frozen source `a5b73db1007ae247fc18d59e08722b1e4bfbc75f`, cohort
`l4-validation-003`, source archive SHA
`6e34d4be03a2861986901a0ad7049137f602f5d10aef9694c5208583a11433a4`.
Multipart upload and complete readback finished before renting (185.034s).
The job manifest binds1938 source files and the worker-built native module.

Reference versus research-only catalog reuse inside one frozen-policy collection:
three seeds10831–10833,64/256streams,4096learner transitions, four paired samples
in ABBA/BAAB order after two-step warmup. First cache computation is timed. Require
exact rollout/journal/reset-seed parity; reject gradients or parameter-version
changes inside the cache. Destroy it before learning. Prediction frozen before
execution: >=15% median collection gain at64streams, smaller relative gain at256.
This tests a mechanism, not a production cache/invalidation API.

Ordinary three-update TrainingRun comparisons at16/64streams use separate
reference/cached processes and the same seed. Retain real exposure/skip counters
and compare final logical tensor states where possible. Do not replace RL with
the model-only diagnostic Adam measurement. Source bundle verification happens
before the existing source owner admits training. CPU preflight exact paired
parity passed in2.846commandseconds,324.2MiB sampled peak; not GPU speed evidence.

## Execution controls and retained history

All allocations used one L4 at$0.59/hour, pool ceiling$0.80, job30minutes/$1.50,
cohort$3,maxoneattempt. Driver caps: setup900s, measurements600s, child90s.
Etude owns source transport and self-deletion; credentials stay in-process.
The first run failed bootstrap before measurements; the second retained two
source-admission failures. Their costs and evidence are included, not discarded.
Exact allocation IDs, cleanup receipts, CLI recovery and complete operational
history are retained in`gpu-probes/f81f288a4bde6776f15cb9d0ce4c7aab6034d08ed3eb07cecd701cb91eace2bc.history.txt`.
The current ledger is`.runs/gpu-validation/budget.json`; its latest snapshot is
`gpu-probes/l4-003/budget.json`. No allocation or observation process remains live.

## Catalog results and remaining uncertainty

All59 returned manifest files verified; archiveSHA
`3a4c377151f535fe5b90863585c3719aebb680f58aed01bb92be7b0b9082bb9b`.
Full small records and tensor comparison: `gpu-probes/l4-003/summary.json` and
`tensor-comparison.json`; raw tensor archive in`.runs/gpu-validation/result-003`.
Third provider pod absent at02:55:38.926UTC. All three allocations are deleted.
Total rental estimate$0.20; conservative charges$0.35 plus ancillary reserve$5
leave$44.65 unallocated. No new allocation is pending. Full ledger is retained
with the third results; billing amounts remain estimates.

Across seeds,64stream collection speedups were1.490/2.181/1.466x (median1.490x);
256stream1.297/1.252/1.228x (median1.252x). All24 timed trajectories matched
exactly within their seed/stream pairs, including rollout, journal and reset
seeds. Extra peak GPU allocation was8192bytes. Both catalog predictions passed.
The2.181x case includes a4.541s reference outlier versus2.285s on its repeat;
retain it. The median result and all other paired comparisons still show gains.

All four three-update training runs completed, each1536learner transitions and
384optimizer exposures;16streams had96optimizer steps/11games,64streams24steps/
zero finished games. No nonfinite or skipped steps were reported. Initial, final
raw and EMA tensor states matched exactly between reference/cache, as did all
scientific diagnostics after excluding timing/resource coordinates.48state tensors
changed during each run, so this was actual optimization, not an empty-filter smoke.
16stream Adam state also matched exactly.64stream Adam state had3differing moment
tensors, maximum absolute delta3.64e-12; no tolerance was preregistered, so exact
optimizer parity is **not** established. Possible explanations include ordinary
CUDA reduction variation and cache-dependent execution/allocation ordering.
A reference/reference plus cached/cached repeatability control is the next small
check before selecting a production implementation. No new tolerance is invented.

Whole-run times were8.825→5.847s (16streams),3.143→2.916s (64streams).
These are single reference-then-candidate comparisons, not a robust training-speed
claim; initialization/host variability contributes. Driver580.159.04 differs
from the prior rental's595.91.07, so use within-rental pairs. Zero completed games
in64stream learning also limits representativeness. This validates catalog reuse
as an optimization direction; production API/invalidation, mature-policy traffic,
and a learner/backpressure/policy-age model remain unimplemented/unmeasured.
