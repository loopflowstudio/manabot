# GPU validation — measured evidence and budget

Jack Heart authorized $50 total. All five rentals are deleted. Current provider
bills/estimates total $0.712507; conservative charges $1.05 plus $5 ancillary reserve
leave **$43.95 unallocated**, with no pending reservation. Exact receipts accompany
`gpu-probes/h100-005/budget.json`; `.runs/gpu-validation/budget.json` is the live ledger.
Historical protocols, predictions and failures remain in Git and hash-named notes.

## L4 evidence retained

Run001 failed bootstrap. Run002 measured model service and nine whole collections;
two learning cells failed source admission. Later runs use the existing verified
source-bundle contract. Failed attempts remain in the cost/evidence ledger.

Run003 tested frozen-weight, no-grad catalog reuse across three seeds and64/256
streams in ABBA/BAAB order. Median collection gains1.490×/1.252×; all24trajectories
matched including journals/reset seeds, with8KiB extra peak GPU allocation.
A4.541s reference outlier remains. Four ordinary three-update runs completed;
initial/final raw and EMA states matched, with3non-exact Adam moments at64streams.

Run004 added reference/reference controls: Adam differed without caching too
(max3.64e-12), ruling out caching as a necessary cause, not identifying the kernel.
Raw/EMA remained exact. Complete8×64/8×512 reference learning rates were2,273/
11,099rows/s; the4.883× improvement exceeded the prediction. Both8×4096 cases
completed two updates then hit the unchanged55second collector deadline. These
provide diagnostic prefixes, not completed whole-loop service. The single512stream
cache pair regressed collection26%; keep the counterexample. Raw data, comparator,
service records and cleanup are under `gpu-probes/l4-002`, `l4-003`, `l4-004`.

## H100 exact-input calibration

Run `6c134ed75f40`, source `cb73f3865b503d9212e9f4a4cf73b9e601fe18da`;
protocol `gpu-highend-protocol.txt`. NVIDIA H10080GB HBM3 at$3.99/hour, driver
580.126.09, Torch2.10.0/cu128, float32/TF32off, one thread. One attempt, unchanged
setup900s/measurements600s/child90s and ordinary stage55s/regime60s limits.
The source archive was completely verified before renting. No production training
rule or model implementation changed. All143returned manifest entries verified;
archive SHA `bd0b1f3c693630e4cd3c8720fc63cb523bbacb2e5f2501d0d9a6a3d12fdbee86`.

Eight model cells used exactly the L4 observation bytes, model architecture,
precision and202attention slots. The35MiB input compresses losslessly to174KiB,
retained as `gpu-probes/frozen-l4-inputs.pt.gz` with both hashes. Different hosts,
GPUs and drivers still prevent a chip-only causal interpretation.

| Model / batch | H100 inference rate relative to L4 |
|---|---:|
|64-wide,1layer /1|0.878×|
|64-wide,1layer /4|0.859×|
|64-wide,1layer /16|0.884×|
|64-wide,1layer /64|1.185×|
|64-wide,1layer /256|2.317×|
|64-wide,1layer /1024|4.999×|
|384-wide,8layers /64|4.014×|
|384-wide,8layers /256|5.489×|

All three preregistered inequalities held: small batch1<2×; small batch1024 and
large batch256>=2×. A uniform faster-GPU multiplier would be wrong. Model-only
Adam remained slower than L4 for small-model batches1–64; it is diagnostic service,
not RL. Full windows, cold times, memory and comparisons remain in `h100-005`.

All ten ordinary three-update cells completed, including8×4096. That largest
reference collected98,304learner rows,24,576exposures,24optimizer steps and63games;
collection16.040s, learning5.003s, run22.481s, peak GPU allocation9,786,454,016bytes.
The512stream reference-A learning time was1.164s for12,288rows, versus1.107s on L4:
a5× model-inference gain does not imply5× learning. Every cell changed48raw state
tensors and reported zero nonfinite/skipped steps. Keep cold and repeated phase
samples separate; method update timing excludes EMA, whole learning includes it.

All initial/final raw and EMA comparisons are exact. Adam is exact across both
reference/reference, cache/cache and reference/cache pairs at64/512streams.
At4096, reference/cache differs in2moment tensors/3elements(max5.68e-14); no
numerical tolerance was added. No exact optimizer-equivalence claim follows there.
Cache improved512stream collection1.254×/1.150× in the two ordered pairs; at4096
it improved only1.007×. The earlier L4 regression and host/time variation remain.

Pod `hahy1m0vsywsro` was confirmed absent at2026-10-15T04:19:06.668747Z.
Its rental estimate is$0.456566. Portable JSON/logs, source snapshots, predictions,
service curves and cleanup live in `gpu-probes/h100-005`; full tensors/SQLite/archive
remain in `.runs/gpu-validation/result-005` and Etude output.

## Laptop integration and proof limits

Named hardware and learner-service arguments now select L4 or H100 explicitly.
The closed original CPU phase remains238.718/240seconds. A separately preregistered
20second calibration-analysis phase consumed4.428seconds; conservative accounting
including all nested attempts is268.812/300seconds. No new local ML was measured.
All27L4 modeled outcomes remain exactly unchanged;27H100 cases each at512 and4096
global streams passed. See `learner-flow.md` and `highend-integration-protocol.txt`.

These are short untrained policies and a one-GPU service calibration. Stationary
cloned traces, assumed links and strict synchronous cohort admission are modeling
choices. Actual distributed execution, evolving trained-policy traffic, target CPU
contention, multi-GPU optimizer/all-reduce and strength per compute remain unmeasured.
