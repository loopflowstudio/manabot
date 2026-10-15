# GPU validation — evidence and budget

Jack Heart authorized $50 total. All six rentals are deleted. Provider bills plus
the sixth rental estimate total **$1.033486**; conservative charges $1.51 plus
$5 ancillary reserve leave **$43.49 unallocated**, with no pending reservation.
`gpu-probes/serving-006/budget.json` freezes this receipt; `.runs/gpu-validation/budget.json`
is the live ledger. Earlier protocols, notes and failures remain indexed or in Git.

## Physical serving: timing prediction rejected

Run `8c51916229c9`, source `b32f4c03aecc757d3fadb5877f8ff30b0ad4c6aa`,
protocol `serving-validation-protocol.txt`. One H10080GB HBM3, driver595.91.07,
$3.99/hour. All six same-rental model calibration cells and nine serving cells
completed within unchanged limits; all37 returned manifest entries verified.
Archive SHA `f4ff5e6692209f8435eb195c71a0d91170f7839bb5ec4841e61a38ae78940fb1`.

Each cell has three finite repeats, one CUDA worker, logical actors waiting for
real CPU responses, and virtual host delays from the laptop's8×64 tape. Actual
fixed observation rows are selected by deterministic indices; these are not the
original game's per-request observations. Host simulation, IPC, network and
learning are absent. Pageable per-field copies and response validation are included.

The preregistered within20% timing prediction failed in **all nine cells**:
actual/model median duration was1.349–1.900×. Batch1024 remained the best at8/64
actors; the one-actor prediction tied all maximum batch settings.

|64 logical actors|maxbatch64|maxbatch256|maxbatch1024|
|---|---:|---:|---:|
|Predicted seconds|5.507|1.592|1.041|
|Measured median seconds|7.587|2.483|1.977|
|Actual/predicted|1.378×|1.560×|1.900×|

Declared repeat1 separates the explanations. Forward time for its actual batch
mix was0.983–1.071× the same-rental calibration across all cells. Actual transfers
were1.146–3.031× the assumed12GB/s+20us/direction model. At64actors/batch1024,
preparation took0.997s, forward0.757s, H2D0.298s, D2H0.014s and validation0.068s.
Those phases describe repeat1, not the median repeat. Preparation is a missing
service component; enlarging batches makes its omission matter more. Batch
histograms match exactly at1/8actors and differ slightly at64actors. Mean host-timer
lateness is0.078–0.164ms; maximum response delivery delay reaches2ms.

Every request and input identity passed. One request's output hash differed in
the second64actor/batch256 repetition; every other cell/repetition matched.
No tolerance or exact cross-batching numerical-equivalence claim was introduced.
Batch-dependent arithmetic and nondeterminism remain distinct possible causes.

**Revised boundary:** the queue is useful for conditional batch ranking here,
but model-only service plus idealized links does not predict absolute serving
throughput. Next calibration must include CPU gather/materialization, per-field
copies and response handling, with request count as well as padded rows. Validate
on withheld actor counts/repeats; fitting these measurements is not validation.

The initial launch stalled in an origin Git fetch before any run/rental existed.
After the computer returned, the exact owned SSH child was terminated; the launch
exited1. A fresh submission with SSH keepalive limits succeeded without changing
source or experiment limits. Pod`ec796h62yytif1`
was provider-confirmed absent at2026-10-15T12:29:48.288561Z; whole-rental estimate
$0.333071 includes setup. Portable evidence is in`gpu-probes/serving-006`; full
archive/input tensors remain in`.runs/gpu-validation/result-006` and Etude output.

## Earlier GPU findings retained

Run001 failed bootstrap. Run002 measured model service/whole collection; two
learning cells failed source admission. Run003's frozen-weight catalog reuse
improved collection medians1.490×/1.252× at64/256streams. All24 trajectories and
initial/final raw/EMA states matched; tiny Adam differences remained. Run004
reference/reference controls reproduced Adam differences without caching
(max3.64e-12), ruling out caching as a necessary cause. Its512stream cache pair
regressed26%; both4096stream cells hit unchanged55s collector deadlines after
two updates. Incomplete loops remain excluded from learner calibration.

Run005 `6c134ed75f40` calibrated exact frozen L4 input on H100, driver580.126.09,
float32/TF32off, one thread. All19 cells and143 manifest checks passed.
Small-model H100/L4 inference ratios at batches1/4/16/64/256/1024 were
0.878/0.859/0.884/1.185/2.317/4.999; large384wide8layer batches64/256 were
4.014/5.489. All three preregistered inequalities held. Different hosts/drivers
prevent chip-only attribution; small-batch diagnostic Adam was slower on H100.

All ten ordinary three-update H100 cells completed. The4096stream reference
collected98,304rows/24,576optimizer exposures/24steps/63games in16.040s collection,
5.003s learning,22.481s whole run, with9,786,454,016peak GPU bytes. At512streams,
learning1.164s versus L4's1.107s shows that5× model inference is not5× learning.
Every cell changed48 raw tensors with zero nonfinite/skipped steps. Initial/final
raw/EMA states were exact;4096 reference/cache differed in2Adam moment tensors,
3elements,max5.68e-14. The64/512 reference/reference and cache comparisons were
exact. Cache improved512 collection1.254×/1.150×, but4096 only1.007×.
Complete receipts, controls, service curves and cleanup remain in`h100-005`.

## Local integration and limits

The original CPU phase remains238.718/240seconds. The separate20second calibration
slice has used7.700seconds; conservative combined measurement use272.084/300.
No local ML was measured in physical-serving preflight. Shared FIFO extraction
preserved all27L4 outcomes exactly; fake-backend request/host-work checks passed.
Earlier27case H100 matrices at512 and4096streams remain conditional scheduling
results. Their absolute rates inherit the now-demonstrated service-boundary gap.
Actual distributed execution, evolving trained-policy traffic, target CPU contention,
multi-GPU optimization/all-reduce and strength per compute remain unmeasured.
