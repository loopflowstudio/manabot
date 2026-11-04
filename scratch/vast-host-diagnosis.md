# Vast admission diagnosis — 2026-11-04

Jack Heart's 22:38 UTC direction: diagnose our launch/training implementation
before assigning host fault, correct fixed-cost extrapolation and add prospective
startup/progress/affordability stops. Original $25 total and $0.50 admission cap
remain; no competing rental, scientific shortening, merge or promotion.
Direction: Task comment `aece955c-e64d-4f51-9336-311898578948`, Session steer
`ab0fe717-1f66-4671-8aec-a802ac927a60`.

## Completed admission, scientific objective still unmet

Run `803bcc26e7de`, source `22c10ca55197ab225546e5880cc53709db985901`,
Etude lock `64e9eaabf016f567cb9567280bcf8730f6f5d524`, succeeded at
22:56:05 UTC. Both excluded seed-74200 learners completed four updates and
1,024 games on the same RTX 3090, CUDA float32, one actual thread, 256 streams.
No scientific seed or score ran. All 46 output files (131,204,931 bytes) match
the worker's SHA256/size inventory. Etude's courier embeds `etude-result.json`
in the attempt record but its fetch path requests an absent standalone object;
that 39-byte file was recovered from transported clocks using the frozen JSON
formatter and matches the original hash. All other files were fetched normally.

Instance `59168556` was terminated through Etude after completed delivery;
provider absence confirmed 22:58:28 UTC. Its reported charge is $0.205 versus
$0.20159 whole-lifetime quote estimate; no final settlement claim. Combined
reported-plus-settled compute is $0.85458 using the prior 21:59 read. Conservative
reserves still leave $16 for science. Evidence is manifest-backed under
`experiments/data/state-representation-gpu-5da5f4/cheaper-3090/`.

## Measured work and limits of attribution

| Host wall seconds / count | Pool | GRU |
|---|---:|---:|
| Executor total | 1,286.32 | 1,419.93 |
| Executor setup | 0.34 | 0.34 |
| Collection | 843.07 | 969.56 |
| Policy inference/action selection, inside collection | 720.10 | 790.30 |
| Environment/encoding, inside collection | 36.51 | 37.40 |
| Batch assembly, inside collection | 38.82 | 97.26 |
| Learning | 440.84 | 448.04 |
| Forward/loss, inside learning | 227.27 | 218.96 |
| Backward/clipping, inside learning | 143.58 | 155.24 |
| Export | 0.61 | 0.58 |
| Learner decisions | 186,007 | 188,735 |
| Optimizer minibatches | 728 | 740 |

Complete admission elapsed is 2,715.66 seconds: 9.41 seconds beyond the two
executor clocks. Setup/native installation and numerical checks precede this
measurement; the whole rental lasts 48.94 minutes. Whole rental less admission
is about 220.63 seconds for setup, numerical checks, delivery and cleanup, not
measured learner idle time. In-stage unclassified bookkeeping is 1.45/1.40s.
The repeated inference/learning phases dominate; fixed overhead does not explain
these rates. GPU sampling during GRU showed 23% utilization, 18,510 MiB memory,
126 W of 250 W, 61 C, 1695 MHz SM; this is one sample, not a utilization trace.
Learner CPU was approximately one core. Actual cgroup quota was 6.4 cores,
versus quoted 13.33; two CPU-pressure samples were zero and cumulative throttling
did not increase between them. Peak learner RSS was 17.55/17.34 GB, below its
24 GiB cap. Neither OOM nor sustained host contention is established.

The source audit identifies per-batch/per-frame loops, `.tolist()`, scalar tensor
reads and repeated CUDA validation in `TypedContinuationEncoder` and
`SharedEffectEncoder` as plausible CPU-dispatch/device-synchronization costs.
These operations occur in both inference and learning. No operator-level GPU
profile isolates their contribution, and no model speedup has been implemented
or claimed. The older 4090 receipt differs in both training and engine source
hashes; it cannot isolate hardware speed even though several recipe/ABI hashes
match. A causal comparison needs identical source/workload and retained profiling.

## Corrected affordability and prospective controls

The former projection multiplied all two-arm elapsed time by the update ratio.
The replacement separately charges recurring collection/learning/bookkeeping
(50% allowance), per-learner initialization/close/export, seven intermediate
exports bounded by each admission export bundle, and one setup/evaluation/teardown
reserve. Unknown outer admission time is charged once per paired seed. Frozen
receipts are unchanged; `phase-projection.json` is a separate derived analysis.

Corrected full-cohort projection is 290.27 hours / **$71.82** at the actual quote.
Recurring training alone, without any allowance or fixed/evaluation costs, is
**$47.58**, versus $16 remaining. Four updates do not guarantee later game length,
but this measured arrangement cannot fund the declared study. Plateau remains
unmet; no replacement science launched.

Future admission binds limits in its job config: 180s to a ready CUDA/one-thread
record, a no-completed-update window derived from four times the affordable mean
update pace (minimum 300s), and after at least two updates a phase-cost rejection
if this arm alone exceeds the whole scientific allowance. These are capacity
policies, not scores or host-defect labels. Durable progress and stop reason
precede a child stop-file request; 120s grace permits evidence flush before hard
process-group reaping. Ordinary nonrecoverable collection now consumes the
existing executor stop/resource callback. This code did not govern the frozen
completed rental and has only fixture validation so far.

Focused validation: 12 projection/watchdog/process tests and two executor
cancellation tests passed; Ruff passed. Native extension rebuilt to match current
source before the executor tests. Gate owns broader suites. Remaining work is
operator-level profiling and a justified implementation improvement, then matched
bounded admission only within the same allowance; the four-paired-seed cohort,
72 curve endpoints, eight held-out endpoints and plateau criterion remain open.
