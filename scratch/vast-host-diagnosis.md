# Vast diagnosis and admission — completed evidence through 2026-11-05 02:34 UTC

The plan owns current scientific progress and accounting; this note retains the
completed diagnosis and admission evidence. Run `e51f61b92931` is still active;
its partial learner receipts do not establish final inventory or qualified
results. Fresh reads and the original launch handoff are retained separately
under `batched-plateau/`, preserving their observation times.

## Retained CUDA diagnosis, admission and scientific handoff

Fresh 01:55:50 accounting confirms all eight preceding Task rentals absent.
The sum of each reported/lifetime maximum plus settled L4 is $0.93290824;
the existing $3.50 compute and $6 ancillary reserves cover it. Diagnostic
`b153cc2e5a77` reserves another $0.50, leaving $15.00 under the original $25.
Source `2b2061c8d` and locked Etude `64e9eaabf` are unchanged. The live quote
is $0.2474074074/hour and $0.18555556 for 30 minutes work plus 15 setup.
Instance `59191189`, session `c8987cae4da6`, was acquired at 01:56:29.
All four retained CUDA oracle comparisons passed (maximum absolute error
7.6294e-5). One of six routing tests failed: pooled parameter gradients differed
in 1/192 entries by 1.3351e-5, exceeding default float32 atol 1e-5. No learner
started. Both output files (4,854 bytes) matched the delivered inventory. The
rental ended at 01:59:30; fresh inventory confirmed absence, charge $0.013
versus lifetime estimate $0.01240595. These are not final invoices.
Evidence is manifest-published under `trace-delivery-failure/` in the GPU study
data directory; local operations remain `.runs/representation/gpu-trace-delivery/`.

`af601877e` compares both float32 paths against independent float64 outputs,
object/header gradients and parameter gradients, with 2e-5 relative/absolute
bounds, retaining duplicate membership and poisoned-padding cases. CPU pool/GRU
passed locally; the following diagnostic supplied the CUDA result. No runtime
encoder, scientific recipe or frozen receipt changed. Failed routing reports
now enter the carried log too.
Fresh 02:01:57 inventory found all nine prior rentals absent; reported/lifetime
maxima total $0.95190824. The $4 compute/$6 ancillary reserve plus a new $0.50
diagnostic commitment left $14.50. Run `911423cce90b`, source `af601877e`,
used the same $0.18555556 whole-rental quote and started on instance
`59191723` (RTX3090/13.33 vCPU/257.509 GB).

That run succeeded: all four frozen output/gradient comparisons and six CUDA
routing cases passed. Pool/GRU each completed 256 games, with collection/learning
42.56/40.15s and 42.38/39.95s. All 59 output files (170,244,006 bytes), including
both inference and complete learning traces, match their worker hashes. The
39-byte result envelope was reconstructed from the frozen formatter and transported
update clock after the SDK requested an absent standalone object; its original
SHA256 matches. All other files were fetched normally. Evidence and derived
trace analysis are manifest-published under `cuda-traces/`. The rental ended
02:08:54; 02:10:42 inventory confirmed absence. Reported $0.030 exceeds the
$0.02577188 lifetime estimate; no final invoice is claimed.

The first learning minibatch's effect/continuation host scopes sum to
13.09ms pooled and 15.45ms GRU; CUDA stream waits inside those scopes sum to
0.099/0.093ms. Across the profiled update, cProfile attributes about 9.15/9.59s
to these encoders. Host routing/dispatch remains measurable, but stream waits
do not dominate these captured scopes. First-step kernel loading (133/137ms
in the learning operator tables), profiler work and sampled gradient/parameter
health prevent extrapolating trace totals as recurring costs. GPU annotation
spans include gaps; they are not utilization or sums of kernel execution.
These profiles selected no further runtime repair before the admission below.

Admission `37c1523a98ed` uses the identical `af601877e` source, four updates per
arm, $14 scientific allowance and $0.26 hourly supervision ceiling. Fresh
02:11:19 accounting confirmed all ten prior rentals absent; reported/lifetime
maxima plus settled L4 total $0.98690824. Prior compute reserve $4.50, new
commitment $0.50 and ancillary $6 leave $14.00. Whole-rental quote $0.371111
for 75 minutes work plus 15 setup fits the attempt cap. Instance `59192537`,
session `46dcc516f702`, reports the same RTX3090/13.33vCPU/257.509GB hardware.
Both arms completed four updates/1,024 games. Their exact collection/learning
times are 136.4431/122.2212s (pool), 132.4886/111.1156s (GRU), with 516.7714s
overall admission. All 49 files (131,228,103 bytes) verified and published under
`batched-admission/`; the result envelope's frozen bytes matched its hash as
above. Rental ended 02:24:19 and was confirmed absent at 02:25:05; reported
$0.052 versus lifetime estimate $0.05139499.

The complete phase projection is 55.66970 rental hours / $13.77310. Before the
result, accounting was corrected to release unused completed-attempt ceilings:
reserve max($3.50, twice fresh per-rental reported/lifetime maxima plus settled
L4), $6 ancillary, and all unknown/live commitments. At 02:25:26 all eleven
prior rentals were absent; the reconciled total was $1.04590824, leaving
$15.50 for science. No final settlement or new allowance is implied.

The matching live quote admitted scientific Run `e51f61b92931` at the same
`af601877e` source. Instance `59194544`, session `2b7ebc4aa353`, has the same
RTX3090/13.33vCPU/257.509GB hardware. Its rental ceiling is $13.77310 and
deadline 2026-11-07 10:09:07 UTC. The full $15.50 allocation remains reserved.
At 02:34:17 UTC the provider confirmed this as the sole Task rental, running.
The carried log reaches `train: typed_pool seed 74211`, proving preceding
CUDA numerical/hardware gates returned success and the first scientific child
was launched. Its completed update count is not carried in the parent log;
there is no final scientific inventory or plateau result yet.
`batched-plateau/` contains the prospectively frozen allocation/source/quote;
the protocol retains the unchanged seeds, horizon, endpoint schedule and criteria.
Etude's documented `Execution.finish` requested delivery and self-deletion after
the sole job; the worker acknowledged `finish when done` at 02:32:32 UTC.
The independent deadline remains a backstop. The local wait client was stopped
after this handoff; eventual final inventory, analysis and teardown remain open.

## Lost diagnostic and delivery boundary

Run `a46e80b31212` / `66ec957fe` completed both instrumented updates and one
ordinary pooled update by carried-log evidence, then lost instance `59185580`
at 01:12:47 UTC without a final exit/inventory. The four reference comparisons
and six routing gates returned successfully, but their detailed reports and
traces were never delivered. Only the attempt record and contiguous log chunks
were recoverable. Cause and final exposure remain unknown; no OOM, host-fault or
code-failure attribution is established. All eight rentals were absent at 01:16:18.
Reported charge $0.038 versus lifetime estimate $0.03984744 remains in accounting.

Etude uploads ordinary files at attempt completion. The maintained diagnostic
therefore ends after its two profiles, before a separate admission can lose its
artifacts, and carries bounded reports in logs. Successful run `911423cce90b`
verified this delivery boundary. Earlier failed evidence remains manifest-backed
under `complete-operator-profile/`. Full commands, timings, cleanup, accounting
and validation: `2784e4283ce12011643b319968131ae5dbba9f6e:scratch/vast-host-diagnosis.md`.
No frozen artifact is rewritten or failed attempt excluded.

## Earlier CPU repair and connectivity evidence

`42709fa1b` introduced ordinary-executor profiling; `59ec1bd24` replaced per-record
CUDA dispatch with host routing admission, batched projections and reducers.
Matched pooled CPU work (256 games, 51,095 decisions, 200 minibatches) fell from
202.86s collection/learning to 121.71s; encoder call time fell from 30.36s to
2.71s. Both used cProfile, the final additionally captured operators. Native
multi-position outputs and all parameter/object/definition gradients agree with
pre-change oracles. Duplicate memberships, canonical ordering, padding and None
gradients for unused Adam parameters survive. The GRU CPU run completed 256 games,
54,184 decisions and 212 minibatches in 150.12s without a matched CPU baseline.
These are CPU engineering observations, not GPU affordability or learning quality.

The initial profiler captured only grad-enabled forward; `6c0e33d4b` expanded it
through backward, clipping, Adam and health. Earlier GitHub/Doppler outages stopped
preparation before allocation; no new compute charge or scientific seed resulted.
The later complete-trace attempt and successful delivery above supersede that
connectivity block. Full commands, contrary interim timing, references and prior
validation remain at `af601877e:scratch/vast-host-diagnosis.md`; no frozen evidence
was rewritten. Manifest-backed CPU profiles and encoder references remain under
the GPU study's `cpu-operator-profiles/` and `encoder-reference/` directories.

## Pre-repair matched work — 2026-11-04

Run `803bcc26e7de`, source `22c10ca55197ab225546e5880cc53709db985901`,
completed four updates/1,024 games per arm, CUDA float32, one thread and 256
streams. All 46 outputs (131,204,931 bytes) verified; rental `59168556` was absent
at 22:58:28 UTC. Reported charge $0.205 exceeded lifetime estimate $0.20159.
The 39-byte result envelope was recovered through the same frozen formatter/hash
check described above. Evidence is manifest-backed under `cheaper-3090/`.

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

The frozen source audit identified per-batch/per-frame loops, `.tolist()`, scalar tensor
reads and repeated CUDA validation in `TypedContinuationEncoder` and
`SharedEffectEncoder` as plausible CPU-dispatch/device-synchronization costs.
These operations occur in both inference and learning. No operator-level GPU
profile isolated their contribution at that source. The later CPU repair above
does not establish CUDA attribution. The older 4090 receipt differs in both training and engine source
hashes; it cannot isolate hardware speed even though several recipe/ABI hashes
match. A causal comparison needs identical source/workload and retained profiling.

## Projection and supervision contract

Total-elapsed scaling overcharged fixed costs. The surviving projection scales
recurring collection/learning/bookkeeping with 50% allowance, charges
initialization/close/export once per learner, bounds seven intermediate exports
by the admission export bundle, and includes outer orchestration once per paired
seed plus setup/evaluation/teardown. `phase-projection.json` is separate derived
evidence; frozen receipts remain unchanged.

The pre-repair full cohort projected 290.27 hours / **$71.82**, or **$47.58** for
recurring training alone without allowances, versus $16 then available. This
excluded arrangement was unaffordable; it did not prove every implementation
would fail. The post-repair admission and scientific allocation above supersede
that execution block, without establishing later game lengths or learning quality.

Prospective admission freezes 180s startup to CUDA/one-thread placement,
no-completed-update time max(300s, four times the affordable mean update pace),
and after two updates a stop if this arm alone exceeds the whole scientific
allowance. These are score-independent capacity policies, not host-defect labels.
Progress/reason precedes a stop-file request, with 120s grace bounded by the
rental and process-group reaping afterward. Controlled tests establish these
paths; no live enforcement claim follows for older frozen rentals.

Current code, acceptance and validation ownership are in
`representation-learning-plan.md`. The outstanding work is the unchanged
scientific cohort, full output inventory, evaluation and confirmed teardown.
Earlier full diagnosis and command/results remain at the precise revision above.
