# GPU training time: startup and steady work

2026-10-15. The basic training report exposed 62% unattributed time in a historical
41-minute L4 run. The retained update coordinates change its interpretation:
most of that time passed before the first update, not throughout collection.

Evidence: [source-bound measurements](data/training-time/historical-startup.json),
run `9162405354044aa7aa86fcc6bc842137`, source
`2555c0e2ab2fb01b1624b5f948a7b5813b0361d2`, job
`etu103-recovery-v2-pilot-1`. The run and evaluation metadata were downloaded from
the current job manifest and verified against its byte counts and SHA256 hashes.

| Observation | Measured time |
|---|---:|
| Complete recorded run | 2,465.680 s |
| First update coordinate | 1,501.394 s |
| Active collection + learning at first update | 2.569 s |
| Initial evaluator process | 1,369.203 s |
| Initial evaluation charged time | 1,369.130 s |
| Recorded interval between first and last updates | 961.225 s |
| Active time within that interval | 937.185 s (97.5%) |
| Full-run collection | 775.588 s |
| Full-run learning | 164.165 s |

The saved job requires initial evaluation admission. In the measured source,
`training/execution.py` waits for the admission file before constructing the
collector. `remote/supervisor.py` creates that file only after the initial
evaluation completes and its snapshot is durably published. This establishes a
blocking evaluation dependency. It does not measure the exact learner wait:
initialization, publication, polling and other startup work also occupy the
pre-update interval. The historical flamegraph therefore retains unattributed
time instead of assigning the entire residual to evaluation.

Collection accounts for 82.5% of the run's active time. That is the next boundary
to explain on a GPU: host observation preparation, policy execution and returned
results, native environment work, and rollout assembly. It does not establish
that neural inference or the simulator alone is the bottleneck. This recipe uses
64 streams and eight transitions per update, with a width-384, eight-layer model.
Earlier throughput experiments used width 64 and one layer. Those timings cannot
be substituted for this model; larger rollouts need their own measurements.
Post-first-update overhead is small in this run, which weighs
against persistence being the dominant explanation here.

Future ordinary runs now record `StageRecord.admission_wait_seconds`. The field
counts attempt-local learner wall time at the admission gate, including failed
waits; it is excluded from active training and partitioned once in the report.
Older records retain an unknown value. This adds a basic host timer, not PyTorch
tracing or evaluation instrumentation.

The larger historical running snapshot, `etu103-recovery-v3-science-2`, supports
the same startup/steady distinction. Its first-to-last-update interval is
3,423.831 seconds, with 3,279.878 active seconds (**95.8%**). Collection is
81.1% of accumulated active time. It contains 1,855 updates and 949,760 learner
rows on an L4 with the same width-384/eight-layer shape. These are observed
snapshot counters, not a completed-run result. [Source-bound measurements](data/training-time/historical-live-snapshot.json)
retain the artifact hash and exact source `45940fe3`.

In that source, phase counters update during training, while `stage.seconds`
is finalized only when a stage ends. The report now shows a running stage's
measured counters as **observed phases** when that stage clock remains zero.
It keeps the enclosing attempt's remaining time unattributed and still rejects
phase sums exceeding the attempt clock. This makes the larger snapshot useful
without inventing its missing stage duration. The historical demo initially
selects both records; it remains a partial inventory rather than a complete
ranking of historical runs.

Ordinary self-play now supports optional collection subphase counters through
`execution.profile_collection`. They record observation preparation, policy/action
handling, rollout recording, opponent policy, environment step/encoding and batch
assembly. The report nests these under collection and retains an unmeasured
remainder. These are host intervals with existing device waits, not GPU kernel
durations. Learning remains a coarse measured phase. Detailed PyTorch traces stay
deferred. The new investigation has a separate $25 GPU allowance.

## Basic-profile-001: evidence delivery failed

The [first protocol](protocols/basic-training-profile.txt) admitted 28 cases on
one RTX5090, covering the small benchmark model and the larger historical shape.
Source `79ea7a0e`, run `70ec76cbf84d`. The job log reports an incomplete study, but
no cell records or training archive reached storage. All 28 planned cells remain
**unknown**, including the proof and observer-effect gates; there are no timing
medians to report. [Retained failure evidence](data/training-time/basic-profile-001.json)
includes every planned case, terminal state and provider-confirmed deletion.

The rental ended at its original 45-minute limit on 2026-10-16 at 02:19 UTC.
Etude recorded `lost / machine_lost`. Estimated rental cost is $0.894; accounting
reserves $1.09 conservatively, separately from the $2 ancillary allowance. Final
provider billing was not yet available. The provider confirmed the pod absent.

Only the claim, job traceback and setup log survived. The launcher uploads every
output before its completion receipt, using a single-object form capped at 5 GiB
and an HTTP timeout of 60 seconds. This explains why a finished local job can
still appear running while delivering artifacts. It does **not** identify the
actual failure: the archive's size, transfer rate and upload exception were not
recovered. Neither this delay nor its rental cost measures a training bottleneck.

The [focused follow-up protocol](protocols/basic-training-profile-002.txt) keeps
the historical width-384/eight-layer, 64-stream comparison: both deterministic
proofs and all three ordinary timer-on/off repetitions at the original work
counts. Timing metadata is independently readable and sorts before model data;
the complete evidence is split into 64 MiB compressed objects with ordered hashes.
Local tests reconstruct every original file and check independently readable
metadata. Training behavior and measured model exports are unchanged. This
repairs evidence delivery; it does not establish a GPU throughput improvement.

## Basic-profile-002: no training start

Run `5f2c526b69b7`, measured source `7d00992a`, ended
`lost / machine_lost_before_start` on 2026-10-16 at 02:23 UTC. Its session output
prefix contains no worker receipt, job claim, log or artifact. The eight planned
cells therefore supply no measurements. The bootstrap failure mechanism remains
unknown; absence of a claim does not identify a failed download or provider fault.
[Failure evidence](data/training-time/basic-profile-002.json) retains the plan and
provider-confirmed cleanup. Estimated rental cost is $0.0453, reserved as $0.25
conservatively. The protocol declares one separate operational replication003
with unchanged measured source, work, predictions, gates and limits.

## Basic-profile-003: policy handling dominates this larger-model window

Run `d51e85f1a805` completed all eight cases on an RTX5090/EPYC9554 host:
width384, eight layers,64streams,8transitions,16ordinary updates, seed10349.
Torch2.10/cu128,FP32, oneTorchCPUthread and one native worker. Measured source
is unchanged `7d00992a`. All40Etude output objects and106archive members were
hash-verified. Provider deletion is confirmed; estimated rental cost$0.2282.
The new campaign's three rentals total about$1.17estimated/$1.77conservative,
plus a separate$2ancillary reserve. Final provider billing remains unavailable.

[Interactive report](data/training-time/basic-profile-003/report.html),
[flamegraph](data/training-time/basic-profile-003/flamegraph.svg),
[all cases and gates](data/training-time/basic-profile-003/analysis.json), and
[source/artifact receipt](data/training-time/basic-profile-003/receipt.json).

The following shares sum the three ordinary instrumented runs' own measurements;
no historical or separately traced subphase percentages are grafted onto them.

| Boundary | Share of recorded training time | Share of collection |
|---|---:|---:|
| Collection, including its children | 62.0% | 100% |
| Learner policy inference and action handling | 22.2% | 35.8% |
| Opponent policy, including its input preparation | 31.6% | 51.0% |
| Environment stepping and observation encoding | 2.8% | 4.5% |
| Learner observation preparation | 2.9% | 4.7% |
| Batch assembly | 1.0% | 1.6% |
| Rollout recording | 0.7% | 1.1% |
| Learning | 20.7% | — |
| Model/optimizer exports | 8.7% | — |

Collection is75.0%of active collection+learning time. Combined policy handling
accounts for86.8%of collection and53.8%of the recorded run. Requests average
only10.95acting rows per policy call, despite64resident environments. There are
67native stepping calls per update. These are host intervals containing dispatch,
model execution, policy logs/sampling and returned-result waits. They do not
separate GPU compute from CPU launch overhead or establish GPU utilization.
Opponent input preparation is nested in its policy interval; the separately
named observation-preparation row covers the learner only.

Uninstrumented complete-run median:18.777s,436learner rows/s. Instrumented:
19.079s,429rows/s, a1.0161duration ratio. First-to-last-update medians are
554.5and542.5rows/s respectively; whole-run time includes startup and exports.
The ratio meets the declared5%descriptive gate, but **exact-state validation
fails**, so it is not a causally isolated measurement of timer overhead.

### Numerical failure: not dismissed as rounding noise

Initial weights and the first complete proof rollout match exactly. Across all
three proof collections, actions/active masks, environment buffers, game seeds
and sampling RNG match. The first learner update already differs in value loss;
subsequent rollout tensors and final raw/EMA/Adam state differ. The largest raw
parameter difference after three deterministic-proof updates is6.37e-5.
The exact gate remains failed; no tolerance is waived.

Ordinary timer-off repetitions also disagree. One contains a finite weight of
293.4111 at `extra_attention.6.mlp.2.weight[362,1461]`; this tensor's maximum
absolute value is0.11701initially and0.11725in another timer-off control.
This is materially different from the tiny Adam differences in earlier small-model
experiments. The first control also has395optimizer exposures and18,240environment
decisions versus390/18,261in the other ordinary cases. All have8,192learner rows.
The first update's sampled numerical-health record agrees between those controls;
later per-step parameter movement was not sampled. The first anomalous operation
and cause are unknown. [Checkpoint audit](data/training-time/basic-profile-003/state-audit.json)
retains all parameter/state differences; [witnesses](data/training-time/basic-profile-003/witnesses.json)
retain every case, including contrary controls.

The [outlier follow-up](data/training-time/basic-profile-003/outlier-audit.json)
rechecks the eight saved checkpoint hashes and maps the affected parameter to
Adam state129. Its own initial value was−0.0429469; after86optimizer steps it
is293.4111, with EMA0.5436766. The other off control ends at−0.0421838 after85steps.
Both use learning rate0.0001, betas(0.9,0.999), epsilon1e-5 and zero weight decay.
For zero-initialized moments, weighted Cauchy–Schwarz bounds the intended Adam
recurrence's cumulative coordinate movement at0.0130134over86steps, regardless
of gradient magnitudes. Observed movement is293.4540. Exactly one element across
133parameter tensors exceeds this bound; the other off control has none.
The formula and an independent equality-case check are retained in the audit.

This is an exact-arithmetic model of the optimizer, not proof that its floating
point implementation executed correctly. Final moments are finite/nonnegative;
both controls also retain tiny moment-inequality exceptions in token embeddings
(first moments below9.1e-26, zero second moments), whose final epsilon-limited
step contributions are below9.1e-25. They do not explain the observed movement.
The separately saved EMA anomaly argues against corruption confined to the final
raw export. Neither final checkpoint locates the first bad operation. Per-update
loss fields describe the last nonempty timestep minibatch, so first-update loss
differences must not be called first-forward differences. No extra GPU was rented
for this analysis, and no numerical gate changed.

The timing hierarchy is therefore **diagnostic evidence**, not validated profiler
adoption or an optimization win. Its dominant policy boundary differs from the
earlier width64/depth1,4096-stream workload, but both model size and batching
changed; this study does not isolate which caused the difference. A serving
benchmark must match the actual model and small-request pattern before predicting
training gains. Making model math faster alone is not the same intervention as
reducing the whole policy-handling interval.

Before another performance intervention, the separating check is repeated
identical deterministic timer-off controls, then matched timer-on controls,
with evidence locating the first divergent learner operation. Training logic,
backend numerical behavior and hardware remain competing explanations. The
finite outlier must be investigated rather than attributed to the profiler or
silently removed. PyTorch tracing remains deferred.

## Basic-profile-004: repeated learner witnesses agree

Run `7d6c152708bc`, source `5a99c7e4`, completed all six predeclared diagnostic
cases. All 31 Etude output objects and 126 archive members were verified, followed
by independent CPU decoding and canonical hashing of all 24 checkpoint payloads.
The RTX 5090 rental was deleted at 04:18:14 UTC on 2026-10-16; estimated cost $0.2380.
The four-study campaign totals $1.4055 estimated / $2.21 conservative, with the existing
$2 ancillary reserve and no live rental. Final billing remains unavailable.

[Comparison results](data/training-time/basic-profile-004/analysis.json),
[checkpoint verification](data/training-time/basic-profile-004/checkpoint-hash-verification.json),
[cross-study comparison](data/training-time/basic-profile-004/cross-study.json),
and [source/output receipt](data/training-time/basic-profile-004/receipt.json)
retain the evidence. Ordered learner JSONL and collection witnesses are alongside
them. The initial source-history fetch failed before admission; no run or rental
was created. An identical bounded read-only fetch passed, after which the single
admission proceeded with the same source and reservation. That failure is retained.

| Comparison | Work per case | Result |
|---|---|---|
| Deterministic timer off/off | 3 updates, 9 Adam steps | All 60 learner events, collections and checkpoint states exact |
| Deterministic timer off/on | 3 updates, 9 Adam steps | Exact |
| Deterministic timer on/on | 3 updates, 9 Adam steps | Exact |
| Ordinary timer off/off | 16 updates, 85 Adam steps | All 505 learner events, collections and checkpoint states exact |

The first four cases have full deterministic algorithms enabled; the last two
do not. Recorded runtime flags confirm this distinction. All cases have cuDNN
determinism enabled, benchmarking and TF32 disabled. The largest observed
coordinate movement is 0.00009993 in the short proofs and 0.00017330 in the ordinary
controls. No movement crossed the diagnostic 0.01 capture trigger.

The new short proofs match study 003's timer-off proof exactly, but not its
timer-on proof. The two new ordinary controls match **all five non-anomalous
ordinary cases from 003**, including both timer settings; only 003's first ordinary
timer-off case differs. Thus the retained evidence supports two exceptional cases
in 003, not a claim that every repetition had distinct learned state.

These diagnostics deliberately hash tensors around forwards, target construction,
backward/clipping and Adam. That adds synchronization, CPU transfers and substantial
overhead; their durations are not performance measurements. The host also changed
from EPYC 9554 / 31.13 CPU quota to EPYC 7543 / 12.75 quota, while GPU family and
Torch 2.10/cu128 stayed the same. Synchronization effects and an intermittent
backend/hardware failure remain materially different explanations. No divergent
operation was reproduced, so no root cause is established and 003's failed gate
remains failed.

This supports returning to matched workload attribution with fresh ordinary
controls, while keeping the anomalous run visible. A same-host witness-off/on
comparison would separate observer synchronization from host changes if the
numerical failure recurs. The missing model-size/batch-size comparison remains
necessary before generalizing the 53.8% policy share to other training regimes.
