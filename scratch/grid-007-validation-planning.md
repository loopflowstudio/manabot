# MTG-135 validation planning — not an admitted allocation

Goal: recommend an exact model, stream count, transition horizon, schedule
mapping and export cadence for MTG-122, with evidence about both cost and learning.
The 25-cell feasibility grid remains the first allocation. This note does not
freeze scientific seeds/counts or authorize additional spending.

## Evidence and constraints

Four cells have complete verified artifacts at 131,072 transitions, seed13571:
small-s4, middle-s4, middle-s64, large-s64. The two 64-stream cells used different
CPU models (EPYC9354 and EPYC7543); the four-stream cells used EPYC7543. Retain
host differences and filtered optimizer exposures in comparisons. The large
64-stream model is16,816,514 parameters,334.868 transitions/s,391.414s,
2,531,847,680 peak CUDA allocated bytes. No strength scores have been used to
select configurations.

Before treating the observed middle4→64 ratio as a batching multiplier, run
bounded pairs on the same rental, with separate processes/outputs and matched
transition budgets. Counterbalance execution order if the remaining budget
supports two pairs. Preserve setup/rental cost separately from training time.

A size-strength comparison must preserve all five architectures. Compare raw
policies against initialization using independent training seeds, paired deals
against greedy and random, and all four deck-role cells. Freeze allocations from
measured training/evaluation costs before scoring. Larger batches alter update
frequency, filtering and staleness even with LR/tau schedule normalization.
A short horizon cannot establish a model's skill ceiling.

## Existing baseline is context, not a drop-in control

Read-only inspection of MTG-137's current operations confirms ongoing Mini/GPU
baselines under its own controllers and budget. Leave them untouched. The older
GPU source4492c786 differs substantially from this grid's f057cfeb in model,
collector, execution and engine code. The ordinary evaluator also binds exact
engine-source identity. Do not combine old checkpoints with the new runtime or
claim a matched-control comparison merely because world and recipe labels agree.
Original evidence stays in its owning checkout; no scientific launch should
require an external checkout path as its working context.

## Candidate learning allocation, pending measured admission

For a common experience horizon across4/64/128/256/512 streams, require multiples
of32,768 transitions. A candidate1,310,720-transition endpoint is5,120 four-stream
updates but only40 updates at512 streams. This is a useful early-learning test,
not a ceiling experiment. It is a candidate, not a frozen target. Cost margins
must include the slower four-stream learning controls, all five sizes, repeated
seeds, retained exports, both evaluation anchors and setup/retention.

The original middle64 timing extrapolates to roughly2,205 training seconds at
that candidate horizon, exceeding the feasibility runner's fixed1,800-second
limit. The new explicit gpu_job training-seconds option carries a planned limit
into the learner and supervisor, bounded by the outer deadline/retention reserve.
Do not silently cut learning short to fit the old feasibility cap.

The retained-evaluation runner now resolves initialization/milestones/endpoints
from observed learner-transition coordinates, verifies relocated artifact bytes
and rejects missing milestones. A possible panel is13 paired deal blocks ×8
legs =104 games per anchor/checkpoint, but evaluation calibration must establish
its cost before admission. Repeated initialization controls, independent-seed
uncertainty and per-role-cell effects belong in the declared analysis. No
best-checkpoint selection or partial-cohort success claims.

## 2026-10-17 13:17 UTC operation

Six cells now have full retained artifacts and finite loaded checkpoints. The
large model at128streams completed131072transitions in365.628s (358.484/s),
versus391.414s (334.868/s) at64streams:1.0705× observed throughput with1.7878×
peak allocated CUDA memory (4.526GB versus2.532GB). CPU host changed from EPYC7543
to EPYC-Genoa; selected optimizer exposures also changed. This is not a clean
causal multiplier. Full result receipt: `.runs/grid-007/feasibility/six-complete-results.json`.

The projected-source large four-stream run bad037591cb2 stopped at its declared
1800-second learner deadline after379updates/97024transitions. Its retained
TrainingRun says interrupted, and the training traceback is specifically
`TimeoutError: collector exceeded training deadline`. This is a censored
feasibility result, not an unexplained CUDA failure or completed131072-transition
cell. Keep the original failure and artifacts; do not retry or extend it silently.
Once the controller has settled current work, continue only the untouched cells.

Evaluation-cost calibration protocol008 and worker are committed at d64cd95c.
Nineteen related tests passed. Source tar and projected upload passed complete
readback verification. The calibration has not been admitted. Add it only at a
safe controller handoff, with one scheduling owner, two rentals maximum, a$2
ceiling and fresh inclusive-ledger admission. Its score cannot select a treatment.

At13:19UTC the stopped controller had settled all rentals, fetched every retained
manifest and exited. Full verification loaded both retained initial and monitoring
policies for the deadline-censored cell and found finite tensors. Seven completed
cells are in `seven-complete-results.json`; small128 completed in129.231s at
1014.244transitions/s on EPYC7543, with0.817GB peak allocated CUDA memory.

A fresh sole controller (PythonPID97240, parent97226) continues the17 untouched
cells and the separate timing calibration. The latter keeps the declared$2 cap
but uses a tighter45-minute command ceiling in the existing55-minute rental
pool. All old sources and results remain unchanged. Conservative settled charge
is$14.61, new caps$19 and ancillary reserve$2: inclusive bound$35.61. No four-stream
large-model replacement was queued. The old stop and receipts are archived in
`.runs/grid-007/feasibility/before-calibration-resume/`.

Calibration628c404a1176 was admitted and observed setting_up on rental
grid-007-feasibility-665ec7536da9 at13:21UTC. Source d64cd95c is committed locally;
publication waits for the already-armed PR299 to merge, so the next change does
not alter that reviewed head. The new PR copy is retained in
`.runs/grid-007/pr-calibration-body.md`. No calibration result exists yet.

## Retained numerical audit (2026-10-17 13:27UTC)

Offline audit of the seven completed cells found finite initial/final parameters
and Adam moments, no negative second moments, no reported nonfinite/rejected
optimizer steps, and no coordinate above the conservative whole-trajectory Adam
movement bound. Largest parameter change ranges0.01767–0.16428. The bound uses
max observed LR and max saved Adam step count with zero initial moments and zero
weight decay; it is deliberately coarse and does not clear the historical outlier
or prove deterministic execution. Script and hashed receipt are retained in
`.runs/grid-007/audit_completed_numerics.py` and
`.runs/grid-007/feasibility/seven-cell-numerical-audit.json`.

A material training contrast is now explicit: small-s4 performed22,001 Adam
steps, small-s64 2,048, and small-s128 1,024 at the same131,072 collected transitions.
Selected exposures were30,784/32,768/32,768 respectively. Four-stream empty
minibatches account for10,767 skipped steps; these were not numerical rejections.
Thus similar selected-example counts do not preserve optimizer-step frequency.
The learning comparison must assess this directly; throughput alone cannot choose
an expanded batch configuration.

## Next allocation cost risk and available parallelism

The ordinary monitor uses CPU policies and runs deal blocks serially. Live phase
receipts show serial evaluation materially exceeds tiny training time, especially
for384×8; exact cost admission still awaits completed calibration008. Preserve
that serial calibration unchanged. `head_to_head.run` already uses isolated
parallel arena units, and `size_frontier_evaluation` selects two workers, but that
runner's frozen entrant/anchor protocol is not a drop-in replacement for our
paired greedy/random MonitorProtocol.

Before reducing scientific replication merely to fit serial scoring, consider a
bounded equivalence/timing probe that runs independent ordinary monitor panels
concurrently in separate processes and output directories. Keep the exact policy,
deals, one-thread CPU games, per-game watchdog and replay contract. Compare actual
retained decisions/results to the serial calibration; do not assume a parallel
speedup or share mutable learner state. A future supervisor must stop active
children on failure/deadline, retain partial results, and cap concurrency from the
observed CPU quota. This is a candidate operational improvement, not an admitted
allocation or a change to the current calibration.

PR299 merged at bc691e76. Calibration worker PR300 is published and armed at
750e2122 after19 related tests passed again following synchronization. Its GPU
run remains pinned to pre-merge scientific source d64cd95c; runtime learning code
was not changed by that synchronization. Eight completed cells and their coarse
numerical audit are retained in eight-complete-results.json and
eight-cell-numerical-audit.json. Middle128:765.042transitions/s,1.861GBpeakCUDA,
1.2871× observed throughput versus64 on a different CPU host.

A concrete candidate parallel probe: on one new bounded rental, create the same
reserved-seed384×8 initialization through one ordinary64-stream update. Evaluate
one eight-leg block per greedy/random anchor serially, then repeat both panels
twice with four concurrent independent CLI processes (48 attempted games total).
Use the same retained checkpoint, protocol and host within the probe, so CPU-host
floating-point differences cannot be mistaken for a concurrency effect. Require
all rows/replays and canonical decision traces to agree with that host's serial
reference before using any speedup in admission. Preserve failed/incomplete cases.
Do not compare only score totals. No extra probe allocation is reserved yet.

Prefer supervising independent ordinary evaluation CLI processes with separate
Worker output directories; existing Worker.command supplies process-group
termination and outer-deadline/retention enforcement. A bounded main-thread
scheduler can dispatch those calls via a small thread pool (the actual games stay
in separate processes). On a failed panel or stop, stop all active child Workers
and start no pending panels. Test real concurrent completion and failure retention.
Keep the default scientific evaluator and learning rules unchanged. This proposed
orchestration still requires implementation, tests, a frozen probe and fresh
budget admission; it must not be described as already validated.

Small256 completed131072transitions in83.821s (1563.709/s), with1.520GBpeakCUDA.
Full manifest, final policy and Adam audit verified; latest aggregate is
`nine-complete-results.json`. CPU model:['AMD EPYC 7543 32-Core Processor']. Observed256/128 speed
ratio1.5417, still no same-rental control. It used512Adam steps
with32768 selected exposures. Report quality against both transitions and elapsed
cost: equal examples do not preserve optimizer-step frequency, and a faster
configuration can collect more experience within the same wall-clock budget.

## Retention repair and next bounded probe (2026-10-17 14:09UTC)

Calibration008 completed all80 games and full verification loaded all five finite
initial policies. Its root manifest incorrectly indexed Etude's reserved
`etude-result.json`, which the courier stores as attempt metadata rather than a
fetchable object. Reconstructed the225-byte original from the retained final log
and checkpoint metadata; its SHA256 exactly matches both recorded manifests.
Explicit provenance is `results/evaluation-cost-calibration/courier-metadata-recovery.json`.
The original manifest is preserved. Source e1fd6029 fixes future manifest generation,
excluding only the root reserved file and retaining nested outputs.

The parallel evaluator supervisor and same-host probe009 are implemented and
covered by28 related passing tests after synchronization, plus lint/whitespace
checks. PR300 merged at f414e729; PR301 is published and armed at1f48f459.
PR301 does not claim measured parallel speedup yet. Full and projected e1fd6029
source uploads passed complete readback verification. Runtime source is pinned
before the merge so scientific bytes remain independent of branch synchronization.
All24 unrelated local files were backed up, stashed during delivery, restored and
verified byte-for-byte; the temporary stash was dropped.

A new middle512 rental69e3ddf705f6 was lost before job claim. Provider logs reached
client_install_ready, but there is no started_at, CUDA admission, learner progress
or output. Root cause remains unknown; `failure-review.json` retains the exact
classification and evidence hashes. This is missing scientific evidence, not a
model/batch feasibility failure. The sole scheduler97240 stopped new admissions
and lets large512 finish before cleanup. Do not retry this failed cell without
new evidence or a repaired cause.

The prepared `resume_after_startup_loss.py` will only run after that scheduler is
gone, finished.json exists, all rentals are provider-confirmed absent, all previous
runs are terminal and a fresh inclusive ledger admits the cost. It continues the
11 untouched grid cells, including all ten user-added64×4/384×2 cases, and adds one
$1.25 parallel evaluation probe. It preserves the missing middle512 cell explicitly;
it does not relabel it complete or silently replace it. The older live-handoff
script `admit_parallel_probe.py` is superseded and must not be executed.

At14:13UTC all17 prior rentals were provider-confirmed absent and the ledger settled
at$17.51. Full local downloads of the two large completed cells were still in
progress. To avoid GPU idleness, recorded an explicit post-cleanup handoff intent
and interrupted only local scheduler97240. No GPU attempt was cancelled; no
complete-evidence claim was made. The new sole scheduler is Python8218, parent8198,
with11 untouched grid cells plus the$1.25 probe reserved. Caps12.25 plus settled17.51
and ancillary2 give inclusive bound31.76. The old status, controller, cleanup and
ledger are archived in `before-parallel-probe/` (no fabricated finished.json).

Queue-order amendment prioritizes small-deep64 and large-shallow64, then the probe
and remaining batched cells; both four-stream references remain at the end.
This gives early results for the user-added architectures while retaining every
source, seed, transition target and cell. The missing middle512 remains explicit.
A separate read-only fetch verifies the two large completed cells while the sole
scheduler owns all admissions and budget writes. Partial files are hash checked
before reuse. Middle256 full verification and coarse numerical audit passed.

A concrete learning-comparison candidate to cost after the probe: use one rental
per independent seed13591/13592, counterbalance treatment order, and compare all
five architectures at common stream counts64 and256. Include the actual64×2 /
four-stream baseline, so expanded batching is tested against the current recipe
instead of only comparing two faster alternatives. A524288-transition horizon
would give22 runs, with common milestones131072/262144/524288 and their mapped
schedules. Reuse an initial evaluation only after exact initialization identity
is verified across stream counts; retain all final policies and paired greedy/
random blocks. This is an unfrozen candidate, not an admission or a learning
claim. Probe-measured cost, all feasibility results, independent evaluation blocks,
retention time and the inclusive ledger must decide whether this horizon and
replication fit. Do not choose treatments from calibration scores or assert a
model's ceiling from this early-learning horizon.

## Second startup loss and stronger evidence capture (2026-10-17 14:26UTC)

The added small-deep64 run e76a6e1883d3 was admitted but lost before job claim.
Unlike middle512, its provider logs end at client_install_start. It was deleted
about131seconds later, consistent with the installed SDK's120-second uv-sync
limit, but the timeout and its underlying cause are not proven by those logs.
Retained review is `results/small-deep-s64/failure-review.json`. No CUDA/learning
record exists; this is not a model-size result. The scheduler8218 blocks new
admissions while sister run e931e7a9d6db, large-shallow64 with the same9a1f source,
continues. The sister job reached ordinary setup, whose600-second limit remains
unchanged. Do not claim it has begun training merely from SDK phase running.

Committed startup diagnostics3769c180 preserve exception type and, for the SDK's
dependency-install timeout, its limit and a bounded sanitized stderr tail before
rethrowing the original exception. Five tests pass, including preservation of the
original timeout and redaction before truncation. This does not repair or assert
the dependency-install cause and is not in the still-armed PR301 head. Publish it
as the next PR after301 merges; do not push it into301 while that head is armed.

A read-only provider-log observer `.runs/grid-007/capture_live_startup.py` runs
against controller8218 (tool session44460), bounded to30minutes. It follows logs
continuously in separate processes, sanitizes URLs/environment values and stops
streams when the scheduler records deletion. It never queues, retries, terminates
rentals or writes the budget. This closes the evidence gap where scheduler ticks
could miss final provider output before self-deletion. It must be restarted with
any later controller; the old occasional provider snapshots remain retained.

The large256/512 full downloads and numerical audits are complete. Twelve-cell
aggregates now retain all results and CPU/quota metadata. Large512:140.565seconds,
932.466transitions/s,17.517GB peak allocated CUDA,256Adam steps,31466selected
exposures. Its observed throughput is1.1832×large256 on a different rental. All12
coarse audits passed, without resolving the historical CUDA/numerical incident.

## Reviewed replacements admitted to the bounded queue (2026-10-17 14:35UTC)

Large-shallow64 e931e7a9d6db did not time out: after an initially successful
nvidia-smi, setup built the native engine and exited255 at555.96seconds because
its final nvidia-smi reported `Failed to initialize NVML: Unknown Error`.
Every fetchable output verifies. `failure-review.json` records the initial GPU
health, setup failure and hashes. CUDA admission and training never began.
NVIDIA documents container loss of GPU access as one possible cause of this
message (including host cgroup updates); this is a hypothesis only here, with no
host evidence proving it or explaining the earlier CUDA incident:
https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/troubleshooting.html#containers-losing-access-to-gpus-with-error-failed-to-initialize-nvml-unknown-error

All19 old rentals were confirmed absent. The conservative total settled at18.16,
with zero open caps before new admission. PR301 merged atae8152f31583572eba02bcf28e7e5fd6cf96ce5d.
PR302 is published and armed atd004d798 with the timeout diagnostics; five tests
and lint passed after synchronization. All24 unrelated files were restored and
hash-verified, and the publication stash was dropped.

`resume_with_startup_diagnostics.py` launched the sole scheduler Python78107,
parent77941, and read-only observer parent77943 at14:32UTC. It queues one manual
replacement each for small-deep64, large-shallow64 and middle512, followed by the
nine untouched grid cells and parallel probe. Each replacement preserves the
original size/streams/seed/source/transition target and retains `replaces` plus an
explicit reason; originals remain failed in history. This is bounded recovery
with improved evidence, not a claimed root-cause fix or automatic retry loop.
The source9a1f successfully reached native compilation on the sister rental,
and the observed GPU management access loss preceded learning. New rentals must
pass all existing GPU checks and CUDA admission. Any new failure blocks admissions.

New caps13.25 + settled18.16 + reserve2 =33.41 within50. The observer is now bounded
to three hours and stops when its owning controller exits with no live rentals.
Diagnostics are pinned to3769c18075f12483483b5e2b3f7d762008d45ddc; existing training
sources remainf057/9a1f and the separate parallel probe remainse1fd6029. The
controller resolves a missing logical cell only after its successful replacement
has been fully fetched and verified. Old control files and ledger are archived in
`before-startup-diagnostic-replacements/`. Do not rerun this admission script.

## Prebuilt-client repair and shared setup (2026-10-17 15:16UTC)

Controller78107 ended after replacement failures; no science was rescued by those
replacements. Large-shallow64 replacement6bef7648d5e4 now has a proven120-second
SDK installer timeout, stderr ending at building etude-sdk. Small-deep64
replacement394601fdf1a5 instead reached the job and exhausted600seconds downloading
CUDA dependencies. Neither reached learning. Both rentals were confirmed absent;
conservative settled total18.91, no open caps before the new canary. The original
CUDA/NVML cause and historical finite-weight anomaly remain unresolved.

PR302 merged at865f289def569371d1e9faffc32cbc0880e25b1b. PR303 is published and
armed for the source-identical prebuilt Etude client adapter. Nine related tests
passed after synchronization. All24 unrelated files were restored and hash-verified.
The original SDK archive, wheel and prepared bundle are retained in
`.runs/grid-007/prebuilt-client/`; isolated macOS installation reproduces all52
SDK Python files plus forwarding pyproject/lock. Guardian bytes and the total
120-second client install budget remain unchanged. Protocol010 freezes a single
$1 canary using original scientific source4d65d477, with project setup1200seconds,
CUDA60, training180 and retention240 inside a30-minute job limit. That larger
setup allowance does not change scientific learner limits.

Canary2ef1d393542c was admitted by the sole schedulerPython69435, tool session33686,
with read-only observer77313, session66008. Both client phases passed: runtime
install10.082seconds, offline wheel0.071seconds, totalclient10.170seconds. It
reached the job at15:02UTC but remains in project setup at15:16UTC. Full CUDA,
two-update training, checkpoint verification and cleanup are still required.
The new `.runs/grid-007/prebuilt-client-probe/verify.py` gates any next admission.
Do not rerun the admission script. Current open reservation is$1, ancillary$2.

Committed8ebcfe93 adds grouped_grid and frozen protocol011: remaining64x4 cells
share setup on one rental,384x2 on another, and the two original512-stream cells
on a third. Every cell retains the original source,seed13571,131072transition
budget,1800second learner limit and separate process/output directory. Group
progress aggregates persisted per-cell clocks and prefixes checkpoint paths;
a new cell cannot reset prior progress. Any failure stops that group and new
admissions; unstarted cells remain pending. Twelve related tests pass. This change
is not yet published because303 is armed. A first64x4 group controller is prepared
at`.runs/grid-007/grouped-small-deep.py`, but NOT admitted: it refuses to proceed
without complete canary010 verification. Other groups and parallel probe009 remain
pending. No grouped result or added-shape training result exists yet.

A separate zero-GPU-cost local timing probe was frozen and started at15:14UTC:
`.runs/grid-007/local-evaluation-probe/`, tool session79397. It uses the retained
384x8 initialization from calibration628c404a1176 and one eight-leg random-anchor
block, same per-game120second/10000command limits and ordinary compatibility/replay
checks. One CPU process/thread,600second outer bound plus30second shutdown. The
local16-core128GB host has capacity; only training/execution.py differs fromf057
among manabot/managym files, and this RSS fix is not called by evaluation. The
probe is distinct fromGPU-only calibration008. It tests whether future evaluation
can use local CPU instead of occupying paid GPUs; no scores select treatments,
no cross-platform numerical identity is assumed, and no future panel has been
admitted from it. Retain its failure if it cannot complete; no automatic retry.

## Verified early-learning screen and dependency repair (2026-10-17 15:52UTC)

Protocol013 completed all24commands and1536games in1309.95seconds locally.
All895manifestfiles,48monitors,checkpointcoordinates,registration/role bindings,
tracehashes,replayreceipts and recomputed scores verify. Report:
`experiments/grid-early-learning-013.md`; machine-readable full pairedblocksummary:
`experiments/data/grid-early-learning-013-summary.json`. Small64x2 scores against
scriptedgreedy at4/64/128/256streams:56.25/53.125/37.5/21.875percent; random:
84.375/84.375/56.25/65.625percent. Initialization15.625/46.875percent. At64streams,
greedy small/middle/large53.125/50/28.125percent. All12eligible cells evaluated;
no selective scoring. Four deal blocks and one training seed do not establish
architecture rankings, noninferiority or a skill ceiling. The longer confirmation
must retain64streams alongside256, allfivearchitectures, independenttrainingseeds
and newevaluationdeals; prioritize quality versus both transitions and elapsedtime.
This is an evidence-driven candidate revision, not an admitted new training plan.

Canary010 2ef1d393542c failed the1200second project setup beforeCUDA/training,
although source-identical prebuiltSDK installation passed in10.170seconds.
Allfetchableoutputsverified and rentalproviderabsenceconfirmed. Conservative
settledrentals19.52; zeroopenrentalcaps; ancillaryreserve increased2to5within
unchanged50total authorization for exact dependency mirror and transfers.

All94Linux3.12lockedwheels were downloaded and individuallyhashverified.
Offlineinstallation and subsequentfrozen reconciliation passed for bothoriginal
4d65 andgridf057/9a source locks. Exactpackageversions/URLs/hashesidentical across
those dependency selections. The4.221GBimmutablebundle remains in bounded upload
and fullreadback; noGPUrented or cachecanaryadmitted yet. Newlocked_gpu_setup helper
verifiesarchive,manifest,sourcelock,wheels andrequirements beforeofflineinstall,
then executesoriginalGPUsetup under the same1200second totalphasebudget. Fourteen
related tests pass sofar. Fullrealbundle helpervalidation is running. Freezeprotocol014
and hash the exacthelper/wrapper before anynewadmission; failure010cannotopenthegrid.

PR303merged b09a42dda32dd448a95c05b8e50968ff1e1d7a9d. PR304merged at15:57UTC
withcommitcfd3ef45dbf0bf439df52e53827278d78d248873. Fullrealbundle helper
validationpassed all94wheels in6.69seconds; GPUruntime remainsunverified.
No liveLF Flow. Taskautomationdisabled andCIwatchernotrunning; GitHubautomerge
isindependent. Taskandgoalremainactive, withlongerlearningcomparisonstillrequired.

## Current handoff (fresh observation2026-10-17 17:36UTC)

The early-learning screen is verified and reported; answering Jack's question,
64streams preserves substantial earlylearning on small64x2, whereas larger
batches can lose sampleefficiency despite throughputgains. Bigger-model ceiling
remains untested. Do not treat this early screen as the final recommendation.

Committed8e39c84a5a8ced7d093fb108370974722f0e8201 adds exactlockedGPUsetup,
optionalcachedsetup arguments, protocol014, tests andverifiedscreenreport.
After sync, PR305headb87d8c1e8b672ea62cf29f7bfc645793093f8f80 ispublished and
GitHubautomergearmed;25relatedtests andruffpass. PR304merged. All24unrelatedfiles
were restored andhashverified; onlythepublicationstashwasdropped.

The original4.22GBupload failed on a closedconnection duringpart398. Its known
multipartuploadwas explicitlyaborted andcleanupreceiptretained. The replacement
`.runs/grid-007/linux-wheelhouse/upload_resumable.py` islivePID29473(parent29472,
doppler29368),toolsession72676. Itretainsatomicper-partreceipts,two uploadwriters,
boundednetworkretries,30-minute totaldeadline,andfullETag-boundreadbackbefore
issuingupload-receipt.json. At17:36UTC,156parts/1,308,622,848byteswere uploaded.
Do not start a second upload or a GPU before its completeverifiedreceipt.
ListMultipartUploads was denied; noaccount/policychangesweremade. Cleanup of
our exact knownfailedID succeeded withoutlistingunrelateduploads.

PreparedbutNOTadmitted `.runs/grid-007/locked-dependency-probe.py`, its observer
andverify.py. It gates onremoteuploadreceipt,all94realwheel/sourceverification,
committedhelperhash,freshproviderabsence/quote/budget and noexistingcanaryhandle.
Protocol014 freezesone$1canary,unchanged1200secondsetupdeadline,twooriginallearning
updates andprovidercleanup. The source remains4d65d477; helpersareembeddedoutside
itscheckout. Oncompletionverifybeforeopeninggroupedgrid011. Neverrertunold010
admissionscript. NoGPUiscurrentlyrented; conservative settled19.52, openrentalcaps0,
ancillaryreserve5 within50. Taskandgoalremainactive. Taskautomationdisabled,
CIwatchernotrunning; GitHubautomergeisindependent.

## Transfer recovery and current state (2026-10-17T20:36:23.044122+00:00)

PR305 is merged as ab32ed6cd378d4b7c7154552b6678185f73169ff; selected setup files are clean and the verified helper hash is unchanged. All historical failed rentals remain charged: conservative settled $19.52, open rental caps $0, ancillary reserve $5, total authorization $50.

The fixed-length upload succeeded: all504parts/4,221,296,640bytes are retained remotely. Its subsequent boto/urllib3 readback failed with a truncated response; PID66107 is absent and no success receipt was issued. Readback-only recovery is live atPID68841 (tool17661), retaining16MiB verified chunks, ETag binding, exact local-byte comparison and final ordered archiveSHA. It never replaces the uploaded object. The reason, script hash and failed-log hash are retained in linux-wheelhouse/readback-recovery-reason.json. No GPU is admitted before the complete receipt.

The single canary014 controller remains prepared/unadmitted. Grouped campaign controller grouped_campaign.py is prepared for the three protocol011 groups with max two rentals, aggregate$5.50 reservation and verified014gate. Its one scheduler handles all pools; a terminal group frees capacity only after provider-confirmed deletion. The read-only observer can query not-yet-created pools (database returns an empty tuple). Offline verifier verify_grouped.py preserves completed versus censored/missing coverage and uses the retained numerical audit definitions without replaying its aggregate-writing top-level code.

A retained-data inspection quantifies the update-frequency difference: small4/64/128/256 has22001/2048/1024/512Adamsteps and30784/32768/32768/32768selected sample exposures at the same131072transitions. This supports describing a change in update grouping, not proving it caused the strength differences. No new scientific scores or training results have been added.

MTG122 existing primary conversation f22259914d814fcc876f887b164e917c was opened in a workspace in the requested cmux window. Its paused goal was left paused; complexity remains deferred and no new science was launched there. This does not pause MTG135. Fresh Task135 reads show active/incomplete, no live Flow, and historical unfinished PIDs absent. Taskautomation disabled; CIwatcher not running. The preceding goal turn was a verified wait on uploadPID66107; this turn repaired its now-observed readback failure.

## User steering (2026-10-17T20:39:25.579956+00:00)

Jack asked whether fewer optimizer steps at larger stream counts should already have been obvious from the code, and whether that coupling is required. Corrected the framing: one timestep-grouped epoch structurally gives at most transitions/streams Adam steps, with empty-filter skips accounting for the four-stream difference. Actual counts and strength results are empirical; the direction of the step-count change is not a discovery. This is a deliberate current-method choice, not a hardware requirement or proof of optimality. The checked final Ataraxos supplement S3.4 uses202simulator-step batches,1536positions/GPU before filtering, one epoch. Our four-stream recipe is an explicit scale adaptation, so preserving its tiny optimizer batches is not automatically desirable. Decoupling collection and optimizer batch sizes is a distinct controlled treatment, not an unannounced edit to the frozen grid.

Jack then explicitly requested starting64x4 and384x2. Preserve allfive stream counts for both, run their two groups concurrently immediately after verifieddependency014canary, within the existing$50cap. No new budget or permission request is needed. This reaffirms the existing authorized sequence; it does not waive bundle integrity, the CUDA/setup check, original deadlines or artifact retention.

Verified dependency mirror at 2026-10-17T20:41:26.764982+00:00: all4,221,296,640bytes read back, each range compared against local bytes and fullorderedSHA matched a5338601. Readback-only recovery took500.99seconds. Protocol014controller is now started (tool96431); its fresh ledger preflight reserved$1, leaving settled19.52+open1+ancillary5 within50. Admission/outcome must be read from its boot-handle/status and verified before the two added-shape groups start.

Canary014 terminal at 2026-10-17T20:49:57.043009+00:00: run6b254f5e2ddd lost before start; providerpodsr74luqazvi9lb confirmeddeleted. SDKclientinstallation passed in89.2246seconds (runtime dependencies89.0285s, source-identicalofflinewheel0.1481s), but no remote worker receipt arrived before the fixedfive-minute prestartcutoff. NoCUDA, lockedprojectsetup, ortraining was reached. failure-review.json andcompleteproviderlogs retained. Conservativeallocationcharge$0.30; settledtotal$19.82, openrentalcaps0, ancillary5. Grouped64x4/384x2areNOTadmitted. Noautomaticretry. The post-install/pre-receipt boundary is not yetlocalized; inspectbootstrapfetch/sourcefetch/receiptpublication and instrument only what newevidencejustifies.


## Startup015 5whys correction (2026-10-17)

Jack requested `$5whys`. Canary015 fe1b153c3183 is terminal, provider absence confirmed, no worker receipt or training. The initial dependency-install interpretation was incorrect: provider timestamps show client runtime install began22:18:44.465UTC and the container stopped22:18:48UTC. The controller claimed at22:05:43.253UTC, and macOS entered idle sleep at22:05:43UTC for772seconds. Source and bootstrap staging resumed after22:18:35UTC; the provider created the pod22:18:39.813UTC, already476.56seconds past the SDK five-minute prestart allowance. Subsequent controller sleep delayed local settlement until22:23:53UTC. This establishes a local launch-liveness/deadline failure; it does not diagnose CUDA or prove another dependency timeout. Exact sleep, provider and object timestamps are in worker-startup-probe/failure-review.json and its referenced records.

PR306 merged as ca04fe2a527c2a1959bc1447404496a2036a0a4e after all CI passed. Its worker diagnostics remain useful but were not reached by015. All24 unrelated files restored and hash-verified; temporary publication stash dropped. Conservative settled20.38, open caps0, ancillary5 within50. Grouped models remain unadmitted. Do not rerun015. A targeted local idle-sleep assertion plus a pre-provider freshness gate is being tested before any separately recorded replacement. No new dependency packaging or longer deadline is justified by015.


## Startup optimization delegation (2026-10-17T23:02:11.177729+00:00)

Jack asked for a maintained training image and/or fewer runtime dependencies, explicitly authorized an autonomous worker with a separate $5 GPU-testing cap, and reaffirmed "launch that task though". Existing Intelligence Task MTG-121 is the same cohesive outcome, so its brief was updated rather than filing duplicate work. MTG-114 is completed; MTG-150 separately owns W&B removal. Accepted outcome: phase-timed before/after startup through a real CUDA update and recovered valid checkpoint, pinned runtime/lock identities, reproducible rebuild/deployment, cleanup receipts, conservative $5 all-attempt ledger, and focused PRs. Mechanism choice is delegated; custom image and dependency reduction are alternatives to evaluate.

LF checkout and run both refused before preparing work: "shared Task first-start admission is unavailable; no recorded execution owner was observed". Read/claim admission confirmed "Task has no established first-start authority; historical absence cannot admit execution". Fresh Task records show no checkout, Flow, Session or unfinished Process for MTG-121; location is unrecorded. Do not modify LF databases or bypass its ownership safeguard. No formal Task Flow, checkout or GPU rental was created. A direct authorized collaboration worker `/root/startup_preparation` is running read-only code/evidence analysis and preparing a precise implementation/test plan while this coordination dependency remains. Its scope excludes repo mutations and paid tests until supported admission is available. This is not a claim that the requested full Task launch succeeded.

Current MTG-135 canary016 remains independently owned by PID9076 with sleep guard PID10439. Run56ac402f2a9f completed transfer of the 4,221,296,640-byte bundle at about23:00:31UTC; project setup remains running, no verified training result yet. Do not tick its shared scheduler from another worker. All25 publication-stashed files were restored and SHA-verified; only the named mtg135-launch-guard-preserve-20261017 stash was dropped. PR307 remains auto-merge armed, with focused36tests passing and CI in progress. Task and goal remain active/incomplete; no scientific groups have been admitted.


## Guarded canary verified; grouped sweep running (2026-10-17T23:08:48.349368+00:00)

Canary016 run56ac402f2a9f completed successfully and verification.json passed: two CUDA updates, all retained study-manifest hashes, exact original source and seed, finite checkpoint reload, worker diagnostics, a 4.420561-second claim at provider creation, guard cleanup and provider-confirmed deletion. Setup took625.2284seconds, CUDA admission4.2051seconds and training7.02085seconds. Estimated rental cost$0.24654; conservative$0.45 is settled in the original50-dollar ledger. This validates the guarded startup path on this attempt, not the old CUDA/NVML causes or the historical weight anomaly.

Started `.runs/grid-007/grouped_campaign.py` under the tested launch guard. Controller PID91501, tool97690; sleep inhibitor PID92293; read-only provider observer tool54043. Two groups are live and reporting dependency-download progress: small-deep64x4 runa89eab426bb0, sessiond699e9afbeca; large-shallow384x2 runf5ece8ba99f6, session08709748be4e. Both retain source9a1faddb96ba518d9b6e63eed545f157d9771c6c and the frozen five stream-count cells. Original middle512/small512 remains pending, to be admitted by this same controller only after capacity frees and no group failure. Do not start another scheduler, group or observer. Fresh provider inventory was empty before campaign admission.

Ledger now: settled20.83, reserved5.50, ancillary5.00, authorized50.00. The separate MTG121 startup allowance has spent0. Do not delete the locked dependency object while these groups consume it. After campaign settlement run verify_grouped.py, then the prepared added-cell learning panel only for verified healthy cells. Longer independent-seed evidence and the recommendation to122 remain required.

MTG121 preparation worker has now finished (no longer running). Its concrete image/dependency plan, baseline evidence and four-attempt5-dollar test proposal are saved in the existing Task brief and local scratch/remote-training-startup-design.md. Formal Task launch remains blocked on missing first-start authority; LOO414 owns recovery, with no installed supported recovery found. No child Flow, checkout or blocking Session was created, and no paid child tests occurred. Do not claim the full autonomous Task launch succeeded.

PR307 is published and auto-merge armed; integration CI was still running on the last read. Task automation remains disabled and CI watcher is not running; GitHub auto-merge is independent. MTG135 Task and goal remain active/incomplete.


## Startup worker running; longer-run infrastructure (2026-10-17T23:37:35.011620+00:00)

MTG121 first-start admission now reports the local execution owner and a dedicated checkout at /Users/jack/src/manabot.mtg-121-reduce-remote-gpu. Another session launched pursue between this operator's fresh status read and launch. Original Flow b3a2aee5-34c3-415d-89a0-15c82d24e62b and Session session_6404e7cda99448b3a83b1738f96291fd are live (Flow PID24337, agent34188 observed). Duplicate Flow8ea93e7c-45d6-447c-ac66-d30877160d1d is verified Stopped after SIGINT to its own Flow process; its implement/compress passes made no repository changes or paid admissions. Full accepted direction and small baseline receipts were copied into startup Task scratch before launch; a saved steer identifies the original worker as sole owner. The separate5-dollar cap remains in force; preparation spent0. Do not launch another worker.

PR307 is merged as fb8ecd6ef0aaf5e81c9efaaabeec4de5782bd7c4. New grouped_learning runner is published in PR308, head3c3681d2b26a7fc8167928fa63fe24a6cd3d058a, with auto-merge armed. It shares setup per independent seed, uses fresh per-cell processes/deadlines, checks exact common transition checkpoints and verifies required initial/final/optimizer/milestone hashes before later admissions. Interrupted incomplete groups now retain nonzero status matching process exit.38 focused tests, Ruff and real retained small-four-stream/large64 manifest checks pass. Live GPU execution of this runner is not yet validated; no longer-learning allocation is admitted. All26 unrelated files were preserved and SHA-verified after publication; only the named publication stash was dropped.

Controller91501 and guard92293 remain live and untouched. Both added groups have completed64/128/256/512-stream cells and are training4-stream references. Preliminary saved clocks for64x4:64=278.79184s,128=168.24679s,256=111.82094s,512=65.95457s;384x2:64=197.75156s,128=120.65585s,256=101.38784s,512=72.17548s. Full artifact/numerical verification and the new-cell learning panel remain pending. Do not change frozen helpers or tick a second scheduler.

Cost planning .runs/grid-007/price_validation.py is implemented but not an admission. It requires verified timing for all five models at64/256 plus small4 and proposes two independent seeds13591/13592. Candidate131072-multiple horizons use linear measured extrapolation,1.35 engineering slowdown allowance, explicit per-cell overhead/deadlines, shared setup and retention,1.25USD/hour quote ceiling and conservative quarter-dollar reservations. With four new-shape timings still unverified, its totals are explicitly lower bounds and no candidate is admissible. The old1.31M proposal already exceeds remaining headroom before those missing cells; freeze an affordable horizon only after full current settlement. Budget remains settled20.83+open5.50+ancillary5 under50. Task and goal remain incomplete; disabled Task automation and stopped CI watcher do not schedule this conversation.


## Storage-read recovery and longer comparison preparation (2026-10-17T23:55:48.188763+00:00)

Original grouped scheduler PID 91501 exited on one SDK StorageUnavailable GET no-reply while observing the completed large-shallow startup receipt. It had already deleted that rental. The same SDK GET subsequently passed at 23:50:47 UTC; recorded run f5ece8ba99f6 is succeeded, while a89eab426bb0 was still running. The old process and its sleep inhibitor were absent. Original source/log/launch/status/ledger snapshots and the recovery reason are preserved in grouped-feasibility/before-storage-recovery/.

Sole scheduler recovery is .runs/grid-007/resume_grouped_after_storage_loss.py, PID 14787, tool 84648. It resumes the same recorded attempts and unchanged $5.50 reservation; it creates no replacement training. Current log is grouped-feasibility/controller-resume-storage.log. The original read-only provider observer is still sufficient. The predeclared pending original-512 group was admitted as run 610a95a1b1af after the successful large-shallow rental was confirmed absent. The existing stop-on-failure rule remains. Inspect launch.json and process liveness before any further scheduler action.

A narrow process-local storage-read guard is implemented in experiments/runners/etude_storage_reads.py, with tests and CI entry, but is not committed/published yet. It retries only the pinned SDK GET no-reply boundary, at most three attempts, preserving errors and truthful absence. Writes, permission/malformed responses and provider errors are untouched. Seventeen targeted tests pass and an actual read-only SDK receipt read passed under the guard (storage-read-protection-check.json). This does not establish the underlying network cause. PR 308 remains armed with passing focused checks and integration CI still running; preserve its head until it merges before publishing this next fix.

Source 3c3681d2b26a7fc8167928fa63fe24a6cd3d058a is fully uploaded/read-back verified: full tar 226467840 bytes, SHA a044d4c43dbdbf19c2ad43b410c994fd34614d7ad1f4a72053e92415f804b965; projected transport 37177238 bytes, SHA e4a6b0e6f71d840f96892549af1f2f94c5e419dd83ebb9b1967365c968031ec5. Projection manifest SHA 94a42bcc7c57434aea7ba19b76751766b7f889106faaaf82afb7a752ecf3eb5e. learning-017/source-config.json retains all 1670 source file identities; the exact locked-wheel manifest admits this source lock. Runtime diff against added-shape source changes only process RSS observation in manabot; model, learning, engine and locks are unchanged.

Unfrozen draft .runs/grid-007/validation-protocol-draft.md preserves all five shapes at 64/256 plus small4, seeds 13591/13592, reversed full order across seeds, common exact checkpoints and 13 new paired deal blocks per anchor. Its operational competence/selection and uncertainty rules are written before new scores. Horizon selection now leaves at least $1 uncommitted after priced reservations. .runs/grid-007/learning_campaign.py is prepared from the known controller with a frozen-allocation gate, capacity observation, bounded sleep guard and narrow storage-read recovery; it is only syntax-checked and still requires completed pricing/allocation and a dry run. No longer-learning allocation, reservation or GPU run has been admitted.


## 2026-10-18T00:12:21.235960+00:00 — grid retained; longer comparison frozen

PR308 merged as 1df750a0726c93e3149654dcd7bce7ab187eb9a2. PR309 (666e06c1962aaf14b564738f000e13bfdb674ab9), the opt-in bounded transient receipt-read guard, is published with auto-merge armed; integration checks were still running at the last read. All 26 unrelated files were restored and hash-verified after publication. The Task remains incomplete.

The recovered sole grouped controller PID14787 finished, retained all artifacts and confirmed all three provider rentals deleted. Conservative campaign charge is $2.67; the total ledger is $23.50 settled, $0 open caps, $5 ancillary reserve before longer admission. The SDK rental estimate alone undercounted occupied time, so settlement used the quoted-rate occupied-time floor plus per-rental margin.

All eleven newly completed 131072-transition cells passed manifests, source/recipe/seed, raw-checkpoint loading and coarse Adam-state checks. This completes 23 of 25 grid cells. Small-deep4 was interrupted at 112384 transitions/439 updates after 1800.0667501820135 seconds with `collector exceeded training deadline`; its retained initial/32768 checkpoints are finite and separate from complete endpoints. The earlier large4 cutoff remains separate. Neither cutoff is a CUDA initialization failure. CPU screen013 panels are admitted once for all eleven new complete cells under a bounded sleep assertion; they are not yet scored or verified.

Protocol017 is frozen in experiments/protocols/longer-learning-017.md, SHA256 f6f22c4ca0ee67848f4d42f1852b0321b7cf18a58c40c1de8d2134a5c2402281. Full verified timing evidence prices 524288 transitions at $17.50 ($8.75 per training seed13591/13592); the next candidate655360 requires $21 and fails the declared $1 uncommitted-margin rule against $21.50 headroom. All five shapes at64/256 and small4 remain, eleven cells per seed, reverse order, no warm start. Raw points0/131072/262144/524288; immutable source3c3681d2b26a7fc8167928fa63fe24a6cd3d058a. The allocation binds26 input hashes and the dry run passed, including exact frozen helper-byte equality and stdlib-only bootstrap import.

Started the learning017 controller command with the exact frozen allocation; admission must still be confirmed from launch.json. It owns the only scheduler ticks. Budget reservations precede admission, provider inventory includes other Tasks, and failures stop new admissions. MTG121's implementation Flow has stopped on registry pull-access uncertainty; a read-only preparation worker is checking whether existing Etude ECR credentials resolve it without new access or public publication. Its separate $5 is untouched in the worker's last ledger. Task automation is disabled; CI watcher is not running.


## 2026-10-18T00:23:44.658929+00:00 — admission evidence and capacity deviation

Learning017 has two admitted live runs: e7acaf5f0c94/seed13591 and d2c61bf03f93/seed13592. Sole controller PID8930 and its read-only observer are active. RunPod pods pvin4z9pqivcfk and ykksxq1x8wqnnk were created at00:15:11 and00:15:18 UTC at$1.19/hour. Both finished the4.22GB runtime download by00:21. The $17.50 reservation remains, with$23.50 settled+$5 ancillary reserve+$4 uncommitted. Exact pre-admission input bytes are retained in learning-017/frozen-inputs, all matching the26 preregistered digests. The original ledger bytes were reconstructed by reversing only this reservation and recomputing outstanding caps, then verified against its previously frozen SHA256; this method is explicit in frozen-inputs-receipt.json.

The account-wide concurrency promise was violated. A foreign pod19j6cpds07kfvv/etude-cb3b3eab7371 was created at00:13:33 UTC while our jobs were queued. Our admission loop counted provider-visible pods and local active jobs separately, so it did not count a queued local job that had no visible provider rental yet. Both local jobs subsequently materialized, leaving three account pods. The complete sanitized provider inventory is retained in learning-017/capacity-observation.json. This is an infrastructure/protocol deviation, not a scientific source, seed, schedule, horizon or spending change. Do not terminate unrelated work or silently hide the deviation. Preserve the already-running paid trajectories; no additional GPU admission is permitted while capacity is full. Before any future launch, capacity must count the union of provider-visible pods and local unmaterialized admissions, with explicit reservation/serialization for cross-controller races. Current startup-task admission is held by its own fresh inventory. Do not modify the frozen live controller as if that retroactively fixed this admission.

MTG121 continuation Flow ac4bc75d-89e5-4a3a-939c-3953cada1913 is verified live (Flow PID80095, implement PID80220). It read the ECR access findings, rebuilt/admitted all94 locked dependencies and is exporting the revised image; no GPU admission was made. Its current note correctly distinguishes offline Python installation from a native Cargo build that still downloads crates. Its exact image and real provider pull/first-update/checkpoint evidence remain outstanding.

Prepared verify_learning.py, evaluate_learning.py and analyze_learning.py for the frozen longer study. They are not new results. The verifier passed an actual retained earlier trajectory and both raw artifacts, and rejected changed seed, source and deadline (verification-readiness.json). The full new-cohort verifier/evaluator/analyzer have not yet run. Evaluation must use bounded caffeinate and place controller.log inside its evaluation directory before manifest finalization. Do not overwrite the live panel or reuse an admission.


## 2026-10-18T00:48:11.060327+00:00 — expanded evidence delivered; startup worker verified live

The expanded short screen is fully audited: 23 complete cells, two retained four-stream deadline censors, 46 commands, 92 panels and 2,944 games. The added eleven cells account for 1,408 games and 820 manifest entries; both cohorts total 1,715 manifest entries. All 92 panel evaluation-runtime identities also match the validated CPU baseline. The standalone runtime receipt is learning-017/evaluation-runtime.json, SHA256 5c14b7f310d7937959aa8baf9e570f16c74f9c41904dd99eb26bb9bbe15142e6. No long-cohort strength has been evaluated.

Local commit 7418728d06b58484a3f58ff1d827594c99624eab retains the expanded report, complete summary, visually checked score chart and exact frozen longer protocol. It is not yet published. PR309 is separately merged as 16c03e5a546225c0c1ad33ef2fb208330c6ef52d after all CI passed; LF delivery reconciliation completed. This does not complete MTG135.

At the 00:47 UTC worker observation, seed13591 completed small64, small256 and small-deep256 at exactly524288 transitions each (758.8388, 288.6445 and361.4690 inclusive training seconds). It started small-deep64. Seed13592's first small4 trajectory reached115456 transitions without a reported failure. These are remote worker receipts, not the final offline manifest/optimizer audit. ControllerPID8930 remains the sole scheduler; no additional admission or tick was made.

Seven exact evidence files and an explicit provisional interpretation were copied into MTG122's own checkout at scratch/mtg-135-size-batching-screen-20261018, with SHA256 manifest and a link in its design. They include both short summaries, report/chart, source recipe and protocols013/017. This is an interim handoff, not the final recommendation. Both tasks use the GRU semantic-card encoder family, but MTG122 requires its common research catalog and exact per-world bindings. No weights, transport-ID buffers or checkpoint identity are transferable by assumption. Its execution hold remains; no Flow was started there.

Jack's request to launch the startup Task is already represented by live Flow ac4bc75d-89e5-4a3a-939c-3953cada1913, verified in fresh Task/Flow/lf-ps and OS records (FlowPID80095, implementPID80220, agentPID80916; Session session_205bd11584c842cc8f80160cd07a08f4). The corrected image passed isolated Python install, native compilation, real training imports, two CPU updates and finite ordinary-checkpoint reload. ECR upload remains active, with no published digest yet. The worker is reviewing its own recovery/settlement while uploading. Paid startup testing remains capacity-gated; its separate$5 has no admitted attempt. Leave this observed live Flow alone. Task automation is disabled and the CI watcher is not running; these launched processes are independent of those automations.


## 2026-10-18T00:57:48.322040+00:00 — analysis readiness and exact recipe export

The previous goal turn made progress by delivering seven verified evidence files into MTG122 without resuming its execution. This turn completed analysis readiness before any long-cohort scores are read. learning_selection.py now isolates the frozen selection rule with typed inputs. Seventeen focused tests pass for both-seed competence, seed-specific initialization improvement, later regressions, worst-seed timing, the10% size/memory preference, missing/duplicate evidence and retained-manifest damage. The analyzer checks complete manifest coverage and reconstructs checkpoint/anchor registrations through the ordinary loader; it retains absolute and initialization-relative paired deal differences separately. The readiness receipt binds exact code/tests, rechecks all895+820 short-study manifest entries, and reconstructs8 real retained registrations. These are machinery checks, not new scientific outcomes.

All11 admitted candidate regimes were exported from source-identical recipe code into learning-017/resolved-recipes. Their index binds allocation, source, complete resolved JSON hashes, seeds, limits and raw checkpoint iteration coordinates. At524288transitions, four/64/256streams run2048/128/32iterations with raw points0/512/1024/2048,0/32/64/128 and0/8/16/32 respectively. All map to2048referenceiterations. The configured LR remains at its1e-4 upper clamp throughout this tested horizon; tau falls from0.05 to0.005076577477226473. Exporting these configurations does not select one or alter training. The actual preferred recipe remains pending the full retained comparison.

Latest direct observation at00:56UTC confirms sole controllerPID8930 and startupFlowPID80095/implementPID80220/agentPID80916 alive. Learning017 seed13591 has3completed cells and is progressing through small-deep64; seed13592 small4 has159232transitions. No new long-cohort scores exist. MTG121's ECR image publication remains in progress; the worker added five passing recovery/settlement checks and is preparing ordinary recovered-checkpoint verification. Its fresh ledger contains no attempts,$0committed,$0settled and$1uncertaintyreserve within the separate$5cap. Neither the image nor local CPU validation proves a startup-speed improvement. GPU comparison remains capacity-gated.


## 2026-10-18T01:03:35.260004+00:00 — deferred 384x2 long-training Task captured

Jack requested a separate Task, explicitly not started, to explore384x2 in more depth against other64-streamGPU models toward~20kupdates. Created MTG-151 in loopflowstudio/manabot's Intelligence Wave after inspecting both Intelligence and Rules task inventories. The canceled MTG103 is historical, and MTG137 owns existing baselines; neither is resumed or duplicated. The self-contained brief includes current short-grid observations/limits, dependency on017findings, comparator candidates, long-run strength/time/cost evidence, independent seeds, precise schedule/update-clock decisions and a separately authorized future budget. No20kextension is added to the current experiment or required before MTG135's initial recommendation.

Explicitly ran lf task pause MTG-151. Fresh Task status confirms execution.status=paused, no checkout, no Flow and no Session. Link: https://linear.app/loopflow/issue/MTG-151/compare-384x2-with-other-64-stream-gpu-models-through-about-20k . Keep that hold until Jack resumes it. In the brief,20kcollection/trainingiterations would mean81,920,000transitions at64streams×64transitions;20kAdamsteps is different and the clock must be resolved before budgeting/launch. No money is transferred from either existing budget. This pauses only151, not the present goal or running studies.

At01:02UTC learning017's sole controllerPID8930 is verified alive. Seed13591 completed its fourth cell, small-deep64, at524288transitions/128iterations in917.5468792060274seconds and started middle64. Seed13592's small4reference reached189184transitions/739iterations. Startup121's Flow remains live; it is validating a first-update timestamp while the image upload continues. No additional training/evaluation launch was made during capture.


## 2026-10-18T01:20:21.248088+00:00 — score/time reporting validated; fifth long trajectory complete

This goal turn made concrete report-preparation progress and verified waits on the original live controller tool20728/PID8930. learning_figures.py now builds per-seed experience, inclusive-time and active-time curves, endpoint score/time figures, and a descriptive Pareto set across all retained checkpoints. Pareto dominance minimizes the worst observed seed's inclusive time while maximizing both anchors' worst observed seed scores; it does not change the frozen competence rule or represent a population confidence bound. The analyzer retains these tradeoffs with exact helper-code hashes.

Nineteen focused analysis tests passed. Real earlier screen013 data (all92points, one training seed) rendered successfully; no synthetic or protocol017 outcomes were substituted. Three PNGs were visually reviewed, and overlapping endpoint text was replaced with distinct stream-color and model-shape legends. All preview titles identify the earlier screen and rendering-check purpose. The receipt binds92source monitor hashes and figure hashes; visual-review.json records the actual inspected files. The readiness check again verified895+820retained manifest files and8ordinary checkpoint/anchor registrations. Actual017result figures still need generation and visual inspection after scoring. The full final report/handoff is not yet written.

At01:19UTC, seed13591 completed middle64 at524288transitions/128iterations in974.2241667569906seconds, bringing that worker to5completed cells. It is progressing through middle256. Seed13592's small4reference reached270080transitions/1055iterations after3360.84seconds, passing the exact262144halfway milestone. These remain remote clocks until the controller's complete retention and offline audit. Both controllers are live; no checkpoint scoring or new GPU admission occurred. The original controller handle was waited on directly in two bounded55second calls without competing scheduler ticks.

The current ledger remains$23.50conservatively settled+$17.50reserved+$5ancillary reserve under$50. Local storage has83GiBavailable; earlier2,944-game panels total93,511,481bytes, giving ample present space for the larger CPU panel and retained trajectories. Recheck actual resources at admission rather than treating this observation as permanent. Startup121's Flow remains live, image upload still active, and its fresh worker reports no rental/reservation. Provider inventory reported by that worker is now two pods, still no startup capacity. MTG151 stays explicitly paused, with no checkout/Flow/Session; it does not expand the current study.


## 2026-10-18T01:33:42.358433+00:00 — resource-accounting readiness; seven long trajectories complete

The preceding experiment continuation made analysis-preparation progress and verified waits; the intervening MTG151 capture/status response did not add scientific results. This continuation made concrete progress with learning_resources.py and check_resource_readiness.py. The new typed resource collector preserves per-cell learner active/inclusive clocks, iterations, actual Adam updates, filtered skips, selected exposures, games, CUDA/RSS memory and wrapper intervals. It records shared setup/CUDA phase intervals and request-to-deletion rental occupancy separately; it does not allocate shared startup to individual recipes or label SDK estimates as provider invoices. It requires complete retention, matching allocation hashes and settled deletion evidence, and only reads SDK records without ticking/admitting anything.

Readiness passed on all eleven actually retained prior-screen trajectories in three groups. All eleven deliberate Adam-counter mismatches and six duplicate/backwards event-log cases were rejected. Ruff passed. learning-017/resource-readiness.json binds actual prior logs, run files, resource outputs and helper hashes. This is preparation, not protocol017 resource/cost verification. After controller retention/settlement and verify_learning.py pass, run the collector with the existing Doppler/Etude configuration; resources.json does not yet exist. No live source, frozen protocol, deadline or allocation changed.

A55-second wait on the original controller tool20728 returned the same live handle. OS evidence confirms controllerPID8930 and MTG121 FlowPID80095/implementPID80220/agentPID80916 alive. At the latest saved worker observation seed13591 has7complete trajectories, including large-shallow256 at524288transitions in327.4322332320153seconds; it is training large-shallow64. Seed13592 small4 has336384transitions. Full offline retention, long-policy evaluation and final recommendation remain pending. Leave the live schedulers alone.

Fresh startup worker output still reports active ECR upload, no published digest yet and$0GPU admissions. Its latest capacity read is two pods. Task automation is freshly confirmed disabled; CI watcher is not running. MTG135 ledger remains$23.5settled+$17.5reserved+$5ancillary under$50. MTG122's execution hold and MTG151's explicit pause remain unchanged.


## 2026-10-18T01:47:49.756348+00:00 — report preparation; current MTG-122 destination direction

The previous goal turn made resource-extraction progress and a verified wait. This turn prepared render_learning_report.py and checked its table functions against all eleven actually retained added-screen trajectories: 44 checkpoint/anchor rows, 11 endpoint/resource rows and 176 deck-role rows. All 11 checkpoint-versus-fit counter/clock checks passed; syntax and Ruff checks passed. The renderer refuses to create a long-study report while its verified analysis is absent. The checks are saved in learning-017/report-rendering-check/table-readiness.json with the input/code/output hashes. They do not constitute protocol017 scoring, a full renderer run or a final recommendation.

The report keeps both seeds, initialization-relative deal intervals, all deck-role results, actual Adam steps/filtered exposure, separate active/checkpoint/fit/rental clocks, frozen qualification and the exact resolved recipes. Earlier real data showed final checkpoint clocks slightly precede full fit bookkeeping (for example 278.7309 vs 278.7918 seconds for small-deep64). Do not force those clocks equal. After the actual analysis and resource collection, render the factual package, inspect the real figures, interpret the full comparisons, and deliver the final recommendation. Retain the scientific analysis helpers with the final evidence so code hashes are accompanied by actual source; the current helpers remain task-local.

Fresh MTG-122 reads found later saved direction: 64-stream GPU training, Mini evaluation, all 36,000 assessment games and frozen controls, removing inline diagnostics, within its separate $35 GPU allowance. The latest scratch/questions.md split-placement direction supersedes the proposed reduced cohort and combined GPU evaluation. Our final recommendation must explicitly address the supported 64-stream choice, preserve the common research catalog/native identity and calibrate its final pipeline across all three pools. A possible 256-stream winner is an alternative to its selected stream count, not automatic adoption. No execution was started or resumed in MTG-122.

A dated scope clarification was delivered to its existing checkout at scratch/mtg-135-handoff-scope-20261018.md and linked from the working design. It records the current destination input hashes and clarifies that the interim handoff's hold sentence was not a new user instruction or additional approval requirement. MTG-122's own accepted direction governs execution; this note changes no existing hold, source, recipe, cohort or budget. The original seven-file interim evidence package was preserved.

A further 55-second wait returned the same live controller handle20728. Seed13591 reports 8 completed cells. Large-shallow64 completed 524,288 transitions in 725.0782296590041 seconds and the worker is now on large64; seed13592's small4 reference has 402432 transitions. No completed long comparison, final local artifact audit or new strength panel exists yet. ControllerPID8930 and MTG-121 FlowPID80095 remain the observed owners.

MTG-121's build receipt now records ECR digest sha256:6ba17f5787314edd7e64cd35e988f38dff1a2b8a2ca75f8af336f9d9484074ae. Its separate access receipt matches the digest and records 10 issuer-downloadable layers / 4,589,519,377 compressed bytes. provider_pull_proven remains false: this is not a RunPod cold-pull or startup-speed measurement. The worker reports candidate preflight passed without GPU reservation, is checking baseline preflight and cleaning unused temporary access while capacity is full. Leave its live Flow alone and inspect its actual ending/effects before any continuation. MTG-151 remains paused and outside the current allocation.


## 2026-10-18T01:53:59.611649+00:00 — verified wait on unchanged admissions

The prior goal turn made report-preparation and destination-handoff progress. This turn is a verified wait: two55-second calls to the original controller handle20728 returned it still running, with fresh OS evidence for PID8930. Saved worker coordinates advanced in both admitted groups: [{"seed": 13591, "phase": "running", "completed": 8, "current_transitions": 249856, "observed_unix_seconds": 1792288420.6047547}, {"seed": 13592, "phase": "running", "completed": 0, "current_transitions": 430592, "observed_unix_seconds": 1792288417.0877795}]. No endpoint, source, deadline or budget was changed, no new scheduler was started, and no long-policy scoring was admitted. This is healthy live work, not an impasse or a reason for replacement.

MTG-121's existing Flow ac4bc75d-89e5-4a3a-939c-3953cada1913 completed implement, compress and sync and is running realign. Its published-image and issuer-layer evidence are retained; both baseline/candidate preflights completed before the sync, and the worker removed unused temporary credentials and stopped its local verifier. The GPU budget still has no attempts or reservation. Realign is checking current build inputs and recording the local launcher's missing use of the newly landed bounded receipt-read guard before any preflight refresh. Leave this live Flow to finish its authored steps; do not relaunch because the GPU comparison is waiting for capacity.


## 2026-10-18T02:06:41.948937+00:00 — scientific source retained; ninth long trajectory complete

The preceding goal turn was a verified wait. This turn made concrete reproducibility progress and another55-second wait on the unchanged live controller handle20728. retain_analysis_source.py now preserves all18 task-local analysis/support scripts, including both legacy numerical-audit helpers and the launcher configuration helper imported by the resource types. The archive contains source only, no credentials, SDK state or paid-run launcher. Its README names the pinned repository, native/runtime identities, original relative paths and necessary retained datasets. It warns against directly executing the historical numerical-audit driver.

Current snapshot: .runs/grid-007/learning-017/analysis-source-ca0059c669f0ded97ecf, contents SHA256 ca0059c669f0ded97ecf538e0f284421e0d1a7127a33b0681f2bb24071b0fe54,18 byte-verified files. Original paths remain in the manifest; stored paths omit the leading.runs directory so normal publication does not silently ignore the archived source. Git check-ignore confirmed all18 proposed report publication paths are unignored. The renderer now includes these exact copies and requires its analyzer/selector/tradeoff source hashes to match the actual analysis receipt. Changed helpers require a new immutable snapshot before rendering. The earlier layout snapshot is retained as preparation history, not the current pointer. No long report or scientific result was produced by this archival step.

Ruff passed for the modified helpers. Table-readiness checks on the real earlier screen passed again:44 checkpoint/anchor rows,11 resource rows,176 role rows and11 checkpoint-versus-fit clock/counter checks. The full renderer still awaits the actual protocol017 analysis and resource files; actual figures and interpretation must be reviewed before delivery.

Seed13591 now reports 9 completed trajectories. Large384x8 at64 streams completed524288 transitions in 1318.1564286709763 seconds, and large256 is running. Seed13592 small4 has 488192 transitions. No endpoint, deadline, source or reservation changed. The sole scheduler remains PID8930. MTG121's existing Flow continues its authored iteration after integrating the receipt-read guard;21 launcher/guard tests passed, expired/deleted credentials are correctly refused, and its paid comparison still awaits capacity. It has no admitted GPU attempts or spending.


## 2026-10-18 02:21 UTC — unsupported cross-Task capacity gate corrected

Jack asked why MTG-121 GPU tests were capacity-gated. The operator traced the
condition to its own unsupported two-GPU provider-capacity claim in the startup
brief and the account-wide len(pods)<2 launcher check. Retained complete inventory
shows three simultaneous rentals. The separate $5 authorization was never used.
The startup Flow stopped at loop-or-next after repeating that false blocker.
Its remaining work will use the corrected own-Task one-rental/reservation limit;
the current Task brief, local plan and unexecuted comparison protocol are updated.
See grid-017-concurrency-amendment.md for the prospective operational amendment;
frozen017 bytes and the running scientific work are unchanged.

The factual report renderer now includes training/evaluation/runtime verification
and rental settlement receipts. Its latest 18-file source snapshot is
analysis-source-70ad4c4319997f59778e, 136983 bytes. No actual017 scores exist yet.


## 2026-10-18 02:27 UTC — startup capacity repair verified and paid attempt launched

The new MTG-121 pursue Flow ea781b00-002c-4ff1-a372-3237b1711f3f is live
(PID32707; implement PID32808, Session session_c6ff0036626241138f7668d1d288de45).
The worker removed the account-wide gate and passed27 focused checks, including
two foreign rentals allowed and own active/queued/unresolved/unsettled work refused.
The root operator verified all5 bound launcher/test/guard/wrapper/helper hashes
and their shared preflight identities. Both current preflights succeeded. The
first baseline attempt was launched with1USD reserved from the separate5USD cap;
no comparison result or completed GPU training is yet claimed. The price read was
1.19USD/hour. The earlier task-run invocation with --directive refused before
creating a Flow; the existing-Task run with --reason succeeded. There is only one
startup Flow and one sweep scheduler; no competing scheduler was started.

The final evidence renderer will carry the prospective concurrency amendment.
Ruff and the real earlier-data table checks passed; full017 rendering remains
pending actual analysis. Latest18-file source snapshot is
analysis-source-69aa8035e38fccb33d3b,137433bytes. Task automation is disabled and
the CI watcher is not running. The launched work owns its ongoing execution.


## 2026-10-18 02:48 UTC — verified wait and first startup baseline outcome

The preceding goal turns were verified waits on the original learning controller
(handle20728, PID8930) and the live startup Flow (handle44912, PID32707); no new
sweep controller or replacement was launched. Fresh receipts show13/22 long
trajectories completed: seed13591 has10 and its small4 reference is at157696
transitions; seed13592 has3 and large-shallow64 is running. Full retention and
all frozen playing-strength evaluation remain pending.

MTG-121 baseline001/run89cddc69d6f3 failed during dependency download after
1073.0895586013794 seconds of setup. Its retained log identifies an SSL socket
read timeout before Python installation, native build or CUDA admission. This
is not evidence about the historical CUDA initialization cause. The worker
verified all5 fetchable output hashes; there is no recovered checkpoint. The
stock image was provider-reported cached. Provider-confirmed deletion settled
a conservative0.57USD charge in the separate5USD startup ledger. Candidate001
has1USD reserved and is in source-reachability preparation under the same
scientific source, seed, GPU class and deadlines. Its prebuilt image may remove
the failing per-rental bundle-download step; actual provider pull and total
startup improvement remain unproven. The existing live Flow owns that comparison.


### 2026-10-18 02:51 UTC — startup candidate creation unresolved

The candidate has no confirmed provider ID; the fresh complete inventory reported
by the existing worker still has only the two sweep rentals. Scheduler renting
state is not proof of provider allocation. The SDK suppresses creation exceptions,
so the current attempt has no retained creation-error cause. The live startup
worker is adding narrowly redacted diagnostics for future attempts. Its1USD
reservation remains held, no additional startup rental is admitted, and no
private-image provider pull or speedup is claimed. The worker owns reconciliation,
retention and cleanup; do not launch another controller or release uncertain cost.


## 2026-10-18 03:06 UTC — startup controller handoff recovered

Fresh OS evidence disproved the startup plan's claim that original controller
PID44248 was alive: it was absent, no matching controller was observed, and
its last log was02:54:39UTC. The exit coincided with the owning implement step
ending; the exact lifecycle cause is not established. Root saved the correction
and scoped recovery direction in the startup checkout. The native Session steer
was saved but delivery was not established; the Flow's realign and loop decision
explicitly consumed the correction artifact and selected recovery of the same
attempt, without a new Flow or new rental.

The existing Flow's second implement pass has now started recover.py, observed
live atPID82481. Root inspected its source: it binds runabcb31376ece, original
session/pool and03:23:29.323805UTC deadline; checks isolated state and the held
1USD reservation; blocks both provider creation and queue_cohort; and preserves
the reservation on stopping. Its fresh recovery receipt still reports no exact-
name provider match. Unknown provider outcome is not a zero-cost refusal. The
worker must stay with terminal reconciliation and cleanup; another attempt
remains forbidden until resolved. MTG135's original controller remains live.


## 2026-10-18 03:13 UTC — durable startup reconciliation ownership

The startup worker deliberately stopped the tool-owned recovery process and
verified a launchd handoff. The new chain is DopplerPID67067(parent1),
uvPID67115, PythonPID67117. The label is
gg.manabot.mtg121-candidate001-reconcile; its plist and domain receipt are in
the startup Task's.runs/mtg-121 directory. Fresh recovery.jsonl observations
continue after handoff. Root inspected the source and plist: exclusive file
locking prevents duplicate recovery owners, provider creation and queueing
remain blocked, run/session/pool/deadline bindings are fixed, the ledger is not
rewritten, and credential cleanup is scoped to the Task-local grant after the
original deadline. It retains unknown outcomes and can detect a late matching
pod; billing currently omits even the known baseline, so it is not complete
settlement evidence. This recovery job, unlike periodic Task automation or the
CI watcher, is actually installed and running. Do not start a second recovery
controller when the old PID82481 is absent; that absence is now an intentional
handoff. Current startup ledger:0.57USD settled,1USD reserved,1USD uncertainty
within5USD; no provider allocation, checkpoint or image speedup is established
for the candidate. The owner stays with the03:23:29.323805UTC deadline.


## 2026-10-18 03:36 UTC — deadline ownership and post-sleep evidence

The startup candidate passed its original deadline with lost/create_unconfirmed
and rental_ended=false. Its Task-local pull credential was deleted at
03:28:49.931625UTC and independently confirmed absent. Its recovery observations
have a668.017909-second gap from03:17:38.059989 to03:28:46.077898; the retained
power log identifies639seconds of clamshell sleep. Launchd survived and resumed
reconciliation. This is not zero-cost-refusal evidence;1USD remains held, with
0.57USD settled. The startup Flow remains live, and its separate launchd job
continues to own recovery. The exact creation failure cause remains unknown.

Root inspected the RunPod guardian and read each admitted training pod's
ETUDE_PROGRAM payload without printing or retaining credentials. Both payloads
match the inspected in-pod guardian source, SHA256
8c52a53335c84aca7014c1781578492f0b1d8df6f90d372f1258bbdc389264ea. The deadlines
are07:00:20.558341UTC and07:00:27.747670UTC; each actual guardian cap is
8.037458333333333333333333333USD, below its8.75USD campaign reservation.
The guardian starts before dependency installation through a ready-marker
handshake, uses absolute and monotonic deadlines and a pod-scoped deletion key,
and does not need the laptop or storage to enforce its timer. The retained
guardian-verification.json proves provider payload binding, not continuing
guardian-process liveness or final deletion. Final retention and settlement
checks remain mandatory. An earlier attempt to find this source in the full
prebuilt-client archive was inapplicable; the guardian is a separate injected
program, and the provider payload check is the relevant evidence.

The evidence renderer now carries this scoped guardian receipt alongside the
existing settlement and prospective concurrency-amendment receipts. No scientific
input, training deadline, endpoint, evaluation allocation or score has changed.


## 2026-10-18 03:50 UTC — independent diagnostics publication

Fresh process evidence confirms the sole protocol017 controller PID8930 is live;
18 of22 trajectories are complete and the remaining cells continue. No offline
evaluation has been admitted, and full retention and settlement are still pending.
The preceding turn was a verified wait on that live controller, alongside a
read-only explanation of the corrected startup capacity gate.

The MTG-121 Flow ea781b00-002c-4ff1-a372-3237b1711f3f stopped at loop-or-next on
missing provider creation/termination/cost evidence. Its separate launchd owner
PID67117 remains live. No replacement Flow or recovery controller was started.
The creation-error retention fix is independently publishable: the exact source
and test diff was moved by Git into a separate checkout based on main, preserving
all startup work. Twelve standalone tests, Ruff and diff checks passed. PR310
is published and auto-merge requested, with required CI and merge still pending.
It claims no startup speedup and does not complete either Task.

Fresh MTG-122 reads still select64-stream GPU training, Mini evaluation and all
36,000 scientific games. Its earlier request to confirm the interim hold is
addressed by the existing handoff-scope clarification: MTG-135 did not impose a
new user approval requirement. The final handoff must name the supported64-stream
option and treat any256-stream preference as an alternative. Periodic Task
automation remains disabled and the CI watcher is not running; explicit current
controllers, launchd recovery and GitHub auto-merge are the actual active owners.


## 2026-10-18 04:12 UTC — all training finished; output delivery progressing

Both worker logs now report phase completed, all11 cells, exit_code0 and no
error: seed13591 at04:03:58.549624UTC and seed13592 at04:09:17.867647UTC.
These are22 completed training trajectories, not yet a retained/evaluated
comparison. The controller's last cells_completed metric remained10 while the
final learner_transitions reached524288; the worker completion log is stronger
evidence than that stale last progress counter. Sanitized read-only observations
are retained in learning-017/seed-*-observed-finalization.log. The same controller
handle20728/PID8930 remains responsible for terminal receipts, deletion, serialized
downloads and settlement; no second scheduling owner or evaluation was started.

Read-only S3 inventories confirm continuing output delivery. At04:10:18UTC,
seed13591 had53 attempt objects/1,031,651,199bytes and seed13592 had16/136,826,685.
At04:12:11UTC those became86/1,314,288,116 and31/475,020,339, with newest writes at
04:12:09 and04:12:03. These are progress observations, not manifest verification.
The optional multipart-list read was AccessDenied; no permission or account
change was attempted. Existing complete-object progress is sufficient to leave
the live delivery owner alone. The observations are in
learning-017/artifact-upload-observations.jsonl. Rentals remain open until
delivery and cleanup finish. Full retention/audit and18,304-game CPU evaluation
remain required before recommending a recipe.

PR310 remains armed at e05fcabf. All completed CI jobs passed, including debug
Rust; both platform integration suites remain running. The previous goal turn
made progress by publishing/arming that independent MLops fix. This turn also
obtained new remote worker-completion and output-progress evidence and performed
bounded waits on the same confirmed-live controller.

The startup Task's03:56:29UTC billing refresh now includes its known paid
baseline for02:00–03:00UTC, superseding the earlier incomplete-billing observation.
It still cannot establish the unknown candidate's provider identity, termination
or zero charge. The Task's working plan was corrected; its1USD reservation and
existing launchd recovery owner remain unchanged.


## 2026-10-18 04:19 UTC — diagnostics landed; both training rentals ended

GitHub reports PR310 merged at04:15:03UTC as
64d18556129fbf3e38c23fc2386d7987c602b424. Every required check passed, including
both platform integration suites and debug Rust. `lf pr reconcile` returned0
and removed only the isolated delivery checkout; it reported its missing file
journal path with a ledger fallback. The original startup checkout and its
dirty files remain present, and recovery PID67117 remains live. This closes the
diagnostic slice, not the image comparison or MTG-121.

The sole protocol017 controller now records both seed jobs done/succeeded with
rental_ended=true. It has begun serialized local downloads; at the first check,
seed13591 had12 local result files and seed13592 had none yet. PID8930 remains
live. finished.json and full settlement/verification are still pending, so no
offline evaluation is admitted yet. Neither GPU training nor a replacement
campaign was launched during these delivery checks.


## 2026-10-18 04:50 UTC — first retained seed passes preliminary checks

Seed13591's download reached all 205 expected files / 1,611,043,232 bytes by
04:42:35 UTC. The same controller then began seed13592's 205-file,
1,607,295,584-byte download; at 04:50:35 UTC it had retained 406,193,402 bytes.
PID8930 / tool handle20728 remains live, and both rentals remain ended.

While the second download continued, a read-only preliminary audit of the first
seed used the existing verify_learning.py functions and verify_grouped manifest
helper. All 11 trajectories passed exact recipe/completed-coordinate checks;
165 manifest-listed files passed their hashes, and all 44 raw checkpoints loaded
through the ordinary loader with finite parameters and expected model sizes.
The receipt is learning-017/seed-13591-preliminary-verification.json, observed
04:50:07 UTC. It binds the allocation and verifier source hashes and explicitly
limits its scope. No training file or live controller was modified.

This is not the full cohort audit, Adam-state health proof, playing-strength
evidence or evaluation admission. Run the complete verifier after the existing
controller finishes retention and settlement, then the declared resource audit
and full CPU evaluation. finished.json and evaluation/admission.json remain
absent. The preceding goal turn was a verified wait; this turn also produced
new first-seed verification evidence. No replacement GPU work was launched.

## 2026-10-18 16:01 UTC — local download resumes after low-power sleep

Fresh macOS power records show Low Power Sleep at 2026-10-17 21:58:03 PDT,
at 1% battery, followed by wake from hibernation at 2026-10-18 08:51:47 PDT
on AC power. The 39,224-second sleep interrupted local retention after both
remote workers and rentals had already ended. Existing idle-sleep protection
does not prevent low-power hibernation. The same controller PID8930 and tool
handle20728 survived; no competing downloader or GPU replacement was started.

At 16:01:23 UTC, seed13591 retained all205 result files /1,611,043,232 bytes;
seed13592 had90 result files /1,375,435,603 bytes, with a newest write less than
one second earlier. Both saved run statuses remain done/succeeded and
rental_ended=true. The read-only observations and exact power-log lines are in
learning-017/local-resume-observation-20261018.json. Full settlement and
finished.json remain pending until the same owner finishes downloading.

Task status is active at the findings node, with no associated Flow process.
This conversation continues the existing direct campaign. No evaluation has
been admitted and there is no new playing-strength result from the overnight
pause. Full cohort verification, resource accounting and the frozen18,304-game
CPU evaluation remain the next steps after retention succeeds.

## 2026-10-18 16:07 UTC — full retention verified; CPU evaluation admitted

The original controller/tool20728 exited0 after finishing both downloads.
finished.json reports completed, no blocked or pending groups, and provider
absence. The settlement records a9.450465872883797USD SDK estimate and a
9.602884628186111111111111111USD quoted-rate occupancy floor. Including the
declared per-rental margin, this cohort consumes10USD conservatively; the
campaign ledger now has33.50USD conservative rental accounting, no committed
rental caps and5USD ancillary reserve within the50USD authorization. These
are accounting estimates/reserves, not a claim of final provider invoicing.

verify_learning.py exited0: all22 trajectories,88 raw checkpoints and330
manifest-listed files passed exact source/recipe/seed/coordinate and numerical
checks. The Adam checks establish retained finite parameters/moments,
nonnegative second moments, the coarse movement bound, and no logged nonfinite
or rejected steps. They do not resolve the historical anomaly or measure skill.
learning_resources.py also exited0, verifying all22 resource rows and both
rentals against settlement. Logs and receipts remain in learning-017.

The frozen CPU evaluation was admitted once at16:07:04.681855UTC:44 commands,
176 checkpoint/anchor panels and18,304 games, using four CPU workers with the
declared deadlines and bounded idle-sleep protection. Its original tool is22094,
Python controller PID77023. At16:07:19UTC all four first commands were live;
there were40 pending commands and no completed panels yet. The runtime admission
check passed before launch. Do not restart or duplicate this evaluator; inspect
its existing status and retained outcomes. Full analysis, figure review and
the exact supported MTG-122 recommendation remain outstanding.

Fresh automation reads: periodic Task automation is disabled and the CI watcher
is not running. The direct evaluator above owns this work; no Loopflow Flow was
started. MTG-121's separate unresolved candidate and MTG-151's explicit hold are
unchanged by this evaluation admission.

## 2026-10-18 16:17 UTC — verified timings and startup-base direction

The verified resource rows show64-to256-stream inclusive-fit speed ratios of
2.068–2.807 across both seeds and all five models at524,288 transitions. The
384x2/64-stream fits took725.08/786.30 seconds, compared with758.84/798.64 for
64x2/64. These are whole-fit timings, with policy-dependent game trajectories,
not isolated inference benchmarks or evidence of playing strength. Largest-model
64-stream runs also had substantially fewer selected optimizer exposures
(58,467/56,973) than most other batched cells; preserve that distinction in the
interpretation instead of treating transition count as identical optimizer work.

The existing CPU evaluator remains live as PID77023/tool22094. At16:17:25UTC it
had four running initialization panels and40 pending commands, with no reported
error. No panel completion or new skill result is claimed. The preceding goal
turn completed retention/audits and admitted evaluation; this turn verified the
same live owner and examined authoritative resource evidence.

Jack asked about MTG-121 progress and reusing deep-learning base images. Fresh
startup records still show the candidate001 launch unresolved, the pursue Flow
stopped, and its launchd recovery PID67117 live. Its prepared image has local CPU
validation but no measured GPU startup speedup. The matching official PyTorch
2.10.0/CUDA12.8 runtime manifest is available; that investigation and exact
registry identity are now recorded in the startup checkout's
scratch/mtg-121-prebuilt-base-review.md and linked from its working plan. No new
GPU attempt, image build or scientific-runtime change was made. MTG-135's
evaluation and MTG-122's accepted64-stream/Mini-evaluation direction are intact.
