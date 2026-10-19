# Study006 execution status and operational amendments

Jack Heart authorized all three experiments, requested a middle size, and asked
that slow initial learning not be mistaken for a lower eventual skill ceiling.
Frozen scientific source: bb235731aca245fe80fbe7fff0eea7f66e3d1449. The protocol,
counts, checkpoints, evaluation panel and reserved deals remain unchanged.
Three sizes (138690,2463362,16816898 parameters) × three seeds,2048updates each.
Independent small fixed-reference seed13560; early-policy performance profiles;
fixed-batch learner replay at64/512streams. This is a bounded early learning study,
not an estimate of asymptotic ceilings or a hyperparameter-tuned size comparison.

Operational changes after admission, decided by the Task operator:
- Three-way provider admission refused its third machine. The first two were
  deleted and provider absence verified; third never existed. Conservatively
  charge$0.41. The accepted queue uses at most two separate GPUs.
- Original profiling965c8138d27a and reference36fbff1af122 both suffered
  machine_lost_before_start. Their commands never began, logs/outputs are empty,
  and deletion/absence are verified. No experiment data was repeated.
- Profiling replacement6c56e92d5eca and reference replacement6b9de076e0dc were
  cancelled before admission during the output-path repair below. Profiling replacement runtime cap is30minutes plus setup/rental bounds,
  $1; complete profile work remains unchanged. Incomplete work stays incomplete.
- Superseding the original blanket no-automatic-retry admission policy, only
  provider-confirmed machine_lost_before_start is eligible for replacement.
  At most two replacements per declared cell; every one requires deleted rental,
  provider absence, no started attempt, and an affordable remaining reservation.
  Failed/partial started experiments never cause automatic retries or new seeds.
  Each replacement has its own unique cohort/run and retains the failed receipt.
  This is infrastructure recovery, not score-dependent scientific allocation.
- New execution reservation is$19.50. Prior conservative charges$3.35 plus$2
  ancillary reserve yields$24.85 maximal reserved campaign exposure under$25.
  Unused job caps retire only after verified terminal deletion; they fund the
  reference replacement without raising the reservation. Authoritative ledger:
  .runs/gpu-bottleneck/budget.json. Every later replacement logs its bound.

.runs/size-frontier/launch.json retains all original and replacement handles.
The independent caffeinate supervisor drives Etude, saves early diagnostic
metadata, verifies complete outputs, confirms provider deletion, and then runs
two local CPU arena workers under a12hour bound. Partial science never gets a
complete size-comparison disposition. Source changes before evaluation block it.
Watcher PID/parent receipt and status are authoritative in watcher.json,
watcher-launch.json,status.json,watch.log. Do not start another controller or
cohort without fresh liveness/admission reads. Infrastructure replacement helper
and all its proof/request receipts are in the same directory.

Task MTG-135 was reopened after PR292 merged; keep it incomplete for this new
research. CI watcher active; periodic Task automation disabled. This conversation
itself schedules nothing. Preserved study005 edits remain uncommitted and intact.

Numerical runbd16b4a8e512 completed. Its authenticated early metadata shows:
64stream repeats agree within each mode; ordinary and deterministic hashes differ.
512streams ordinary repeats
form three learned-model and Adam groups; deterministic repeats form one.
The first difference between ordinary0and1 is the gradient witness immediately
before Adam step1; earlier observed inputs/targets/forward/loss/model/RNG agree.
This bounds the discrepancy to backward plus clipping, not one identified kernel.
No causal claim about the old293 parameter jump. Full binary rehash remains
pending; exact preliminary analysis is numerical-preliminary.json.

The launch wrapper's shared /tmp/frontier-study path collided when Etude reused
a GPU machine. Four subsequent commands failed at args.out.mkdir, before source,
CUDA or training admission. Retained stack traces prove no model trained in those
failed commands. Queued commands were cancelled; the already-running small13561
was preserved. The repaired command uses $ETUDE_OUTPUT_DIR/../frontier-study,
which is unique per attempt. No source/scientific count/seed changes. Cancelled
queued and pre-training path failures are retired with receipts; a corrected
10-job cohort supplies only remaining work. unique-output-admission.json records
all proofs and a conservative$18.899remaining+settled bound within$19.50reserved.

At 2026-10-16 18:10 UTC the corrected ten-job cohort was accepted. All 24
original/replacement handles remain in launch.json; no live scientific run was
cancelled. The repair sentinel is removed. Supervisor parent55235/worker55927
was observed alive and adopted the corrected queue. Its recovery loop now
retains never-admitted cancellations without crashing on missing attempt logs.
Worker implementation confirms ETUDE_OUTPUT_DIR is attempts/<attempt>/outputs.
Fresh state: small13561 running, corrected profile5aec98e566b2 setting_up,
remaining nine corrected jobs queued; numerical already succeeded. Task shared
state is active, completion_pending null, no live Loopflow Flow. Disposition:
acted (queue repaired and fresh status read); awaiting GPU experiment results.

Interpretation agreed in conversation: larger may reach competence later but
ultimately improve farther. Compare curves against experience and wall time,
retain the full declared horizon for all sizes, and do not treat an early tie
or loss as evidence of equal ultimate ceilings. No new horizon extension or
size-specific hyperparameter tuning was authorized by this discussion.

2026-10-16 18:45 UTC progress check: numerical and full three-size profiling
completed. Profile training reached128updates in135.8/197.3/235.7seconds for
small/middle/large; these are training timing observations, not strength results.
Original small13561 remained running with no stdout progress counters available.
All nine corrected learning commands failed CUDA admission on the same machine
size-frontier-006-0169511dddf4. Verified authenticated metadata has no training
files and status rejects missing RTX5090; full stack traces and output hashes
are retained in cuda-failures/. Provider deletion/absence confirmed. These are
untrained cells, not bad-score restarts. Proof: cuda-recovery-proof.json.

Remaining nine cells now use serial admission: one replacement, then only after
its full success admit the next. Any unsuccessful replacement blocks further
admission. Source, horizon, seeds and recipes remain unchanged. This prevents
another broken GPU from consuming the whole queue. Reconciled settled rentals
plus active-run reservation and all nine future caps conservatively total
$16.648 within the existing$19.50 reservation (whole campaign ceiling$25).
No new spend allowance. Independent supervisor parent47520 owns the pending
plan in cuda-recovery-plan.json, with replacement IDs appended to launch.json.
Task remains active/incomplete; no live Loopflow Flow. No intelligence/size
conclusion is available yet. Current disposition: acted, infrastructure recovery
and fresh Task/Sessions reads; awaiting experiment execution and retained data.

18:46 UTC: first serial replacement admitted, reference-cuda-recovery3b882c4c1dd1; eight further cells remain held in the local plan until its success. Fresh status is saved in latest-live-status.txt. Periodic Task automation disabled; CI watcher active.

2026-10-17 04:21 UTC (Oct16 21:21 Pacific) failure explanation, fresh evidence: original small13561
and reference-cuda-recovery both ended out_of_time at their3600second command
limits, exit-15, with no output artifacts or progress clocks. Therefore neither
completion percentage nor slow-training-versus-hang can be established. The
one-hour outer deadline included setup and cut off before the wrapper retained
partial evidence; this is an observability/timeout design defect. Eight serial
replacements remain unlaunched; serial admission is blocked, not still running.
All seven subsequent rentals have deletion/absence receipts. No GPU job relaunched.
CUDA unknown error is only the observed symptom: driver/container/provider/hardware
root cause has not been established. Nine failures were on one machine and before
training, not nine physical GPU failures or evidence against model size.
The local full-artifact downloader also failed because it shared the Etude SQLite
connection across threads. Fixed downloads to use the owning thread, and restarted
only recovery of the two successful diagnostic jobs. Immutable remote bytes are
reused; no measurement is repeated. Incomplete training prevents arena evaluation.
Task remains active; disposition acted (download repair and fresh shared reads).

5whys analysis completed at scratch/size-frontier-006-incident.md. Two local one-second timeout probes reproduced late-only retention failure; no GPU/network. Timeout hierarchy, machine reuse amplification, and missing lifecycle validation are evidenced; CUDA hardware/driver cause remains unknown. Findings ready for review; no scientific relaunch.

Jack Heart explicitly requested carrying the deadline/progress/retention/CUDA
repairs through and launching the missing runs. Sourcea23d74acb1ddf4469070eaeb7a4609a96ed1bbf5
adds a stdlib worker deadline adapter, isolated CUDA admission diagnostics, and
TrainingRun persistence-to-Etude progress/checkpoint publication. Five focused
tests pass, including forced timeout retaining files, same-host attempt isolation,
CUDA diagnostic failure, and exact same-seed learning with loadable checkpoints.
CPU CLI smoke passed. Source upload/readback is underway. Ten full learning cells
remain (including original small and reference timeouts); preserve all old runs.
New prospective cap16.35 plus settled6.55 plus ancillary2 equals24.90 under25.
First release is a separately labelled two-update seed13599canary; admit at most
two scientific jobs only after fetch/load verification, and stop new admissions
on any failure. No score-based allocation, changed model/seed/horizon or newbudget.

Frozen repair source uploaded and fully readback-verified: a23d74ac,
SHA25613b4adb3cc264903f2961e66930d93e196769ec21fea31da1fe50aed7ff58bf8.
Additional two-second test through installed Etude outer timeout retained the
checkpoint, training log and interruption status (incident-probes/repaired-outer).
Initial provider quote reported unavailable capacity; no rental was created.
Bounded capacity controller subsequently observed eligible5090stock at$1.19/hour
and is publishing the canary source. Controller parent49388, worker receipt in
.runs/size-frontier-repaired/controller.json. New cap16.35 is reserved; all prior
attempts retained. No scientific run is yet claimed admitted or successful.

2026-10-17 fresh recovery: complete numerical (49 files) and profiling (72 files)
archives are now hash-verified. Their bounded scientific findings stand; zero
full learning curves. Canary 2d3cccee6ca7 passed CUDA initialization and backward
on RTX5090/driver570.153.02 but failed before training: gpu_job resolved the
virtualenv executable symlink, selecting base Python without pydantic. Setup
log proves pydantic installed; this is an interpreter-selection bug, not another
CUDA failure. Retained logs made the failure diagnosable; all rentals deleted.
Repair 4d65d477 preserves the venv executable path. Six focused tests pass,
including a real symlinked virtualenv regression. Ruff passes.

Replacement operational allocation: fresh settled conservative charges7.05,
ancillary reserve2, one canary0.35 and ten scientific caps1.56 each =25 total.
This supersedes the prior1.60 per-cell/16.35 campaign reservation, which was
released after canary failure/verified cleanup. Scientific seeds, horizon,
training deadline and incomplete-outcome rules remain unchanged; dollar caps
can stop a cell early. No partial survivor comparison. New controller/results
will be under .runs/size-frontier-repaired-v2, preserving old failure receipts.
Admission requires verified source upload and canary fetch/load as before.

Replacement canary b52e0fd1927d admitted on source4d65d477 after complete
226191360-byte source readback SHA256
9f9b9fc82bb2701467d1a4166fccd82d4f86f7f4a1f623b68a302c1a595a181a.
Controller PID30991/parent30864 observed alive. Provider machine
size-frontier-006-repaired-v2-96c0039f5702 is setting_up as of10:50UTC,
no measured training progress yet. All ten scientific cells remain pending.
Fresh lf status: MTG-135 active/incomplete, no Task Flow; original research
process lookup unavailable (no Flow found), no competing Flow launched.
Task automation disabled. CI watcher is now NOT running (last poll stale),
although MTG-135 repair is enabled; earlier active-watcher statements are stale.

Full numerical replay verification now additionally loads all12after.pt payloads
and reproduces canonical model/optimizer digests against recorded witnesses.
Receipt: .runs/size-frontier/results/numerical/replay-tensor-verification.json.
At512streams ordinary has3model/Adam groups; deterministic1; at64eachmode1.
The first observed ordinary difference remains gradients after backward/clipping
beforeAdam; exactkernel and old293jump remain unresolved. This strengthens
evidence retention, not a new experiment or proof of universal determinism.

2026-10-17 latest direction supersedes separate queued study006 replacement plan:
Jack Heart explicitly assigns this conversation the batching sweep, then says
"No, no, combine. do a grid sweep on batching and size", raises the inclusive
campaign budget to$50, and authorizes continuing to land necessary MLOps PRs.
Prepare a joint small/middle/large ×4/64/128/256/512-stream study, anchoring the
small model to MTG-122 corrected baseline (width64/depth2/masked mean,64transitions
per stream, cross-balanced). Do not substitute old study006 value-token recipe.
Existing MTG-122 baseline executions are untouched. Prior size-only jobs remain
unlaunched; retain all failed attempts and completed diagnostics. New grid
protocol must declare schedule mapping and fixed transition budgets, measure
full-loop throughput and health, and separate feasibility from strength claims.
Last canary b52e0fd1927d exhausted3pre-start attempts at the SDK five-minute
startup cutoff, with no command start/output. All3machines deletion-confirmed.
Conservative settledcampaign7.96, ancillaryreserve2, no active reservation.
Need startup evidence/repaired cause before another identical retry.

Grid007 implemented/committed f057cfebb3e9be07addfff9ae7e04c0106b85ef3.
Three capacities(masked-mean small64×2, middle192×4, large384×8), five stream
counts4/64/128/256/512,64transitions perstream; full15cell feasibility seed13571
target131072transitions each, originalfourstreamLR/tau with endpoint mapping for
othercounts. Nine focusedtests pass plus two-update CPU harness checkpointload.
Source226211840bytes, fullreadbackSHA256
8bdc81fa3f3c05a181079cc6ffccda2f1de0122ca1942cb55a1c79e54d30075f.
Canary5c29eca30f2c succeeded:2updates/1024transitions,live clocks, complete
retained artifacts independently hashverified, checkpointloaded, providerabsence
confirmed. This resolves the interpreter failure; earlier CUDA/startup causes
remain unknown. Provider logs now retained before deletion (oldpods logs404).
Feasibility controller13916(parent13795) owns .runs/grid-007/feasibility.
New$15cap admitted under$50inclusive. Firstcellsmall-s4 is8d626f299347.
Priorcanary fullestimatedrental0.058155USD; conservativeledgercharge0.26.

User separately requests landingPR295and296 whenready. PR295 passed allchecks
and merged571f57dee590a820d57d4c589f4b3d2ae3ed6bec. PR296 lifecycle repair
head9c0beeb12816dead28f1999580543ece80813047 isarmed; integrationchecks pending.
31local lifecycle/recovery/artifacttests passed aftersync. Gridcommit is local
and not added to already-armedPR296; do not accidentallypushit over approvedhead.
All23unrelatedmodified/untrackedfiles restored/hashverified after deliverystash.
Do not completeMTG135 from infrastructuremerge: jointgrid/recommendationremain.

2026-10-17 11:25UTC: both grid small-s4/middle-s4 workers started; small
reports measured updates and both initial policy outputs retained. Fixed local
controller source-verifier accidental second argument before result collection,
restarted only local controller from unchanged launch handles (parent70525);
remote jobs not cancelled/retried. Source/grid/allocations unchanged.
Fresh Loopflow reads now fail: /Users/jack/src/manabot repository lacks Git
locator and installedlf0.14.3 cannot read database migration0.14.4.001_release.
No repository rebinding, upgrade, account change, or competing Flow attempted.
GitHub/Etude authoritative reads remain available. Task status/completion cannot
currently be reconciled throughlf. PR295 verified merged; PR296 still autoarmed
at9c0beeb, integration Ubuntu/macOS running.

11:30UTC: local grid controller counters resumed after its predecessor pool
lease expired at11:29:30. Direct read-only remote observations proved workers
continued during the five-minute fence (small98/middle93 updates at11:29);
normal controller now sees118/111. This was expected abandoned-owner lease
behavior after local SIGTERM, not training/courier failure. No lock bypass or
second GPU attempt. Both initial artifacts retained. Future local controller
restart should use graceful interrupt so its pool-lock finally can release.

11:34UTC: GitHub confirms PR296 merged08921254f9bd32593223d4758e0178ee45c9afd2
at11:33:44Z, all9checks passed; PR295 previously merged571f57d. Both requested
landings delivered. lf pr reconcile still refuses schema0.14.4.001_release
with installedlf0.14.3; freshTaskstatus also fails missingrootGitlocator. Do not
claim sharedTaskcompleted/reconciled, and grid outcome remains unfinished.
Gridcontroller70623 observed live with small/middle fourstream180/512updates,
no failures. Independently downloaded retained monitoring checkpoint0 (planned
update128) from each run, matched remoteSHA256/bytes, loadedpolicy and checked
allparameters finite. Proofs in .runs/grid-007/feasibility/early-checkpoints/.
Small188610parameters; middle2463170. LiveEtudeSDKfetch requiresfinishedattempt,
so used read-only checkpointobject and currentremoteoutputreceipt for verification.
These are health checks, not strength or deterministicrepeatability conclusions.

2026-10-17 11:38UTC direction: Jack Heart asks to add64×4 and384×2.
Implemented names small-deep and large-shallow across allfive stream counts;
25feasibility cells total. Same131072transitions,seed13571,schedules and caps.
Added10cells to the existing solecontroller queue after original pending cells;
original15 source f057cfeb remains pinned, new10 source9a1faddb96ba518d9b6e63eed545f157d9771c6c.
Expansion only adds capacity choices, CLIvalidation, tests and protocol amendment.
10focusedtests pass; two-update CPU smoke forboth completes with finite/loadable
checkpoints (288578 and6169730parameters). CPUtimings are not GPU evidence.
lf commit succeeded with ledger warning; Taskstatus/reconcile remain unavailable.
Newsource226222080bytes fullyreadbackverifiedSHA256
d4b458f9b7c0d56337761603dea51b2d0e8f8032b751c57cf2bac93cc4fdf57f.
Reservedfeasibilitycap25, priorconservative8.22, ancillary2 =35.22 exposure
underexisting50total;14.78uncommitted beforeunusedcapsrelease. Ledgerupdated.
Stopped only localcontroller withSIGINT, verifiedpoollockreleased, replaced
controller and restartedfromsameGPUrunIDs; neitherremotejob cancelled/retried.
Parent94827; expansionreceipt .runs/grid-007/feasibility/expansion-receipt.json.
Preparedsourcehelper acceptsperrevisionreceipts and controller verifieseachcell
against its ownsource. CurrentGPUruns remain originalsmall/middle fourstreams.

2026-10-17 activegoal continuation: previous turn madeprogress byexpanding
25cellqueue; freshremoteobservations confirmedbothworkersaliveandadvancing.
GoalremainsenoughexperimentsforclearMTG122trainingrecommendation,notmerely
throughput orinfra delivery. Addedfutureexplicittransitioncheckpoint support,
keepingcurrentpinnedGPUgridunchanged.16focusedtests plusactualCPUexport at
256transitions/1update with512transitionendpointpassed. Syncedmainincluding295;
resolvedtwoadd/addconflictsbyretainingonlynewrunneroptions/sharedcheckpointAPI
afterconfirmingpublishedincomingcodeidenticalotherwise.19focusedtests passed
includingchild-exitmemoryguard; addedallthreefiles to explicitCIselection.
PR297 published/autoarmedhead df04e348c1308d1ce101c20b53596889f687ff0f.
Preserved/restoredall23 unrelatedfileswithSHA256verification.
lfexternallyupdatedto0.14.4andTaskreadsworkagain; prreconcileverified296merge
andretainedTaskactive/incomplete. NoTaskcompletionfrominfrastructuredelivery.

11:56UTC firsttwofullGPUcellsverified: small-s4 andmiddle-s4 both512updates/
131072transitions completed; allmanifestfileshashchecked and finalpoliciesloaded.
Small1777.975s/73.7198transitions/s; middle1756.371s/74.6266transitions/s.
Collection1056.879vs1385.339s, learning695.973vs348.725s. Selectedoptimizer
samples30784vs10083; Adamsteps22001vs8842. Thus similarend-to-end throughput
doesnotshowequalmodelcomputecost: middlecollectsslowerbutfiltersmuchmorework.
No playing-strengthclaim. Analysis .runs/grid-007/feasibility/first-complete-results.json.
BothworkerdeletionsrecordedbyEtude; large-s4 admitted4253228d7531, nextmiddle-s64
admissioninprogressbysolecontroller. Do not duplicate pendingcohorts.

11:57UTC solecontroller94999admittednextcells large-s4/4253228d7531 and
middle-s64/d001fc45b738;21pending. FirsttwocompletedhardwarebothEPYC7543/
RTX5090,sameCPUquota, samefloat32TF32off; oneGPUperjob. Fullmetricsin
first-complete-results.json retainoptimizerworkconfound. PR297CIactive,
automergeenabled; do notdeclaregoalcompletebeforebatchcomparison/learning
validationandexplicitrecommendationto122. Thisgoalturnmadeprogress: shipped
newPRhandoff, addedvalidatedmatched-experienceexports, recoveredtwofullresults
andidentifieddifferentselectedlearningworkbehindnearlyequalthroughput.

2026-10-17 compressed-source recovery and grid resumption:
The source-free probe 2c94d4c8a545 succeeded, while the full uncompressed-source
phase probe df40f846452d stopped before job claim. Both retained logs and have
provider-confirmed deletion. The phase probe reached client_install_ready, so
its stall followed client installation; it is not evidence of a CUDA failure.
Compressed-source canary 5f7715ecdd57 then succeeded: two updates, CUDA admission,
12 manifest entries independently verified, loaded finite final policy, exact
4d65d477 source, and provider absence. Estimated rental cost 0.168821413 USD;
conservative ledger charge 0.37 USD. This supports compressed-source admission,
not a definitive startup root cause or resolution of prior CUDA initialization.

Prepared gzip objects for f057cfeb and 9a1faddb preserve every original tar byte
on decompression. Complete ordered range readback verified both compressed
objects. The remaining grid uses these objects through the tracked
experiments.runners.prepared_source adapter and instrumented bootstrap. Already
finished cells keep their original transport history; scientific source and
recipe are unchanged for each cell. Do not attribute setup-cost changes solely
to batching.

The 21 pending cells plus large-s4-replacement are resumed by the sole bounded
controller, parent 63963 / Python 63976. Failed original large-s4/4253228d7531
is retained and marked retired, not erased or relabeled successful. Earlier
controller receipts are archived at .runs/grid-007/feasibility/before-compressed-resume.
Remaining caps are 22 USD; settled conservative charges 11.57 plus ancillary 2
bound campaign exposure at 35.57 of the authorized 50 USD. Two rentals maximum,
one attempt per new cohort, no automatic failed-cell retries. Fresh ledger and
SDK/provider state remain authoritative.

PR297 merged fb0cb3b05abfee3803e3df43cc120b285c03b7ed. PR298 publishes bootstrap
phases, verified source transport and current-self EMA clarification, head
20e79a3c015ee8f0f0a459d9693e75a20ba698e8, auto-merge enabled. 25 focused tests
passed. Publication initially auto-staged unrelated files; that generated commit
was removed with an exact-head lease before a PR was created, and all 23 files
were restored with SHA256 checks. PR298 contains only the intended six files.
Keep unrelated work stashed during future lf pr publish/land: this installed
command auto-stages it. Task remains active; infrastructure delivery is not the
research outcome. Task automation is disabled and CI watcher is not running.

Three full grid cells are verified: small-s4, middle-s4, middle-s64. Middle4→64
observed throughput 74.6266→594.3977 transitions/s (7.965x); CPU host changed
EPYC7543→EPYC9354, so this is not a clean batching-only causal multiplier.
Selected optimizer exposures also differ. Before a concrete MTG122 training
recommendation, finish the feasible size/batching screen, run a same-host paired
batch comparison, and validate learning at matched transition milestones with
independent seeds and paired greedy/random evaluation. No strength results yet.

2026-10-17 projected-source recovery:
Full-source gzip did not reliably fix startup: large-s4-replacement/5497aeb9ebc9
again stopped before job claim, after client_install_ready. Large-s64/015c0dd4c829
succeeded alongside it; all artifacts are verified and the final policy loads.
Its131072 transitions took391.4137s (334.868/s);16,816,514 parameters,2.532GB
peak CUDA allocation,9773 selected exposures and1997 optimizer steps. CPU is
EPYC7543, unlike middle64's EPYC9354. Four-complete-results.json retains this
confound and every completed cell. Both failed startup attempts stay historical.

Source projection omits only experiments/data/, preserving every other member's
bytes, modes and links. Full source archives and retained/omitted manifests remain
available. Projected canary ddee82337fbf succeeded with all12 manifest entries,
loaded finite policy, CUDA admission, exact retained source hashes, and confirmed
provider deletion. It retained1628 files and omitted536 historical data files.
Its estimated rental was0.0523554 USD; conservative charge0.25. This packaging
change is explicit, not claimed lossless full-source compression.

Verified projected source objects/configurations are prepared for original
f057cfeb and expansion9a1faddb. Sole grid controller parent38972/Python39072
resumed21 remaining cells after all previous rentals were confirmed gone.
Third large4 cohort is large-s4-projected/bad037591cb2; small-s64/55cf95c904d3
also admitted. No competing controller or automatic failed-cell retry. All new
cells identify history-excluded-gzip transport, preserving their frozen runtime
source and recipe. Controller receipts before this resume are archived in
.runs/grid-007/feasibility/before-projected-resume.

Billing reconciliation reduced some historical SDK cost values below their
quoted-rate occupied time. The ledger now retains the larger quoted-rate bound
plus existing per-rental margin; raw billed/estimated values remain separate.
Conservative settled charges12.54 plus21 remaining caps plus2 ancillary bound
exposure at35.54 USD of50. Never release spending room based on partial billing.

Local commit6d1008e7 adds explicit source projection and exact-transition retained
panel selection, with30 focused tests passing. Further GPU-worker limit wiring
passes11 worker tests. PR298 is still auto-armed on its earlier20e79a3c head;
do not push newer local commits into that armed PR. Publish the next serial PR
after verified merge, stashing unrelated files before publish/land. The draft
validation planning note names remaining scientific evidence and baseline source
compatibility limits; it does not authorize a new allocation or claim learning.

2026-10-17 fifth completed cell and PR299:
PR298 merged with all checks green. Synced its merge, resolving only the known
prepared-source/CI additions;35 focused tests passed on the resulting tree.
PR299 is published and auto-armed at a090897574501842cb78eceef30b1afdd5b3accd:
source projection, exact-transition evaluation selection, and explicit bounded
training horizons. All24 unrelated files were stashed during publication and
restored with SHA256 verification. The initial PR opening over-attributed the
startup stall to transfer; its body was corrected to state only the observed
pre-claim stall and archive size. Root cause remains unproven.

Small-s64/55cf95c904d3 succeeded. All manifest files are now independently
verified and the final checkpoint loaded.131072 transitions took296.876s,
441.504/s, with449450496 peak CUDA allocated bytes. Observed4→64 ratio5.98895,
but CPU changed from EPYC7543 to EPYC-Genoa; this is not a batching-only causal
multiplier. Evidence five-complete-results.json preserves host/transport and
optimizer-work differences. The sole live controller39072 admitted large-s128/
8643e483dd58 while large-s4-projected/bad037591cb2 continues. Do not duplicate
these runs. The latter is much slower and may be censored by its declared
1800-second training limit; retain that negative result if it occurs, not a
silent shorter-budget success or an automatic retry.

The future learning source differs from frozen f057cfeb in core runtime only by
the child-exit memory-guard fix, while the added runners/control options support
new explicit allocations. MTG137's older baseline source has broader model,
collector and engine differences, so it cannot silently replace a matched
four-stream learning control. Validation budgets and protocols remain to be
frozen from the completed feasibility and evaluation-cost measurements.
