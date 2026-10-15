# Single-GPU inference serving

2026-10-15: Jack Heart redirects optimization from managym simulation to clearing
the inference-service bottleneck for single-GPU training. Existing total paid
budget remains $50; .runs/gpu-validation/budget.json is the live ledger. PR292
contains the actual worker benchmark and preserved failures, open/manual merge.

## Evidence and boundary

RTX5090 run012: fixed32768 learner rows,64actors,1/2/4persistent processes,
three repetitions: median28.2593/23.1009/23.5961s,1.223x/1.198x. Service occupies
97–99% of multiworker wall. This includes host preparation, H2D, forward, D2H and
output delivery; it is NOT GPU utilization. Ordinary4096-stream training:
98304rows/43.1665s,2277.3rows/s,~6.89Mrows/rental-dollar at$1.19/h.

The shared process is an experimental worker harness. Production TrainingRun
still uses SeatRoutedCollector directly on CUDA. An isolated service win is not
acceptance; require ordinary collection and full training evidence. Current
Agent.forward_distribution recomputes SemanticCardEncoder.definitions and performs
GPU-dependent input checks. No evidence yet divides current w6 service time among
catalog, validation, compute and copies. Shared-vs-independent GPU ownership,
coalescing requests, and eliminating repeated fixed-weight work remain distinct
hypotheses. Keep strict viewer/mask/sampling/rollout witnesses.

Earlier frozen w4 evidence lives in git c5cef8b7bfb5b326d0a536bb8a079d23da62ef63:
scratch/staging-validation-results.txt (isolated1.23–1.42x buffer-reuse gains),
scratch/collector-staging-results.txt (ordinary integration failed performance
gates, including two4096-stream timeouts), and scratch/cpu-findings.md (initial
CPU catalog diagnosis). Do not repeat these as unmeasured ideas or transfer their
coefficients to w6. Full historical scratch remains preserved in that commit.

## Next discriminating work

Finish H100013 and corrected CPU-only5090-014; both predeclared, frozen source
a8574ddb056f6af54ef43b7e2d9e00303d8b5ba8. Original012 cross-platform CPU replay
failed after input hashes passed. New named mismatch and host-local capture keep
portability separate from exact within-host worker verification.

Profile complete serving phases on the economical selected host, separately from
throughput timing. Compare independent per-worker models on one GPU against the
shared single-request service, retaining original request shapes and exact output
checks. This directly tests whether the central process is the limit without
conflating the result with changed batch arithmetic. A paired no-grad catalog
reuse intervention can address repeated inference work in both the service and
ordinary training. Freeze its protocol before renting; no new rental yet reserved.

Acceptance: exact discrete trajectory, masks, journal/seeds; explicit numerical
and gradient checks for any changed arithmetic; memory and whole-attempt limits;
paired fixed-work collection and complete-training gain with all failed samples
retained. No simulation changes or claim of improved playing strength.

Delete—do not maintain: no production predecessor selected for removal yet.
Research baseline must remain reproducible until a measured replacement passes.

Update19:12Z: H100013 completed and podogr8gh4wg2fumu deleted;199outputs/198manifest
files verified. CPU median1/2/4=5.7372/3.0373/1.5005s (3.8235x); CUDA45.7451/23.8276/
20.5705s (2.2238x). Four-worker service20.2s, workerCPU20–22summed seconds. Ordinary
4096-stream2546rows/s,~2.30Mrows/$;5090~3x cheaper per row. H100 CPU isXeon8480+,
23.8core quota. Only cross-platform batchhash differs, all other finalchecks pass.
014 CPU-only5090 launchedbe06db3fd5b3 froma857 withdirtylocalcandidate excluded by
verifiedgitarchive; monitor28463. Do not change preparedsource until014terminal.
Candidate implemented but not GPU-validated: CudaPolicy shared forward owner,
worker-local models ononeGPU, exactsamebatch audit, per-worker inference counters.
New prospective ownership protocol and separateCPU/CUDAphaseprofiling. Three local
worker tests passed8.44s, Ruffpassed. No production change, no performanceclaim.
Oldw4 catalogreuse at4096:22.4806→22.2904s (~0.85%); deprioritize as a whole-training
solution unless new attribution contradicts this. Newgoal remains active.
014terminalsuccess/providerdeletedtypnr6jb8l996e. WholeCPUmode264.280s versusH100
60.996s; hostload232→249 (not itself a causal diagnosis). Recoveringoutputs.
New ownershiptests repeatedaftercontrollerCPUcounterchange:3passed8.13s. NoGPU
candidate yet tested. Upcoming015needs publishedsource,verifiedarchive,quote and
014terminalledger. Allocation015reserved$4/max$3/30min; sourcependingpublication.
014verified47outputs/46manifestfiles,CPUmedians20.7198/13.6759/9.4356s;2.1959x.
HostXeonGold6530,13.6corequota,highload232–249. All576actorsmatch,substantial
between-repeatvariation. CannotpairCPU014withGPU012asonehost. Wholeestimate$.29989.
Completedbaseline evidence nowretainedexperiments/data/worker-scaling/gpu012–014.
015protocolkeeps180s trialcap andsame-hostpairedownership. AvoidanyuniversalGPU
recommendationfromonegoodrental. Newanalysishelper.runs/gpu-validation/analyze-ownership.py
runsAFTERanalyze-worker.py015andrecovery;checksall1152actors,cross-modehashesandgates.

Ownership candidate published PR292 head91b42f4677ee64a2ddfeb4f3861cdd7619cdfff6.
Code: worker_inference.CudaPolicy sharedforward+timing;worker_scaling --device
cuda-local, controller/workerCPU andperprocessGPUpeaks; --profile; new
worker_service_profile.py; worker_experiment --serving-ownership ordersABBAAB,
three curves eachmode, each180s/24GiB, separate45sprofile, ordinarycontrolsunchanged.
All014andpriorpods deleted. 015reserved$4/$3job/30min, exactheadabove, poolworker-5090.
Sourceupload/fullreadback live tool65496; quote/priorabsencecheck live96067. DO NOT
launch015untiluploadverified. Command samecachedruntime launch-worker.py015;
helperchecks014deleted,sourceledgerandreceiptmatch,noexistingcohort. Budgetexisting$50.
PublishedPRheadreceipt.runs/gpu-validation/pr-ownership.json.
Profile trace excludesIPC/obs preparation and usesexplicitphaseCUDA sync; don't
substituteits timings forunprofiled throughput. Ownershipgates>=15%fourworkergain,
<=5%oneworkerregression, allcross-modecounts/hashesmatch; no productiontrainingclaim.
Recovery+analyze-worker.py015thenanalyze-ownership.py;retain-worker.py015packages
originalJSON/logs+manifests/lifecycle; its receiptsarchive mayincludeCUDAtrace.
Latestfocusedworker tests3pass8.13s afterCPUcounter addition. Ruff/syntaxchecks pass.
015source187084800bytesSHA491cf2fa09fdc0ab7032331e8da954e89e962b46d60a9ff1b07b7563088151f9,
fullreadbackverified107.7106s. Quote1.19/h/all14priorpodsabsent19:28:03Z. Launching
oncewithlaunch-worker.py015; do notresubmitifmonitor fails. NewheadCI Python,Play,
Protocolpass; Rust/Visualpending. Currentallrentalestimates5.1277,conservative7.00,
ancillaryreserve5,pending0154,unallocated34ofexisting50.
Newsourceanalysis: Exchange explicitly caps64rows/actor. With64actors there are
~4288policyrequests per fixed32768rows. Ordinarytraining uses much larger request
batches; currentownershiptest intentionally holds these original shapes constant.
A later actor-size sweep(64x64,16x256,4x1024 residentstreams) could discriminate
per-request overhead frommodelthroughput, but cross-shape cloned trajectories are
not identical. Do not claim this small-request serving limit applies to ordinary
4096-stream collection or a maximumsingle-GPU rate. Exactwithin-shape parity would
remainrequired; cross-shape workloadcounts/seeds needseparateprospective treatment.

CRITICAL015ADMISSIONFAILURE: launchmonitor47450terminated1 BEFORE anyrun admitted:
LauncherError cannotlaunchworker-5090:create replyunknown. No launch-015.json/no
run/cohort recorded. DO NOT RESUBMIT. Exactsession1e46a8e9a3d1, machine
worker-5090-1e46a8e9a3d1, requested19:30:43.312338Z, provideridNone. Fullprovider
inventorylookup ofactualnameetude-1e46a8e9a3d1 foundNone (wrongnamesfirstchecked,
thenactualnamefrominstalledprovider source). s.tick doeslookup-onlyforclaimed
unknowncreate; s.terminatepublicAPI cannotmarkanabsentunconfirmedcreateended.
NoEtudesource/authchanges. All14previouspodsremainconfirmeddeleted.
Watch-admission-015.py nowownscontinuouspublicterminate+namedlookup until20:16:43Z
(max2700srental+60s). It nevercreates; iflatepodappearsitendsownedunusedrental.
Full$4reservation retained ledgerstatusadmission-unknown-no-run-provider-currently-absent.
Currentgoalstillactive andnotachieved. Candidate015notGPU-tested. Blockerisprovider
admission, notcode/testfailure; oneconsecutiveblockedgoalturn, do notmarkblocked
until3consecutivegoalturnswithsameimpasse. Do notsilentlyrerent/expandlimits.

Executiveadmissiondecision (supersedesstoppingat015,not its no-resubmit rule):
ProviderfullinventorypositivelyconfirmsNOPOD; no015run/cohortwasadmitted. Usergoal
andexisting$50authorizeactualtest. Keep015'sfull$4reservedandcleanupguard89254
until20:16:43Z, but allowONEreplacementadmission016innewisolatedpoolworker-ownership.
No measurementhasbeenrerun; code/protocol/180scapsunchanged. Newreserve$4,total
pending8;conservative7+ancillary5+pending8 leaves30unallocated. Exposed thisdecision
inconversation; no permissionrequest, no auth/Etudesource/store repair.
Preflight016helper checksactualoldprovidernameabsent andno015run, quotes<=1.25,
createspoolbycopyofworker-5090 withnewnameonly. Livepreflight40842. Source91b42f
stillpublishedandfullarchiveverified. launch-worker.py016 checksabsenceagain.
Ifthereplacementadmissionalsofails,do notsilentlyrepeat. Inspector/analyzersnow
support016; runanalyze-worker.py016thenanalyze-ownership.py016afterverifiedrecovery.
Inspectorretainsunknown015reservationwhenrecomputingpendingbudget; skipsbilling
unconfirmedprovider-lessrentalsratherthanassertingtheywerecharged. Candidate remains
unvalidated; no productiontraininggainclaim. Goalactive,notcomplete.
016ADMITTED run45a304b45d64; launchmonitor69201live19:40Z. Sourceunchanged91b42f.
Monitor015emptyreservation89254stilllive; itloggedoneProviderUnavailable readand
continues; no resubmission. Inspect016startedafteradmission torecordactualpod/rate.
Neverrerunlaunch016ifmonitorreadfails; resume-worker.py016sameid instead. All
remainingpaidworkownedbytheseexplicitreservations; no thirdadmissionif016fails.

Openimplementationbranches,notselectedorimplemented: a largerper-actor VecEnv
(4x1024 ratherthan64x64) reducesrequestcount butchangesclonedworkloaddistribution;
truecross-actorbatching requires concurrentoutstandingrequests, notmerelyshared
ownership. Synchronousone-requestperworker currentlylimitsbatchopportunities.
Ifusingthreads toparkactors atGPUwaits, explicitlyboundCPUexecutionperprocess:
PyO3 releasesGIL, so otherwise1"worker"mayalreadyusemanyCPUcores. Alsoaudit current
w6collector._self_rng versus PreparedReplay's globalRNG assumption beforethreading.
ProductionTrainingRun hasnoActorPool/inferenceservice. Integratingone requires
policy-version barriers, consistentglobalbatch ordering, stats and composite
collector snapshots/recovery—not just monkeypatchingcollect and ignoringresume.
Potentialfastpaths(inputadmission/GRUreuse/batching) mustpreservegradient-enabled
learning andwhole-trainingacceptance. Wait for016ownership+trace evidence first.

## Integration boundary checked during result recovery

`manabot/training/execution.py:1288` calls the ordinary collector with the selected
raw or EMA behavior agent, then runs learning at the update boundary. Recovery
(`manabot/training/recovery.py:106`) saves the collector together with learner,
EMA, optimizer and RNG state. A replacement service must bind all requests in a
collection to that behavior-policy version and preserve these recovery facts.
The experimental pool is not called from this path.

`manabot/sim/net_opponent.py:397` finalizes learner transitions per stream and
uses its own sampling generator. Ordinary CUDA collection samples on its device;
the experimental service returns CPU logits/values and samples on CPU. Thus an
ownership result cannot establish exact equivalence to the ordinary CUDA training
trajectory, even when all benchmark modes match each other. Preserve this boundary
when designing the eventual training comparison.

The serving profile labels only real initial observations and synchronizes each
phase. Interpret GPU event unions as time with traced device work, not SM occupancy;
catalog host spans can include launch/synchronization costs. Neither this trace nor
the four-worker service interval alone identifies one cause for every idle gap.

19:59Z: all six ownership trials and the separate profile completed. Outputs are
uploading; provisional downloaded timing files are not yet authoritative-verified.
No new allocation or production change. Guard015 continues checking the unknown
admission; its latest successful provider observation still finds no pod.

## Verified ownership result and next decision

016 succeeded; provider deletion confirmed for 7jr3i6c7v7stt8 at20:03:50Z.
Recovery verified757 authoritative outputs/756 manifest files (105,490,360 bytes).
All18cases/1152actors match across modes. Shared median1/2/4 seconds:
31.6642 /21.7110 /21.7949; local28.3280 /20.5925 /18.8711. Four-worker local
1736.41 rows/s versus1503.47 shared,1.15493x: narrowly passes the>=15% screen.
One-worker duration ratio0.89464 also passes. Local peak summedRSS9.88–10.09GiB
versusshared6.76–6.98GiB. The last local4worker sample used70.47 summedCPU seconds
in17.83 wall seconds; service ownership helps modestly at additional CPU cost.

Separate profile atB1/16/32/64: forward medians7.130/6.995/6.973/7.166ms,
H2D0.411–0.640ms,D2H0.101–0.108ms. CPU user_annotation catalog spans2.91–3.03ms;
15 scalar reads and3equalchecks per forward. Union of kernel/copy/memset intervals
inside each synchronized CPU forward scope0.882–1.150ms. Ignore duplicate
gpu_user_annotation spans when selecting host windows. Profiler overhead and
initial-state-only inputs limit inference; busy intervals are not SM utilization.

The evidence selects amortizing fixed per-request inference work (batching and
removing repeated work/check-related synchronization with equivalent admission)
over more ownership variants. Do not treat3ms catalog host span as a guaranteed
3ms saving or erase prior0.85%whole-training catalog result. Next implementation
must expose a policy-version-scoped inference boundary usable by ordinary training;
batching changes arithmetic shapes and needs explicit numerical/discrete-trajectory
checks. No new production implementation or rental launched in this research pass.

Ordinary samehost512/4096 controls1655.49/3454.43 rows/s. Large run98304rows/28.4574s,
collection22.2193s,learning4.2691s. These unchanged controls are NOT a worker-local
training gain. Actual goal remains active and incomplete.

Durable evidence: experiments/data/worker-scaling/gpu-016 (summary with derived
trace analysis, original receipts/trace archive, verification, preflightadmission,
budget snapshot). Full experimenthead91b42f CI all10checks passed. Only evidence
and prose changed after run; no new correctness tests needed for these artifacts.
Current ledger: estimatedrentals5.50789,confirmedbills4.73122,conservative7.63,
ancillaryreserve5,unknown015reserve4,unallocated33.37 oforiginal50. Run016 whole
estimate0.47678; no new admission. Guard89254stillruns until20:16:43Z, no015pod
observed and no015experiment admitted. Do not silently release its reserve or
claim its local launcher record is settled; retain watcher and evidence.

Results published in PR292 at8351961c (evidence commitda9059d1); all measured code remains91b42f. Publication changes only retained evidence/prose. Guard015 tool89254 remains live; latest20:08:48Z provider absence confirmed. No new rental. Next turn should reconcile guard before any new admission, read this note, and preserve the original$50 budget.

## Implementation decision — policy-scoped preparation

First integrate a PolicyInference scope into ordinary SeatRoutedCollector. It
admits semantic/generic definition references on CPU before transfer and reuses
semantic catalog embeddings only within one unchanged behavior-policy version.
Prepared observations retain the ordinary dictionary tensor interface so existing
forward instrumentation and recorded-output replay continue to work. The scope
owns prepared references/catalog and rejects use after close, parameter/buffer
mutation, model replacement or in a gradient-enabled forward. Public ordinary
Agent.forward_distribution remains the single computation path; learning with
ordinary tensor dictionaries preserves its existing validation and gradients.
Catalog computation is lazy, so recorded-output replay never runs a neural
encoder merely to prepare an observation. Sampling devices/order, request shapes,
rollout layout and snapshots remain unchanged in this first intervention.

Delete — do not maintain: replace repeated inference-time GPU definition admission
and repeated catalog construction within a collection with the scoped preparation.
Keep their ordinary gradient-enabled equivalents as required learning behavior,
and keep the frozen benchmark baseline in git. No configuration switch or shadow
collector. An experiment-only reference prepares plain tensor dictionaries through
the same collector to compare against the optimized prepared observations.

Acceptance for this slice: forward parity; invalid CPU inputs rejected; stale and
closed scopes rejected; optimizer changes observed by a fresh scope; gradients
unchanged outside serving; exact rollout and recovery parity at unchanged request
shapes. Local focused correctness allowance remains bounded at180s; new focused
suite cap45s before any GPU run. A paid protocol must compare full TrainingRun,
not only isolated requests. This slice does not declare the whole goal complete;
if insufficient, batching remains required investigation. No new rental reserved.

Implementation complete, awaiting GPU validation: new PolicyInference/PreparedObservation, CPU definition admission shared with ordinary checks, lazy per-policy catalog, unchanged public Agent forward, ordinary collect scope and fixed-policy CudaPolicy scope. External/recorded placeholder policy bypasses duplicate preparation in pipeline_benchmark; exact replay tests remain passing. Experimental --reference-policy returns original plain tensor dictionaries. --serving-preparation interleaves six shared-service curves plus twelve ordinary512/4096 training cells; canonical model/EMA/Adam witnesses are computed outside timing. Full prospective protocol is experiments/protocols/inference-preparation.txt.
Local checks:13initial tests passed9.97s; worker tests passed9.42s after external-policy preparation fix; final15model/training tests passed3.16s. WholeCPUTrainingRun raw/EMA/Adam and diagnostics match exactly. Two intermediate test failures were witness-reader recovery-envelope selection and a historical-mean test fixture incorrectly expecting masked-mean rejection; receipts retained under.runs, corrected without weakening semantics. Ruff/diff checks pass. No GPU performance claim yet.
015watcher89254finished20:16:54Z: provider absent after deadline, no experiment admitted; unknown launcher reservation remains held.017reserved$4, poolworker-ownership, job30min/$3, oneattempt; sourcependingpublication. No rental yet. Preflight checks prior provider deletions/quote; new outputs packaged as complete archive (all raw files preserved) to avoid hundreds of upload calls. Existing original$50 ledger now pending8/conservative7.63/ancillary5/unallocated29.37. Do not resubmit a launch after an observation failure.

Candidate/protocol published PR292 head61e7149a707940e56eece42836cd912a6be4247e.017source frozen in ledger; preflight20:28:50Z confirms all prior admitted pods deleted, unknown015absent, quote$1.19/h. Archive upload/readback required before launch. Same cachedEtuderuntime. launch-worker.py017 onlyonce; resume-worker.py017 if observation fails. Output is one complete worker-evidence.tar.gz; recover-worker.py017 verifies upload then safely extracts and validates internal manifest.017notyetadmitted.

Source61e7149archive189347840bytes SHA5cb3549b03c0810d22eacec396cc40da18de9d82e3276834537394807dbc73d5 fullyreadbackverified137.01s. CI Python/Protocol passed. Clean-machine Play failed downloading Torch before application startup at90s; rawjoblog retained .runs/gpu-validation/ci-preparation-play.log, one same-limit CI rerun requested. No code fix or source change inferred from download timeout.017paidrun stillunadmitted until launch helper returnsrunid.

017launch submitted once at20:34Z; live monitor tool35755 owns admission/wait/deletion. DO NOT call launch again. Latest preflight20:33:50Z all15prior admitted pods absent, unknown015absent, quote1.19. Source unchanged61e7149. If admitted use inspect-worker.py017; if monitor dies resume-worker.py017 SAME run. New analysis helper analyze-preparation.py017 directly after verified recovery (not analyze-worker thenownership). It checks18worker cases plus12training cases and exact model/EMA/Adam/diagnostic parity before speed gates. CI rerun command rejected while run38802377581 stillinprogress; retry only once it becomes terminal, no CI rerun admitted yet.

017 ADMITTED run71bfad424b47; monitor35755 live. Source61e7149. No retry. Inspector running to record actualpod/rate. Newgoalturn is progress: production-path candidate + tests + published frozen protocol/source + admitted paired actualGPU test. Goalnotcomplete; keep serving and complete-training gates.

Inspector confirms017running, session207c36f35881, podtnle36pwpz9282, actualrate1.19/h. Setup log shows pinnedRustinstallation, no benchmark result yet. Monitor35755 live; no other paid experiments active. Contextscratch5052tokens/21710bytes below12000/98304 limits.

While017runs, source boundary checked: Execution.workers is Literal[1] (models.py:181); execution.py constructs oneSeatRoutedCollector, and UpdateSnapshot.collector is oneCollectorSnapshot (formats2/3). Parallel training would require an admitted worker count, fixed actor/stream/seed identity independent of active worker count, ordered RolloutBatch concatenation, aggregate counters with global stream IDs, composite collector snapshots and orderly resource/deadline teardown. Do not treat existing benchmark actors as already integrated. No architecture change selected until017paired results; no source edits while this run executes.

## Verified017 result — ordinary adoption rejected

Run71bfad424b47 succeeded; podtnle36pwpz9282 provider-deleted20:50:17Z. Whole rental estimate$0.30715. Complete archive32,633,252bytes and783manifestfiles verified. All18worker cases/1152actors exactly match; four-worker median26.4652→14.3532s,1238→2283rows/s,1.84386x. Ordinary4096 median24.6011→24.4009s,3996→4029rows/s,only1.00820x;512 median6.28184→6.32450s,0.68%slower. This fails10%whole-training target. All12cells completed; no exclusions.

Strict learned-state parity alsoFAILED.4096allraw/EMA/diagnostics exact; lastpreparedrun oneAdam element differs3.55e-15.512firstREFERENCE differsfromlaterREFERENCE (rawmax2.38e-7,EMAmax7.45e-9,Adammax2.79e-9anddiagnostics); preparedtrial2hasfourAdam elementdifferences max1.82e-12 despiteexactraw/EMA. Within-modevariation is observed; causeunknown, no posthoc tolerance or gate relaxation. Fullposthocparity-audit.json retained. Do not callgoalcomplete or accept production adoption. Source61e7149 remains frozen evidence; currentPRcode stillcontainsprovisionalordinary integration and mustbe reconciledbeforeadoption.

Next discriminating work: instrumentactualordinarycollection512/4096streamphaseboundaries andrequestshapes, separateunprofiledthroughput fromCUDAtrace. Existing016smallrequestprofiledoesnotattribute017ordinarycollection. Profileinference/assembly/transfers/sampling/sync beforechoosingbatching/parallelcollectorarchitecture. No newrentaladmitted, no simulationoptimizationselected. Currentresearchpass changesonlyevidence/docs; implementationreconciliation remainsnextauthorizedgoalwork.

Ledgerafter017: confirmedbills4.86773,estimatedrentals5.25091(updatedbilling),conservative8.09,ancillaryreserve5,unknown015reserve4,unallocated32.91oftheoriginal50. Alladmittedpodsdeleted;015providerabsentafterdeadline,localunknownrecordunsettled. CI38802377581Rust/Python/Protocol/Visual/Semanticpass; integrationslive;Playdownloadtimeoutretry NOTyetadmitted(waitworkflowterminal).

017evidencepublishedPR292head87ebfab4; evidencecommit9b949df0, measuredsource61e7149unchanged. TitleMeasure inference-serving gains and ordinary training limits. No newrental, nomerge, goalactive. Thisturnprogress:recovered/audited017,localizedparityfailure,publishednegativewholetrainingresult. Currentresearchboundarymeansno sourceimplementationchangesinthisturn. Nextgoalworkmustreconcileprovisionalordinarycollectoradoptionandmeasureactualordinarycollectionbeforeanotheroptimization.

## Ordinary collection attribution design — 018

017 changed the decision: retain scoped preparation as a small-request experiment, but do not accept ordinary integration as the optimization target. Next instrument actual512/4096-stream TrainingRun at unchangedw64d1,8steps,3updates,seed10349. Four unprofiled controls(reference/prepared xstreams),four Torch CPU/CUDA traces of secondcollection,andtwo separate cProfile runs at4096(reference/prepared). All70sprocess/55sstage/60srecipe caps remain; no retries. Profiles are diagnostic, not throughput; preserve traces, pstats, per-phase windows/histograms, learned-state witnesses and failedcells. CPU windows include launch/synchronization; device interval union is not SMutilization. Ordinary path remainsprovisionalinPR, no deployment.

Delete—do notmaintain: no runtime predecessor removedforan instrument; do notcreateanother training runner, call existing cuda_performance.child and observe collection via scoped patches. Existing capacityprofile remainsdistincthistoricalevidence. Tests require instrumentationrestore andexactCPUrun learned-state parity, bounded20s. Budget018max$4reservation/$3job/30minoneattempt,sourcepublished+readback+priorproviderabsencebeforeadmission. This is attribution, not a renewed performanceclaim.

018instrumentimplemented incollection_attribution.py andworker_experiment --collection-profile. Fourinitialtests passed5.30swhole; twoattributionmode tests passed5.47swhole,each20scap. TorchCPUtracepreserveslearnedstate; actualCUDAprofilingpending. Ruff/diffchecks pass. Scratch6136tokens26406bytes below12000/98304budget.018reserved4,unknown015held4,unallocated28.91. NoGPUadmissionbeforepublishedsource/readback.
