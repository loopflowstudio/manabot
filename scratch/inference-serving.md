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
