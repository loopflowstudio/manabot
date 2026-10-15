# Single-GPU inference serving

2026-10-15: Jack Heart selected clearing the inference-service bottleneck for
training on single GPU machines, rather than optimizing managym rules simulation.
The original total GPU authorization is $50. The goal remains active and unproven.
An isolated service or collection win does not establish faster complete training.

## Work and authoritative evidence

Existing Task task_8f83a201c5b04faf97b8d916fe6a886e, supplied checkout, PR292:
https://github.com/loopflowstudio/manabot/pull/292. Manual review/merge; no auto-merge,
Task settlement, unrelated Task, worktree, agent or auth repair selected.
Current published head4015ce56 (rollback/config-borrow4d0388cf);
020evidence6e7f4bde/published20a31a61; measured storage candidate
head e41edb26be24b0de9d68b46f9edaadfb28c6fcb6;
implementation73317666. Prior producer source a07b3cd3794df658368918742ef25b661b28bb38;
019 evidence771d0c25, published05117ccc2c4edc782bc5931aa69fb4f53bc76853.

Canonical findings/protocols/raw structured receipts are in
experiments/worker-scaling.md, experiments/protocols/, and
experiments/data/worker-scaling/gpu-012 through gpu-019. Full model/profile binaries
remain in each verified immutable Etude archive; retained receipts bind them.
Historical scratch through019 and the storage design is preserved in commit73317666.
Earlier w4 staging/catalog failures and initial research are in
c5cef8b7bfb5b326d0a536bb8a079d23da62ef63. Uncommitted monitoring history before this
context compression is copied to .runs/gpu-validation/inference-serving-before-context-compaction.md.
No historical failed sample or gate has been superseded by this shorter note.

## Completed experiment020 — verified negative result

Run a22592d08d8b, session2f11e00fee28, attemptaae9236e73bb, poolworker-ownership.
Sourcee41edb26be24b0de9d68b46f9edaadfb28c6fcb6, one attempt. Finished22:41:20Z;
provider pod4pjpy18md4xym9 deleted22:41:42Z. Observer38987 completed and confirmed
provider absence. No paid run is active and no021 allocation/admission exists.

All32declared cells completed at exact work counts, with no producer leaks.
All eight deterministic full trajectory/learned-state proofs match exactly.
Ordinary raw/EMA/scientific diagnostics match, but Adam has fourgroups per shape,
including variation within frozen1. Exact ordinary-state gate remains failed.
Posthoc audit retains every optimizer group against first frozen1:512differences
4/5/71elements,max7.28e-12;4096oneelement pergroup,max9.09e-13. No gate relaxed.

Ordinary full-training medians (three repetitions each):
-512 frozen1/direct1/direct4/frozen4:5.308/4.805/4.830/5.189s.
-4096 frozen1/direct1/direct4/frozen4:23.084/24.913/23.690/24.935s,
  4258/3946/4150/3942learner rows/s.
-4096 direct4/frozen1=0.9744x (target1.25); direct4/frozen4=1.0526x(target1.15).
-512durationratio0.9098 passes1.05ceiling. Performance/adoption gates fail.
Direct storage is not a validated large-workload optimization. The subsequent implementation removes it from production and restores the
pre-storage collector plus experiment owners froma07b3cd3. The exact source/control
and negative evidence remain retained. Native config borrowing is implemented
and locally validated; its GPU benefit is unmeasured.

HostRTX5090/AMD EPYC9554,31.13corequota,~62GBmemorylimit,driver610.57.04,
Torch2.10/cu128. Do not compare absolute speed against019's different host.
Archive84,061,387bytes,SHA256be13accb98333705c4d2d943f4b3ddf87ab2ba570ebc1c466f707d125c9de905;
all443manifestfiles verified. Complete analysis in experiments/worker-scaling.md
and experiments/data/worker-scaling/gpu-020; fullmodelbinaries in verifiedEtudearchive.

Private operational helpers remain in .runs/gpu-validation. Cached Etude runtime:
```
doppler run --project etude --config prd -- env VIRTUAL_ENV=/Users/jack/.cache/uv/archive-v0/vs7MnbtcaP-dGy8d-5EAc uv run --active --no-project python .runs/gpu-validation/inspect-worker.py 020
```
Never relaunch020. inspect-worker records authoritative status/deletion/costs;
recover-worker verified immutable outputs; analyze-storage audits all32cells;
retain-worker packages public structured evidence. cli.configure isolates the
validation namespace/credentials; never print secrets or signedURLs.

## Budget and cleanup

.runs/gpu-validation/budget.json is authoritative. After020: estimatedrentals
$6.30585,confirmedbilling$5.66757,conservativecharges$9.32,ancillaryreserve$5.
Unknown015 retains$4reserve; unallocated$31.68oforiginal$50.020wholeestimate$0.25127.
All admitted pods through020 have provider-confirmed deletion. No021reservation.
015 never admitted a run; unknowncreate session1e46a8e9a3d1/providername
etude-1e46a8e9a3d1 was absent through bounded watcherdeadline20:16:54Z and later
preflights. Localunsettledstate/fullreserve persists; no resubmission orstore repair.

## Implemented pipeline and proof limits

Production has one SeatRoutedCollector/GPU/sampling owner. Persistent CPU threads
own disjoint native VectorEnv shards and contiguous NumPy row views. Buffered
PyO3 calls release the GIL. A barrier completes all shards before global policy
batches, sampling and rollout handling. Global initial seeds/reset strides preserve
stream identity across worker counts. Execution.workers supports1(default)/2/4
for self-play; other stages/diagnostic roots require1. Pools close on failure,
interruption and completion. The existing global recovery snapshot is unchanged.
This is production orchestration, not a rules-engine rewrite or async GPU overlap.

Direct storage is removed after020's failed gates. The immutable old collector
fixture remains evidence; its runtime loader, exclusive test file and CLI modes
are removed. Historical020 reproduction uses measured sourcee41edb26. Production
again uses pending transitions and final batch assembly, preserving the existing
update-boundary/recovery API.

Native VectorEnv now borrows immutable stream configs during step/reset iteration;
the callback receives one config slice while mutating only its environment/seed.
This removes full deck/config clones on every micro-step, even when most rows are
inactive. Rules, encoding, action masks, seed strides and callback/error ordering
stay fixed. Infrequent recovery setup replacement remains owned as before.

Local validation of rollback+borrowing:9Rust debug vector tests pass0.89s; root
cp312release rebuild7.85s;20Python producer/recovery/inference/collector tests
pass11.29s. Ruff/cargo fmt/diff checks pass. New test compares masked heterogeneous
setups through terminal autoreset against independent scalar environments.
No GPU benefit claimed; next work needs paired native-build provenance and a
prospective same-host protocol before rental. Full detail in collection-tail.md.

## What actual GPU evidence establishes

012/013/014: native VecEnv loops serially; releasing the GIL alone is not parallel
execution. Matched persistent-process CPU-only scaling reached3.82x on the H100
host, and2.20x on a more variable RTX5090 host. Shared GPU four-worker scaling
was1.20x on012RTX5090 and2.22x on013H100. The measured5090host delivered roughly3x
learner rows per rental dollar versus H100, motivating this economical choice;
it is not a universal hardware optimum. Cross-host timings are not paired trials.

016: independent worker-local GPU models versus shared ownership improved the
four-worker benchmark15.5%, with exact1152actor witnesses. Small-request profile
showed substantial per-request host work; it did not establish a production
training gain or GPU SM utilization. All native inactive rows are skipped before
stepping; native boundaries also include encoding/bindings, not pure rules cost.

017: policy-scoped preparation reused the catalog during fixed behavior weights
and moved definition admission to CPU. Four-worker serving improved1.844x,
1238→2283rows/s. Ordinary4096training improved only0.82%(3996→4029rows/s);512 was
0.68%slower. Strict parity failed:4096oneAdam element3.55e-15,512baseline itself
varied(rawmax2.38e-7,EMA7.45e-9,Adam2.79e-9), plus another prepared Adam difference.
No tolerance relaxed or repetition excluded. Production adoption was not accepted.

018: all10ordinary attribution cells completed, source3afa1ac26f277f70dea997e288c28dcf3b77e66f.
Prepared4096second-collection Torch host windows:8.132s total; native stepping+
encoding4.073s; input selection/preparation/H2D1.239s; forward0.616s; policylogs
0.308s; sampling0.045s; batch assembly0.894s; unmarked0.956s. Catalog was nested
inside forward(199calls→1), not an additive phase. Separate cProfile supports
large native/assembly costs. Device-work interval union0.579s is not SM utilization.
199policyrequests/67275actingrows,142requests<=64 despite4096residentstreams.
Profiles alter scheduling/cost; their fractions are not throughput coefficients.
All raw/EMA hashes agreed across these modes but Adam varied. This selected CPU
producer/input/rollout data flow feeding the GPU, not another model-only change.
Earlier pinned-buffer/staging attempts failed multi-seed/large cases, including
4096timeouts; old isolated1.23–1.42x gains are not ordinary training acceptance.

019: actual production1/2/4producer test on RTX5090/AMD EPYC7543,14.375corequota,
driver570.153.02. Run978556688139, sourcea07b3cd3, archive83,040,230bytes/452entries
verified.30of34cells completed; large deterministic proofw2/w4/finalreference and
first large ordinary reference hit unchanged deadlines. All prepared timing cells
completed.4096medians29.359/26.806/26.233s =3348/3667/3747rows/s,1.11917x4vs1,
below20%target.512prepared7.285/6.757/6.699s. Missing ordinary reference repetition
prevents a complete primary comparison; no survivor-only median is substituted.
SummedCPU29.480→39.527s while collectionwall22.580→20.060s. Cause of limited scaling
is not established. The native profiler is disabled and per-Env, so a shared
profiler-lock explanation is unsupported on this path.

All five512deterministicproofs exactly match full trajectories and learned states.
Every available4096collection witness matches(5/5/3samples atcollections0/1/2),
but only firstreference/prepared1 completed learning. Matching prefixes do not
certify unfinished runs. Completed ordinary raw/EMA/diagnostics match withinshape;
Adam has two groups:512outlier71elements,max3.64e-12;4096oneelement2.84e-14, including
single-worker reference/prepared variation. Strict gate remains failed. Provider
pod a71mb90ubjolw8 deleted; whole rental estimate$0.38017. No019 rerun was launched.

## Unresolved numerical question

scratch/questions.md records the parameter mapping and competing explanations.
The019differences map to semantic token embedding, semantic GRU weights and
attention.mha.in_proj_weight;4096affects attention projection alone. Artifact
mapping is saved at .runs/gpu-validation/result-019/optimizer-parameter-names.json.
Experiment._setup_random sets cuDNN deterministic but not Torch's full deterministic
algorithm mode. Diagnostic settings change several controls together, so current
proofs do not identify one cause. The actual learner is ataraxos_move through
update_iteration/update_move_iteration/OptimizerHealth.step, not stock PPO's
Trainer._optimize_step. A controlled repeated backward on one saved loss graph
could separate CUDA gradient variability from rollout/storage/Adam if needed.
No such probe has been implemented or admitted; no gate is waived.

## Read-only follow-up

scratch/collection-tail.md records a new source-backed hypothesis: VectorEnv's
buffered step deep-copies every stream's owned player configs before testing the
active mask. The verified018 histogram has142/199calls carrying only815/67,275rows;
chronological late windows still contain substantial native work. Copy cost and
its contribution to019's scaling are unmeasured. No change to020 or new admission.
The previous goal turn completed020verification/analysis/publication. This turn
removes the failed storage candidate and implements/tests config borrowing.

020evidence published through lf pr publish to existing PR292; body rewritten
around complete scope and latest failed gates. That evidence publication did not change production; the subsequent
rollback/config-borrow implementation is ready for publication. No Task settlement, merge, auto-merge or new admission.

Rollback/config-borrow implementation published to existing PR292 via lf pr publish.
No021 reservation/admission, no paid job active. Source4015ce56 is ready for
paired native-build experiment instrumentation; it has not been GPU measured.

Paired native experiment instrumentation is implemented for021, protocol
experiments/protocols/native-config-borrowing.txt. Native builds differ only in
vector_env.rs; the e41control is a hash-checked immutable fixture. Local two-build
fresh-process comparison passed all4cells with fullcollection/learned-state parity;
7selector/existing inference tests pass. Next requires publication/source readback,
provider+budget preflight, then one$4reserved run; no admission yet. Setup failures
retain logs/manifest. analyze-configs.py checks everycell's selected binary and
source receipt against both retained libraries before all32cell gates.
