# Why study006 did not deliver learning curves

Observed 2026-10-17 04:24 UTC (Oct16 evening Pacific). Requested by Jack Heart
with `$5whys`. This analysis concerns the GPU campaign authorized later in the
MTG-135 conversation, not a new launch from the older CPU-research brief.

## Outcome and recovery

The requested result was a small/middle/large learning comparison with three
seeds, alongside profiling and numerical replay. Profiling completed128updates
for each size; numerical replay also succeeded. No2048-update learning trajectory
has been recovered, and no strength evaluation has run. The original small run
and replacement reference both ended `out_of_time`. Eight further replacements
were never submitted after the serial recovery stopped on its first failure.
All campaign rentals have deletion/absence receipts. No GPU workload is running.
The Task remains incomplete. Successful diagnostic metadata is retained; full
binary recovery is running locally after fixing a downloader threading error.
This is recovery of immutable results, not repeated measurement.

“GPU failures” was an imprecise description. These were separate failures:

| Event | Evidence | What it establishes |
|---|---|---|
| Third concurrent rental refused | `admission-refusal.json` | Capacity admission failed; the first two rentals were deleted and concurrency reduced to two. It says nothing about model training. |
| Two subsequent machines lost before command start | `profile-failure.json`, `verified-prestart-losses.json` | Etude recorded `machine_lost_before_start`; no training executed. Provider/root cause remains unresolved. |
| Four commands rejected an existing study directory | `retired-*.log`, `unique-output-admission.json` | My wrapper reused `/tmp/frontier-study` across attempts on a reused machine. `exist_ok=False` correctly stopped before training. This was a launcher bug. |
| Nine commands failed CUDA admission on one machine | `cuda-recovery-proof.json`, `cuda-failures/*/job.log` | PyTorch reported `CUDA initialization: CUDA unknown error`; the RTX5090 admission check rejected the environment. Authenticated metadata contains no training files. These were nine jobs, not nine independently broken GPUs. |
| Two long commands reached3600seconds | `results/small-13561/attempts.json`, `results/reference-cuda-recovery/attempts.json` | Both ended `out_of_time`, exit-15, empty clocks and outputs. Their last phase and completed update counts are unknown. |
| Full diagnostic downloads failed locally | `recovery-failures.json` | A ThreadPoolExecutor used a Scheduler whose SQLite connection belonged to the main thread. This did not invalidate successful remote runs. |

All receipt paths above are under `.runs/size-frontier/`.

## Causal chain

1. **Why is the learning comparison missing?** Most jobs never passed admission;
   the two remaining long attempts hit their command deadline without retained
   results. These are distinct causal branches. There is no evidence here that
   large models failed to learn, exceeded GPU memory, or were slower than small.

2. **Why did one bad environment affect nine jobs?** The queue treated failed
   experiments as completed attempts and reused the available machine. The
   installed Etude scheduler explicitly reuses machines after an attempt finishes
   (`launch/scheduler.py`, `_next_run`). My supervisor observed failures but only
   had automatic recovery logic for `machine_lost_before_start`; it did not stop
   admission or withdraw a machine after CUDA admission failed. All nine commands
   were already queued. The application-level admission check prevented invalid
   science, but it was not connected to a machine-health or cohort-stop decision.
   A clean process exit from a failed job did not make the machine usable.

3. **Why did timeouts leave no usable progress?** The launcher set learning jobs
   to `max_hours=1.0`, which Etude converted into a3600second limit for the entire
   shell command. That command first ran setup, then invoked a4320second inner
   timeout; the runner separately allowed3600seconds of training. The documented
   training/command/rental hierarchy was therefore not the effective hierarchy.
   The outer deadline could expire before the training allowance, and did expire
   in both recorded attempts. This is a proven configuration mismatch, not proof
   of what kept either particular command busy for the hour.

   Setup output went into `/tmp/frontier-setup.log` and was copied to retained
   outputs only after the command returned. Scientific files went into a study
   directory outside `ETUDE_OUTPUT_DIR`; `size_frontier.main` packaged them only
   in its finalization path. The installed worker stops the shell process group
   at the deadline and records files under `outputs/`. A Python SIGTERM handler
   alone cannot guarantee completion of that finalization or a large archive.
   Both attempt receipts show empty outputs and clocks. We cannot distinguish
   setup delays, a hang, slow training, or late checkpoint/export work from those
   receipts. I should not have described scheduler state `running` as proof that
   training was making progress.

4. **Why were incompatible limits and late-only retention possible?** The
   campaign wrapper duplicated lifecycle responsibilities across shell timeouts,
   Etude job limits, the training recipe and a post-run packager. They were
   assembled independently rather than derived from one deadline with reserved
   checkpoint/upload time. TrainingRun already persists local progress, but the
   wrapper did not connect it to Etude's `etude-result.json` clocks/checkpoint
   contract. Thus useful local persistence was not durable remote observability.
   The repo also has an Allocation type deriving pause/checkpoint/final deadlines
   in `manabot/remote/plan.py`; this study did not use such an allocation.
   Reusing that ownership model is preferable to adding another deadline owner;
   choosing the exact adapter still requires implementation review.

5. **Why did validation and supervision miss the lifecycle?** The recorded local
   checks exercised short CPU training, checkpoint milestones, profiling, replay
   and arena loading. They did not establish same-machine sequential execution,
   forced deadline retention, queue behavior after GPU admission failure, or
   threaded full-artifact recovery. The preflight checked provider price/capacity
   and source identity, not those contracts. I expanded to the full cohort before
   proving that one long-run-shaped attempt could be observed, interrupted and
   recovered through the real launcher. Later repairs addressed the immediate
   symptom and resubmitted remaining work without first validating that complete
   lifecycle. The watcher kept scheduling and writing local status, but was not
   an experiment-health gate. This is the operational design gap that amplified
   the initiating failures; it is not an explanation of the CUDA driver's error.

## Direct local check

Before execution, declared two local probes: no GPU/network, at most15seconds
total, all files under `.runs/size-frontier/incident-probes/`. They invoked the
installed worker's `_run` with a one-second command limit. Each child wrote one
progress file, then waited beyond the deadline.

- Publish only after work: the file existed outside outputs, outcome was
  `out_of_time`, and published outputs were empty. Elapsed1.05seconds.
- Write progress into outputs before waiting: same timeout, but the finished
  record retained `progress.json`. Elapsed1.04seconds.

Script: `incident-probes/retention.py`; receipt: `incident-probes/results.json`.
This reproduces the output-location/late-publication failure, not CUDA behavior
or the missing trajectories. It tests the local worker journal, not remote
upload completion. The courier's source shows that live clocks/checkpoints use
`etude-result.json`; arbitrary output files are carried at completion. Merely
moving all files into outputs is therefore not a complete live-recovery design.

## Unanswered questions and contrary evidence

- CUDA initialization's root cause is unknown. Its warning mentions environment
  changes as an example; that is not evidence that CUDA_VISIBLE_DEVICES changed.
  Driver state, device exposure and host problems remain possibilities.
- Profiling successfully trained all three sizes on a different machine with
  the same frozen source. That argues against a universally incompatible model
  or software configuration, but does not prove a physical GPU defect.
- Neither timed-out attempt has remote phase evidence. Their actual completed
  work is unavailable in the recorded outputs. Do not extrapolate128-update
  timings into a claimed2048-update endpoint or call either one a trained model.
- Numerical non-repeatability at512streams is a separate measured result. It
  is not evidence that it caused admission errors or these timeouts.
- Successful remote artifacts still need full local verification. The download
  exception is not evidence that their remote payloads were corrupted.

## Prevention and proof

| Change | Status | Proof required |
|---|---|---|
| Unique attempt-owned study paths | Implemented in private launch commands | Two attempts on the same machine both start and retain separate files. |
| Stop subsequent admission after the first unsuccessful replacement | Implemented in private serial controller; timeout blocked eight remaining cells | Retained `cuda-recovery-blocked.json` and unchanged pending plan. This limits damage; it does not diagnose a machine. |
| Keep Scheduler/SQLite access on its owning thread | Fixed in local artifact recovery; full recovery still in progress | Hash-verify complete outputs for both successful diagnostic runs. |
| One derived deadline with time reserved for setup/checkpoint/upload | Proposed, not implemented | Reject inconsistent plans before launch; interrupt a bounded local run and recover its checkpoint before the outer deadline. Do not just increase all caps. |
| Live phase/update counters and durable checkpoint publication through existing Etude contract | Proposed, not implemented | Observe counters while running, force interruption, fetch and load the retained checkpoint independently. Report last observed progress age. |
| Failed GPU admission stops the affected queue or withdraws its machine | Proposed beyond current serial stop | Inject admission failure; prove no next scientific attempt is assigned there and retain diagnostic evidence. |
| Full lifecycle canary before cohort expansion | Proposed | On the actual transport, prove setup, CUDA operation, one update, interruption, retained outputs and cleanup; then permit the declared cohort. Never use its score to select science. |

The useful next action is to repair and test the deadline/progress/retention
adapter with the existing training and Etude APIs, then run one bounded GPU
lifecycle canary under the existing authorization and remaining budget. The
2048-update scientific study must not be silently shortened, re-seeded, extended,
or treated as resumed unless a valid continuation checkpoint exists. This
analysis does not itself launch a canary, approve a new budget, or claim recovery
complete. Stop here for findings review; retain the two successful diagnostics.
