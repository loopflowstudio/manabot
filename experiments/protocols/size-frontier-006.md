# Capacity, inference costs, and learner replay — 006

Jack Heart authorized all three tracks on 2026-10-16, requested a middle
size, and recalled an earlier negative size comparison in a broken world.
He asked specifically whether larger models could learn more slowly but reach
a higher ceiling. No size is eliminated by early score. This bounded study
cannot determine asymptotic skill ceilings or tune each size independently.

Use the existing MTG-135 checkout and reopened Task. PR292 is already merged;
this experiment does not re-land or change that delivery. Preserve prior results.
All jobs bind the same exact source archive, native build and w6 world; no
world compatibility override. The w6 branch includes action/decline and object
identity fixes missing from the old w4 capacity comparison. This does not assert
that every world defect is fixed. Native/runtime admission must pass.

## Common controls

Small: width64/depth1/FF256. Middle: width192/depth4/FF768. Large:
width384/depth8/FF1536. Four attention heads, scalar value-token, no history,
float32 with TF32 disabled, current-self opponents, existing Ataraxos MOVE
learning rule and iteration schedule. FF scales at four times width; this
intentionally differs from study005's FF1536 small model. Record actual parameter
counts; width labels are not parameter counts. One native producer and one CPU
thread own each job. Same selected Lessons/Allies match and observation schema.
No optimizer, batching, reward, inference, or production defaults are modified.

## 1. Learning curves

Three capacities × training seeds13561/13562/13563, 2048 iterations each,
64 streams ×8 learner transitions per iteration. Equal planned experience is
1,048,576 learner transitions; native decisions and filtered optimizer exposures
can differ and are reported. Initial,128,512,1024,2048 raw checkpoints are retained.
Same schedule policy for all sizes; changing the rollout setting is deferred.
An independent small reference, seed13560 at2048, is frozen by hash for evaluation;
it is not a competing scientific seed. Never choose the reference by its score.

Compare scores against measured training seconds, learner transitions, native
decisions, optimizer exposures and allocated dollar bounds. Rental startup and
actual charges are separate from training time. No extrapolation outside observed
support and no comparison of unmatched endpoints as equal compute. Three seeds
are limited evidence; deal uncertainty is not training-seed uncertainty.

Use ordinary replay-validated arena, both seats and deck assignments per deal.
Fixed panel: source-pinned scripted greedy plus the independent frozen trained
reference. Initial and intermediate checkpoints use8 monitoring deals
1916135000–007. Endpoint uses25 held-out deals1916136000–024, never used for
selection or extensions. Report paired deal intervals per seed and variation
across three training seeds. A flat greedy result alone does not establish equal
strength. Compare endpoint to initial against the same panel as a sensitivity
check; if none learns or the panel fails to discriminate, report unresolved.
Evaluation is bounded to two local CPU workers and12hours, sequential monitoring
and endpoint phases (6hours each), with all incomplete evidence retained.

A slow-starting large model receives its whole predeclared allocation. If still
improving at the endpoint, a longer continuation is a separate allocation, not an
inferred ceiling. No score-dependent early stopping, checkpoint selection or
adaptive expansion. Failed or timed-out runs remain failed, not equivalent full
training trajectories. All checkpoint/run identities must agree before scoring.

## 2. Inference and batching

All three sizes, fresh and after128 ordinary updates, fixed seed13570. Capture
real CPU observation rows and chronological requests from64 and512 streams;
capture overhead is excluded from speed evidence. Separately measure clean frozen
collection windows and real128-update training. Fixed-observation sweep:
1,4,16,64,128,256,512 requests, cycling captured rows only when needed and declaring
that repetition. Compare device-resident prepared forward with CPU preparation,
transfer, forward, legal sampling and result materialization. Three warmups and
three synchronized latency windows. These are latencies, not GPU utilization.
GPU processes never share a device during timings. Initial and128-update models
only cover early-policy workloads; mature-policy refresh remains future work.

## 3. Numerical replay

Large model,64 and512 streams, seed10349. At iteration16 retain the exact learner
batch, model, Adam, EMA bootstrap and RNG states. Restore before each of three
ordinary and three deterministic replays. Record target/selection, forward,
gradient, optimizer and parameter witnesses with the existing diagnostic observer.
EMA is frozen during replay; normal training advances it once afterward. No
recollection between replicas. Full payloads and maximal parameter movement are
retained; observer synchronization means this track supplies no throughput result.
Differences within an ordinary replay are separated from self-play trajectory
variation. Exact agreement in a small probe does not clear the old outlier.

## Admission and retention

Existing $25 campaign cap, conservative prior rentals$2.94 and ancillary reserve$2.
New maximum commitment$18.50: ten learning/reference jobs at$1.50 each, profile$2,
replay$1.50. Total bounded exposure$23.44. One attempt per job; no automatic
retry. RTX5090 quote at most$1.25/hour; max three separate rentals simultaneously.
Per learning job:3600seconds training,4320seconds command,4500seconds rental cap;
job dollar cap can stop earlier. Profile/replay command caps2700/1800seconds,
plus bounded setup and artifact upload. Preserve all failures and verify provider
deletion. Run source and hashes bind archived evidence. No paid evaluation beyond
this reservation. No change in scientific allocation based on observed scores.
