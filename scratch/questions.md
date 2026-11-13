# MPS retest: decisions and remaining work

The initial adoption threshold and prediction are superseded by the final
2026-11-12 steer below. Torch is the incumbent; the report will interpret
measurements and offer a reasoned recommendation, never trigger a rule.

Implementation choices (reversible, not quotes):
- w64l2 uses four heads; w96l4 and the other shapes use eight.
- Capture w384l8 at 256/512/1024 streams with the same seed and recipe.
  Every width replays those rows; other widths start from seeded fresh weights.
- MLX stays float32 because its backend explicitly owns float32 arithmetic.
- MPS driver peak is sampled every 50 ms and labelled a lower bound.
- Per-call comparisons use the 256-stream capture. All three captures feed
  whole-update replays and their own parity blocks.

Unresolved execution: installed Etude SDK 0.2.16 declares rental providers
`runpod`, `vast`, `lium` (`etude/launch/records.py::Provider.kind`), no AWS Mac
provider. The requested USD25 capped AWS allocation/release path has not been
identified. No host was allocated, no paid experiment launched, no timing
performed. Local load read at 16:13 was 263.74; it was not used for timing.
Credentials were not tested or printed; this is a launcher capability issue.

The small CPU/MLX two-row smoke and synthetic chart checks are controlled
implementation fixtures, not performance evidence. MPS execution is unverified.
Actual results, final charts, artifact publication, cost and recommendation
remain required. Do not represent this implementation step as the completed
experiment. No optional L4 arm was launched.

Steer from Jack Heart (Task comment d2803d55, 2026-11-12), verbatim: "make sure
the torch MPS half-precision arm is actually run, not skipped. It is a required
arm at every shape: bfloat16 autocast on MPS if torch 2.15 supports it,
otherwise float16 autocast with a grad scaler; report which dtype ran, its
gradient cosine against CPU float32, and its time next to MPS float32 and MLX.
If bfloat16 and float16 both work on MPS, run both." The report's Arms section
carries this.


## Steer 2219841a-07f2-4888-9d41-71dbbb9891b4

Comment 42df6eb6-2b86-4f21-a757-752af29acd51 by Jack Heart:

Jack: add one shareable chart as a named deliverable, a 'full context' counterpart to a colleague's forward-only benchmark (decisions/second for one policy call at 64 and 512 boards: PyTorch CPU, PyTorch MPS, PyTorch MPS half precision, MLX). Ours shows relative performance of the CURRENT TRAINING RECIPE as a whole update, not an isolated call: both-seat self-play, 256 streams, at w160l8 and w384l8, one captured real update replayed through every arm on the same host in the same hour (the MTG-321 method). Chart: grouped bars per width, one bar per arm (torch CPU8 float32, torch MPS float32, torch MPS half, MLX), each bar stacked by phase: engine stepping, collection policy calls, learning. Y axis in seconds per update, and a second small panel or secondary label giving learner decisions per wall-second of the whole update so it is in the same unit as the colleague's chart. Caption: machine (Apple M2 24 GiB, torch 2.15, MLX version), median of three replays, which half-precision dtype ran, and that all arms produce the same decisions (parity bars). If an L4 replay of the same captured update is cheap (under USD 1 on Etude) add it as a reference bar labelled by device; otherwise leave it off rather than mixing captures from earlier reports. Human-readable labels, no internal arm ids, light and dark safe, PNG at experiments/data/mps_retest/full_context.png published with manabot.data, embedded in experiments/mps_retest.md, and linked first in the PR body so it can be shared on its own.

<!-- loopflow-requester:"Jack Heart" -->

<!-- loopflow-steer:42df6eb6-2b86-4f21-a757-752af29acd51 -->

## Steer cf2540f2-cbea-44ec-844d-5d3fd756e69c

Comment 3329f511-722a-4261-9a69-28a659205712 by Jack Heart:

Jack: you may run measurements on either this laptop or a rented M2. This laptop is an Apple M4 Max, 16 cores, 128 GiB, but it is Jack's working machine and its load average is often above 100. Plan: (1) run the comparable arms on a fresh rented AWS mac2-m2 (24 GiB), same host class as the MTG-321 pilot, so the chart's main bars match earlier evidence; (2) use this laptop only for the arms the M2 cannot hold, the wider widths (w512l8, w768l8, w1024l8 as memory allows) and the largest minibatches, and only when it is quiet: before every timed block check the 1-minute load average is under 8 and wait otherwise, record the load in the receipt, use at most 8 CPU threads, and never run both hosts' measurements on the laptop at once. Never start a timed block on the laptop without that load check, and stop if Jack's interactive work is affected. If the laptop never gets quiet, report the wide arms as not measured rather than as noisy numbers. Keep the per-host provenance on every chart and table (M4 Max 128 GiB vs M2 24 GiB are different machines; never put their bars side by side as if comparable).

<!-- loopflow-requester:"Jack Heart" -->

<!-- loopflow-steer:3329f511-722a-4261-9a69-28a659205712 -->

## Steer ea022e5b-b3f8-427a-a111-6b6776c79c8c

Comment 52f4f0fd-a3fc-4508-9c78-f6af18fda27a by Jack Heart:

Jack: run the whole-update arms at 256, 512 and 1024 streams, not only 256. Capture one real both-seat update at each stream count (same seed and recipe otherwise) and replay each capture through every arm on the same host. The full-context chart then has a group per (width, streams) or a second row of panels for streams, with the per-update seconds stacked by phase and learner decisions per wall-second alongside, so it shows whether larger batches close the MPS gap and how each arm scales. Memory permitting on the 24 GiB M2 at w384l8; if 1024 streams does not fit an arm, say so on the chart rather than dropping the group silently.

<!-- loopflow-requester:"Jack Heart" -->

<!-- loopflow-steer:52f4f0fd-a3fc-4508-9c78-f6af18fda27a -->

## Steer c4acf262-54ca-4350-bbfb-5c1c38c0b476

Comment dbeb9bd7-cbaf-4e49-8197-a3ec6fd4bfa9 by Jack Heart:

Jack: widen the model axis both ways. Smaller: add w64l2 (the batched-baseline agent shape: hidden 64, 2 attention layers) and w96l4, so the chart shows where the GPU arms stop paying off at the small end. Bigger: beyond w384l8, run w512l8 and w768l8 and, if memory allows, w1024l8; on the 24 GiB M2 run what fits and mark the rest 'does not fit' per arm, and use the M4 Max laptop (under the load rule already given) for the widths the M2 cannot hold. Every width runs the full set of arms (torch CPU8, MPS float32, MPS half, MLX) at the stream counts 256/512/1024 where the capture fits. The full-context chart gets one panel per width; keep hosts visibly separate.

<!-- loopflow-requester:"Jack Heart" -->

<!-- loopflow-steer:dbeb9bd7-cbaf-4e49-8197-a3ec6fd4bfa9 -->

## Steer 1f3506a2-09b9-45d1-b5f7-3350fd5601fe

Comment f8b011bd-45ad-4c96-a58e-47d3dcaaedfd by Jack Heart:

Jack: when you reach a timed block that must run on this laptop, do not wait indefinitely for the load to drop. Post a progress note naming the block and its memory need, then pause your own Graph step until the operator quiets the machine, or proceed if the 1-minute load is under 8. The operator (this session) has Jack's authorization to pause other Tasks' Graphs and stray builds for the duration of a timed block and will resume them after; you only need to say when you are ready to time.

<!-- loopflow-requester:"Jack Heart" -->

<!-- loopflow-steer:f8b011bd-45ad-4c96-a58e-47d3dcaaedfd -->

## Steer ff0a995d-202e-42b6-8335-81cd692bb4cb

Comment 22c452b8-9bc9-4d39-aa24-fe1f21716a24 by Jack Heart:

Jack: drop the pre-registered decision rule from the brief; it was backwards and he does not want decisions made before the results exist. Torch is the incumbent: one backend is simpler to keep, and MLX is extra code that has to justify its own existence with a clear advantage where training actually spends its time. So the report does not decide. It measures, then gives the reader what they need to decide: at each width and stream count, how each arm compares on the whole update and on each phase, whether torch MPS (float32 or half) gets close enough to MLX that the extra backend is not worth keeping, where MPS overtakes CPU, what memory each needs, and whether gradients are correct. State plainly what the numbers support and what they do not (noise, single host, replayed update), and give a recommendation with its reasoning as a recommendation, not as a rule that was triggered. The full-context chart and the arms stay as steered.

<!-- loopflow-requester:"Jack Heart" -->

<!-- loopflow-steer:22c452b8-9bc9-4d39-aa24-fe1f21716a24 -->
