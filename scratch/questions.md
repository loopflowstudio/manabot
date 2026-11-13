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
- Cross-host replays retain and label their capture engine origin. The new
  direction also permits independent same-host laptop captures and all widths.

The earlier rental-lifecycle investigation is superseded by the operator's
explicit ownership below. Its observations remain in commit 0ed4f2cb1; no
rental tooling should be built. No Task-owned allocation occurred.

Reversible report choice: readable comparisons are generated Markdown beside
the PNGs (`comparisons.md`, hostname suffix for supplemental hosts), with all
replay repetitions and the latest per-call/parity attempts. Unknown provenance
is labelled rather than inferred; synthetic proofs stay under `.runs/`.
For the allowed laptop-only fallback, a sole host uses the canonical artifact
names, including `full_context.png`; its captions still identify the M4 Max.
Combined M2/M4 reports keep the M4 hostname suffix. This filename choice does
not treat the two hosts as comparable.

Completed measurements, published artifacts and remaining design work are in
`scratch/mps-retest.md` and `experiments/mps_retest.md`. Controlled fixtures
are implementation checks, never performance evidence. No optional L4 arm ran.

Superseded load thresholds, pause requests, preflight observations and the
earlier laptop-only reporting fallback remain verbatim in
`adeb2b164:scratch/questions.md` (comments 3329f511, f8b011bd, 17d97d4c and
56378d1f). The current load and replay direction is quoted below.

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

## Steer ff0a995d-202e-42b6-8335-81cd692bb4cb

Comment 22c452b8-9bc9-4d39-aa24-fe1f21716a24 by Jack Heart:

Jack: drop the pre-registered decision rule from the brief; it was backwards and he does not want decisions made before the results exist. Torch is the incumbent: one backend is simpler to keep, and MLX is extra code that has to justify its own existence with a clear advantage where training actually spends its time. So the report does not decide. It measures, then gives the reader what they need to decide: at each width and stream count, how each arm compares on the whole update and on each phase, whether torch MPS (float32 or half) gets close enough to MLX that the extra backend is not worth keeping, where MPS overtakes CPU, what memory each needs, and whether gradients are correct. State plainly what the numbers support and what they do not (noise, single host, replayed update), and give a recommendation with its reasoning as a recommendation, not as a rule that was triggered. The full-context chart and the arms stay as steered.

<!-- loopflow-requester:"Jack Heart" -->

<!-- loopflow-steer:22c452b8-9bc9-4d39-aa24-fe1f21716a24 -->

## Operator rental ownership and laptop execution

Comment d2be0c51-82be-44c7-8adc-2ef13bfcf107 by Jack Heart:

Operator answer to the missing input: the rental and release path is the operator's, not the Task's. The operator allocates the mac2-m2 host with the AWS CLI, launches the instance, writes ~/.manabot-mps-bench/access.md (host, instance, SSH command, machine facts), and releases it with the 24 h guard in ~/.manabot-mps-bench/release.sh; the cost receipt is the allocation and release timestamps plus the 24 h minimum (about USD 21), which the operator records and you cite in the report's cost section. No capped-lifecycle tooling is needed and none should be built. Right now AWS has no mac2-m2 capacity in any US zone (InsufficientHostCapacity in us-east-1b/c, us-east-2b/c, us-west-2a/b/c at 01:20 UTC); the operator's retry loop tries every 10 minutes and will rewrite access.md when a host comes up. Meanwhile continue: this laptop (M4 Max, 128 GiB) is an allowed measurement host under the load rule, and its load is expected to fall over the next 1-2 hours. Prepare the laptop venv (torch 2.15, mlx), run the capture and the arms that fit here when the 1-minute load is under 8, and keep the M2 arms queued for the host. If the host never arrives within 12 hours, report the laptop measurements with their provenance and say the M2 arms were not run.

<!-- loopflow-requester:"Jack Heart" -->

<!-- loopflow-steer:d2be0c51-82be-44c7-8adc-2ef13bfcf107 -->

Reversible execution choice: retain this checkout's prepared CPython 3.13
venv for laptop runs; the M2 still requires its fresh Python 3.12 venv.
Use separate per-host capture/ledger directories so the M2 comparison can
proceed independently. Interpret the 12-hour window from comment creation
2026-11-13T01:12:38.971Z (deadline 2026-11-13T13:12:38.971Z).

## Latest direction: no laptop load threshold

Comment 902d2d02-c77a-40e6-bab0-9c73aaa046b0 by Jack Heart:

Operator: drop the load threshold for laptop blocks entirely. This machine hosts many agent sessions and its 1-minute load will not go under 16 tonight (41-50 with the heavy test binaries paused). Jack authorized measuring here anyway. The operator keeps Rust test binaries paused for the next four hours. Proceed now, unconditionally: freeze the source, run the w384l8 256-stream capture with four CPU threads and the 20 GiB budget, then the three interleaved replays within one hour, and so on through the matrix. Interleaving call by call and step by step is what makes the relative numbers trustworthy under shared load; record the 1-minute load at the start and end of every block and print it in the receipts, tables and chart caption. Do not stop again for host availability: if the M2 arrives (access.md changes), add its arms; if not, the laptop numbers with their load labels are the deliverable, and the report says the M2 arms were not run.

<!-- loopflow-requester:"Jack Heart" -->

<!-- loopflow-steer:902d2d02-c77a-40e6-bab0-9c73aaa046b0 -->


## M2 access supplied

Comment dc46aaf7-05b3-42f0-a34c-74a9c7fce64e by Jack Heart:

Operator: the rented M2 is up. Host h-067a779e551e75812 (us-east-1b), instance i-01bad7ce91e5de2d7, public IP 34.229.142.55; SSH command, key path and machine facts are in ~/.manabot-mps-bench/access.md (rewritten 02:56 UTC). It is a fresh macOS 26.7 image with nothing installed; allow 5-10 minutes from 02:56 UTC for first boot, then install Homebrew, uv, Python 3.12, torch 2.15 and mlx, and clone the repo at your frozen revision. Run the M2 arms there (all widths that fit in 24 GiB, 256/512/1024 streams, four arms interleaved, three replays within one hour); keep the laptop for the widths the M2 cannot hold. The operator releases the host when you post that the M2 blocks are complete, or 24 h after allocation at the latest; do not release it yourself. Cost receipt: allocated 2026-11-13T02:56:06Z, 24 h minimum about USD 21.

<!-- loopflow-requester:"Jack Heart" -->

<!-- loopflow-steer:dc46aaf7-05b3-42f0-a34c-74a9c7fce64e -->

Reversible interpretation: “four arms” names backend/precision families; keep
both working half dtypes as explicitly required earlier (five executed arms).
Finish the already-running laptop w384 comparison; subsequent fitting widths
belong on M2, with laptop supplemental widths only if M2 cannot hold them.


## Back-to-back replays replace the one-hour window

Comment 3b899965-3f27-426c-8c7a-8afe14a666be by Jack Heart:

Operator: drop the one-hour window; it was my constraint and it does not fit a 40-minute replay. Rule now: the three interleaved replays of a capture run back to back on the same host, each with its start and end wall-clock and 1-minute load in the receipt; no time cap. Repeat a replay only if the load more than doubles within that single replay, at most once, and keep both in the receipt with the quieter one in the chart. Do not stop the laptop replays; continue the w384l8 256-stream replays now and then the rest of the matrix. On the M2 the same rule applies, and nothing else runs there so the load note will be trivial. If the whole laptop matrix would exceed the four-hour quiet window the operator holds (ends 06:17 UTC), prioritise in this order: w384l8 at 256/512/1024 streams, then w768l8 and w1024l8 at 256 streams, then the smaller widths, and report what was not reached.

<!-- loopflow-requester:"Jack Heart" -->

<!-- loopflow-steer:3b899965-3f27-426c-8c7a-8afe14a666be -->

Implementation choice: preserve prior capture and parity/replay evidence at its
source. Restart three repetitions under one new freeze after the retry-policy
change; do not combine runs separated by cancellations. Per-call comparison
blocks use the same directional load-increase retry on both hosts; laptop
captures retain their retry with this directional predicate.

## M2 comparison design counterexample, 2026-11-13 05:29 UTC

Five resident arms on the 24 GiB M2 sustained paging, reaching 10,518.25 MiB
used swap with no complete replay after about 105 minutes. Raw counter samples
and the cancellation receipt are retained in `m2/host-observations.jsonl`.
Etude `877fa3cf766d` was cancelled at 05:29 UTC; no OOM was observed or inferred.
The implement instruction says to stop dependent work when evidence breaks
the design. Further M2 timing is stopped pending review of a comparison that
does not keep five allocator working sets resident on a 24 GiB machine.
Pairwise interleaving would change the steered all-arm comparison and its
single CPU baseline, so it is not silently substituted. Captures and parity
remain evidence; no M2 performance conclusion or full-matrix completion is
claimed. The explicit instruction to continue laptop replays still applies.

Laptop scheduling choice: the completed three-replay w384/256 series took
6,870.95 seconds. Do not start another full series that cannot fit the
remaining window ending 06:17 UTC. Capture 512 and then 1024 streams while
time remains, preserving failures or unused captures; report their replays
and lower-priority widths as not reached rather than substituting a single
replay for the required median of three.

Scope refinement after the w384 M2 stop: the counterexample blocks that
resident working set and larger ones, not every smaller model. Attempt w64l2
on M2 independently under the unchanged all-arm method; watch for active
paging and retain any failure. Do not extrapolate the w384 failure to a
smaller working set without measuring it. No pairwise substitution is made.

Final scope observation: w64l2 completed on M2, but the full setup/replay/cleanup
interval still recorded 7.90 GiB of swap-outs with about 257 MiB used swap at
the end. Those timings are retained with this qualification. No additional
M2 widths or per-call matrix were launched; further scaling awaits a revised
memory-bounded comparison. The laptop w64 series finished its required retry
at 06:19:45 UTC, then all measurement workers exited.
