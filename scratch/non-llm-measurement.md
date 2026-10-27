Jack Heart redirected this Task on 2026-10-27: move further Qwen inference into the existing MTG-191 follow-up and focus MTG-188 on measuring evaluation time for non-LLM models. This supersedes the earlier requirement to finish a Qwen 100-game campaign as part of MTG-188. Preserve all existing software, historical LLM receipts, failed attempts and publication history.

Problem
Evaluation time mixes process/model startup, policy inference, simulation, serving/IPC and evidence work. We need measured evidence to decide how our trained ActorCritic models should be evaluated on CPU or GPU, and where batching/residency helps.

Outcome
A reproducible non-LLM evaluation timing baseline, using existing evaluation/Perf owners and retained trained checkpoints, that explains single-game latency, many-game throughput and their costs. Report a slower GPU or ineffective batching as a valid result. Do not claim playing strength from speed.

Acceptance
- Pin representative trained checkpoint bytes, world/decks, deal and policy seeds, input shapes, software/native versions, hardware, precision and concurrency before each measurement. Start with admitted MTG-183 final raw u5950 bytes already obtained (SHA-256 4f2ad656ba036a33d527212a7ac2badf2c5b9db136916f730846b07eef2fbea4, 1,071,166 bytes); retain MTG-181 u616 as an additional reference where its exact bytes and world admission are available. Do not substitute random-weight fixtures for trained results.
- Separate process/model preparation and load from warm execution. Break evaluation time into policy encoding/inference/sampling, native simulation, serving/queue/IPC, replay/evidence and cleanup where observable; label combined/unavailable spans honestly. Keep overlapping worker/service intervals distinct from wall time.
- Compare ordinary scalar CPU, resident CPU batch-1 and existing shared/batched CPU evaluation with fixed-input witnesses and complete games; obtain equivalent CUDA measurements when authorized hardware is available. Preserve action masks, per-game RNG, checkpoint identity and replay. Use declared numerical tolerances for equivalent-execution comparisons and report action differences.
- Report per-decision latency distributions, full completed/failed/unattempted game counts, game throughput using actual allocation/attempt wall, startup amortization, memory and inclusive costs. Keep failures and all attempts. Bound local work using the existing recipe limits (one Torch thread/worker, at most four workers/rows, 120 seconds/game, 300 seconds/cell, 1,800 seconds/invocation and 8 GiB process-tree ceiling). Never induce artificial load.
- Produce the existing offline Perf report and reusable retained inputs; publish immutable evidence bytes before committing artifact manifests. Controlled tests verify interpretation, not empirical performance.

Current implementation and handoff
Existing checkout: /Users/jack/src/manabot.mtg-188-measure-inference-startup. Current PR #381: https://github.com/loopflowstudio/manabot/pull/381 . Receipt/reporting and bounded measurement infrastructure already exist; continue them without replaying completed Graphs. The earlier direct-session workaround for Loopflow settlement failures remains applicable. Shared model residency/batching ownership is MTG-189; ActorCritic optimizations are MTG-190. Measure their current paths here rather than implementing another serving framework or training changes. Open PR #383 (adaptive-curriculum Task 1d7884) changes simulation rollout/training and retains its own experiment choices.

Qwen boundary
MTG-191 now owns further Qwen reasoning/generation/backend diagnostics and long-running LLM experiments, coordinating MTG-192 for prefix caching. Preserve source through 6ea6e0bef45d and all seven MTG-188 campaign attempt identities in its handoff; no silent deletion of LLM support or evidence from this branch. Run 7b8c16de1b60 is being cancelled and its rental cleanup/partial outputs reconciled as part of this refocus. The prior $10 RunPod authorization was for the Qwen campaign; it is neither replenished nor automatically reassigned to non-LLM GPU experiments. Begin independent local CPU measurements; surface unavailable CUDA evidence before spending on a new allocation.

## GPU direction — 2026-10-27

Jack Heart subsequently requested: “do the inference perf testing on a gpu for
policy v policy batched inference.” This supersedes the CPU-first execution
plan and authorizes this finite GPU measurement within the original cumulative
$10 cap; it does not replenish that cap. All seven Qwen rentals have ended,
with $2.917705820125419603078713051 finalized rental charges. Preserve $0.50
for ancillary billing. The first GPU policy job reserves at most $1.00 and one
hour on the existing RTX A6000 supply; no concurrent allocation.

Both seats use admitted MTG-183 u5950 with independent player registrations and
policy RNG. Compare resident CUDA batch-1 against four concurrent games/rows:
three rotated repeats, one single game and two eight-game cells per repeat
(51 game opportunities), then the first 64 retained roots from a completed
many-game cell, two warmup passes and one measured pass per replay cell.
Each phase is bounded at 900 seconds, each cell 300 seconds, each game 120
seconds. Record actual batch distribution; four games is not a GPU saturation
claim. Preserve failures and unattempted reservations. No artificial load.

The CPU-versus-heuristic invocation was interrupted on this direction. Its first
cell completed one replay-verified native game (150 decisions), but collection
was interrupted, so the raw attempt ledger remains unsettled. Retain it under
`.runs/policy-baseline`; do not report it as a completed GPU measurement.

The A6000 quote reported no stock. The replacement is NVIDIA L4, available at
$0.59/hour, with the same $1.00 run ceiling. Dedicated supply
`mtg188-policy-l4` copies the existing L4 machine/image/storage settings and
limits this measurement to one rental. Prior Qwen pod absence was reverified
before creating the supply; creating a supply rents nothing.

The L4 quote subsequently lost capacity before launch (no rental). A fresh A6000
quote succeeded at $0.59/hour, at most $0.74 including setup allowance under the
$1.00 cohort ceiling. Run `4f8f34d188a7` launched from `bb47aafe0d3c` on
`mtg188-a6000-bbb813fb57a2`; first read confirms `setting_up`, one attempt.
The unused L4 supply has no rental. Observe only this run and the dedicated
`mtg188-a6000` supply; never tick unrelated Etude work.
