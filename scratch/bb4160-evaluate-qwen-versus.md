# Qwen versus MTG-183 with vLLM and explicit policy versions

Requested by Jack Heart, 2026-10-28, in MTG-209's primary conversation. Create one Task and launch now. Prototype, test and reasonably optimize vLLM evaluation of Qwen against MTG-183, beginning with one game and aiming for up to 100 games if feasible within a new $10 GPU budget. Use this work to introduce Prime-style policy-version accounting and other clearly beneficial operational patterns. A verifiers-versus-custom fine-tuning trial is accepted future direction, not execution in this Task.

## Outcome and execution authority

Implement and exercise the actual evaluation path, retain results and publish a reviewable change. Do not stop at a design or simulator-only proof. This request authorizes GPU rental and iterative repairs/optimizations within a fresh cumulative $10 ceiling for this Task; no further permission is needed for those steps. Previous canceled campaigns remain historical and their budgets are neither reused nor reopened. Include setup, warmup, failed attempts, storage/transfer and cleanup in commitments, reserve cleanup headroom, and stop before exceeding $10. Reconcile charges and verify allocation termination. Prefer one resident allocation and one GPU service; retry only after diagnosing/repairing a cause and accounting for previous commitments.

Start with controlled offline contract checks, then one real complete Qwen-versus-MTG-183 game. Inspect answer completion, repairs/fallbacks, replay, memory, elapsed time and cost before scaling concurrency in bounded stages. Target at most 100 scored games; reaching 100 is a goal, not permission to overrun budget or substitute random fallback for useful Qwen play. If infeasible, preserve the honest partial outcome and diagnosis. Real task execution is authorized; artificial load/stress generation is prohibited on every host.

## Opponent, model and scientific contract

Use the existing Qwen3.5-9B path and the MTG-183 policy. Historical pinned opponent: run f3748bcabac8, final raw update 5950, checkpoint SHA256 4f2ad656ba036a33d527212a7ac2badf2c5b9db136916f730846b07eef2fbea4 (1,071,166 bytes; 196,482 parameters). Qwen model/tokenizer revision c202236235762e1c871ad0ccb60c8ee5ba337b9a. Verify bytes, current native/world/deck admission and the authoritative MTG-183 checkpoint selection before renting; never silently substitute a candidate or force incompatible admission.

Inspect experiments/qwen_mtg183_vllm.py, experiments/qwen_mtg183.py, experiments/protocols/qwen-mtg183-188.md and existing llm/session, vllm_runtime, vllm_server, sim/serving and experiment/Etude owners. Reuse their repaired runtime rather than recreating infrastructure. Start from the retained reasoning treatment and investigate its failure causes; do not repeat the known failing greedy/short-output setup blindly. Reasonable serving, batching, sampling, cache and answer-completion experiments are in scope, but declare each treatment before running it. Preserve complete viewer-safe observations/legal choices, native Command binding, stale rejection, EOS/thinking evidence and replay. Do not silently disable thinking, truncate prompts, quantize weights or change opponents to inflate game count. Material quality tradeoffs must be explicit in the report; ask for consequential judgment if required.

Freeze seeds, decks, seat assignments, opponent/model digests, source/native/runtime, renderer, sampling, limits and stopping policy before each scored cohort. Separate development pilots and changed treatments; no pooling unlike results, success-only reporting or replacements hiding failures. The previous 100-game Elemental ordered-pair schedule is a useful starting point, not authority to rewrite old evidence. Account independently for scheduled, admitted, completed, failed, canceled and unattempted games. Report wins/losses/draws, replay coverage, valid initial answers, repairs, fallbacks, startup-inclusive useful games/dollar, latency, token throughput and whole-device memory. Token throughput alone is not success.

## Prime-inspired improvements

Implement immutable behavior/model version identities propagated through admission, requests, replies, game/rollout receipts and recoverable state where applicable. Bind checkpoint digest, model/tokenizer and inference configuration to the version; reject mismatched or late responses. A scored game always evaluates frozen candidates. Preserve one owner for each identity, extending existing PolicyKey and training snapshot contracts rather than inventing mutable aliases.

Adopt bounded producer/consumer handoffs, explicit admission acknowledgement and snapshot retirement, backpressure and cancellation accounting when they simplify or measurably improve this path. For structured-policy training, explicit batch provenance, collection-time probabilities/values and learner-update identity may be added with controlled tests while preserving the synchronous algorithm. Asynchronous PPO, stale-data learning, changed GAE/per-seat credit and adaptive dropping of scored games are not automatically improvements and are not implied by this request. Keep a concise adopted/deferred rationale. No second trainer or generic distributed rewrite; no Prime install is required to serve Qwen.

## Evidence and overlap

MTG-209 reviewed Prime RL 6a32e72dc0d3e739cf1d676d8125fb7575125683 and its verifiers gitlink 60290d436fcb4c2ad6dd7d0986bcf00ff2366031. Current upstream uses verifiers.v1 and a newer vLLM than our pinned 0.17.1; do not accidentally mix API families. Full assessment is handed into this Task's scratch before execution.

Historical evidence: vLLM achieved about 129 aggregate output tokens/s at four requests versus 34 at one, but that attempt completed no games. A later 2,048-token attempt completed/replayed 18 games with 519 random fallbacks among 793 selections. A 16,384-token attempt completed no games before cancellation. These are distinct treatments, not a matched backend benchmark. Raw receipts were unavailable in the research checkout and require recovery/verification for new claims. MTG-191/192 are canceled; this new Task explicitly owns the newly authorized evaluation continuation, without restarting them.

Open overlap at capture: PR #389 / MTG-193 (task_19d240c18e2c4ec7b8f45d00dbb5ae9f) changes shared evaluation batches, serving protocol, Torch-free workers, collection and execution. PR #383 / MTG-175 changes curriculum execution, collection and learning rules. Preserve their work and reserved decisions; inspect current delivery before editing common code. MTG-212 separately evaluates MTG-183 against Jev via OpenRouter; do not interfere with its run, budget or treatment. MTG-210 is an unstarted behavior-version design proposal: its relevant implementation scope is now included here, so do not launch a competing effort. MTG-211 remains an unstarted controlled verifiers adapter proposal for the later fine-tuning comparison. No additional Tasks are requested now.

## Validation and delivery

Read AGENTS.md, TESTING.md and docs/development/validation.md. All Python commands use uv. Test version mismatch/stale replies, budget exhaustion, cancellation/cleanup and complete native game/replay through controlled fixtures; then verify the real one-game path before scaling. Preserve frozen evidence, publish experiment bytes via the artifact store, commit manifests and keep results untracked as required. Deliver reproducible commands, pinned protocols, artifact references, all attempt costs/failures and a clear outcome for the up-to-100-game goal. No fine-tuning or training rental is authorized under this budget.

## Implementation plan — 2026-10-28

The accepted directive above is the delivery boundary. Extend the current LLM
schema/session and exact-token vLLM owners with immutable behavior identity,
request binding and explicit admission. Preserve PolicyKey as the structured
policy authority; no training overlap or learning-rule changes are needed for
this frozen evaluation. PRs #389 and #383 remain open at inspection.

Delete — do not maintain:
- Model-name-only vLLM routing; bind the served name to the immutable inference
  configuration and validate the returned model before accepting tokens.
- Implicit greedy-only sampling configuration, if the pilot requires sampling;
  carry explicit settings through both existing runtimes and evidence.

Preserve historical receipt parsing, exact tokens/EOS, thinking completion,
complete choices, repair/fallback evidence, deadlines and native replay.
Preflight checkpoint bytes and all 100 deck-pair admissions before renting.
Then declare a separate one-game development treatment, inspect completion and
quality, and scale only within a reconciled fresh $10 cumulative budget.
Optional training provenance/async work and Prime installation are deferred.

## Implementation result and revised boundary — 2026-10-28

**Blocked before paid execution.** Verified the exact requested R2 checkpoint
bytes; its world is w13 and policy-input version 1. Freshly built current source
reports w14. Ordinary `CheckpointOpponent` admission rejects it before weights
are admitted. This is a counterexample to the directive's current-world matchup,
not permission to relabel the checkpoint, weaken admission or substitute a
historical engine. A design decision selecting historical w13 execution or a
new opponent remains necessary. The exact failure and artifact are in
`scratch/questions.md`. GPU commitments/spend: $0; no rental, no games started,
no running child processes. No 100-game cohort or scientific outcome is claimed.

The independent completed cut replaces unbound LLM IPC on the existing pipe:
InferenceConfig hashes pinned model/tokenizer/config; RuntimeIdentity binds that
to effective runtime/template; selection receipts retain the behavior version.
The worker acknowledges admission, each request has a unique ID and expected
version, and progress echoes the request ID. The parent validates version,
configuration, runtime, decision and deadline before publishing any action.
Failures retire the session. Historical receipts may omit version identity;
readers do not invent one. Native execution and training are unchanged.

Canonical hashing has one local helper, preserving the existing digest bytes
without importing infra's training/runtime dependencies. Fault-injection peers
receive their scenario explicitly instead of repurposing model identity. The
unbound IPC deletion is complete; the remaining targets above belong to the
unfinished evaluation path, not this session-only cut.

Check: `uv run --no-sync pytest tests/llm/test_llm_session.py tests/llm/test_llm_selection.py -q` — 44 passed; changed-file Ruff check/format and `git diff --check` passed. Gate owns broader offline-reader and native replay checks. `lf context` confirms scratch and supplied wave memory fit both limits.

Remaining delete-list cuts: vLLM model-name routing and explicit sampling remain
unmodified. Remaining accepted outcome: resolve world/runtime launch admission,
finish server/opponent version propagation, declare a quality-focused pilot,
execute and inspect one real game, scale if feasible within the new $10 cap,
retain/publish all outcomes and inclusive charges, and verify rental termination.
Etude 0.2.10 also refuses the existing launcher database as written by another
SDK version; that shared database was left intact. No publication or landing
was attempted by this implementation step.
