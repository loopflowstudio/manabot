# Qwen versus MTG-183 with vLLM and explicit policy versions

Accepted direction: Jack Heart, 2026-10-28, in MTG-209's primary conversation.
Reconciled 2026-10-28 against implementation commit
`02fccd93e1983319d355a1866c101d7d0e01565b` and the retained preflight result.
The original directive is preserved at that commit in this file.

## Accepted outcome and constraints

This Task owns implementation, testing and reasonable optimization of the actual
Qwen3.5-9B versus MTG-183 vLLM evaluation path, retained results and a reviewable
published change. One real complete game precedes a goal of up to 100 scored
games, if feasible within a fresh cumulative **$10 GPU budget**. Session-only
implementation or simulator-only proof does not satisfy that outcome. An honest
partial result and diagnosis are acceptable if the game goal is infeasible;
the world/opponent contract has not been relaxed.

GPU rental and iterative repairs are authorized within that ceiling, including
setup, warmup, failed attempts, storage/transfer and cleanup. Commitments include
cleanup headroom; charges and allocation termination require reconciliation.
One resident allocation and one GPU service are preferred. Further attempts
require a diagnosed/repaired cause and accounting for earlier commitments.
Historical canceled budgets remain closed. Artificial load/stress tests are
prohibited on every host. Fine-tuning and training rentals are outside this
budget; a verifiers-versus-custom fine-tuning trial is accepted future direction,
not execution here.

The immutable matchup inputs are:

- MTG-183 run `f3748bcabac8`, final raw update 5950, checkpoint SHA256
  `4f2ad656ba036a33d527212a7ac2badf2c5b9db136916f730846b07eef2fbea4`,
  1,071,166 bytes, 196,482 parameters.
- Qwen3.5-9B model/tokenizer revision
  `c202236235762e1c871ad0ccb60c8ee5ba337b9a`.
- Verified checkpoint bytes, authoritative selection and current native/world/deck
  admission before rental. Substitution or forced incompatible admission is not
  authorized.

Each scored cohort freezes seeds, decks, seats, both player identities,
source/native/runtime, renderer, sampling, limits and stopping policy. The
historical 100-game Elemental ordered-pair schedule is a starting point, not
permission to rewrite evidence. Development pilots and changed treatments remain
separate; failures are neither replaced nor excluded from reporting. Scheduled,
admitted, completed, failed, canceled and unattempted games have distinct counts.

The quality contract retains complete viewer-safe observations and legal choices,
native Command binding, stale rejection, EOS/thinking evidence and replay.
Reasoning remains enabled; prompt truncation, weight quantization or opponent
changes are not silent optimization options. Material quality tradeoffs require
an explicit treatment and, when consequential, an accepted decision. Sampling,
batching, caching and answer-completion experiments remain in scope. The known
failing greedy/short-output treatment is not a qualified default.

The report covers wins/losses/draws, replay coverage, initially valid answers,
repairs/fallbacks, startup-inclusive useful games/dollar, latency, token throughput,
whole-device memory, all attempt costs/failures and cleanup. Token throughput or
random-fallback completions alone cannot establish useful Qwen play.

## What exists

The schema/session/selection change replaces unbound LLM IPC on the existing
pipe. `RuntimeIdentity.behavior_version` hashes the full pinned inference
configuration, including limits and fallback seed, together with the reported
runtime/template/configuration. There is no separate configuration version.
The worker reports its runtime identity; the parent derives the version from it.
Requests carry a unique ID and progress echoes it. The parent checks ID, version,
configuration, runtime, decision and deadline before forwarding progress. Protocol failures
retire the session. Selection receipts retain the version, including when nested
in game evidence; historical receipts may omit it and retain an unknown version.
Canonical hashing remains local to the schema, without training/native imports.

The vLLM server now exposes the existing behavior hash as its sole served name.
Startup and client admission require `/v1/models` acknowledgement; requests use
that route and completions require the same returned model. Missing/mismatched
models and late replies fail without an action. The existing session retires on
runtime failure. Base argv participates in the hash; the derived name is excluded
to avoid circularity, and `vllm-command.json` retains the full launch command.
Startup reads the separately pinned tokenizer snapshot. This binds the supervised
service declaration, not independently measured GPU weight bytes.
`InferenceConfig` supports explicit seeded sampling as well as greedy decoding.
`CheckpointOpponent` retains a digest and validates admission.
`GameSummary` now binds both players: `model_behavior_version` (resident
sessions only; hosted/random/historical receipts leave it absent) and
`opponent_sha256`, recorded before opponent admission so failed admissions keep
it. The vLLM campaign freezes the cohort's model behavior version from the
acknowledged server identity before admission and stops admission, retaining the
game, when a receipt names a different behavior version or opponent digest.
Played games missing either identity also stop admission; failed preparation may
omit identities. The first game cannot establish or replace the expected version.
There is no resume path to re-check: each campaign is one Etude attempt and
sessions cannot resume; a new attempt re-runs admission from the frozen schedule. Native execution and training behavior are unchanged.

The existing campaign already has bounded game concurrency, retained admissions,
stop-on-failure queue handling and process cleanup. These are reuse points, not
newly completed behavior-version features. `PolicyKey` remains the structured
policy identity owner; no parallel trainer or mutable alias has been introduced.
No new producer/consumer framework is justified for this frozen-player evaluation.

## Accepted world decision and execution path (Jack Heart, 2026-10-28)

Retain the exact MTG-183 checkpoint and evaluate Qwen against it in a separately
pinned **historical w13** experiment; results are w13 evidence. Main keeps w14
admission (the preflight's `checkpoint requires an explicit compatible w14 world
binding` rejection still reproduces on this branch).

Implemented 2026-10-28: campaign branch `jack-heart/bb4160-w13-campaign` = this
branch plus `git revert e742e8bff` (the w14 commit; the only conflict was
scratch). Its engine/content source equals `f5e7289b1a6d`, which built the
reused Linux native input `1d1b674f…` (6,555,520 bytes). Locally on that tree:
checkpoint bytes verified from R2, all 100 scheduled games admitted (w13,
policy-input v1, 196,482 parameters), `tests/llm` 129 passed. Launch commits
are cherry-picked from this branch onto the campaign branch; it is never merged.
The Etude launcher database is schema 11 again, so SDK 0.2.10 launches.

Treatment and attempts live in `experiments/protocols/qwen-mtg183-bb4160.md`.
Seeded sampling (`InferenceConfig.decoding="sampled"` + `Sampling`) reaches
both runtimes; Transformers rejects a presence penalty rather than substituting.
`runs()` sizes the job's hours and watchdog from its dollar cap.

## Delete — do not maintain

Removed the report reader's greedy-only inference projection. The existing
historical reader now preserves sampling, thinking budgets and behavior versions;
there is no second campaign report implementation. Historical raw token counts
remain unchanged; generated totals exclude retained forced-close spans. Missing
split/version evidence stays unknown, with attempt denominators retained.

Removed the unversioned `config.model_id` HTTP/served-name route, completion
parsing that ignored model identity, and startup's use of the model snapshot
as the tokenizer authority. Existing consumers use the sole behavior-version
route; historical receipt reading remains unchanged.

`tokenization.load_tokenizer` owns pinned tokenizer resolution for both backends
and server startup. Its required offline flag preserves Transformers' acquisition
and vLLM's cache-only contract. `tokenization.eos_token_ids` replaces both copies
of decoder/chat terminator handling; the template stays a startup local instead
of duplicate runtime state. Preserve exact tokens and both Qwen EOS authorities.
`SelectionResult.invalidate` owns failure updates for selection and both
supervisors: retain evidence and revoke selection, origin and fallback together.
Selection cannot retain an action when publishing its evidence fails. Token
decoding counts closing thinking delimiters once, preserving unknown splits for
ambiguous output and all-reasoning counts for unfinished output.
Selection now publishes all terminal actions through the same guarded path;
failed fallback publication revokes the action just like an initial or repaired
answer. `_prepare_schedule` replaces `_schedule` and the duplicated schedule/index
writes, keeping retained inference, deal and game-index files together. Startup
alone prepares the cohort; the worker reads the saved schedule and configuration.
The worker's second generation/write path is removed so current source defaults
cannot redefine the prepared cohort. `_Choice.text` and `_Choice.stop_reason` are
removed: local decoding of retained token IDs owns answer text and EOS, while
the server's `finish_reason` must still agree with that EOS evidence.
No exclusive tests or fixtures remain to delete. Keep the one-shot runner's
loading-inclusive deadline separate from resident sessions' per-decision deadline.
The vLLM startup probe takes the interpreter prefix at `-m`, avoiding an argv
length dependency. `vllm_runtime.SERVICE_PORT`/`SERVICE_URL` own the loopback
address for server argv, health, admission and completions; the unused client
`port` parameter was removed.
Removed the session's admission and request version echoes: both sides derive
the version from the same identity and configuration, so those checks were
tautological. Reply version/config/runtime/decision checks remain authoritative.
Removed first-result initialization of the campaign's expected behavior version;
the acknowledged server identity is its sole source. Preserve raw game outcomes
and failure accounting when identity rejection stops further admissions.
Removed `_Ready`: the startup message is the `RuntimeIdentity` itself, which
already distinguishes successful admission from `_LoadFailure`. Canonical hashing
lives directly in `RuntimeIdentity.behavior_version`; the single-use hash helper
is gone and serialized hash inputs are unchanged. Campaign tests share one inline
executor stub and supply game outcomes through `play_game`, retaining their
admission and accounting assertions without separate pool implementations.

The full research audit lives at
`02fccd93e1983319d355a1866c101d7d0e01565b:scratch/prime-rl-assessment.md`;
its scratch summary retains applicable findings, contrary evidence and the
later accepted disposition.

## Remaining implementation and execution

Implemented and qualified: w13 campaign tree and admission; seeded sampling;
budget forcing (`thinking_budget`, `generate_response`, per-response
`forced_close_at`/`inserted_token_ids`, game `forced_closes`); per-attempt
`QWEN` and relayed `VLLM` stdout lines; `single_choice_decisions` and
`skipped_trivial` receipts; MTP draft as a hashed server option; treatment arms
in `runs()`. Attempts, charges and the frozen cohort are in
`experiments/protocols/qwen-mtg183-bb4160.md`.

1. **Scored budget cohort `cdb0c09f90c6`** (100 games, c=32, no MTP, $7.00 cap):
   still running at 2026-10-29 05:50 UTC, with admissions stopped after a failed
   game and admitted games settling. `8f68c0bc488b` finished with 32 failed games,
   68 unattempted and no completion; its cleanup receipt confirms deletion.
   The existing `budget-64k` declaration is a possible separate treatment, not
   a launched replacement. Final reconciliation precedes further commitments.
2. **No-thinking floor:** about 10 games, after the cohort, labeled, well under
   $1 (minimum job cap $1.05 for the 1.5-hour window).
3. **Evidence and delivery:** publish run outputs with
   `uv run --extra artifacts python -m manabot.data publish <path>`, commit the
   manifest, report W/L/D, replay, initial/repair/fallback/forced-close counts,
   reasoning by decision kind, latency, tokens/s, memory and inclusive cost.
   Note in the PR that skip_trivial was already enabled by the schedule.

Historical reporting is implemented for sampled and budget-forced receipts,
including generated reasoning length by retained continuation stage, inserted
token exclusion, mixed-treatment rejection and unknown split/version coverage.
Verified fetched pilot evidence and MTP failure summaries are published under
`experiments/data/bb4160/` through the manifest. The canceled sampled pilot has
41 selection files: 38 initial, 2 repaired and 1 interrupted, not 41 completed
selections. Its 43 attempts contain 42 responses and no fallback.

## Adopted and deferred ideas; related work

Jack Heart accepted Prime-style behavior accounting for this evaluation on
2026-10-28. Explicit immutable identity, acknowledgement and rejection fit the
existing session and frozen-policy owners. Bounded handoffs, backpressure,
cancellation accounting and snapshot retirement are adopted only where the
existing path needs them; a scored game always keeps frozen candidates.

Structured training batch provenance, collection-time probabilities/values and
learner-update identity are optional future work preserving synchronous rules.
Asynchronous PPO, stale-data learning, changed GAE/per-seat credit and adaptive
scored-game dropping are not authorized consequences of this evaluation.
Prime installation and a second trainer are unnecessary. The inspected Prime
revision `6a32e72dc0d3e739cf1d676d8125fb7575125683` uses verifiers gitlink
`60290d436fcb4c2ad6dd7d0986bcf00ff2366031`, `verifiers.v1` and vLLM 0.32.0;
this campaign retains vLLM 0.17.1. The assessment summary remains in
`scratch/prime-rl-assessment.md`; these are pinned research facts, not a fresh
upstream audit or permission to mix API families.

Historical treatments remain separate: approximately 129 aggregate output
tokens/s at four requests versus 34 at one completed no games; the later
2,048-token treatment completed/replayed 18 games with 519 random fallbacks
among 793 selections; the 16,384-token treatment completed none before
cancellation. Raw receipts were unavailable in the research checkout. These
retained reports do not establish a matched backend ranking or healthy reasoning
configuration, and new claims require verified evidence.

On 2026-10-28, GitHub still reports PR #389 / MTG-193 open at `064d2395c`
(shared evaluation/serving and collection) and PR #383 / MTG-175 open at
`d0f48ca96` (curriculum/learning). Main remains `c5d406e3d`, this Task's base.
Their shared-code decisions remain reserved; this work changes no structured-policy
training or shared serving implementation. MTG-212 owns the separate Jev/OpenRouter
comparison. The accepted directive includes MTG-210's relevant version-accounting
scope here and leaves MTG-211's controlled verifiers proposal for later.
MTG-191/192 remain canceled historical campaigns; no competing Task or execution
is selected by these references.

## Wave context and validation limits

The supplied training chapter seeks repeatable playable checkpoints, full-game
improvement and repeatable human wins, with frozen identities and complete attempt
accounting. A session protocol proof satisfies none of those scientific outcomes.
The selected `.lf/wave/training/` directory, GOAL.md, MEMORY.md and child memories
are absent from this checkout; the Wave API reports an empty goal. Thus memory
coverage and size cannot be certified, and no substitute Wave was chosen or
memory created. Scratch is within both configured limits.

Proposed durable lessons for that missing memory: admission must precede rental;
a version hash binds the fields it covers, not a remote server's loaded weights;
and replayed fallback-heavy games cannot qualify useful Qwen reasoning. These
are proposals retained here, not accepted Wave-memory edits.

Service admission and completion checks are implemented with controlled fixtures;
the earlier mismatch fixture preserves all five IPC reply checks and wrong-admission
coverage. The fresh-process LLM-only reader import boundary passes for sampled
receipts. Gate still owns the actual mixed-report builder/Perf proof and broader
native replay checks. Current cohort settlement, budget reconciliation and final
allocation cleanup remain pending.

Check: `uv run --extra dev --extra reports pytest tests/eval/test_inference_receipts.py -q -k 'sampled_report or sampled_sequence or cli_never_imports'` — 7 passed (2026-10-29); retained sampled-pilot projection succeeds. Earlier runtime qualification: w13 `tests/llm` 129 passed, this branch vLLM 42 / Transformers 2 passed. Gate owns broader acceptance.
