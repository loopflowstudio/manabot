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
`InferenceConfig` still permits only greedy
sampling. `CheckpointOpponent` retains a digest and validates admission.
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

## Accepted world decision (Jack Heart, 2026-10-28)

Retain the exact MTG-183 checkpoint and evaluate Qwen against it in a separately
pinned **historical w13** experiment: w13 source/native build, w13 decks,
policy-input v1, results labeled as w13 evidence. Current w14 admission stays
unchanged; no relabeling, forced admission or engine rollback on main. Next:
pin the w13 build for the campaign, repair the Etude SDK 0.2.10 launcher-database
incompatibility, then the offline contract checks, one real game, and bounded
scaling within the $10 budget as originally directed.

## Failed assumption and unresolved decision

The implementation preflight fetched and verified the requested checkpoint from
`r2://manabot/jobs/f3748bcabac8/checkpoints/learner_transitions-331180341.pt`.
It binds **w13**, policy-input v1. Current source/native builds **w14** and ordinary
`CheckpointOpponent` loading rejects it with
`checkpoint requires an explicit compatible w14 world binding`.
`validate_checkpoint_world` confirms that rejection is the admission contract.
There is no evidence of compatibility that would justify weakening it.

The accepted pair of requirements—this exact opponent and current-world
admission—cannot currently be met. **Unresolved for review:** retain the exact
MTG-183 checkpoint in a separately pinned historical w13 experiment, or select
an explicitly identified current-world opponent and revise the matchup. Neither
is accepted. Keeping the original requirements leaves paid execution pending;
no engine rollback, relabeling or substitute is assumed.

A separate launch prerequisite failed: pinned Etude SDK 0.2.10 refused the
existing launcher database as written by another SDK version. The database and
dependency pin were left unchanged. A compatible launch environment remains
necessary even after the scientific decision; fixing it cannot repair world
admission. This is retained implementation evidence, not a newly repeated probe.

Recorded Task GPU commitments/spend are $0. No allocation or game was started;
no scored cohort was frozen. There is consequently no 100-game denominator,
strength estimate, measured speedup or games/dollar result for this Task.

## Delete — do not maintain

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

1. **World and launch prerequisites:** an accepted matchup/world revision and
   compatible Etude launch path, preserving frozen evidence and other Tasks.
   Exact byte/native/world/deck admission must cover the declared schedule before
   rental. The current rejection happens before deck-pair qualification.
2. **Opponent and campaign identity:** service routing, acknowledgement and
   returned-model checking are implemented through the existing runtime identity.
   Game summaries bind both frozen candidates and the vLLM campaign stops on a
   departure from either, including its first game. The expected opponent digest
   remains the MTG-183 constant; only an accepted opponent revision may replace
   it. Historical-world selection would instead preserve that digest.
3. **A declared quality-focused pilot:** recover/verify the retained reasoning
   evidence and distinguish repetition, incomplete thinking and output limits.
   Explicit sampling, if selected, must reach both existing runtime implementations
   and receipts without changing historical defaults. Exact tokens, complete
   choices, repairs/fallbacks and replay remain acceptance requirements. Reusing
   the existing campaign avoids a second launcher or generic distributed rewrite.
4. **Controlled qualification, then real execution:** fixtures now cover version
   mismatch, stale/late replies, attempt-ceiling refusal, campaign expiry/stop and
   a complete native game/replay. Live Etude charge reconciliation and allocation
   termination remain unproven. Then one real development game after prerequisites;
   inspection of answer quality, replay, memory, elapsed time and inclusive cost
   before bounded concurrency increases. At most 100 scored games within the
   fresh $10 cap; unlike treatments are not pooled.
5. **Evidence and delivery:** reproducible commands, frozen protocols, artifact
   references, all outcomes and inclusive charges, verified allocation termination
   and a published reviewable change. Experiment bytes belong in the artifact
   store via `uv run --extra artifacts python -m manabot.data publish <path>`;
   manifests are committed and results remain untracked. Publication, landing
   and Task completion are not claimed here.

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
coverage. Gate still owns the fresh-process
LLM-only reader import boundary, actual mixed-report builder/Perf proof and native
replay checks. Real service/sampling, budget and campaign cleanup qualification
remain part of the unimplemented path above.

Check: `uv run --extra dev pytest tests/llm/test_llm_vllm.py -q` — 40 passed; focused ruff and `git diff --check` passed (2026-10-28). Prior unaffected session/selection checks remain applicable; gate owns broader acceptance. No live GPU/service proof; paid execution stays blocked on the unresolved world/opponent decision above.
