# Qwen versus MTG-183 with vLLM and explicit policy versions

Accepted direction: Jack Heart, 2026-10-28, in MTG-209's primary conversation.
Reconciled 2026-10-29 against checkout `39573ed973a9be766bf9eddc63f465d45ea2b39a`,
including merged main `6d069d6e5f0755b392c796aaf6d754913e8996c2`, and Etude's
retained log and direct provider reads refreshed at 06:40 UTC.
The cohort remains live; final settlement is not yet available. The original directive
is preserved in this file at `02fccd93e1983319d355a1866c101d7d0e01565b`.

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
Reasoning remains enabled in the main arm; Jack Heart accepted a separate small
no-thinking floor on 2026-10-28. Prompt truncation, weight quantization or opponent
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

Implemented 2026-10-28: campaign branch `jack-heart/bb4160-w13-campaign` was
derived from this Task before the main sync, with `git revert e742e8bff` (w14).
The launched no-MTP run pins source `cfcd947042c4`; its engine/content source
equals `f5e7289b1a6d`, which built the
reused Linux native input `1d1b674f…` (6,555,520 bytes). Locally on that tree:
checkpoint bytes verified from R2, all 100 scheduled games admitted (w13,
policy-input v1, 196,482 parameters), `tests/llm` 129 passed. Launch commits
are cherry-picked from this branch onto the campaign branch; it is never merged.

Treatment and attempts live in `experiments/protocols/qwen-mtg183-bb4160.md`.
Seeded sampling (`InferenceConfig.decoding="sampled"` + `Sampling`) reaches
both runtimes; Transformers rejects a presence penalty rather than substituting.
`runs()` sizes the job's hours and watchdog from its dollar cap.

## Ownership and completed simplification

The campaign treatment table no longer repeats the inference files' output and
context limits. `Treatment` names the file and any explicit context override;
`runs()` resolves those into the existing Campaign fields before schedule
preparation. The saved schedule/configuration remain the worker's authority.
The four declared arms retain their existing effective values. This removes
duplicated limit ownership without changing a frozen treatment.
`_ADMISSION_RESERVE_SECONDS` owns the shared reserve used by the job constructor,
queue admission and game preparation. Per-attempt token/timing totals and live
diagnostics live in `_record_inference_usage`, beside hosted-usage accounting,
so the game loop owns action execution and replay. Receipt counters and log
fields retain their meanings, including harness-inserted tokens.

The surviving ownership boundaries are: `RuntimeIdentity.behavior_version`
owns hashing; `tokenization` owns pinned tokenizer resolution, EOS and budget
forcing; `SelectionResult.invalidate` revokes failed actions while retaining
evidence. The acknowledged service identity freezes the cohort before admission.
The historical report reader remains independent of executable admission and
preserves unknown splits/versions and failed-attempt denominators. One-shot
loading deadlines and resident per-decision deadlines remain distinct.

Earlier removals and their rationale are retained at
`5016ad1337c47f0854e0c0376b920c5d1bc5e1df:scratch/bb4160-evaluate-qwen-versus.md`.

## Completed process integration and recovered evidence

Game supervisors use bounded threads and `ChildProcess`; resident sessions and
one-shot inference use the same child owner. Children avoid re-executing parent
main, acknowledge game admission, reject wrong/late replies, retain pending
evidence and reap on failures. Games inherit the outer watchdog's process group
for descendant cleanup. Campaign receipts distinguish failed, canceled and
unattempted games and retain IDs of interrupted admissions. These changes are
review-branch code, not the launched source.

Internal one-shot and game replies now use typed records on the existing
Connection, as resident-session replies already do. Redundant JSON transport
and the game supervisor's two-state loop are removed; persisted receipts remain
JSON. The game handshake is sequential, with unchanged admission acknowledgement,
scheduled-game binding and hard deadline. Lifecycle tests share lightweight child
fixtures instead of importing the native test module in an inference child,
and verify reaping for one-shot success, timeout and exit.

Implemented: w13 campaign tree and admission; seeded sampling;
budget forcing (`thinking_budget`, `generate_response`, per-response
`forced_close_at`/`inserted_token_ids`, game `forced_closes`); per-attempt
`QWEN` and relayed `VLLM` stdout lines; `single_choice_decisions` and
`skipped_trivial` receipts; MTP draft as a hashed server option; treatment arms
in `runs()`. Attempts, charges and the frozen cohort are in
`experiments/protocols/qwen-mtg183-bb4160.md`.

**Recovered evidence:** 64 budget-pilot/MTP game archives and their receipts
are digest verified and manifest-published. Pilot: 95 initial, 3 repaired,
0 fallback, 32 interrupted selections (130 files); MTP: 106 initial, 5 repaired,
0 fallback, 32 failed (143 files), 0 completed games. Per-kind reasoning and
token evidence are in each `receipt-accounting.json` and the protocol. The
pilot vLLM log failed fetch identity validation and remains unverified.
The offline reader now counts exactly the extra failed terminal selection
left by historical close errors; frozen bytes are unchanged. Final no-MTP
qualification remains separate from these recovered attempts.

Historical reporting is implemented for sampled and budget-forced receipts,
including generated reasoning length by retained continuation stage, inserted
token exclusion, mixed-treatment rejection and unknown split/version coverage.
Verified fetched pilot evidence and MTP failure summaries are published under
`experiments/data/bb4160/` through the manifest. The canceled sampled pilot has
41 selection files: 38 initial, 2 repaired and 1 interrupted, not 41 completed
selections. Its 43 attempts contain 42 responses and no fallback.

## Delete — do not maintain

Completed: remove multiprocessing spawn / ProcessPoolExecutor launch paths and
unbound IPC, preserving admission, identity, deadlines, failure evidence and reap
coverage on `ChildProcess`. No alternate executor remains.

Completed: remove `runs()`'s fixed 1.5-hour minimum. The surviving check requires
setup/delivery reserves plus a complete decision admission window. No remaining
deletion targets; frozen receipts, source trees, models and worlds are retained.

## Remaining execution and qualification

1. **Scored budget cohort `cdb0c09f90c6`** remains running. The refreshed retained
   log at 2026-10-29 06:43 UTC has 49 admissions, 30 completions, 3 failures and
   16 unsettled games; 51 schedule entries remain unattempted. No admissions
   follow the first failure. Etude retains zero output files. The 30-second log
   refresh remains partial evidence, not final outcomes.
   Preserve the frozen settle-admitted-games rule. Final summaries, W/L/D and
   retained replay coverage cannot be qualified from these logs.
2. **Accounting and cleanup:** direct provider reads still confirm all three
   ended pods absent and $1.47606228455 billed for them. Current pod is present;
   its whole-hour bills now total $1.71744000551. Partial-hour and ancillary
   charges are unknown. Published `accounting-20261029T064055Z.json` retains
   provider bills, rental deadline, output count and transfer-coverage gaps,
   alongside the associated log under `experiments/data/bb4160/`. The active
   rental deadline is 14:04 UTC; no new rental, controller or watcher was started.
   Final inclusive accounting and current pod deletion remain outstanding.
3. **No-thinking floor:** a bounded $0.80 pilot is now constructible with
   `runs(..., arm="nothink", games=1, concurrency=1, attempt_usd="0.80",
   decision_seconds=300)`. It preserves 15 minutes of machine acquisition and
   45 minutes of setup/delivery reserves, leaving 20 minutes of gameplay.
   The deadline reaches the saved inference configuration and behavior identity.
   Configuration checks pass, but no GPU pilot or scored floor has run. This
   is a declared new deadline, not qualification from the thinking arm. The
   $1.02393771545 uncommitted balance covers this pilot alone; the requested
   settlement/billing ordering still prevents a new commitment now. After
   settlement, qualify the pilot before freezing an affordable floor cohort;
   final budget-limited infeasibility has not been established.
4. **Late-game context fit:** short early-game pilots did not qualify full-game
   16k context. Final receipts are still required to classify every failure.
   `budget-64k` remains unlaunched and unqualified; it does not replace the
   frozen cohort's evidence.
5. **Final outcome and review:** final outcome accounting, retained replay,
   startup-inclusive useful games/dollar and cleanup remain execution work.
   Gate owns mixed-report builder/Perf and broader replay/review checks.
   Publication remains the caller's boundary; no PR is recorded.

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

Main's shared evaluation/serving and collection work (MTG-193) is now present at
`6d069d6e5f0755b392c796aaf6d754913e8996c2`, merged by `41f33aae3`; the former
open-overlap/base description is obsolete. It introduces the child-process owner
and preserves LLM-only offline reporting through lazy evaluation projection.
This Task adds no structured-policy training or shared serving changes.
The last captured MTG-175 / PR #383 state was open at `d0f48ca96` on 2026-10-28;
its current delivery was not re-queried. MTG-212 owns the separate Jev/OpenRouter
comparison. The accepted directive includes MTG-210's relevant version-accounting
scope here and leaves MTG-211's controlled verifiers proposal for later.
MTG-191/192 remain canceled historical campaigns; no competing Task or execution
is selected by these references.

## Wave context and validation limits

The supplied training chapter seeks repeatable playable checkpoints, full-game
improvement and repeatable human wins, with frozen identities and complete attempt
accounting. A session protocol proof satisfies none of those scientific outcomes.
The selected `.lf/wave/training/` directory, GOAL.md, MEMORY.md and child memories
are absent from this checkout; the earlier Wave API read reported an empty goal.
Thus memory coverage and size cannot be certified, and no substitute Wave was chosen or
memory created. Scratch is within both configured limits.

Proposed durable lessons for that missing memory: admission must precede rental;
a version hash binds the fields it covers, not a remote server's loaded weights;
and replayed fallback-heavy games cannot qualify useful Qwen reasoning. These
are proposals retained here, not accepted Wave-memory edits. Further proposed
lessons: short serving pilots do not qualify late-game context fit or mixed
speculative batches; provider billing can contradict launcher estimates; process
integration on a review branch does not change a running attempt's frozen source.

Service admission and completion checks are implemented with controlled fixtures;
the earlier mismatch fixture preserves all five IPC reply checks and wrong-admission
coverage. The fresh-process LLM-only reader import boundary passes for sampled
receipts. Gate still owns the actual mixed-report builder/Perf proof and broader
native replay checks. Current cohort settlement, final inclusive billing and allocation cleanup remain
pending; stopped-attempt provider billing and deletion are reconciled.

Check: `uv run --extra dev --extra artifacts pytest tests/llm/test_llm_vllm.py tests/llm/test_llm_native_game.py -q` — 69 passed; Ruff and `git diff --check` passed. Gate owns mixed-report/Perf and broader replay/review; implement owns cohort settlement, inclusive accounting, cleanup and no-thinking qualification.
