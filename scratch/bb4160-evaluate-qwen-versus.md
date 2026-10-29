# Qwen versus MTG-183 with vLLM and explicit policy versions

Accepted direction: Jack Heart, 2026-10-28, in MTG-209's primary conversation.
Reconciled 2026-10-29 against checkout `41f33aae3938ece3cc96bcf78e66352d15aa4149`,
including merged main `6d069d6e5f0755b392c796aaf6d754913e8996c2`, and Etude's
incomplete retained snapshot observed at 06:20 UTC. The original directive is
preserved in this file at `02fccd93e1983319d355a1866c101d7d0e01565b`.

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

The surviving ownership boundaries are: `RuntimeIdentity.behavior_version`
owns hashing; `tokenization` owns pinned tokenizer resolution, EOS and budget
forcing; `SelectionResult.invalidate` revokes failed actions while retaining
evidence. The acknowledged service identity freezes the cohort before admission.
The historical report reader remains independent of executable admission and
preserves unknown splits/versions and failed-attempt denominators. One-shot
loading deadlines and resident per-decision deadlines remain distinct.

Earlier removals and their rationale are retained at
`5016ad1337c47f0854e0c0376b920c5d1bc5e1df:scratch/bb4160-evaluate-qwen-versus.md`.

## Remaining implementation and execution

### Delete — do not maintain

No remaining deletion targets. The spawn/Pipe and ProcessPoolExecutor paths
have been replaced by `ChildProcess` and bounded supervisor threads. Runtime
admission, reply bindings, deadlines, pending evidence and retirement remain on
that path; there is no compatibility executor to maintain.

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

1. **Scored budget cohort `cdb0c09f90c6`** (100 games, c=32, no MTP, $7 cap):
   still running at 2026-10-29 06:20:40 UTC. The retained log now has 49 admissions,
   24 completions, 2 failures and 23 unsettled admissions, with 51 unattempted.
   Admissions stopped on failure. Etude has no recorded output files yet;
   final game/replay receipts cannot be recovered before the existing run settles.
   No new rental, watcher or controller was started in this implementation pass.
2. **Accounting and cleanup:** direct RunPod bills total $1.47606228455 for
   the three stopped attempts, and all three pods are verified absent. The
   provider totals supersede the understated retained launcher costs. The active
   cohort has $1.47195755 whole-hour billing at the retained snapshot, with its
   partial hour, cleanup and ancillary costs still unknown. The narrowed provider
   receipt is published at `experiments/data/bb4160/accounting-20261029T0620.json`.
   Final inclusive costs and current allocation deletion remain outstanding.
3. **No-thinking floor:** reserving the current $7 commitment and $0.50 ancillary
   headroom leaves only $1.02393771545, below the constructor's $1.05 minimum.
   The floor is budget-deferred until settlement; no shorter bounded launch is
   qualified. `budget-64k` remains unlaunched, never pooled with the current arm.
4. **Process integration implemented:** game supervisors use bounded threads and
   `ChildProcess`; resident sessions and one-shot inference use the same child
   owner. Children avoid re-executing parent main, acknowledge game admission,
   reject wrong/late replies, retain pending evidence and reap on failures.
   Games retain the outer watchdog's process group for descendant cleanup.
   New receipts distinguish failed/canceled/unattempted games and retain IDs of
   interrupted admissions. This is review-branch code, not the launched source.
5. **Recovered evidence:** 64 budget-pilot/MTP game archives and their receipts
   are digest verified and manifest-published. Pilot: 95 initial, 3 repaired,
   0 fallback, 32 interrupted selections (130 files); MTP: 106 initial, 5 repaired,
   0 fallback, 32 failed (143 files), 0 completed games. Per-kind reasoning and
   token evidence are in each `receipt-accounting.json` and the protocol. The
   pilot vLLM log failed fetch identity validation and remains unverified.
   The offline reader now counts exactly the extra failed terminal selection
   left by historical close errors; frozen bytes are unchanged. Final no-MTP
   W/L/D, replay qualification and inclusive games/dollar still require its output.
   No PR is recorded; publication remains the caller's later boundary.

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
are proposals retained here, not accepted Wave-memory edits.

Service admission and completion checks are implemented with controlled fixtures;
the earlier mismatch fixture preserves all five IPC reply checks and wrong-admission
coverage. The fresh-process LLM-only reader import boundary passes for sampled
receipts. Gate still owns the actual mixed-report builder/Perf proof and broader
native replay checks. Current cohort settlement, final inclusive billing and allocation cleanup remain
pending; stopped-attempt provider billing and deletion are reconciled.

Check: `uv run --extra dev --extra artifacts pytest tests/llm/test_llm_runner.py tests/llm/test_llm_session.py tests/llm/test_llm_vllm.py tests/llm/test_llm_native_game.py -q` — 77 process/vLLM checks passed; all 6 native-game checks passed on focused rerun after rebuilding the stale local extension with `uv run --extra play maturin build --release -i .venv/bin/python -m managym/Cargo.toml -o managym/target/wheels` and atomic installation. Changed-file Ruff checks pass. Reused reporting results: failed-terminal/pending-work 5 passed, artifact-untracked passed, offline reporting-boundary 2 passed. Gate owns mixed-report builder/Perf, broader native replay and review checks; implement still owns cohort settlement, inclusive billing and allocation cleanup.
