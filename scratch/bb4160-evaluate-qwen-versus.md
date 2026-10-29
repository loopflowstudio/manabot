# Qwen versus MTG-183 with vLLM and explicit policy versions

Accepted direction: Jack Heart, 2026-10-28, in MTG-209's primary conversation.
Reconciled 2026-10-29 against checkout `bbc06ce93fd780929b073b031f65046ee17e20ac`,
including merged main `6d069d6e5f0755b392c796aaf6d754913e8996c2`. The latest
retained log was collected at 07:25 UTC; the 07:27 scheduler read still reports
the cohort running on `cfcd947042c4`. Direct provider reads remain dated 07:14
UTC: the 07:27 retrieval reported missing `RUNPOD_API_KEY`. Final settlement is
not evidenced; this reconciliation verifies retained bytes, not later run status.
The original directive
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

## Implementation and ownership

- **Identity and admission:** `RuntimeIdentity.behavior_version` hashes the full
  inference configuration (including limits, sampling and fallback seed) and
  reported runtime/template. Hashing stays in the schema without training/native
  imports. The parent derives the version from the worker's reported identity;
  each request has a unique ID. Before forwarding progress, the session checks
  ID, version, configuration, runtime, decision and deadline. Failure retires
  the session; `SelectionResult.invalidate` revokes actions but retains evidence.
  Historical receipts without versions remain unknown.
- **Service:** the behavior hash is the sole served name, acknowledged through
  `/v1/models` at startup and client admission and checked on completions. Base
  argv participates in the hash; the derived name is excluded to avoid circularity.
  `vllm-command.json` retains the full argv. Startup reads the separately pinned
  tokenizer. This binds the supervised declaration, not measured GPU weight bytes.
- **Frozen players:** `CheckpointOpponent` verifies bytes and world admission.
  `GameSummary` retains `model_behavior_version` for resident sessions and
  `opponent_sha256` before opponent admission. Hosted/random/historical model
  versions remain absent. The campaign freezes the acknowledged server version
  before admitting games; a receipt cannot redefine it. Changed identities or
  played games missing either identity stop admissions and remain in results;
  failed preparation may omit identities. There is no session/campaign resume:
  a new Etude attempt re-runs admission from its frozen schedule.
- **Treatments and accounting:** inference files own sampling and token limits;
  `Treatment` adds only a declared context override. `runs()` resolves Campaign
  fields and sizes hours/watchdog from the cap. `_ADMISSION_RESERVE_SECONDS`
  owns the constructor/admission reserve. `tokenization` owns pinned tokenizer
  loading, EOS and budget forcing. Seeded sampling reaches both runtimes;
  Transformers rejects unsupported presence penalties. `_record_inference_usage`
  counts all attempts and inserted tokens; offline reports count generated tokens
  separately. `single_choice_decisions`, `skipped_trivial`, forced-close receipts
  and per-attempt `QWEN`/relayed `VLLM` logs support interpretation.
- **Processes:** bounded supervisor threads own one `ChildProcess` per game;
  resident sessions and one-shot inference reuse that child owner without
  re-importing parent main. Typed IPC leaves persisted evidence as JSON.
  Sequential admission acknowledgement, scheduled-game binding, hard deadlines,
  evidence retention and reaping remain required. One-shot loading deadlines
  differ from resident decision deadlines. The outer watchdog process group
  cleans descendants; interrupted admissions remain canceled, distinct from
  failures and unattempted games. `retain(admitting=game)` persists admission
  before submission; the futures map holds submitted work or a settled submission
  error, without placeholder reservations. Concurrency and stop-on-failure queues reuse
  the existing campaign. `PolicyKey` still owns structured-policy identity;
  no training or native execution changes are introduced.
- **Reports:** `_selection_statistics` assembles identity coverage, token counts
  and latency distributions for games and sequences. The historical reader
  stays independent of executable admission/native imports. Missing splits and
  versions remain unknown; failures stay in denominators. Exactly one failed,
  uncommitted terminal selection can exceed a historical close-error summary's
  counter, without implying success or rewriting evidence.

These review changes do not alter the running rental's source `cfcd947042c4`.
Native `skip_trivial` remains ON for every Qwen campaign (Jack Heart, 2026-10-28).

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
checkpoint bytes verified from R2, all 100 schedule entries passed preflight
admission checks (w13, policy-input v1, 196,482 parameters), `tests/llm` 129 passed.
That preflight is distinct from actual game admissions counted below. Launch commits
are cherry-picked from this branch onto the campaign branch; it is never merged.

Treatment and attempts live in `experiments/protocols/qwen-mtg183-bb4160.md`.

Recovered evidence is manifest-published under `experiments/data/bb4160/`:
the sampled pilot and all 64 budget-pilot/MTP archives. The protocol and each
`receipt-accounting.json` hold the detailed counts and per-kind reasoning.
The sampled pilot retained 38 initial, 2 repaired and 1 interrupted selection;
budget pilot: 95 initial, 3 repaired, 32 interrupted; MTP: 106 initial, 5 repaired,
32 failed. None used fallback or completed a game. The budget pilot's vLLM log
failed fetch identity validation and remains unverified. These attempts do not
qualify the pending no-MTP cohort.

Detailed implementation history remains at
`f01fd6a4554b2880b48c77b1b657ed040851313c:scratch/bb4160-evaluate-qwen-versus.md`;
earlier removal rationale is at
`5016ad1337c47f0854e0c0376b920c5d1bc5e1df:scratch/bb4160-evaluate-qwen-versus.md`.

## Execution outcome (settled 2026-10-29)

Jack Heart stopped paid execution at about 14:35 UTC; no further cohorts. The
outcome is `cdb0c09f90c6` (budget arm, 16k context): 49 admitted, 45 completed
and replay-verified, **Qwen 3–42**, 4 `ContextOverflow` failures, 51
unattempted. `81769c1a29c7` (budget-64k) was stopped by Jack Heart: 51
completed (Qwen 1–50), 30 canceled, 19 unattempted; never pooled. The
no-thinking floor never ran. Write-up, biases, encoding caveat (`manabot-text-2`
only; `jack/jev-encoding-w13` untested) and per-run provider bills ($6.98 billed,
plus 81769's final partial hour) are in `experiments/protocols/qwen-mtg183-bb4160.md`.
All five pods were absent at 14:47 UTC. Receipts and game archives are published
under `experiments/data/bb4160/` (cohort dirs `cdb0c09f90c6/`, `81769c1a29c7/`,
analysis scripts in `analysis/`); 81769's `vllm.log` failed Etude's range check.

## Remaining work

1. **Final billing:** reread 81769c1a29c7's last partial hour once the provider
   bills it; no other charge is outstanding beyond unknown ancillary storage.
2. **Delivery:** gate owns the mixed-report builder/Perf and broader replay and
   review checks; then publish the reviewable change. The w13 campaign branch
   `jack-heart/bb4160-w13-campaign` is local evidence of launched source and is
   never merged.

## Adopted and deferred ideas; related work

Jack Heart accepted Prime-style immutable identity, acknowledgement and rejection
for this frozen-player evaluation on 2026-10-28. Bounded handoffs, backpressure,
cancellation accounting and retirement belong to the existing owners where
needed. `scratch/prime-rl-assessment.md` retains the adopted/deferred rationale,
pinned sources and contrary historical evidence. No Prime installation, second
trainer, asynchronous PPO, stale GAE/per-seat learning or adaptive scored-game
dropping is authorized. Structured training provenance remains optional future
work; training and fine-tuning rentals are outside this budget.

Main's shared evaluation/serving and collection work (MTG-193) is now present at
`6d069d6e5f0755b392c796aaf6d754913e8996c2`, merged by `41f33aae3`; the former
open-overlap/base description is obsolete. It introduces the child-process owner
and preserves LLM-only offline reporting through lazy evaluation projection.
This Task adds no structured-policy training or shared serving changes.
Remote `main` was checked read-only on 2026-10-29 and still points to
`6d069d6e5`; no newer main changes were available. The last captured MTG-175 / PR #383 state
was open at `d0f48ca96` on 2026-10-28; its current delivery was not re-queried.
MTG-212 owns the separate Jev/OpenRouter
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
An affordable pilot configuration alone does not establish an affordable scored
cohort; both setup windows and the remaining cleanup reserve affect that decision.

Validation retained from preceding implementation: 128 vLLM/receipt tests and
the subsequent 64-test vLLM pass covering submission/child launch failures,
focused Ruff/compile checks, the sampled fresh-process import boundary and the
artifact-untracked test passed. The 07:27 audit verified prior snapshot digests/sizes,
the frozen schedule, 49-game admission prefix, disjoint 100-ID accounting and
stopped admissions. Documentation consolidation changes none of those bytes or
experiment outcomes; their receipt and report checks remain applicable.

Check: `uv run python -` (retained-evidence audit), `git diff --check`, `lf context` — passed: five manifest SHA256/size matches, disjoint 100-ID partition, unchanged 49-game admission prefix, five additional completions since 07:14, no admission after failure, and scratch within both limits. Gate retains mixed-report/Perf and broader review checks; settlement and floor qualification remain execution work.
