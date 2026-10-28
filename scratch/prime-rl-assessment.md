# MTG-209 — Prime RL, manabot, and Qwen serving

Research · 2026-10-28 · Recommendation, not an implementation decision.

## Answer

Keep manabot's structured-policy trainer. Borrow explicit behavior-version
accounting and bounded producer/consumer interfaces before considering asynchronous
learning. Prime RL is a credible candidate for a **separate LLM fine-tuning path**,
but wrapping managym does not replace our native admission, replay, evaluation or
cost-accounting contracts. First prove the adapter with controlled responses.

Use the existing **vLLM service as the preferred direction for sustained,
concurrent Qwen GPU evaluation**, retaining Transformers as the reference and
small sequential path. Do **not** change the default or claim a speed/cost win yet.
The reasoning campaign demonstrates batching and substantial answer-completion
problems, not a matched comparison with Transformers. OpenRouter is useful for
hosted-model comparisons, but cannot stand in for an immutable local Qwen BF16
checkpoint. No new execution or spending is proposed as part of this research.

These answers separate three decisions: serving an unchanged Qwen model,
fine-tuning its language-model weights, and training our small actor/critic.
Installing Prime RL is necessary for none of the current inference paths.

## Scope and source identity

Local code inspected at `c5d406e3d2a701525c15c8500a30f7da7896cc40`.
Primary upstream repositories were fetched read-only, without installing or
executing their code:

- [Prime RL `6a32e72dc0d3e739cf1d676d8125fb7575125683`](https://github.com/PrimeIntellect-ai/prime-rl/tree/6a32e72dc0d3e739cf1d676d8125fb7575125683).
- [verifiers `60290d436fcb4c2ad6dd7d0986bcf00ff2366031`](https://github.com/PrimeIntellect-ai/verifiers/tree/60290d436fcb4c2ad6dd7d0986bcf00ff2366031), exactly Prime RL's `deps/verifiers` gitlink.
- Prime RL also pins renderers to `b6ab8c7d67229be6fb99d57a8395f978d94b0566`.
  That renderer's implementation was not independently audited here; exact Qwen
  template equivalence remains an adapter acceptance requirement.

The source has moved beyond the question's older API description: current Prime
RL consumes `verifiers.v1`, with `Taskset`, `Env`, `Agent`, `Episode` and message
graph `Trace` objects. It uses renderer-owned token input/output for training;
OpenAI-compatible endpoints remain available. Do not implement against an old
`MultiTurnEnv` tutorial without deliberately pinning the corresponding release.
Current Prime RL's Linux dependency is vLLM 0.32.0, whereas our campaign pins
0.17.1. Adopting Prime is therefore also a runtime migration, not simply reusing
our server environment. [Dependency declaration](https://github.com/PrimeIntellect-ai/prime-rl/blob/6a32e72dc0d3e739cf1d676d8125fb7575125683/pyproject.toml),
[verifiers environment API](https://github.com/PrimeIntellect-ai/verifiers/blob/60290d436fcb4c2ad6dd7d0986bcf00ff2366031/verifiers/v1/env.py).

No benchmark, model inference, paid API call, GPU rental, training run, or
upstream test suite ran. Historical measurements below are retained reports,
not remeasurements. The MTG-185 raw corpus is not present at its manifest path
in this checkout; the manifest binds bytes but does not supply them.

## System map

```text
EXISTING MANABOT TRAINING
TrainingRegime → execute_regime
                  ├─ SeatRoutedCollector → native vector games
                  │    frozen behavior → per-seat RolloutBatch
                  └─ Learner + learning rule → Adam → EMA/checkpoint/recovery
                       collection and learning occur in sequence

EXISTING MANABOT EVALUATION / LLM PLAY
admitted world + seed + player identity → isolated native game
  ├─ structured observation → shared frozen-policy service → action index
  └─ captured viewer-safe text → LLM session → validated legal reference
               ↓ native-bound Command → transition → replay + evidence

UPSTREAM PRIME RL
tasksets → isolated environment workers ↔ inference fleet (vLLM)
                         ↓ Episode/Trace, sampled tokens + logprobs
orchestrator → group credit / staleness masks / packing → trainer (FSDP2)
      ↑                  ZMQ or filesystem                     │
      └──── weight watcher / handshake ← versioned weights ────┘
                 NCCL / filesystem / optional NIXL

PROPOSED LLM ADAPTER BOUNDARY
Prime Env → manabot capture/selection/Command/replay owners → managym
        ↘ token traces for training; immutable native receipts for evaluation
```

### 1. What Prime does differently

**Process ownership and asynchrony.** Prime separates inference, orchestration
and training. The CPU orchestrator dispatches environment episodes while the
trainer optimizes previously sampled tokens. Rollouts can span several weight
versions. Our `execute_regime` calls collection and then `update_iteration`;
both-seat collection finishes games and keeps trajectories separate by player.
Its no-banked-action update boundary is deliberate, not an incidental lack of
parallelism. Native producer processes already exist, so “add workers” is not
the missing concept. [Prime architecture](https://github.com/PrimeIntellect-ai/prime-rl/blob/6a32e72dc0d3e739cf1d676d8125fb7575125683/docs/overview.md),
[execution](../manabot/training/execution.py),
[collector](../manabot/training/rollout/collection.py),
[producers](../manabot/training/rollout/producers.py).

**Staleness is data.** At trainer step `n`, a token sampled under policy `k`
has staleness `(n - 1) - k`. Prime records token policy versions, falling back
to dispatch version when unavailable. `TrainSink._drop_stale` masks old tokens
and drops samples with no remaining loss. The current default bound is eight;
staleness alone does not cancel a rollout. This is materially different from
keeping one immutable policy throughout a game. Importance weighting and
staleness diagnostics address distribution mismatch; they do not make the
rollout on-policy or prove unchanged learning quality.
[Algorithms](https://github.com/PrimeIntellect-ai/prime-rl/blob/6a32e72dc0d3e739cf1d676d8125fb7575125683/docs/algorithms.md),
[actual admission/masking](https://github.com/PrimeIntellect-ai/prime-rl/blob/6a32e72dc0d3e739cf1d676d8125fb7575125683/src/prime_rl/orchestrator/train_sink.py).

**The estimator differs too.** Current Prime defaults combine group-relative
credit with IPO, an importance-weighted gradient that masks absolute probability
changes outside a trust region. IcePop ratio masking is another supported rule;
it is not the default in this revision. Manabot has actor/critic GAE and its
explicit PPO/damped-policy rules. Prime's causal-token losses and group reward
normalization are not replacements for per-seat returns. In particular, a
same-prompt group with identical rewards supplies no relative signal; this
matters for sparse game wins. Prime also has a self-play RAE algorithm with
per-agent baselines, so “Prime cannot do adversarial play” would be false.
[Loss implementation](https://github.com/PrimeIntellect-ai/prime-rl/blob/6a32e72dc0d3e739cf1d676d8125fb7575125683/src/prime_rl/trainer/rl/loss.py),
[RAE](https://github.com/PrimeIntellect-ai/prime-rl/blob/6a32e72dc0d3e739cf1d676d8125fb7575125683/src/prime_rl/orchestrator/algo/rae.py),
[local objectives](../manabot/training/learning/objectives.py).

**Weight publication has a commit boundary.** Every transfer offers a version,
waits for receiver readiness, starts transfer, then marks completion. Missing
receivers time out. NCCL sends tensor buckets, with conversion from trainer
representations; filesystem and NIXL transports also exist. The watcher updates
the active policy version after application. Engine pause/transfer/resume is
explicit, including ordering for separate prefill/decode engines. Broadcast
directories are transient transport state, not durable recovery checkpoints.
This is worth borrowing as a protocol, without importing distributed collectives
for a roughly 196k-parameter policy.
[Weight transport](https://github.com/PrimeIntellect-ai/prime-rl/blob/6a32e72dc0d3e739cf1d676d8125fb7575125683/src/prime_rl/transports/weights/base.py),
[NCCL implementation](https://github.com/PrimeIntellect-ai/prime-rl/blob/6a32e72dc0d3e739cf1d676d8125fb7575125683/src/prime_rl/transports/weights/nccl.py),
[watcher](https://github.com/PrimeIntellect-ai/prime-rl/blob/6a32e72dc0d3e739cf1d676d8125fb7575125683/src/prime_rl/orchestrator/watcher.py),
[engine administration](https://github.com/PrimeIntellect-ai/prime-rl/blob/6a32e72dc0d3e739cf1d676d8125fb7575125683/src/prime_rl/orchestrator/clients.py).

**Batching has three distinct units.** Environment episodes run concurrently;
vLLM batches token generation; the orchestrator packs token sequences into
trainer microbatches. The default ZMQ transport waits for each trainer rank's
request, limiting trainer-side queued batches. Current inference admission
uses an adaptive controller driven by KV usage, queue pressure and preemptions;
it can reduce admission and cancel excess work on hard pressure. Those
cancellations are algorithmically relevant when long games correlate with
outcomes. Manabot should retain fixed, declared evaluation cohorts rather than
copy that policy into scored games.
[Concurrency controller](https://github.com/PrimeIntellect-ai/prime-rl/blob/6a32e72dc0d3e739cf1d676d8125fb7575125683/src/prime_rl/orchestrator/concurrency.py),
[ZMQ transport](https://github.com/PrimeIntellect-ai/prime-rl/blob/6a32e72dc0d3e739cf1d676d8125fb7575125683/src/prime_rl/transports/batch/zmq.py).

**Data, evaluation and recovery.** `TrainingSample` holds token IDs, sampled
masks, inference logprobs, temperatures, advantages, optional loss weights,
trace/branch identity and staleness. `MicroBatch` carries aligned typed buffers.
Trace branches retain multi-turn token attribution; shared sampled nodes are
trained only once. This is richer than “a prompt, answer, reward,” but still
not a native Command trace.
[Wire types](https://github.com/PrimeIntellect-ai/prime-rl/blob/6a32e72dc0d3e739cf1d676d8125fb7575125683/src/prime_rl/transports/batch/types.py),
[trace conversion](https://github.com/PrimeIntellect-ai/prime-rl/blob/6a32e72dc0d3e739cf1d676d8125fb7575125683/src/prime_rl/orchestrator/trajectories.py).

Prime provides online and standalone evaluation, task-content matching and
resolved-config checks on resume, metrics and traces. It also exposes an explicit
skip-checks escape hatch. Checkpointing is opt-in: trainer model/optimizer/
scheduler/progress and orchestrator source/progress/config state are saved;
inference weights are repushed on restart. The inspected checkpoint payloads do
not preserve live native games or the entire in-flight rollout set. Process
supervision stops child processes on failure, and weight handshakes are bounded.
These are useful recovery facilities, not evidence of transparent exact replay,
automatic cluster fault recovery, or an inclusive rental-cost ledger.
[Trainer checkpoint source](https://github.com/PrimeIntellect-ai/prime-rl/blob/6a32e72dc0d3e739cf1d676d8125fb7575125683/src/prime_rl/trainer/ckpt.py),
[orchestrator checkpoint source](https://github.com/PrimeIntellect-ai/prime-rl/blob/6a32e72dc0d3e739cf1d676d8125fb7575125683/src/prime_rl/orchestrator/ckpt.py),
[process supervision](https://github.com/PrimeIntellect-ai/prime-rl/blob/6a32e72dc0d3e739cf1d676d8125fb7575125683/src/prime_rl/utils/process.py),
[evaluation](https://github.com/PrimeIntellect-ai/prime-rl/blob/6a32e72dc0d3e739cf1d676d8125fb7575125683/docs/eval.md).

**Recommendation:** treat Prime as a substantial LLM training system with
explicit operational semantics. Reuse its ideas selectively; it does not
supersede Etude's experiment ownership or manabot's scientific admission.

### 2. What transfers to non-LLM RL

| Pattern | Existing local owner | Proposed application and limit |
| --- | --- | --- |
| Explicit behavior versions | `training/rollout/collection.py`, `training/learning/objectives.py` | Bind batches to immutable behavior snapshot, world/ABI, collection probabilities and learner update. Add age/rejection accounting before allowing overlap. Keep synchronous control. |
| Separate actor and learner | `training/execution.py`, `training/rollout/producers.py`, `training/state/` | Only after evidence shows worthwhile idle time: bounded handoff and snapshot acknowledgement. Preserve seat trajectories, complete-game quarantine, original deadlines and durable attempt ledger. |
| Shared inference | `sim/serving.py`, `model/inference.py` | Already implemented for frozen evaluation; extend its ownership if needed. A training service additionally needs values, behavior distributions, immutable version lifetime and recovery semantics. Do not overload frozen `PolicyKey` with mutable weights. |
| Pressure-aware admission | `sim/execution.py`, `sim/serving.py`, `sim/performance.py` | Observe actual batch rows, queue delay, memory and useful completions. Use fixed bounds in scored evaluation; adaptive training admission is a new declared treatment. No synthetic load tests. |
| Versioned weight handoff | Serving plus `training/state/` | Stage a small policy snapshot, validate identity, acknowledge admission, retire only after users finish. No need for FSDP/MoE/NCCL infrastructure at today's scale. |
| Structured terminal accounting | `training/state/game_errors.py`, `sim/` supervision, `experiment/` receipts | Keep attempted, completed, failed, canceled, stale-discarded and unattempted distinct. Preserve failure/cost records across learner rollback. Much of this already exists. |

The [existing shared service](../manabot/sim/serving.py) binds requests to game,
actor, revision and ordinal, validates replies, groups compatible policies and
samples each row from its own retained CPU seed. Its immutable key includes
checkpoint and input identity. It is a stronger starting point than a new
OpenAI-shaped service for numerical tensors.

There is already empirical motivation for separating inference from native
games: the [A6000 policy report](../experiments/inference_baselines/policy-gpu-20261027.md)
records an eight-game median of 27.21 s at four active games versus 68.75 s
serially within one rental. But live batches averaged only 1.13 rows. This
supports bounded shared **evaluation**, not asynchronous learning or sustained
GPU saturation. An approximately 1 MB policy checkpoint also changes the economics
of weight transport radically from a multi-billion-parameter LLM.

**Recommendation:** capture a small contract/design follow-up for behavior
versions and overlap feasibility. Do not implement asynchronous PPO merely
because throughput can increase. Old-policy importance ratios alone do not
repair off-policy GAE/value targets, changing opponents or outcome-biased drops.
Any later learning comparison needs independent seeds, matched cost and the
existing frozen full-game evaluation protocol.

### 3. Direct Prime / verifiers adoption

For an LLM, the smallest useful integration is one task per declared game
opportunity: immutable world/content/source/native/model identities, deal seed,
deck assignment, player seats, opponent digest, policy RNG and limits. A custom
`vf.Env.run` owns a native game and asks only the acting player's agent for a
response. Capture through `semantic.text_observation`; validate through the
current selection policy; submit the captured Command; retain and replay each
transition. The non-LLM opponent runs through existing admitted simulation code.
Native state stays in its supervised worker, rather than being serialized into
the token training batch.

This is feasible in principle, not a working adapter. The upstream
[Kuhn poker environment](https://github.com/PrimeIntellect-ai/verifiers/blob/60290d436fcb4c2ad6dd7d0986bcf00ff2366031/environments/kuhn_poker/kuhn_poker/taskset.py)
already deals from a seed, keeps seat-specific private information, alternates
live interactions and attaches opposite terminal rewards to seat traces. Its
invalid-answer forfeits are a counterexample to copying examples verbatim:
manabot requires one protocol repair followed by a marked seeded fallback, while
runtime failure remains failure. Its accumulated interaction history also
cannot silently replace our fresh bounded observation treatment.

| Contract | Can survive? | Adapter responsibility |
| --- | --- | --- |
| Viewer-safe observation and complete legal choices | Yes | Use the existing captured observation. No opponent-private trace, whole-deck filler or growing chat memory in a player's prompt. |
| Stale rejection and Command binding | Yes | Keep live native capture authority outside the model; validate before every commit, including repairs. Trace metadata cannot authorize an action. |
| Seed → retained Commands → native replay | Yes | Store native receipts and verify replay with admitted engine/world bytes. Token-generation reproducibility is a separate property. |
| Exact tokenizer/template/output evidence | Requires integration proof | Preserve exact token sequences, EOS/thinking boundaries and per-attempt settings; compare Prime renderer output against the pinned harness. Text round trips alone are insufficient. |
| Repairs and fallbacks | Requires explicit training masks | A random fallback is not sampled by the LLM. Never assign it an invented model logprob. Decide whether invalid/repair token trajectories train, and bind that decision to a new protocol. |
| Per-seat terminal credit | Requires explicit algorithm choice | Fixed-opponent candidate traces can use declared group credit. Two learning seats need separate attribution; do not mix opposing rewards into one group baseline. |
| Exact learning recovery | Not inherited | Preserve/reconstruct native prefixes, behavior versions, RNG, pending samples and failure ledger, or declare weaker restart semantics. |
| Evaluation denominators and cost | Not inherited | Keep all scheduled opportunities, failed/canceled attempts and paid setup/cleanup. Generic resume reruns and success-only charts cannot replace our attempt ledger. |
| Frozen checkpoint identity | Yes for exported weights | Digest/admit exports independently. A training inference fleet that updates during a game is not evaluation of one frozen candidate. |

Recommended first proof: no weights, no paid endpoint, controlled model responses;
one complete tiny native game, both seat assignments, illegal output/repair,
fallback, stale submission, engine exception and cancellation. Inspect native
receipt parity, private-information boundaries and token loss attribution. Pin
Prime/verifiers/renderers together in a separate environment, and disable
external monitors for private fixtures. A full Qwen fine-tune needs a separately
approved objective, reward/credit protocol, hardware plan and spend limit.

For the **non-LLM policy**, direct use is not a practical drop-in. Prime's
trainer consumes causal-token sequences and vocabulary logprobs. Our model
consumes object arrays/action-focus tensors and produces masked legal-action
scores plus a critic. Replacing its loaders, inference routes, batching, loss
and recovery would amount to retaining orchestration while rebuilding most
useful machinery. A custom verifiers harness might evaluate such a policy, but
adds little over our existing replay-verified simulator.

**Recommendation:** consider only the isolated LLM adapter proof. Keep numerical
policy training in manabot and experiment launching/recording in Etude.

### 4. Qwen backend decision

#### Evidence before extrapolation

| Evidence | What happened | What it does not establish |
| --- | --- | --- |
| MTG-185 `4c2a54fdfaac`, Transformers, L4 | One non-thinking game, 160 initially valid selections, no repairs/fallbacks, 308 replayed Commands. Game 280.55 s; reported median decision 1.96 s. Preparation 87.52 s; session readiness about 10.04 s. Largest prompt 4,329 tokens. | Current-renderer performance, reasoning throughput, 8,192-token worst-case fit, concurrency or complete billing. |
| MTG-188 `b6a4716bb321`, Transformers, A6000 | Second reasoning selection exceeded 120 s; zero complete games. | Matched backend ranking; this is a longer-output treatment than MTG-185. |
| vLLM startup failures then `dcd5044dd2e0` | Offline file-pattern, venv-path and EOS-metadata integration failures; later health in 79.136809 s, games still failed in client preparation. | A comparable model-load timer or useful game throughput. |
| vLLM `12f050b52057` | Logs approximately 129 aggregate output tokens/s at four requests versus 34 at one; initial/repair deadlines still failed; zero complete games. | Complete-game speedup, matched Transformers comparison or successful action throughput. |
| vLLM `b1d7ed75492f`, 16 games, 2,048 output tokens | 18 completed/replay-verified games, 8 timeouts, 74 unattempted. 793 selections: 33 initial, 241 repair, 519 fallback. | Unrestricted Qwen strength or a healthy reasoning configuration. |
| vLLM `7b8c16de1b60`, 8 games, 16,384 output tokens | Canceled; no complete games. Retained progress: 78 selections (27 initial, 16 repair, 35 fallback), 122 policy moves. Eight archives verified; server-log fetch failed integrity validation. | Benefit of the longer allowance or a complete server timing decomposition. |

Sources: [historical L4 report](../docs/llm-harness.md),
[digest inventory](../experiments/inference_baselines/mtg185-l4.json),
[campaign protocol and attempt history](../experiments/protocols/qwen-mtg183-188.md),
[MTG-191 settlement and handoff](https://linear.app/loopflow/issue/MTG-191).
The last row comes from the authoritative Task description read during this
research, not from raw archives reopened here. Seven finalized Qwen rental
charges total $2.917705820125419603078713051; final ancillary billing is unknown.
Provider absence for the final run was recorded at
`2026-10-27T14:03:49.843896Z`. These historical authorizations do not permit a
new run in MTG-209.

The L4 allocator peak was 19,063,865,344 bytes; sampled whole-device occupancy
reached 22,494 / 23,034 MiB. “Load 10.04 s” in the older report includes session
imports/IPC; it is not isolated weight loading. Preserve the original median's
definition rather than mixing it with later floor-index quantiles.

#### Comparison

| Dimension | Transformers path | Existing vLLM path | OpenRouter path |
| --- | --- | --- | --- |
| Throughput | Resident sequential model session; no cross-game scheduler in the default harness | One resident service batches independent games; already observed aggregate token scaling | Provider schedules requests; concurrency/rate limits and Qwen-specific results unmeasured here |
| Startup | Local snapshot verification, import/load; simpler reference | Separate locked Linux environment, subprocess, compile/load/health and client admission | No local model load; network/provider readiness remains external |
| Prefix reuse | No retained cross-request prefix state in current implementation | Explicitly disabled by server arguments; enabling is a separate treatment | Provider/model dependent; adapter defaults cache off |
| Output constraints | Current exact-reference validation, one repair, seeded fallback | Same validator; no constrained generation currently requested | Same recovery intent; provider structured-output support is conditional |
| Identity and replay | Pinned model/tokenizer bytes and raw tokens; native replay is independent | Same pinned tokens/weights plus server identity; scheduling may change numerical outputs | Slug/provider identity and usage, without immutable local weight/tokenizer identity |
| Operational burden | Weight cache, worker deadlines, GPU cleanup | Those plus server lifecycle, IPC/HTTP binding, scheduler/cache fit and a second dependency lock | API credentials, usage reconciliation, rate limits, remote cancellation uncertainty |

Local implementation sources:
[Transformers](../manabot/llm/transformers_runtime.py),
[resident sessions](../manabot/llm/session.py),
[vLLM exact-token client](../manabot/llm/vllm_runtime.py),
[vLLM server](../manabot/llm/vllm_server.py),
[hosted adapter](../manabot/llm/openrouter.py).

vLLM's [prefix caching documentation for 0.17.1](https://docs.vllm.ai/en/v0.17.1/features/automatic_prefix_caching/)
states that reuse saves prefill, not generated-token decoding. Consequently it
cannot by itself solve thousands of reasoning tokens or missing final answers.
The local [prefix-layout probe](../docs/openrouter.md) found a median of only
63 proxy tokens shared by consecutive same-seat prompts in its tiny fixture;
this is not Qwen tokenizer evidence or a cache hit rate. Retain exact visible
definitions and all choices; renderer reordering requires a versioned treatment.
Qwen's hybrid attention/recurrent state needs backend-specific validation, not
an assumption that a conventional KV cache is sufficient.

vLLM supports [choice, regex, JSON and grammar constraints](https://docs.vllm.ai/en/latest/features/structured_outputs/);
OpenRouter supports [structured outputs on supporting models/providers](https://openrouter.ai/docs/guides/features/structured-outputs).
An enum over the entire legal reference set may improve formatting, but
restricting the whole reasoning response would change the treatment. Constrain
only an explicitly designed answer phase, preserving thinking and EOS contracts.
Neither a valid string nor a grammar replaces native stale/legality checks.
These upstream features are not already enabled in manabot's adapters.

Greedy decoding does not establish cross-backend or cross-batch bit identity.
[vLLM 0.17.1 reproducibility guidance](https://docs.vllm.ai/en/v0.17.1/usage/reproducibility/)
requires additional settings and limits reproducibility to the same hardware
and version. Verify model support for any batch-invariance option before adopting
it. Replay of retained Commands can remain exact even if regenerating the LLM
responses is not. Hosted model slugs provide an even weaker regeneration contract.

#### L4 / A4500 economics

There is no matched L4/A4500 backend trial in the inspected evidence. NVIDIA's
[RTX A4500 specification](https://www.nvidia.com/en-us/products/workstations/rtx-a4500/)
lists 20 GB memory. The observed L4 process used more whole-device memory than
that capacity, so **do not admit the same BF16 treatment on A4500 on the strength
of the L4 receipt**. This is an admission inference, not proof that every possible
vLLM configuration fails. The 48 GB A6000 long-reasoning campaign cannot certify
24 GB-class L4 fit either. Quantization, shorter context or offloading would each
need a separate identity, quality check and performance result.

Use total allocation cost, not advertised GPU hourly price:

```text
rental cost = quoted hourly rate × all billed allocation seconds / 3600
              + separately billed storage / transfer / other charges
cost per useful game = all attempt costs / replay-verified completed games
```

Count setup, warmup, failed runs, repairs and cleanup. Keep fallback rate beside
cost/game; a replayed random-fallback game is not evidence of useful Qwen play.
Hosted cost uses reported uncached/cache/output token charges plus applicable
fees, with missing usage left unknown. No current rental quote or Qwen provider
price was obtained, so no numerical break-even or games/dollar winner is claimed.

**Recommendation:** standardize the interface and evidence contract around the
existing resident vLLM service for future sustained Qwen work, conditional on
qualification. Retain Transformers for sequential checks and exact-input
comparison; retain OpenRouter as a separately identified hosted option. Default
promotion requires a frozen same-input comparison and complete games with
declared startup-inclusive cost, memory, latency, useful legal selections,
repairs/fallbacks and cancellation outcomes. Hold model, renderer, thinking,
sampling, output budget and opponent fixed when comparing backends. Investigate
sampling separately: greedy repetition and insufficient output allowance remain
different plausible explanations; more concurrency alone distinguishes neither.

## Tests and proof limits

Source review found relevant local tests for collection boundaries and probability
retention (`tests/training/test_collection.py`), complete-state recovery
(`tests/training/test_recovery.py`), exact token binding, EOS fallback, admitted
game settlement and environment selection (`tests/llm/test_llm_vllm.py`), and
native text stale rejection (`tests/semantic/test_text_observation.py`). These
tests were read, not rerun for a documentation-only assessment.

Prime has staleness, train-sink, loss and evaluation-resume unit tests; verifiers
has a real Kuhn self-play integration test. Those are relevant contract examples,
not proof of our renderer, world ABI, failure denominators or GPU fit.
[Prime staleness test](https://github.com/PrimeIntellect-ai/prime-rl/blob/6a32e72dc0d3e739cf1d676d8125fb7575125683/tests/unit/orchestrator/test_rollout_staleness.py),
[verifiers multi-agent tests](https://github.com/PrimeIntellect-ai/verifiers/blob/60290d436fcb4c2ad6dd7d0986bcf00ff2366031/tests/integration/test_multi_agent.py).

The source model was checked against counterexamples: existing native producer
processes and frozen serving, Prime's self-play support, its bounded stale-token
masking, and failed Qwen attempts as well as successful token batching. None
supports replacing our trainer or announcing a measured backend winner.

## Follow-up disposition

Current `lf task status` reads found **MTG-191 and MTG-192 canceled**. The
intelligence Task list also marks MTG-189/190 canceled. Historical handoff text
is context, not permission to reopen them. No canceled Task is restarted and no
Task receives steering in this step. Proposed continuations are recorded below;
capture does not accept their design or authorize execution.

Captured with `lf task create --wave intelligence`, both confirmed **unstarted**
with no checkout or running Graph: [MTG-210](https://linear.app/loopflow/issue/MTG-210)
and [MTG-211](https://linear.app/loopflow/issue/MTG-211). The descriptions carry
the essential scope, acceptance and source pins into their future checkouts.

1. **MTG-210: Behavior-version contract for decoupled policy collection.** Design the
   smallest immutable snapshot/batch interface and decide whether measured phase
   imbalance justifies a bounded pilot. Preserve current synchronous control,
   per-seat credit and recovery. Do not implement async learning in that design
   Task. Reject the pilot if transport/idle-time evidence does not justify it.
2. **MTG-211: Controlled managym–verifiers adapter proof.** Pin the inspected API family;
   demonstrate one native game and failure paths with stub responses, loss masks
   and receipt parity. No LLM weights, paid calls or training. Stop if preserving
   the existing observation contract requires a new renderer decision.
3. **Qwen serving qualification — proposal only, defer.** If Qwen work is
   selected again, use MTG-191/192's retained scope and settlement above instead
   of creating parallel backend/prefix implementations. First recover verified
   evidence, then separate sampling/allowance from backend/cache treatments.
   A new paid protocol needs explicit authorization. No new backend Task is
   created while that direction remains canceled.

## Decisions left open

1. For policy collection: (a) retain synchronous ownership, as recommended now,
   or (b) select a separately reviewed overlap pilot after phase evidence?
2. For LLM training: (a) defer fine-tuning, or (b) select the controlled adapter
   proof before any model training or rental?
3. For Qwen serving: (a) keep vLLM as the conditional sustained-serving choice,
   or (b) reopen a broader backend search? Default promotion still requires the
   same qualification evidence either way.
