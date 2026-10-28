# MTG-209 — Prime RL, manabot, and Qwen serving

Research · 2026-10-28 · Pinned source assessment, not a fresh upstream audit.

Jack Heart's later 2026-10-28 directive accepted Qwen evaluation and relevant
behavior-version implementation in bb4160 under a new $10 budget. Its current
accepted constraints and unresolved world decision are in
`scratch/bb4160-evaluate-qwen-versus.md`. The research evidence below predates
that acceptance; its lack of execution/spending describes this assessment only.

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

## Findings retained for implementation

Keep manabot's structured-policy trainer and synchronous collection/update
boundary. Prime's causal-token learner is a candidate for a separate LLM
fine-tuning path; installing it is unnecessary for Qwen inference. The existing
vLLM service is the selected evaluation direction, with Transformers retained as
the sequential reference and OpenRouter as a separately identified hosted path.
No matched backend comparison or default promotion has been established.

Prime separates inference, orchestration and learning. It records token policy
versions, masks stale tokens (default bound eight at the inspected revision),
and acknowledges weight publication before switching versions. Its default IPO
loss and group-relative credit differ from manabot's actor/critic GAE and
PPO/damped-policy rules. Prime also supports self-play RAE with per-agent
baselines; lack of adversarial support is not a reason to reject it.

Borrow immutable behavior identity, acknowledgement, bounded handoffs and
terminal accounting where existing owners need them. Native producer processes
and frozen shared serving already exist. `sim/serving.py` binds requests to game,
actor, revision and ordinal, groups compatible policies, and retains per-row
sampling seeds; extend its `PolicyKey` ownership rather than adding mutable
aliases. The historical A6000 eight-game comparison reported median 27.21 s at
four games versus 68.75 s serially, but mean batches were only 1.13 rows. That
supports bounded evaluation, not asynchronous learning or GPU saturation.

Prime's adaptive admission can cancel excess work. Do not transfer that policy
to scored games: outcome-dependent dropping changes the cohort. Token version
ratios alone do not repair stale GAE, changing opponents or per-seat credit.
Training overlap remains a separately designed treatment requiring phase and
transport evidence. Prime checkpoints save trainer/orchestrator state and repush
weights; they do not establish recovery of live native games, all in-flight
rollouts, or the rental ledger.

## Separate verifiers proof

A future adapter should own one declared native game through existing capture,
selection, Command and replay APIs, keeping native state outside token batches.
Freeze world/source/model/opponent identities, seed, decks, seats, RNG and limits.
Preserve viewer-safe observations, complete choices, fresh bounded prompts,
stale rejection, exact tokens/template/EOS and per-seat credit. Native receipts
remain authoritative; token traces cannot authorize actions.

The inspected Kuhn example demonstrates seeded private observations and
opposite terminal rewards, but its invalid-answer forfeits and accumulated
history are not manabot's one-repair/marked-fallback contract. Random fallback
has no model logprob. Training masks for invalid/repair tokens and restart
semantics require an explicit design, including failure and cost accounting.

First prove a complete tiny game with controlled responses, both seats, repair,
fallback, stale submission, native exception and cancellation; inspect receipt
parity, privacy and loss attribution. Pin Prime/verifiers/renderers together in
an isolated environment and disable external monitors for private fixtures.
Exact renderer parity remains unverified. This is a proposed proof, not an
implemented adapter or authorization for training rentals.

## Historical serving evidence

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

## Serving qualification limits

vLLM batches concurrent games; Transformers currently runs sequentially. The
campaign disables prefix caching and requests no constrained output. Cache reuse
can save prefill, not long reasoning decode; the separate proxy-token layout
probe's median 63 shared tokens is not a Qwen cache hit rate. An answer-only
grammar may help formatting but needs a declared treatment preserving thinking,
EOS and the entire legal set. It cannot replace native legality checks.

Greedy output is not guaranteed identical across backends or batches. Retained
Commands can replay exactly without regenerating identical language-model
responses. Hosted slugs provide weaker weight/tokenizer identity than local
pinned checkpoints. The session hash also does not attest the remote server's
loaded weights; the current evaluation plan records that implementation gap.

No matched L4/A4500 trial exists. A4500's 20 GB capacity is below the observed
L4 whole-device occupancy; the same BF16 treatment is not admitted by that
receipt. This does not prove every vLLM configuration fails. A6000 evidence also
cannot certify L4 fit. Quantization, shorter context and offload are separate
quality treatments, not silent cost optimizations.

Compare fixed model, renderer, thinking, sampling, output budget and opponent.
Report all billed setup, warmup, failed attempts, repairs, storage/transfer and
cleanup, plus replay-verified completions and fallback rate. Missing hosted
usage remains unknown. No fresh quotes or break-even result were obtained.
Recover raw evidence before new claims, and distinguish greedy repetition from
insufficient output allowance before scaling concurrency.

## Evidence location and proof limits

The full source audit, upstream file links, backend comparison and original
proposal list are retained at
`02fccd93e1983319d355a1866c101d7d0e01565b:scratch/prime-rl-assessment.md`.
Local references include `training/execution.py`, `training/rollout/collection.py`,
`training/learning/objectives.py`, `sim/serving.py`, `llm/session.py`,
`llm/vllm_runtime.py`, `llm/vllm_server.py` and the policy GPU report in
`experiments/inference_baselines/policy-gpu-20261027.md`.

The assessment read local collection/recovery, vLLM and text-observation tests,
plus upstream staleness, train-sink, loss, resume and Kuhn self-play examples.
It ran no tests or paid work. Those examples do not establish this adapter's
world ABI, token parity, failure denominators or GPU fit. Existing native workers,
Prime self-play support and failed Qwen attempts remain contrary evidence to
replacing the trainer or claiming a measured backend winner.

## Disposition after accepted direction — 2026-10-28

Jack Heart selected the existing vLLM evaluation continuation and relevant
behavior-version implementation in bb4160, with a fresh $10 ceiling. That
supersedes the earlier proposal-only serving disposition; MTG-191/192 remain
canceled and their budgets stay closed. MTG-210's relevant scope is included in
bb4160 rather than a competing implementation. MTG-211 remains the separate,
unstarted controlled managym–verifiers proposal; the verifiers-versus-custom
fine-tuning comparison is future direction, without training authorization here.

Synchronous structured-policy training remains the control. Overlap, asynchronous
learning and default backend promotion still lack the necessary evidence and
accepted design. This assessment does not authorize a Prime/runtime migration.
The current evaluation plan owns the unresolved world/opponent choice and launch
prerequisites; no backend performance or scientific outcome has been established.

The original proposal list and research-time open decisions are preserved at
`02fccd93e1983319d355a1866c101d7d0e01565b:scratch/prime-rl-assessment.md`.
