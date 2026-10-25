# Shared evaluation compute — 2026-10-24

- Accepted refinement, Jack Heart: checkpoint-versus-checkpoint at the current,
  growing model size is the main capacity workload. Mixed models/heuristics are
  secondary cases. Local small-model-versus-heuristic checks must share arena
  semantics without requiring training metadata, managed queues or cloud access.
  The generic local admission/entry-point seam is implemented in this checkout.
- Proposed compute study compares scale-to-zero, warm pool and baseline-plus-burst;
  no topology, provider, rate commitment or rental is selected. Open inputs are
  checkpoint arrival/cadence and distinct active-model count, useful study backlog,
  acceptable quick-check/checkpoint/study latency, growth-size receipts and budget.
  Existing width/depth ladders are stress cases, not proof of future demand.
  The current one-device allocation cannot place different policies on CPU and
  GPU independently; cluster/routing changes need a subsequent design.

- Jack Heart additionally requires monitoring that matches/exceeds the latest
  training report prototype and explains the next optimization. It is now part
  of the first delivery, including real report projection, concurrency timelines,
  cost/latency/throughput and explicit measurement gaps. Runtime rewrites and
  custom kernels remain hypotheses to test after boundary attribution, not
  approved implementation choices.
- Open: production Etude reporting integration availability; use the existing
  experiment report prototype as the executable reference, without claiming it
  is production or inventing another report framework. Benchmark plans must pin
  instrumentation-overhead tolerance as well as numerical/performance gates.
- Current design: `scratch/batched-eval-inference.md`, accepted for pursuit by
  Jack Heart on 2026-10-24 (“all sounds good. lets pursue”). Deliver the full
  single-machine inference/report keystone as one PR; later increments remain
  follow-ups and paid execution remains unauthorized. Jack Heart wants
  one budgeted pool of eval compute, prioritizing inference batching/caching,
  with a controllable queue API and a language for default priorities. Studies
  can span a wide priority range; currently training checkpoints should receive
  a middle default. No exact priority numbers, capacity or spending approved.
- Proposed sequence: bounded multi-block shared inference first; durable Etude
  priority queue and producer cutover next; steady capacity and adaptive eval
  allocation later. This sequence and the Manabot/Etude API boundary need review.
- Open: representative checkpoint sizes/mix, desired result latency, minimum
  throughput or cost improvement, and machine/budget window. Do not select a GPU
  or claim always-on is cheaper before whole-allocation measurement.
- Implemented first coverage is flat/stateless checkpoint policies and existing
  anchors. Compound, belief and search batching need separate stateful contracts.
- First priority language is ordered metadata rules with explicit overrides;
  affinity only breaks ties with bounded waiting. Score-driven allocation,
  implicit aging across priorities and automatic stopping are not selected.
- Numerical mode/batching changes receive explicit execution identities; exact
  historical sampling streams are preserved per request, but floating-point
  differences may change trajectories. A numerical tolerance and adoption target
  must be fixed for the representative benchmark before claiming equivalence or
  performance acceptance. Replay proves execution, not policy-output equality.

## Implementation decisions — 2026-10-24

- Explicit execution opt-in preserves frozen scalar registrations. New arena
  keys retain the original seed compute envelope separately from the new serving
  envelope, so batch/device/configuration changes do not silently reseed panels.
- Pinned Etude uses spawned callback processes. A private authenticated Unix
  endpoint delegates panels to one allocation supervisor; native game threads,
  queued submissions and slots are globally bounded. This is local transport,
  not a durable queue or a second evaluation authority.
- CPU row sampling is the first numerical contract for both CPU and CUDA serving.
  Whole-batch logits return to CPU. Default counters/spans add no CUDA waits;
  bounded diagnostic traces/event timings explicitly include diagnostic waits.
- Baseline/cold/warm and boundary costs are measured through host stages; pure
  FFI overhead, ordinary kernel counts, device clock alignment and billing stay
  unavailable when not measured. Worker elapsed totals never become a wall pie.
- New source files need Git intent-to-add before Etude identity tests, because
  the pinned SDK hashes `git ls-files`. Documented in AGENTS.md; no commit or
  publication is part of implementation.
- CUDA hardware and a representative frozen benchmark plan (model mix, tolerance,
  adoption threshold and allocation rates) are unavailable. The implemented
  benchmark accepts those declarations; CPU fixture calibration does not choose
  them. Full suites and hardware/performance acceptance remain for gate/CI.

- Identity decision: absolute allocation deadlines and instrumentation/diagnostic
  switches describe attempts, not policy behavior. `EvalCompute` owns stable
  registration/benchmark identity; complete allocation settings remain in receipts.
  Equivalent windows can reuse completed Etude deals. Reuse is not independent
  evidence or a fresh performance sample; calibration uses direct fresh attempts.
  Device, batching, residency/cache, memory and request limits remain identified.
  Duel seeds retain the original opponent cohort before shared execution binding.
- Shared-monitor fixture allowance: on this busy host, three games terminated
  while one reached its 120-second limit after 419 decisions, without forfeit
  credit. The shared correctness fixture now allows 240 seconds per game and
  360 per allocation. These are test allowances, not accepted performance targets;
  production defaults and short deadline regressions retain their own limits.

- Local API decision, 2026-10-24: reuse `EvalExecution` for CPU limits and the
  existing registration/arena/row types. Unknown checkpoint training seeds are
  nullable metadata; exact bytes, world, input and setup admission remain required.
  Frozen monitor fields retain their values. CLI defaults are one four-leg deal,
  heuristic opponent, two workers and a 300-second whole lifetime. They are quick
  check defaults, not scientific sample or performance acceptance thresholds.
- Local preparation is separate from typed `evaluate_local`; the CLI supervises
  it in a killable process within the same absolute deadline. API users can admit
  requests ahead of allocation. Local execution rejects CUDA and never falls back
  to a provider. The managed seam can still use an explicitly owned CUDA runtime.
- Compression assumption, 2026-10-24: public evaluation requires a fresh panel
  directory, including for repeats of failed attempts; reject even an existing
  empty directory rather than infer permission to reuse it. One-deal uncertainty
  is qualified in CLI presentation, preserving stored/frozen bootstrap values.
  The new unshipped calibration schema wraps `EvaluationRequest`; no frozen study
  statistics or evidence are migrated. Ordinary staging and sync subsequently
  succeeded; earlier stash errors are historical.

Inherited Task assumptions (MTG-174/176/180/182) are historical, not current work.
Their complete tracked text is preserved at
`07a091bca:scratch/questions.md`; original attribution, unresolved historical
reporting limits, recovery receipts and earlier git/stash references remain there.

## Upstream integration — 2026-10-24

Current base is `884249024`; original implementation is retained at stash
`187271122326af8955746647d11d3790975c649a` (base `07a091bca`).
Upstream historical MTG-183/card assumptions remain at
`884249024:scratch/questions.md`; they do not select work for this branch.

Reversible ownership decision: `sim` owns serving configuration, resident policies,
game supervision and performance counters. `eval` owns comparison binding,
checkpoint/anchor registration, request/results, local CLI, calibration and callback
transport. `experiment.checkpoint_evaluation` retains the training wrapper.
Shared card evidence is captured during bounded replay, using upstream APIs;
subset-leg comparisons remain available through `eval.match.play_cell`.
No compatibility arena package or supervised-training path is restored.

Integration verification exposed a worker-exit race: the timed pipe poll can miss
a phase/result queued before the exit check. Game and replay supervision now
drain queued messages after observing exit, preserving failure attribution and
completed replay receipts. The forced-race test reproduced the attribution loss
before the fix; no player is inferred when no phase was received.

Final reconciliation: the CLI's preparation deadline covers more than the measured
cell/runtime intervals. Earlier checkpoint admission, final receipt writing and
provider paid lifetime cannot be inferred from serving-load counters or cell
throughput. Whole-job latency/cost and end-to-end speedups remain unmeasured.
