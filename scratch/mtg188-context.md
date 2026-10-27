# Scope update — accepted 2026-10-27

Jack Heart redirected MTG-188 to non-LLM evaluation timing and requested a
Qwen inference follow-up. The existing unstarted MTG-191 is that owner. This
supersedes the Qwen campaign as a completion requirement here. Preserve the
existing reporting/runtime contributions and all earlier receipts; do not
replay completed Graph work. The current briefs are retained locally in
`scratch/qwen-inference-handoff.md` and `scratch/non-llm-measurement.md`. Their
Task edits were saved locally; planning-provider confirmation is pending.

The 16k Qwen run 7b8c16de1b60 was cancelled and its provider pod confirmed
absent. All seven SDK-finalized rental charges total $2.91770582012541960308;
ancillary final billing remains unavailable. The original $10 was a Qwen
campaign ceiling, not new authorization for non-LLM GPU rentals. Continue
local CPU measurements with the existing bounded runner and admitted MTG-183
final raw u5950 checkpoint. CUDA remains an empirical dependency unless an
authorized GPU is available. The earlier accepted design follows as history
and implementation detail; its broader LLM scope is superseded above.

# MTG-188 — Comparable inference performance receipts

Status: accepted for implementation, 2026-10-26. After reviewing this file,
Jack Heart said “seems fine, pursue” in the primary Task conversation. This
authorizes the pursue Graph, including implementation and publication; new paid
execution remains unauthorized. The problem and outcome are in MTG-188 revision 5.
No prior implementation or PR existed at acceptance.
Inspected base: `a0dead2b5195853e0931bf244d84529c08944acb`.
Reconciled with integrated MTG-184/185 at
`558e0186b585228dd2130cf14d9cb40de217e66f`, 2026-10-26.
Reconciled again against the execution continuation and retained fixture evidence
on 2026-10-26: the runner and separate LLM host observations exist. The earlier
feedback describing them as absent predates this implementation. Collection now
shares the original supervised budget. The software publication gate passed; empirical baselines remain open.

Current continuation, 2026-10-26: Jack Heart asked to work around Loopflow's
SQLite launch/settlement failures. Finish the remaining review and publication
in this primary conversation, preserving the completed implementation. No new
Graph or worker is needed. Direct synchronization with main succeeded; the
mixed CLI/headless Perf proof and changed-file Ruff checks passed. The affected
evaluation/LLM regression run passed: 161 tests, one CUDA-only skip. Empirical baseline dependencies
remain unchanged.

## Problem and observable outcome

Startup, prompt processing, model computation, environment work and evidence
collection are easy to conflate. A developer choosing a backend, batch size or
cache needs comparable receipts before choosing an optimization. Retain a
baseline for trained policies and LLM inference: cold preparation, warm execution,
single-game latency and bounded many-game throughput, with every attempt,
missing measurement and cost boundary visible. A slower GPU is a valid result.

This serves the intelligence Wave's reproducible, versioned training and
inference-cost objective. Its current chapter requires pinned identities and
accounting for every attempt, but specifies no performance target. Throughput
satisfies neither its playing-improvement nor human-challenge KRs. Preserve
world admission, viewer-safe observations, complete legal actions, revision-bound
Commands, replay and training coordinates. No world or observation change.

## Demo

Implemented offline command:

```sh
uv run python -m experiments.report_prototype.build \
  --llm-receipt .runs/llm/l4-result \
  --out .runs/inference-baseline/report
```

It prints a report path and concise measurement/coverage summary. The existing
Perf surface shows 160 LLM selections, 308 Commands, one replayed game, 87.52 s
cache preparation and 10.04 s worker readiness. Tokenization, prefill/decode,
queue delay and inclusive billing are unavailable in these old receipts.
Evidence links reach the immutable inputs. Building the report runs no model,
native replay, download or GPU workload.

The implemented mixed-report path adds `--eval-receipt <policy-panel>` and a
baseline manifest. Controlled fixtures cover linked policy/LLM attempts,
contradiction rejection and missing/restored evidence; the mixed CLI/headless browser proof passed on 2026-10-26. The empirical demo still requires CPU/GPU policy cells within their
declared cohort, startup separated from warm work, and finite many-game
denominators. LLM and trained-policy workloads remain separate cohorts, never
one speedup ratio. Generated data/text are inspectable headlessly.

## Findings and current architecture

### Existing owners

- `sim/performance.py::EvalPerformance` owns bounded host spans, full counters
  and request samples. Its first-2,048 sample is not random; p50/p95 use
  `sorted_samples[floor((n-1)*q)]`. `allocation_cost` distinguishes measured
  dollars from estimates and avoids charging each job the entire allocation.
- `sim/serving.py::PolicyStore` owns byte admission, residency and catalog reuse.
  `_infer` measures assembly, combined admission/transfer, host forward, output
  transfer, sampling and latency. Historical `batch_queue` ends **after**
  `store.get`, including cold loading. Schema-2 `dispatch_queue` now ends before
  loading and separates submission-to-receipt transport; old data keeps its
  original interpretation.
- `sim/match.py`, `sim/supervision.py` and `eval/match.py` own game execution,
  durable Commands, replay and output. `native_step_and_encoding` is inclusive;
  engine/FFI/encoding separation is unavailable without further instrumentation.
  `eval/local.py` admits checkpoints before Perf cell/allocation timers: its
  whole-process deadline covers startup but does not measure it in those files.
- `eval/serving_benchmark.py` already provides killable probes, retained roots
  and numerical/instrumentation checks. Its four fixed ablations and speedup
  gate answer an adoption question. Reuse useful primitives, not an optimization
  tournament as a prerequisite for measurement-only CPU/GPU comparisons.
- `reports/evaluation.py::project_evaluations` and
  `experiments/report_prototype/{build.py,report.html,model.js,charts.js}` own
  the existing Perf projection/view. The builder now accepts evaluation, LLM
  and baseline-manifest inputs; `reports/inference.py` owns historical LLM
  projection. LLM-only reports do not import the evaluation projection or
  manufacture score rows.
- Training's `PhaseTiming` and collection/learning/export/diagnostic records
  already feed `reports/time_profile.py`. Keep that authority and its clocks.
  Environment decisions, learner transitions and generation attempts are
  distinct units; training observations are contextual evidence, not new samples.

### What the retained L4 evidence actually establishes

Unchanged evidence is copied from `/Users/jack/src/manabot.llm-harness` into
`.runs/llm/l4-result/`; its historical design is `.runs/llm/mtg185-design.md`.
Original Etude run: `4c2a54fdfaac`; source:
`e747f0445e31ac08d674f855ad0128a4a722b100`. The accepted base lacked the renderer and LLM runtime. MTG-184 renderer/native
input code is integrated at `ce43a5cf76d52b99c4c13c6debb21aab79ac9f82`;
MTG-185's `manabot/llm`, optional dependency group and LLM full-game implementation
are integrated at `558e0186b585228dd2130cf14d9cb40de217e66f`.
The historical receipt still belongs to its original source and pre-capture-v2
renderer; integration supplies execution owners, not new performance evidence.

Qwen/Qwen3.5-9B model/tokenizer revision
`c202236235762e1c871ad0ccb60c8ee5ba337b9a`, BF16, Transformers 5.3.0,
Torch 2.10.0+cu128, one NVIDIA L4, greedy exact-reference output, thinking off,
8,192 context tokens, 32 output-token cap. Game seed 23, opponent seed 24,
model seat 0, public history v1/capacity 32; each deck contains 24 Mountain,
24 Gray Ogre and 12 Lightning Bolt.

| Quantity | Observation and boundary |
| --- | --- |
| Cache preparation | 87.5167 s acquisition **plus verification**, not network-only time |
| `model_load_seconds` | 10.0418 s spawned session start-to-ready, including imports/IPC, not isolated weights load |
| Game wall | 280.5549 s including reset, rendering, selection, writes, native execution and immediate clone replay |
| Selections | 160 initial-valid, no repairs/fallbacks; summed selection time 276.3684 s |
| Latency | Historical median 1.9557808 s; floor-index p50 1.9556065 s, p95 2.4596925 s, max 2.518225 s; n=160 |
| Commands/games | 308 Commands, including 148 opponent decisions; one completed game; supplied audit records local replay of all 308 |
| Input size | 1,219–4,329 actual prompt tokens; no maximum-context capacity proof |
| Memory | Torch peak allocated 19,063,865,344 bytes; 291 device samples peak at 22,494/23,034 MiB |
| Cost | Historical design reports approximately $0.0800 and rental cleanup; copied files lack billing evidence, so measured inclusive cost is unavailable |

The 4.1865 s difference between game wall and selection sums is a derived
remainder, not environment time. Selection includes encoding, generation and
IPC; game time excludes session construction and outer final collection.
The first selection remains in the population; no warmup was separately recorded.
One model stayed loaded, with no cross-request prefix state. The remote cache
ended with the rental. Neither CPU/GPU comparison nor concurrent fit follows.

### Mechanisms and constraints resolved

Host forward timing measures launch/wait behavior, not necessarily device work.
Use the existing profiler/CUDA-event diagnostic path on separate retained-input
attempts. Do not synchronize every ordinary request to manufacture additive
charts. CUDA events measure stream intervals, including possible gaps, without
host epoch alignment. See the pinned
[PyTorch 2.10 timing contract](https://docs.pytorch.org/docs/2.10/notes/cuda.html#asynchronous-execution).

Allocator peaks omit some CUDA allocations; see
[PyTorch memory visibility](https://docs.pytorch.org/docs/2.10/torch_cuda_memory.html).
Record allocated/reserved process peaks separately from whole-device used/total
bytes, device UUID, sample count/interval/window. A bounded 1 Hz `nvidia-smi`
sampler reuses the historical CSV convention; each invocation needs a timeout
and cleanup. Missing utility means unavailable, not zero. No mandatory NVML
package is needed. Shared-device occupancy remains a confound.

The inspected Transformers runtime uses one `generate` call. Record encode and
generation host intervals and tokens initially. Separate TTFT/prefill/decode is
unavailable; do not infer it from token count or rewrite generation to obtain it.

### Related accepted direction — 2026-10-26

Jack Heart accepted the Resin deferral and inference-reuse priorities recorded
at `50184b6f937588c29d661f6d07f0f780282df939:docs/resin-assessment.md` and its
intelligence Wave memory. This related branch is evidence of direction, not
integrated runtime code. The order is MTG-188 measurement, MTG-189's serving
ownership contract, then MTG-190's measured buffer/capture/overlap changes;
supported LLM backends and prefix reuse remain MTG-191/192 work.

Slice 2 therefore measures completion boundaries through existing owners,
without adding asynchronous serving ownership or graph capture here. Where
execution already exposes them, capture graph-hit/fallback counts without
forcing synchronization on each metrics read; unavailable counters stay
unavailable. Downloaded weights, compiled artifacts and live model state have
distinct cache identities. Resin adoption, an L4 port and changed hardware/model
treatments remain outside this Task. No new performance result or paid authority
follows from the assessment. The checkout includes MTG-184 at `ce43a5cf7` and
MTG-185 at `558e0186b`; the runtime integration dependency is resolved.

### Delivered LLM owners — inspected 2026-10-26

`llm/cache.py::prepare` now retains acquisition and shard verification separately,
with the original inclusive interval preserved. `llm/session.py` already
owns one spawned resident worker, load and decision deadlines, progress retention
and termination without restart. Its `load_seconds` includes runtime construction,
unlike policy service readiness, which precedes admission. `llm/selection.py`
owns encode/generate/validation and repairs. Optional host intervals now retain
encode, generation and validation separately, including failed calls, through its
existing progress channel. No second runtime or per-token generation loop exists.

`verify/llm_game.py::play` owns rendering, durable Commands, clone replay and the
game window. Its L4-specific `run` is not the bounded baseline runner: device
inspection has no timeout, and sampler cleanup has a terminate/wait path without
kill/reap escalation. The new experimental runner supplies whole-attempt and
preparation/collection boundaries, planned-attempt retention before loading,
and a bounded optional device sampler. It does not relaunch that historical job. Missing inclusive billing remains unknown
despite the approximate cost documented by the delivered harness.

## Chosen architecture and contract

### Source of truth and consumers

Existing raw files remain authoritative: evaluation requests/rows/traces and
allocation/serving Perf; LLM config/cache/runtime/selection/Command/transition;
provider settlement; training PhaseTiming. A baseline manifest binds those
artifacts by digest, size and source, declares workloads/cells, and references
every attempt. It does not duplicate Commands, selections or invoices.

Extend typed measurement metadata at `sim/performance.py` and the current
simulation writers. Extend the delivered native LLM receipt types at their
existing owner. The report is a read-only projection into the
existing Perf model, not a converted authoritative store. Do not make the
lightweight LLM parent import simulation/infra just for a timer: infra's package
initializer imports runtime/Torch. Its worker sends observations through the
existing progress channel.

`experiments/inference_baseline.py` owns the typed experimental plan and
explicit local runner. Policy execution delegates to `EvaluationRequest`,
`play_cell` and the existing policy store; LLM execution uses delivered
`InferenceSession` and `verify.llm_game`. Validation, retained-input replay and
execution are distinct commands. None provisions hardware, repairs credentials,
or downloads missing models implicitly. Reuse supervision primitives instead of
copying loops; extract a private common helper only where both callers use it.

The report builder accepts `--llm-receipt` and `--baseline-manifest` alongside
`--eval-receipt`, extending the existing Perf view and evidence copier.
Report generation must not require Transformers or model execution.
Existing monitoring/Etude evaluation callers receive compatible optional Perf
fields; registrations, training coordinates, callback DTOs and scientific rows
keep their meaning. Training reporting remains on its current path.

Persist planned attempts before starting workers; include workload/cell ID,
repeat, ordinary/diagnostic mode, status, termination reason and evidence refs.
Keep planned, attempted, completed, failed, unknown and unattempted distinct.
Reuse reserved output directories and atomic writes. Parent/worker failure must
retain received evidence and reap owned children without automatic retry.

### Evidence dependencies — implemented correction, 2026-10-26

Concept review found one further consequence of Jack Heart's accepted digest-bound
evidence and once-only allocation accounting: a pinned report must bind the
evidence it consumes, including shared files outside an attempt directory.
This is a design correction implied by existing requirements, not a new product
choice. The declaration/observation/availability distinction remains unchanged.

Counterexample checked with the existing controlled policy/LLM fixtures:
`report-run.json` pointed to a sibling allocation directory, as the production
writer in `eval/match.py` permits. Changing only that directory's `execution.json`
from measured USD 1 to USD 2 produced two successful builds against the identical
manifest: games/dollar changed from 1 to 0.5. Neither policy inventory included
the execution receipt, and neither report copied it. No inference or performance
workload ran. Thus a directory inventory alone cannot represent the accepted
measurement evidence boundary.

The v1 manifest adds optional `allocations` (identity, root, file inventory).
Attempt metadata's serving path connects cells to that declaration, which now
travels intact through the builder and evaluation projection rather than parallel
identity/file maps. `EvidenceReader` exposes consumed snapshots, validates shared
pins and retains unavailable dependencies with reasons. `_retain_evidence` verifies
bytes again when copying and rewrites links for a portable report. Each allocation
is charged once. Serving identity contradictions and conflicting pins fail.
Direct inputs receive read-time inventories; legacy policy manifests cannot
consume unpinned external measurements. Historical LLM manifests stay unchanged.

Missing execution removes cost; missing allocation wall removes allocation rates;
missing serving removes its timings; missing diagnostics remove their links.
Local observations and attempt links survive, including shared files nested
beneath a cell. The declaration preserves allocation identity without serving
evidence; exact restoration recovers the projection. Historical serving trace
paths retain their working-directory basis and are resolved once before copying;
manifest inventory paths resolve from their declared root.

Controlled fixtures cover two cells/one charge, pinned mutation (including copy
time), partial pins, nested shared files, missing/restored identities, and portable
evidence after source removal. The mixed CLI/headless proof uses these fixtures;
its execution remains with gate. Rebuilding executes no inference or billing.
No unresolved product choice arose in this review. The collection-bound repair,
remaining gate checks, empirical acceptance and paid-execution authorization are
documented below.

### Measurements

Every metric has value/unit, clock/resource, boundary, coverage and status:
`recorded`, `derived`, `unavailable` or `not_applicable`, with a reason. Zero
means observed zero. Missing files are unavailable; malformed input or digest
mismatch is a validation failure. Partial experiments can render but cannot
claim complete acceptance. Failed-only cohorts have no success latency quantile.

- **Whole attempt:** parent entry through preparation, readiness, games, replay,
  final output and cleanup. Etude acquisition/idle/drain intervals are separate
  external records; a local timer cannot supply them.
- **Preparation:** acquisition and verification separately where observable;
  requested/hit/missing files and bytes, cache state before/after. Distinguish
  artifact cache, model residence, policy catalog, within-generation KV and
  cross-request prefix caches. Fresh process does not imply cold disk; OS page
  cache stays unknown. Never evict shared caches to simulate cold startup.
- **Readiness:** process start-to-ready with child admission/load spans; model
  placement is inclusive when it cannot be separated. Nested load is not a
  second wall-time sibling. Retain first-request latency separately.
- **Requests:** submission-to-reply, dispatch queue wait, assembly/render/encode,
  admission/transfer, forward/generation, output transfer/decode, validation and
  evidence writes. Correlate request/game/attempt/batch IDs. Repairs are extra
  attempts within one model decision. Parent/worker intervals may overlap.
  Stamp server receipt and dispatch separately: dispatch queue is receipt-to-
  dispatch; client submission-to-server receipt includes transport. Neither
  interval includes a subsequent model load. Cross-host clock subtraction is
  forbidden; current serving uses a single host.
- **Native work:** existing reset/step-plus-encoding and Command validation/
  durability, replay and collection. No additional policy RNG draws for profiling.
- **Distributions:** n, observed population, sample coverage, p50/p95/max using
  the floor-index convention. Game/request/attempt latency are separate
  populations. Keep original historical medians; explicitly label recomputed
  quantiles. Counters cover all calls even when spans/samples hit their cap.
- **Rates:** committed native decisions, completed model decisions and completed
  replay-verified games per wall second, each labeled. Failed work consumes the
  same wall denominator. Show both game-window and whole-attempt rates when
  available, never dividing by summed concurrent worker time. Warmup is retained
  and charged inclusively; only the named warm population excludes it. Unknown
  historical warmup is not zero. Report all individual repeats, not pooled
  quantiles across unrelated workload/hardware cells.
- **Memory/cost:** whole-device samples plus process allocator/RSS observations.
  One allocation identity owns one inclusive charge (setup, idle, failures,
  cleanup and known storage/network costs). Estimated, partial and settled
  totals differ. Missing inclusive billing makes games/$ unavailable.

Preserve sequential timing-tree overlap checks. Queue, nested load, request
wall, device intervals and worker sums live on separate resources. Deduplicate
by source digest/run identity so archive/extracted copies do not add samples
or duplicate charges. A report cannot turn replayed historical outputs into
fresh inference throughput.

## Baseline protocol and limits

Freeze the resolved manifest before measurement: source/native hashes,
world/content/ABI, exact decks/sideboards, model/tokenizer/checkpoint bytes,
input corpus/masks, seeds/derivation, decoding, dtype/TF32, threads,
runtime/driver/device facts, cache policy, warmup, repeat ordering and collection
settings. Changed backend is a declared treatment, not continuation under the
old identity. No minimum speedup is required by this Task.

Proposed small trained reference: MTG-181 seed 18101 raw u616, 1,071,166 bytes,
SHA-256 `cc78fe1d44fcc65e93fe5673d8d78b6e44b85bf7433b5624a52a0be33410915e`,
recorded by `experiments/mtg186_eval.py::MTG181`. Its frozen recipe is width 64,
`elemental-world-v2`; use its admitted UR/GW authored pair. Obtain exact bytes
through existing artifact tooling and verify world/setup and flat stateless
admission. No trained bytes are present here yet. Missing/incompatible bytes
block that empirical cell; a random-weight CI fixture cannot replace it.

Two complementary observations:

1. **Fixed retained inputs:** capture the first 64 real decision roots of the
   declared baseline with full masks, seeds and raw outputs. Replay identical
   inputs across CPU/CUDA and batch settings. Two warmup passes over those
   roots and one measured pass per repeat; three repeats in manifest-fixed
   interleaved order (rotate the declared cell order by repeat index). Keep
   warmup and output witnesses. No synthetic enlargement,
   resource contention or busy-loop timing. Check logits with predeclared
   FP32 absolute tolerance 1e-5 and relative tolerance 1e-4; report action
   agreement, including ties. These are draft diagnostic tolerances, not
   permission to alter outputs. Failed agreement retains the cell but forbids
   a numerical-equivalence claim; no cross-device bitwise promise.
2. **Complete games:** single-game cell with seed 18801 and one declared leg;
   many-game cell with seeds 18801–18804 and both candidate seats (8 games),
   against a frozen heuristic identity. Compare ordinary scalar CPU, resident
   CPU/CUDA batch-1, and current shared CPU/CUDA execution with four active games,
   four batch rows and zero added batch delay. Preserve all Commands/replays
   and decision counts. Numerical differences may change trajectory length;
   disclose that and use fixed-input comparisons for like-for-like diagnosis.

Small-policy engineering ceilings: one Torch thread per worker, at most four
game workers plus one serving worker, 120 s/10,000 Commands per game, 300 s per
cell including cleanup, 1,800 s total local invocation, 8 GiB summed process RSS,
three declared repeats, no automatic retry. Stop at limits and retain unfinished
cells. These are execution bounds, not paid authorization. Compare instrumentation
on/off on the same bounded retained inputs; require unchanged same-device
outputs and disclose observed overhead without machine-speed CI thresholds.

LLM reusable corpus is all 160 complete retained prompts/token IDs and legal
references in ordinal order, with native captures/Commands and exact config.
Prospective single/many-game validation keeps the existing sequential resident
runtime: seed 23 control and seeds 18801–18804, one session at a time, max 2,000
Commands/game, 120 s decision and 180 s load deadlines. No concurrent LLM serving
is added. Game/allocation ceilings and a total cost cap must be fixed under
separate execution authorization; do not regenerate old prompts with a changed
renderer or interpret old captures under an incompatible world.

CPU/GPU tradeoffs require same-host CPU and one-GPU cells with exact device facts.
The old L4 LLM result is no proxy for a trained-policy GPU cell. No rental is
selected. Any future paid validation must declare hardware, whole-rental time,
inclusive dollars including startup/cleanup, one attempt, no retry and resource
termination/absence verification. No paid run or inherited allowance is currently
authorized. Missing empirical evidence stays pending, while independent software
slices remain actionable.

## Alternatives and failure analysis

A separate dashboard duplicates the existing Perf authority; rejected. A broad
Modal/Resin/vLLM or network-serving comparison still needs these baselines;
deferred. Model-only microbenchmarks omit startup/native/replay/output, while
whole-game timing confounds trajectory length; retain both observation types.
Per-operation CUDA synchronization changes the workload; separate diagnostics
and explicit unavailable attribution preserve honest ordinary measurements.

Wild success: a backend proposal names a fixed corpus and explains why a small
policy belongs on CPU while a large model benefits from residency. Wild failure:
charts omit startup, exclude failed attempts, compare different games or label
allocator bytes total device headroom. Typed boundaries, source links and the
negative acceptance fixtures below address those failure modes.

## Scope and deletion

Included: receipt completeness, instrumentation, reusable baseline/replay inputs,
bounded measurement orchestration and existing Perf presentation. Excluded:
backend optimization, prefix caching, compile/quantization, LLM batching, provider
adoption, network serving, scheduling/startup gates, renderer redesign, training
changes, strength claims and worst-case 8,192-token capacity certification.

**Delete — do not maintain:** duplicate single-game/sequence latency reducers in
`reports/inference.py`, the builder's second attempt-observation read/inventory
validation, and repeated witness-archive/runtime reads in `inference_baseline.py`
are removed. The shared latency reducer preserves each population; attempt binding
checks declarations and outer/game outcomes once. Projected inventories retain
missing shared dependencies as unavailable even when an outer attempt exists;
copy-time digest verification remains mandatory. Supervised collection hashes durable
raw files once, then the parent pins its final observation separately. No exclusive fixtures
are obsolete; retained failure, replay and evidence tests exercise the surviving
paths. No remaining deletion targets from these reductions.

The duplicate `inference_baseline._pin` and `reports.inference._digest` paths
are removed: `reports.inference.pin_file` binds both protocol and receipt files,
with explicit relative inventory paths. The frozen protocol's one pin also supplies
its manifest source digest. Interrupted-hashing fixtures inject at this surviving
owner. The process supervisor builds its JSON validator once per call, and cache
preparation uses one completion stamp for its total and nested verification time.
The builder uses an explicit deferred evaluation import, preserving offline
LLM-only isolation. No remaining deletion target or exclusive obsolete fixture.

No runtime replacement. The report's exclusive
evaluation-input requirement is removed; evaluation rows/scores remain intact.
Removed per-opportunity copies of `worker_request_latency`; one per-game
collection now lives on each run, where the Perf view consumes it, even without
a measured optimization phase. Duplicate measurement accumulation and execution
receipt writers now share `Measurement.observe` and `InferenceRuntime._write_execution`,
preserving counts, boundaries and failure evidence. No remaining deletion targets
in this implemented slice.
This correction removes the builder's unlinked attempt metadata and the LLM
projection's replay-dependent `status`. Preserve declaration bytes, direct inputs,
raw evidence and replay coverage on the linked attempt/projection path.
The builder's parallel source/inventory/projection lists and duplicate run-index
calculation are replaced by `_Receipt` records and each projection's assigned run.
Parallel allocation identity/file maps are replaced by the existing
`BaselineAllocation` declaration; duplicate copy validation and output-location
guards are consolidated. `_retain_evidence` owns copying separately from assembly.
The shared-evidence correction removes
`project_evaluations`' unguarded sibling receipt reads and the builder's unverified
serving/trace copier. `EvidenceReader` owns dependency byte snapshots, digest
validation and availability; no unchecked cost path survives. Local observations,
raw bytes, historical v1 manifests and once-only allocation charges are preserved.
Implemented: new schema-2 policy receipts no longer emit ambiguous `batch_queue`;
readers retain old bytes and label the historical interval through model readiness.
`dispatch_queue` ends before loading; overlapping measurements are explicitly
excluded from sequential timing trees.
Do not repair/relaunch MTG-185's historical ad hoc job. No duplicate LLM runtime,
policy store, fake evaluation rows, parallel dashboard or dual receipt authority.

Implemented LLM hooks extend the integrated MTG-184/185 owners. `e747f0445`
identifies historical mechanisms and frozen receipt provenance; current code is
at `558e0186b`. Historical receipt projection remains independent of executable
LLM imports. No runtime copy or additional integration step is needed.

## Internal slices and Done when

1. **Implemented slice — retained evidence to Perf; software gate passed.** Typed readers,
   builder, projection and existing UI validate/deduplicate sources and show LLM
   latency, startup, memory, coverage and failure states without fake score rows.
   Compact schema-faithful fixtures and a real corpus manifest are present.
   Large/private evidence belongs in the existing artifact store, never source
   control; publication of these historical bytes remains pending. Focused
   test (new): `uv run --extra dev pytest tests/eval/test_inference_receipts.py -q`.
   Done when the demo command explains the actual L4 run offline without
   modifying raw bytes, importing a model or zero-filling missing measurements.
2. **Instrumentation and execution — implemented; software gate passed.**
   Policy readiness, admission, transport/queue, success/failure latency, residency
   misses, process RSS and allocated/reserved CUDA peaks now use existing owners.
   The experimental runner now freezes recipes, replays retained roots, records
   whole-attempt preparation/collection/cleanup, optionally samples devices and
   extends the existing LLM observations. Collection/comparison share the original
   supervised deadline and RSS ceiling. Focused validation is retained.
   LLM hooks now use the delivered runtime. Preserve old readers and local CLI import
   isolation. Done when a complete game, injected failure and overlapping workers
   yield correct counts/denominators in the same report; timeout/worker death
   retains evidence and reaps all owned processes.
3. **Measured baselines.** Freeze the protocol, obtain/admit the pinned checkpoint
   and record bounded CPU observations. GPU/new LLM work needs authorization.
   Keep individual repeats, output comparisons and complete-game counts. Task
   acceptance requires actual small-policy CPU/GPU tradeoffs and LLM startup/
   single/many-game coverage; explicitly unresolved cells prevent a completion
   claim. Before committing result manifests, publish bytes using the existing
   `manabot.data` operation and verify experiment data remains untracked.

### Acceptance at gate

Primary end-to-end scenario: a bounded CPU fixture creates a policy panel with
Commands/replay and Perf; an LLM fixture includes an initial response, repair,
fallback and runtime failure. Feed both through the real report CLI, then inspect
its generated data and headless Perf view. Verify exact counts, latency method,
startup nesting, overlapping resources, once-only allocation cost, visible warmup
and unknowns, and evidence links. Digest mutation and duplicate import fail.
Malformed data cannot silently become an empty run. Controlled clocks supply
asserted durations; fixtures do not establish real performance.
The manifest-backed mixed proof also requires terminal contradiction rejection
before output, missing completed evidence with unknown observations and no run,
and restoration with unchanged attempt identities and declaration bytes. Both
policy and LLM measurements link back to their declarations; replay coverage
remains independent of observed completion.

Passed gate checks for the implemented slice (2026-10-26):

```sh
uv run --extra dev --extra evaluation --extra artifacts pytest tests/eval/test_inference_receipts.py tests/eval/test_inference_baseline_execution.py tests/eval/test_eval_performance.py tests/eval/test_batched_eval.py tests/eval/test_local_evaluation.py tests/test_package_boundaries.py -q
./scripts/verify-eval-performance-report
```

Expected: all checks pass; the existing report proof uses the generated mixed
report, not only handwritten JS data. Its Python policy and LLM receipts are
controlled fixtures. They establish report interpretation, not live execution,
native replay, overlapping-worker lifecycle or cleanup. Separate execution fixtures
now cover native games, retained-input parity, failure accounting and cleanup;
the combined CLI/headless scenario remains a gate obligation. CUDA-specific tests may skip on CPU CI
with explicit missing-hardware reporting; that is not GPU acceptance. For changes
to the now-integrated LLM owners, the affected-suite gate also includes:

```sh
uv run --extra dev --extra llm pytest tests/llm/ -q
```

Those cache/session/selection/native-game tests use controlled fixtures, no
pretrained download. The existing `tests/eval/` CI selection already includes
`test_inference_receipts.py`; combined collection now finds 235 tests in the exact evaluation CI invocation
without basename collisions. Further new modules require combined collection
with their exact CI paths.
Run `uv run --extra dev pytest tests/infra/test_ci_selection.py -q` if CI selection
changes. The existing local
CLI test must still reject training/cloud/callback imports. If modules/CLI paths
move, apply the repository's release/package checks; no move is currently planned.
Run the report proof on capable headless CI when Chromium is unavailable locally.
No display, paid GPU allocation or full-weight download is ordinary gate work.

## Implementation state — 2026-10-26

### Attempt and evidence relationship — implemented 2026-10-26

Concept review changed the design on 2026-10-26 as a consequence of Jack Heart's
accepted raw-evidence authority and every-attempt accounting. The controlled
fixture had reproduced declared `completed`, observed `failed` and no projection
attempt ID despite matching inventory digests. This correction preserves the
accepted Task boundary; it does not establish baseline execution or completion.

The manifest owns declarations; raw receipts own observations. The builder's
`AttemptProjection` now binds each available policy/LLM projection to its declared
attempt ID and report run. Perf labels declared outcome, observed outcome and
evidence availability separately and links both directions. The original manifest
`status` bytes stay unchanged. Direct receipt inputs have no invented declaration.
A missing directory leaves even a completed declaration visible with unknown
observations and no run; restoring exact pinned files rebuilds the same attempt
link without executing inference.

Readers own outcome semantics. An LLM terminal summary may establish game
completion without certifying replay or full receipt coverage. A failed game can
contain successful selections. Policy panels retain successful/failed/partial
work together: any failed row establishes failure; all retained blocks must have
their terminated legs and no declared unattempted games to establish completion.
Other nonempty panels are partial; empty panels remain unknown. Replay validity
and billing stay independent. Opposing completed/failed declarations and observed
outcomes fail before report output is written, with attempt ID and source path.
Nonterminal declarations and partial observations need not have matching strings.

The focused fixtures cover both contradiction directions for policy and LLM,
nonterminal declarations gaining observations, missing completed evidence,
restoration of both kinds with unchanged attempt IDs and declaration bytes,
partial panels, unknown replay coverage and direct receipt inputs. The existing
headless proof now generates a manifest report through the actual CLI, rejects
the initial contradiction, corrects only the declaration, removes/restores pinned
inputs, and checks links and labels on the generated mixed Perf page. Its browser
execution remains with gate.

### Current implementation

The explicitly named first slice is implemented: `reports/inference.py` reads
historical v1 selections without importing the LLM/native/model stack; the
existing report builder accepts LLM and baseline-manifest inputs, preserves
separate populations and copies digest-bound evidence. The existing Perf page
shows preparation/readiness, success/failure latency, Command/game coverage,
allocator/device memory and explicit unknowns. Evaluation score rows and
training clocks are unchanged. The evaluation-only input restriction is removed.
The projection owns its evidence inventory: baseline validation compares that
inventory once, and the shared copier verifies each file's bytes against it.
`_project_allocation` keeps shared dependency interpretation separate from the
cell/worker aggregation loop; allocation identity and once-only accounting remain
with the caller.
This avoids a second manifest-driven directory scan without weakening the
copy-time digest check. `_Receipt` keeps each source, inventory and projection
together through validation and copying; attempt links reuse the projection's
assigned run index. Output must stay outside declared source directories even
when those directories are absent, so a report cannot contaminate restored
evidence. Browser report scaffolding belongs to the builder;
the historical reader owns receipt interpretation and model identity validation.
The historical corpus is pinned in
`experiments/inference_baselines/mtg185-l4.json` (1,708 files, 160 ordered
selections); raw bytes remain under `.runs/llm/l4-result`, outside source control.
The generated offline demo is `.runs/inference-baseline/report/report.html`.
The v1 manifest remains the retained evidence inventory and attempt ledger.
`inference-execution-v1` separately freezes the executable recipe, local source,
native/content/input byte pins, installed runtime, device facts, offline cache
policy, one-thread/TF32 settings, ordering and bounded execution budgets.

### Policy instrumentation continuation — 2026-10-26

`EvalPerformance` schema 2 marks whether a measurement partitions elapsed time.
New queue and latency observations overlap; timing trees omit them, while the
Perf timeline and distributions retain them. Full call counts and maxima survive
the bounded prefix sample. Disabled timing keeps counts and reports unknown
latencies. The first observed completion/failure stays in its population.

`PolicyStore` records residency misses and byte verification separately from
loading. Serving stamps receipt and dispatch before loading; it keeps completed
and failed request distributions separately. Game workers measure the actual
reply round trip; supervisor validation and service reply send remain distinct
boundaries. The report retains per-game distributions without pooling quantiles.
Parent start-to-ready includes spawn/import/IPC, excludes model admission and
may overlap admission round trips. Cleanup remains within allocation elapsed.
Startup failure retains an unready attempt; service death retains admission and
parent observations and reaps the owned service. RSS is sampled at admission and
batch boundaries; CUDA allocated/reserved peaks are process-allocator values,
never whole-device occupancy. No shared caches are evicted for measurements.

Remaining: gate owns affected suites and the extended mixed CLI/headless proof.
The original bounded fixture used truncated games; the execution continuation
adds complete games, failed-attempt receipts and retained-input comparisons.
Slice 3 still needs trained-policy bytes/admission, empirical CPU/GPU and LLM
single/many-game observations, and artifact publication. No new paid execution is
authorized; this remains one unfinished Task and delivery boundary.

Historical evidence check before this correction: rebuilding the real manifest
in a temporary directory matched the existing
`.runs/inference-baseline/report/data.json` exactly, verifying all 1,708 files.
The real manifest demo reported 160 selections, 308 committed
Commands and one replay-verified game, with 87.516725 s preparation, 10.041803 s
readiness and 2.459692 s floor-index p95. No fresh inference/native replay ran;
full affected suites and browser proof remain with gate.


### Execution continuation — 2026-10-26

`experiments/inference_baseline.py` owns freeze/validate/run/replay. All attempts
are persisted before any worker starts; validation and preparation execute inside
the bounded attempt, so their failures and wall time survive. Fixed repeat order
rotates cells. Policy games use `EvaluationRequest` and `play_cell`'s existing leg
selection, allowing one leg or four seeds/both seats without changing score
semantics. LLM games use the delivered sequential session and native game writer.
A multi-game LLM workload reuses one sequential resident session and charges
readiness once. Per-game distributions survive in the same report; no concurrent
LLM serving is added.

Retained policy replay stores exact arrays, masks, row seeds and output witnesses
for both warmup passes and the measured pass. It checks same-device repeat
identity and compares legal logits/actions across cells using the declared
absolute/relative tolerances. Retained LLM replay checks the original prompt text
and token IDs before generation, without invoking the renderer. Diagnostics have
pinned evidence and no invented game counts. Frozen source changes require a new
protocol; historical manifests and copied L4 bytes are unchanged.

**Delete — do not maintain:** replaced `serving_benchmark._bounded_probe`'s
private polling/cleanup loop with the shared lightweight `manabot._process`
supervisor. Preserve numerical calibration and its existing adoption contract;
only process ownership is shared with the measurement runner. The new runner
does not execute the ablation tournament. The historical L4 job stays unchanged.

The parent supervises worker RSS/deadlines, terminate/kill escalation and a process group.
Whole-attempt wall includes verification, spawn/readiness, work, worker output,
collection and cleanup; its boundary explicitly excludes the final ledger write.
It never sums concurrent worker intervals. The report copies these digest-bound
receipts, including failures before game creation; prefill/decode and inclusive
billing remain unavailable. Optional device sampling is bounded, identifies UUID
and driver, and reports missing utilities as unavailable. No cache eviction occurs.

### Collection boundary — implemented 2026-10-26

**Delete — do not maintain:** the parent's synchronous `_compare_outputs` and
`inventory` calls are removed. After worker cleanup, one supervised collection
child receives the original absolute cell/invocation deadline and RSS ceiling.
Raw-file pins stream before comparison, so a deadline/RSS stop retains received
digests without rehashing in the parent. Collection cleanup and whole-attempt
wall remain inclusive; worker/collection supervision observations remain separate.
Only final self-describing receipt and ledger writes remain in the parent.
Incomplete child cleanup stops collection and leaves later reservations
unattempted; it cannot start another writer against unstable evidence.

An interrupted scan or comparison marks `inventory_complete=false`. Raw progress
stays on disk; the manifest binds received file pins and the final failed attempt
observation. Reports copy those pins and retain failure accounting, without
deriving game metrics from incomplete inventories. Unpinned extra files are not
consumed; mutation of any retained pin still fails. Complete/historical inventory
equality and separate ordinary/diagnostic timing populations are unchanged.
Controlled clocks, injected RSS and sleeping-child synchronization exercise
hashing/comparison interruption, cleanup and later reserved attempts. No load or
new empirical inference is induced. The accepted execution ceilings are unchanged.

Fixture results establish accounting and cleanup, not trained baselines.

Prior collection guards and LLM/report fixture checks passed; their retained evidence remains at the paths below. Current focused verification is recorded with the pairing correction.

Collection-bound evidence is retained in `.runs/inference-collection-verified/`
and `.runs/inference-collection-followup/`; final cleanup/deadline guards are in
`.runs/inference-collection-guards/`. Controlled clocks expire the invocation
or individual cell during hashing/comparison; injected RSS crosses the same cap
without allocating pressure. Received digests remain pinned, children are reaped,
invocation exhaustion leaves later directories absent, and failed work retains
its worker result and whole-attempt denominator. Partial policy/LLM reports
ignore malformed unpinned game files and reject mutation of pinned files.

Retained controlled evidence is under `.runs/inference-execution-verified/`;
repeated execution evidence for these reductions is under
`.runs/inference-compress-verified/`:
`test_complete_overlapping_game0` holds two completed policy games (275 Commands),
native replay and overlapping recorded worker intervals, scalar/shared retained-input witnesses and instrumentation-on/off parity;
`test_llm_recovery_and_failure_0` holds real native Commands with a scripted
initial/repair/fallback/runtime failure and separate latency populations.
Parent-clock fixtures assert failed work keeps its 14-second denominator despite
20 summed worker-seconds, and later reserved cells become unattempted. Readiness
synchronization precedes injected timeout, with child cleanup checked. The
negative digest fixture deliberately corrupts its final input; its already-built
portable report retains the original verified copy. No empirical baseline or paid
inference ran. New software evidence does not close slice 3.

### Comparison pairing — implemented 2026-10-26

Inspection found that retained-input comparison always selected the first completed
cohort cell. A scalar-first recipe could therefore compare shared instrumentation-off
against scalar execution, omit its available matching shared instrumentation-on cell,
and never check unchanged outputs or report instrumentation overhead. This is an
implementation gap in the accepted instrumentation comparison, not a changed protocol.

**Delete — do not maintain:** removed first-completed-only reference selection in
`inference_baseline.run`. The supervised collector selects a completed opposite-
instrumentation cell with otherwise equal compute and diagnostic settings when one exists, using
the latest such observation; otherwise it retains the first completed cohort
reference. Preserve exact input/mask/seed equality, numerical comparisons, output
witnesses, explicit reference identity, and the original collection deadline/RSS cap.

Controlled witnesses cover both instrumentation orders, the latest matching repeat,
exclusion of differently diagnosed cells, measured overhead and rejection of changed
outputs. The end-to-end native fixture now starts with scalar execution before
shared instrumentation-on/off cells. No remaining deletion target in this correction.

Check result, 2026-10-26: `uv run --extra dev pytest tests/eval/test_inference_baseline_execution.py tests/eval/test_inference_receipts.py tests/llm/test_llm_cache.py -q --basetemp .runs/inference-compression-review/tests` — 72 passed; targeted Ruff check/format, Python compilation and `git diff --check` passed. Reuse the preceding exact evaluation CI collection (235 tests; no new modules or test cases). Full affected suites and mixed CLI/headless proof remain with gate.

Remaining: trained checkpoint acquisition via `ArtifactStore.download` failed because `R2_ACCOUNT_ID`
is absent; no auth repair or empirical execution ran. Gate and empirical obligations
above remain open.

Compression handoff: continue sync, realign, review and authorized publication
without replaying completed implementation. The historical manifest builder
command with `--out .runs/inference-compression-review/historical-report` passed,
verifying all 1,708 pins and 160 corpus entries; it reports 160 selections,
308 Commands and one replay-verified game. Numeric historical measurements match
the older demo; that demo predates the implemented declaration/observation and
host-timing projection fields, so complete JSON equality is not expected.
Pre-compression uncommitted notes and changed source snapshots remain in
`.runs/inference-compression-review/before/`. Scratch is within both configured
limits; shared Wave memory was left as supplied. No empirical run, auth repair,
Task completion or publication acceptance follows from these fixture results.

Direct review, 2026-10-26: normal report generation, failed/partial attempts,
missing/restored evidence and once-only shared allocation accounting agree with
the accepted design. The completed mixed CLI/browser proof exercises those
interactions. No further concept change or unresolved product choice was found.
Runtime review preserves existing game/LLM owners, separate clock populations,
reserved attempts and supervised collection; remaining empirical evidence is
still required for Task acceptance. Automatic Task checks are disabled and the
CI watcher is not running. The optional broader evaluation run was interrupted
during the existing eight-game card audit after 26 passes and one CUDA skip;
the publication gate uses the design's affected suites instead. Earlier failures
were source admission of the new untracked helper, repaired with Git intent-to-add.

Final software check, 2026-10-26: affected receipt/execution, Perf, batched/local evaluation, all LLM and package-boundary tests — 161 passed, one CUDA-only skip in 688.55 s; `./scripts/verify-eval-performance-report` passed; changed-file Ruff check/format and `git diff --check` passed; exact evaluation CI collection found 235 tests. No empirical or paid run occurred.
