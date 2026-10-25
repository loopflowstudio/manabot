# Evaluation design evidence — 2026-10-24

## Final implementation reconciliation — 2026-10-24

The supplied outcome records successful sync after ordinary staging. The checkout
has base `884249024`, staged integration/compression and no unresolved index
conflicts; earlier stash errors are historical. Software is complete. Reused
evidence: 77 focused integration checks, 3 process regressions, 137-test collection,
then 9 compression checks plus lint/format/compile and calibration CLI help.
These are prior outcomes, not new test runs or full-suite certification.

Source inspection confirms:

- `eval/local.py` owns `checkpoint_request` and the module CLI;
  `eval/evaluation.py` owns `EvaluationRequest`, `EvaluationResult`, `evaluate`
  and `evaluate_local`. `sim.execution.EvalExecution` is the shared limit type.
  Local imports omit training/experiment, callback transport and hosted adapters.
  The old `manabot.arena` package is absent; frozen arena wire fields remain.
- `InferenceRuntime.infer` blocks on bounded multiplexed requests; `submit_game`
  returns a future. Proposed public `submit`/`receive` and `ServingReceipt` names
  never became APIs; `EvalPerformance` snapshots own the receipts.
- `local.main` sets the deadline before checkpoint preparation, then calls the
  evaluator. `match.play_cell` starts its timer after initial validation and
  snapshots after replay. Runtime construction/shutdown owns allocation elapsed;
  neither interval includes earlier CLI admission loads. `evaluation_seconds`
  stops before cohort statistics and final receipt persistence. These intervals
  are not total CLI latency, paid lifetime or whole-job cost. Runtime cost fields
  have no billing/rate inputs. The report's numerator includes only supplied cells.
- Etude 0.2.2, upstream card-evidence capture/projection, subset-leg execution and
  study-specific Duel/monitor rules remain intact. No source or test repair was
  needed in this reconciliation; durable usage docs now preserve the named later
  compute-policy/cluster, queue and adaptive-allocation increments.

Check: `uv run --no-sync python -m manabot.eval.local --help` and a fresh `uv run --no-sync python` import probe passed (documented symbols, excluded managed imports, absent arena); `git diff --check` passed. No games or broad tests rerun. Broad suites/report browser proof remain with gate/CI; CUDA and representative performance acceptance need hardware and a declared benchmark plan. No whole-job speedup, capacity choice or automatic adoption follows from CPU fixtures.

## Compute and local-entry follow-up — 2026-10-24

- `experiments/regimes/elemental-uniform-174-v1.json` declares width 64, two
  attention layers and four heads. It is an inspected recipe, not an assertion
  that this is the latest successfully trained or future default model.
  `model/agent.md` documents the 64/1, 64/2, 128/2 capacity ladder and optional
  384/8 stress case; architecture receipts own actual parameter counts.
- `experiments/worker-scaling.md`, physical single-GPU results: historical
  width-64/one-layer training showed a better rows-per-rental-dollar projection
  for its RTX 5090 host than its H100 host, excluding setup. Different CPUs,
  short runs and untrained policies limit extrapolation. Those historical rates
  are not current quotes or checkpoint-versus-checkpoint eval measurements.
- `sim/execution.py::EvalCompute` selects one CPU/CUDA device for
  all checkpoint policies in the allocation. `serving.py` groups by exact policy
  identity and executes batches serially in its service process; heuristics run
  in game workers. Multiple resident policies do not imply concurrent forwards.
- `eval/match.py::play_cell` provides reusable execution. The training monitor entry
  requires TrainingRun and coordinates; Duel has study-specific equality/cohort
  restrictions. That refinement-time local API gap is now closed by
  `eval/evaluation.py`, shared registration admission and `eval/local.py`.
- Compression inspection found that public `evaluate` accepted existing output
  directories, allowing a repeat to replace `evaluation.json` and traces even
  though local wrappers rejected repeats. It now owns fresh directory creation
  for all panel consumers. The one-deal CLI previously labeled its degenerate
  bootstrap range a 95% interval; it now reports unavailable single-deal
  uncertainty while retaining stored/frozen statistics. Calibration's separate
  completeness test could accept replayed partial panels; its named panels now
  wrap `EvaluationRequest` and use the shared evaluator/validator.
- [AWS purchasing options](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/instance-purchasing-options.html)
  separates usage-based On-Demand billing from 1/3-year Savings Plans commitments.
  Always-running is a lifetime policy, not itself a discounted purchasing option.
- [AWS G6](https://aws.amazon.com/ec2/instance-types/g6/) describes full L4 GPUs
  with 24 GB; [accelerated instance families](https://aws.amazon.com/ec2/instance-types/accelerated-computing/)
  lists L40S G6e examples with 48 GB per GPU. These illustrate memory classes,
  not a price/performance recommendation, capacity guarantee or selected vendor.
- [NVIDIA concurrent model execution](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_execution.html)
  distinguishes per-model scheduling/batching from concurrent model instances.
  This is mechanism evidence, not a proposal to adopt Triton or a claim that the
  current Manabot runtime overlaps models on the GPU.

## Initial inspection

Read-only source inspection at base `07a091bca` for [the draft](batched-eval-inference.md).
Historical measurements establish hypotheses, not permission or current speed.

- `manabot/arena/match.py::_execute_game`, `_bounded_game`, `play_cell`: fresh
  spawned process and player construction per game; serial deal/leg traversal;
  parent retains/fsyncs pending decisions before acknowledging native execution;
  actor decision seeds derive from player seed plus ordinal. Native timeouts and
  unattributed failures are distinct from player forfeits. Replay finalizes rows.
- `arena/monitor.py::checkpoint_manifest` and `arena/models.py::PlayerRegistration`:
  frozen explicit CPU, batch-size-one inference registration. New execution must
  change this admission contract explicitly. `arena/benchmark.py` and `duel.py`
  call `play_cell` for complete eight-leg deals and currently use one Etude worker.
- `sim/search/flat_mc.py::AgentMatchupPlayer` delegates flat decisions to
  `verify/util.py::_select_agent_action`; this uses ordinary tensor inputs, not
  `PolicyInference`. The checkpoint factory also supports compound and belief
  players, which cannot be treated as stateless flat forwards.
- `model/inference.py::PolicyInference` already owns a lazy immutable semantic
  catalog cache and CPU input admission. It rejects changed weights, model mode,
  config and closed scopes; preparation cannot enter autograd. Current collection
  uses it. `tests/model/test_inference.py` covers parity and invalidation.
- `sim/rollout.py::BatchedSampler` and `run_vector_games` already batch network
  calls, but use shared sampling, fixed mirror setups and aggregate `GameRecord`
  outputs rather than arena Commands/receipts. Preserve their existing consumers.
- `experiment/evaluation_queue.py::CheckpointQueue` owns a bounded single-worker
  checkpoint queue, machine lease, immutable job bindings, conservative crash
  charges and no retries. `arena/monitor_worker.py::follow_checkpoints` is another
  checkpoint follower. Neither inspection establishes a shared priority API.
- Local Etude source at `/Users/jack/src/etude/src/etude/eval/{match,tournament}.py`
  and `docs/evaluation.md` has complete-match workers, match-key result reuse,
  tournament scheduling and correlated match statistics. `launch/scheduler.py`
  owns machine/run admission and budgeted rentals. This is local upstream source,
  not proof that Manabot's pinned `etude-sdk==0.2.0` has every capability. No
  installed SDK was inspected or upgraded. Proposed API names are not existing APIs.
- `experiments/protocols/local-evaluation-route-012.md` records a historical
  same-host 2.8321× equal-work gain from four independent CPU evaluators, with
  canonical trace agreement. This was process concurrency, not shared inference.
  The preceding paid parallel probe009 was unlaunched; preserve its disposition.
- `experiments/worker-scaling.md` records prior shared-serving, preparation and
  CPU-producer investigations. One profiled training collection reduced catalog
  encoding from 199 calls to one; end-to-end gains did not follow automatically.
  Many actual batches were small despite thousands of resident streams. These
  training results do not certify current eval throughput, GPU superiority or a
  reusable production inference service. Historical `worker_experiment` code is
  not present under that name in this checkout.
- `docs/development/validation.md` supplies affected checks, including collective
  collection of new test modules. No model/world/native changes are proposed.

Context lookup: `lf task status --json` on 2026-10-24 returned
`Error: this checkout has no Task`. The supplied context manifest has no Wave.
No placement lookup, Task creation, launch, rental, training or evaluation followed.

## Historical performance-report and boundary inspection

These observations describe the pre-implementation design baseline. The final
implementation replaces the missing-child aggregation noted below; source
references and external mechanism evidence are retained, not newly reverified.

- `experiments/report_prototype/{README.md,definitions.py,build.py,report.html}`:
  Python-authored report definitions, per-job seed aggregation, baseline styling,
  resource-selectable icicles with shared drill-down and total/share/per-million
  learner-decision normalization. `TimeNode` separates recorded, derived,
  uninstrumented and hypothetical measurements. Real seat-comparison inputs do
  not have the detailed training split; the Elemental timing example is synthetic.
  The prototype is explicitly not the production Etude report.
- At that baseline, `report.html::jobTiming` substituted zero for missing child nodes
  when averaging across seeds. The eval projection must expose coverage rather
  than copy this behavior. No prototype code was changed during design.
- `manabot/infra/profiler.py` owns hierarchical host timers;
  `tests/training/test_phase_timing.py` verifies state preservation without added
  synchronization. This is a reusable contract, not a GPU execution timer.
- `managym/src/python/bindings.rs::step` includes native stepping and conversion
  into Python observation/info values. `vector_env_bindings.rs::step_into_buffers`
  already offers a coarser native boundary, including encoded output writes.
  Neither inspection nor a whole-call timer measures pure FFI overhead. Preserve
  arena evidence before adopting a vector transport.
- `uv.lock` pins training Torch 2.10.0 (CUDA 12.8 wheel on Linux). Consulted
  [PyTorch 2.10 CUDA semantics](https://docs.pytorch.org/docs/2.10/notes/cuda.html):
  GPU work is asynchronous, device timing needs events or synchronization, and
  graph replay can reduce host launch overhead. Routine host timers must not be
  labeled device compute time; synchronized diagnostic trials change execution.
- [PyTorch 2.10 compilation](https://docs.pytorch.org/docs/2.10/generated/torch.compile.html)
  documents `reduce-overhead` CUDA graphs, extra workspace costs, applicability
  limits and recompilation. These support a bounded experiment before a custom
  runtime, not a promised speedup on Manabot's dynamic batch shapes.
- [PyO3 performance guidance](https://pyo3.rs/main/performance.html) identifies
  conversion/allocation and interpreter attachment costs. This is mechanism
  guidance, not a benchmark of this repository's pinned extension or evidence
  that moving model inference to Rust would remove CPU/GPU transfers.
