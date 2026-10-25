# Evaluation design evidence — 2026-10-24

## Intermittent capacity price check — 2026-10-24

[Runpod's pricing page](https://www.runpod.io/pricing), marked updated October 8,
2026, advertises Pods including A5000 24 GB / 9 vCPU / 25 GB RAM at $0.27/h;
A6000 48 GB / 9 vCPU / 50 GB RAM at $0.59/h; 5090 32 GB / 9 vCPU / 35 GB RAM
at $1.19/h. These are displayed configurations, not live reservations or
guaranteed available offers; storage and other charges are additional.
Illustrative 12-hour weekly compute totals are $3.24/$7.08/$14.28, versus
$45.36/$99.12/$199.92 at 168 hours. No experiment runtime is inferred.

[Vast pricing](https://vast.ai/pricing) describes market-dependent rates and
interruptible instances that may be reclaimed. Its static page did not expose
a usable GPU-specific quote. It is a comparison candidate, not evidence of a
lower total cost for this workload. Continuation and off-host evidence retention
must work on the chosen route before interruption savings can be assessed.

Historical `experiments/worker-scaling.md` favors its measured 5090 host over
its H100 host in training rows per rental dollar, excluding setup. The document
also records sensitivity to host CPU and native/input work. This supports a
commodity-GPU/CPU-balanced eval trial, not a measured eval winner. The current
software has no representative CPU/CUDA eval calibration or whole-bill result.

Check: documentation-only refinement; `git diff --check` passed. No runtime,
hardware or paid evaluation checks run.

## Published foundation

PR #370 at `79506de9cae0182653e251767d35ffba5427c066` owns the completed
runtime/report keystone. Full reconciliation and prior checks are retained in
Git blob `7bae0e735e83f08273f093dbea10422ed6f416db`. Its focused CPU evidence
never established current GPU throughput, a capacity choice or whole-job gain.

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

Historical report/boundary inspection is preserved in Git blob
`7bae0e735e83f08273f093dbea10422ed6f416db`; current ownership and measured limits
are in the design and `docs/shared-evaluation.md`.

## Publication check — 2026-10-24

The first publication attempt stopped before `lf pr publish`: `lf sync` succeeded without conflicts,
with main `66b8fb3e7` already integrated through `003df4e6f`. The accepted
continuation and activity-chart changes remain staged. PR #370 is open, ready,
at `003df4e6f`, with auto-merge absent. The local demo on port 8767 remains running.

The broader publication-time run was:

```sh
uv run --no-sync pytest tests/eval/test_local_evaluation.py tests/training/test_etude_evaluation.py tests/training/test_head_to_head.py tests/eval/test_card_evaluation.py tests/eval/test_eval_performance.py tests/infra/test_ci_selection.py tests/infra/test_release.py tests/training/test_monitor_evaluation.py::test_checkpoint_admission_failure_is_saved -q
```

It reached **37 passed, 1 failed**, then was interrupted to inspect the failure;
this is not a completed suite. `test_native_evaluation_resumes_delivers_and_audits`
retained eight 60-second timeouts with successful replay, then `card_data` failed
with `BinderException: Referenced column "win" not found`. Its all-excluded
Etude table lacks the nullable win column. `card_query.py` and
`card_evaluation.py` are unchanged from main; no baseline reproduction was run,
so this observation does not establish when the bug was introduced. The native
test also expects eligible card statistics, which this all-timeout run cannot
provide. Neither the query nor the success assertion was weakened for publication.

The original failure was under pytest-579's temporary output; later pytest
retention cleanup removed it. The query failure was reproduced cheaply through
Etude with all-excluded fixture games before fixing it. `card_data` now returns
the retained exclusions and no rates when there are no eligible player-games,
without requiring an all-null `win` column. Eligible-game queries still require
their win data. Regressions cover crash, timeout, truncation, replay failure and
empty filtered selections through Etude's real schema inference.

Native success was verified separately with the unchanged 60-second limits:
all eight games terminated in 1.7–5.2 seconds, replayed, delivered and audited.
Evidence is in `.runs/card-native-audit-proof`; a preceding one-leg diagnostic
is in `.runs/card-native-probe`. The earlier timeouts' cause is unestablished;
the query fix does not change execution or time limits. All card tests now pass
across two focused invocations; this is not a completed rerun of the original
combined command. The separate required
two-hour useful-backlog scheduler gate remains unimplemented as documented.
Scoped Ruff checks, formatting (28 files), and staged diff whitespace checks passed.
The remaining lightweight selection (performance projection, CI selection,
release contracts and the corrected admission-failure fixture) then passed:
`uv run --no-sync pytest tests/eval/test_eval_performance.py tests/infra/test_ci_selection.py tests/infra/test_release.py tests/training/test_monitor_evaluation.py::test_checkpoint_admission_failure_is_saved -q`
— 48 passed in 1.52 seconds.

Check: `uv run --no-sync pytest tests/eval/test_card_evaluation.py::test_native_evaluation_resumes_delivers_and_audits -q --basetemp .runs/card-native-audit-proof` — 1 passed in 50.28s; `uv run --no-sync pytest tests/eval/test_card_evaluation.py -k 'not native_evaluation_resumes' -q --basetemp .runs/card-publication-proof` — 33 passed in 65.37s; scoped Ruff and whitespace checks passed.
