Comparing training frameworks previously required maintaining a separate MLX loss alongside the PyTorch implementation. The damped update now runs through either backend, with the same admission rules, independent GRU optimizer parameters, and a gated clipped-Adam comparison.

<!-- loopflow:task-pr-context:start -->
> [!NOTE]
> **Task:** [Make PyTorch the first matched backend and write the updates once over it: manabot/backends/torch, manabot/policy, and backend-generic damped/PPO updates · MTG-325](https://linear.app/loopflow/issue/MTG-325/make-pytorch-the-first-matched-backend-and-write-the-updates-once-over)
> **PR lifecycle:** The pull request is published for review; no Task settlement is requested.
<!-- loopflow:task-pr-context:end -->

## What changes

- Move PyTorch policy, checkpoint, loss and learner code into `manabot.backends.torch`. Updates take a backend through explicit Protocols; MLX differentiates the shared minibatch objective.
- Share NumPy admission, preparation, returns and diagnostics. Validate raw bootstrap and learner rows before mask conversion or indexing; cache admission and observation bucket needs across minibatches and epochs.
- Keep both GRU input and hidden biases as separate MLX parameters, preserving Adam moments and the clipping norm. Gate every weight after one clipped Adam step at `1e-6` absolute tolerance.
- Load supported containers through PyTorch with both `bucketed_shapes` and `live_tokens` honored. Repair the experiment checkpoint helper's removed imports, include backend update tests in CI, and remove committed scratch artifacts.

Production training, evaluation and Academy remain on PyTorch. MLX supports the scalar damped update for measurement; PPO and categorical critics remain PyTorch-only. Saving still produces `.pt`, and PyTorch's forward does not yet consume shared `Prepared` indexes directly (MTG-333). Removed import paths have no compatibility aliases; external callers must update their imports.

## Validation

After the GRU/admission corrections, fresh Apple Silicon parity passed on eight learner minibatches and eight collection calls. The capture contains 150 learner minibatches and 302 collection calls from a reduced 16-stream × 16-transition run. The old capture was refused for its stale world binding; no evidence was rebound.

| Comparison to PyTorch | Torch backend | MLX backend | Bar |
| --- | ---: | ---: | ---: |
| Learner logits, max absolute difference | 4.83e-9 | 5.88e-9 | 1e-4 |
| Learner values, max absolute difference | 1.79e-7 | 3.87e-7 | 1e-4 |
| Loss, relative difference | 0 | 4.34e-6 | 1e-3 |
| Gradient, minimum cosine | ~1 | 0.999999999758 | ≥0.999 |
| All weights after clipped Adam, max absolute difference | 7.45e-9 | 1.19e-7 | 1e-6 |
| Collection logits / values, max absolute difference | 0 / 0 | 4.89e-9 / 2.68e-7 | 1e-4 |

Clipping was active: pre-clip norms 3.4389486 (torch) and 3.4389496 (MLX), limit 0.267. The weight comparison includes near-zero-gradient tensors; only gradient cosine excludes negligible tensors. This establishes these numerical tolerances, not identical optimizer trajectories.

- PyTorch equivalence rerun: all **3,248 tensors match the retained A4 capture byte-for-byte**, covering outputs, losses, gradients, inference and trained weights. All five update-capture JSON records also match B4 byte-for-byte (diagnostics, residuals and recorded parameter sums).
- Focused follow-ups: **49 passed / 1 skipped**, then **7 passed** for the stricter actual updates and malformed-row rejection tests, then **172 passed** across conformance, model/inference, naming, boundaries, CI selection and release staging. MLX ran locally; the actual-update tests force clipping and compare every post-step weight.
- CI's exact training file list collects **358 tests**. The repaired experiment helper saves and loads a tiny policy. Ruff passes on all 201 changed Python files; actionlint and diff whitespace checks pass. No configured repository-wide type check.
- Earlier real-path QA covered recipe training, simulation, conversion, Academy opponents and interrupt/resume. Hosted CI and Linux/macOS CPython 3.12/3.13 package builds passed at `65fa8c7aa`; **CI for this follow-up remains for delivery**.
- Retained L4 measurement (Etude `8a376219e430`, revision `7820d1e8d`): three processes per arm with identical learner rows; damped throughput 9.63 → 9.94 steps/s, inference 1,282 → 2,296 observations/s. PPO median was 0.9% lower, within observed variation. This predates the admission/GRU follow-up and covers that configuration only. The current host was heavily loaded, so the new capture supplies numerical evidence, not speed evidence.

## Remaining limits

- Converted containers omit scalar `value_semantic`; PUCT refuses the converted scalar policy. Use its original `.pt`.
- Legacy non-ZIP `.pt` serialization is misclassified as `NotAContainer`.
- Health reporting and rollout storage still import Torch, so the MLX update requires Torch installed.

PR #501 remains the single PR; #503 is closed as superseded.
