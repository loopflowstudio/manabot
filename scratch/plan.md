# 855bff — standing pipeline speed

Implemented on `scripts/bench_training_update.py`: a bounded local suite with registered
training/evaluation workloads, the frozen four-size ladder, retained stage and
shared-serving receipts, warmup-excluded update timing, aggregate batching
utilization, behavior identities and supplied-price rates. No learning change.

Comparison reruns a clean reference checkout and candidate in alternating order
for three repetitions, retaining every sample and selecting whole minimum-duration
samples only after identity/behavior admission. The reference checkout is explicit;
the harness does not create worktrees or launch rentals. A complete clean-source
reading may be recorded; historical timings alone never produce a speedup.

Delete — do not maintain: no replacement subsystem. Keep the single-recipe path
and `_execute` as the sole training executor. Frozen experiment declarations,
receipts, recipes and native/model/collector semantics survive unchanged.

Compression: remove duplicated CLI lifetime/price admission and parallel attempt
lists; `run_suite` owns one progress report, while `_measure_attempt` owns worker
execution and sample admission. Samples and source identities remain immutable.
Training totals use their existing phase sums. Evaluation hashes the initialized
models directly instead of reloading both checkpoints only to hash their weights.
No remaining deletion targets.

Validation: `uv run --locked --extra dev --extra evaluation pytest tests/training/test_pipeline_speed.py tests/infra/test_ci_selection.py tests/infra/test_release.py -q` — 78 passed. The evaluation CI job now includes the benchmark tests; its complete invocation collected successfully (579 cases before the added checkpoint-identity test). Python compilation and focused Ruff checks passed.

Implementation decisions and remaining delivery work are in `scratch/questions.md`.
Gate owns broader suites and full-ladder measurement/reference adoption. Controlled
projection/comparison fixtures and bounded real CPU workload proofs passed; the
first complete reference and real interleaved comparison remain acceptance work.
