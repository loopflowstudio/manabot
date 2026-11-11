# 855bff — decisions and remaining work

Inherited b015d5 execution, billing and delivery notes are historical, at
`402dfd5b2c8aaca6084b5dec446d058d249cddad:scratch/questions.md`. They grant no
execution/spending authority here. No paid runs or artificial contention.

Steer, Jack Heart (2026-11-11): "The benchmark should report mean rows per inference
call (environment_decisions / policy-inference calls from collection_timings)
and the idle fraction (idle_stream_steps / (calls x streams)) per size."
About `experiments/training_throughput_sizes.py`: "prefer not to edit that file".

Reversible headless choices:
- Local CPU is the initial reference candidate; recording is explicit and requires
  a complete clean-source run. No machine has been designated/adopted by this step.
  The reference-host question remains for review before committing measurements.
- Shared evaluation admits CPU/CUDA, not MPS (`EvalCompute.device`). Keep MPS
  training-only through explicit workload selection; reject a whole MPS suite.
  Adding an evaluation compute contract is outside this Task.
- Comparison requires an explicit pre-existing clean reference checkout with this
  suite interface and the recorded commit. No automatic worktree management.
- Stage timing rows already exclude warmup by reading Etude update-time series.
  Inference-call and idle counters exist only cumulatively; report those over all
  updates, explicitly labelled, rather than adding a second metrics store or
  modifying the training hot path just for this harness.

Implementation and focused checks are complete. Remaining for gate/delivery:
full-ladder runtime measurement, reviewed host selection, a committed real
reference, and an interleaved real-source comparison. Recording requires clean
source; checkpoint this implementation before measuring/adopting that reference.
No baseline JSON or speedup claim has been fabricated. MPS/CUDA were not exercised.

Validation: `uv run --locked --extra dev --extra evaluation pytest tests/training/test_pipeline_speed.py -q` — 18 passed; after the final receipt/printing changes, the same command with `-k 'reference or supplied_price or suite_interleaves or timeout'` passed 5 (19 total cases now present). Focused Ruff lint/format and `git diff --check` passed.

Bounded real CPU proofs (local receipts under `.runs/`, not experiment evidence):
- `uv run --locked --extra evaluation python scripts/bench_training_update.py --suite --size w64 --streams 2 --updates 2 --seconds 180 --out .runs/pipeline-speed-proof`: training completed two updates; its measured second update was 6.144 s / 399 learner decisions. Whole-stage batching was 541 calls, 1.743 rows/call and 0.1285 idle fraction. Evaluation then rejected an incorrect authored role name; the coordinator retained the failed receipt and exited nonzero after 32.34 s. Corrected the panel to `elemental_bg`/`elemental_gr` and added native admission coverage.
- `uv run --locked --extra evaluation python scripts/bench_training_update.py --suite --size w64 --workload evaluation --seconds 180 --out .runs/pipeline-evaluation-proof`: four games completed and replayed; coordinator elapsed 13.15 s, evaluation rate 0.5197 games/s, replay 1.729 s. These are narrow integration proofs, not default-ladder readings or GPU evidence.

Both supervisors completed cleanup; no started process remains. Frozen experiment
files, native rules, training semantics and existing evidence were unchanged.
