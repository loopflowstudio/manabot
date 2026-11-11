# 855bff — decisions and remaining work

Inherited b015d5 execution, billing and delivery notes are historical, at
`402dfd5b2c8aaca6084b5dec446d058d249cddad:scratch/questions.md`. They grant no
execution/spending authority here. No paid runs or artificial contention.

Steer, Jack Heart (2026-11-11): "The benchmark should report mean rows per inference
call (environment_decisions / policy-inference calls from collection_timings)
and the idle fraction (idle_stream_steps / (calls x streams)) per size."
About `experiments/training_throughput_sizes.py`: "prefer not to edit that file".

## Earlier headless choices (host choice superseded below)
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

The harness exists; acceptance is incomplete. `scratch/plan.md` owns the remaining
full-ladder measurement, committed reference and real-source
comparison. No baseline JSON or speedup claim has been fabricated. MPS/CUDA were
not exercised.

Validation: `uv run --locked --extra dev --extra evaluation pytest tests/training/test_pipeline_speed.py -q` — 18 passed; after the final receipt/printing changes, the same command with `-k 'reference or supplied_price or suite_interleaves or timeout'` passed 5 (19 total cases now present). Focused Ruff lint/format and `git diff --check` passed.

Bounded real CPU proofs (local receipts under `.runs/`, not experiment evidence;
both report dirty `402dfd5b2c8aaca6084b5dec446d058d249cddad` source before main sync):
- `uv run --locked --extra evaluation python scripts/bench_training_update.py --suite --size w64 --streams 2 --updates 2 --seconds 180 --out .runs/pipeline-speed-proof`: training completed two updates; its measured second update was 6.144 s / 399 learner decisions. Whole-stage batching was 541 calls, 1.743 rows/call and 0.1285 idle fraction. Evaluation then rejected an incorrect authored role name; the coordinator retained the failed receipt and exited nonzero after 32.34 s. Corrected the panel to `elemental_bg`/`elemental_gr` and added native admission coverage.
- `uv run --locked --extra evaluation python scripts/bench_training_update.py --suite --size w64 --workload evaluation --seconds 180 --out .runs/pipeline-evaluation-proof`: four games completed and replayed; coordinator elapsed 13.15 s, evaluation rate 0.5197 games/s, replay 1.729 s. These are narrow integration proofs, not default-ladder readings or GPU evidence.

Both supervisors completed cleanup; no started process remains. Frozen experiment
files, native rules, training semantics and existing evidence were unchanged.

## Main sync — 2026-11-11

Assumption (reversible): keep the pipeline-speed decisions and remaining work
active in this checkout. Main's MTG-271 dependency and release notes remain at
`9f69a96a1ca5f0c03a9df5af7217138390d415d9:scratch/questions.md`; their task scope
does not select work here. Main's code, dependency and evidence changes merge
unchanged.

## Realign — 2026-11-11

The reference-host recommendation in the Task brief is still a choice, not a
designation. Preserve the local CPU candidate assumption until demo resolves it;
the full-ladder and comparison acceptance remain required.

Merged Python 3.13 support exposed a comparison gap: the same machine/Torch
identity could cover different Python versions. Samples now include
`Host.python_version`; old smoke reports without it remain historical evidence,
not admissible references. None was a complete reference.

Validation: `uv run --locked --extra dev --extra evaluation pytest
tests/training/test_pipeline_speed.py -k 'different_host or suite_interleaves or reference'
-q` — 5 passed, 16 deselected. Focused Ruff lint/format and `git diff --check`
passed. No workloads were rerun. `lf context` reported memory and scratch below
both limits; no child Infrastructure memories were present.

## Concept review — 2026-11-11

Open product choice: which owned machine and device anchors the standing
reference? The mini is a recommendation in the Task brief, not a designation.
Do not turn the local CPU candidate assumption into reference adoption.
The working design records the question for demo and the full-reference admission
clarification. The host choice does not prevent the admission repair or its
controlled validation.

Reversible presentation choice: keep the existing command and receipt model,
and show the review in `scratch/concept-review.html`. No new CLI mode, service,
measurement, Task or execution is introduced by this review.

## Reference completeness repair — 2026-11-11

Validation: `uv run --locked --extra dev --extra evaluation pytest tests/training/test_pipeline_speed.py -k 'reference or suite_interleaves' -q` — 16 passed, 18 deselected in 3.24 s. Controlled fixtures cover subset rejection before workers, persisted admission errors, CPU/CUDA/MPS full-device admission and eight-row interleaving. Focused Ruff lint, formatting, Python compilation and `git diff --check` passed after applying Ruff formatting.

No measurements or host adoption occurred. Reference-host designation, clean
full-ladder elapsed time, committed baseline and real interleaved comparison
remain open in `scratch/plan.md`.

## Compress — 2026-11-11

Kept the shared reference-completeness rule and simplified fixture construction:
an attempt derives its workload and size from its sample. No further deletion
targets remain; the three receipt types preserve failed attempts without samples.

Validation: `uv run --locked --extra dev --extra evaluation pytest tests/training/test_pipeline_speed.py -q` — 34 passed in 8.56 s. Focused Ruff lint/format and `git diff --check` passed after formatting. `lf context` remains under both token and byte limits. No workload measurements were run.

Realign after the repair: the shared completeness check, early rejection and
persisted errors agree with the tests and development docs. Reused the 34-test
result above; no code or workload changed. Removed the stale deletion section
and routing verdict from the active plan. The reference-host choice remains for
demo, with all measurement obligations intact.

## Reference decision and shipping docs — 2026-11-11

Jack Heart, comment `262b941c-2188-4426-8ff0-35f7e52eaa10`, on the offered
standing hosts: "i dont really thin kany of these options are acceptable tbh".
On the reference-commit alternative: "no i think you have it right. this is a
strong reason to have at least one more stable gpu. maybe aws gpu perf is more
consistent?"

The supplied direction specifies no reference host: commit plus suite interface,
fresh interleaved reference/candidate measurements in one job on one machine,
and the same pod for a rented GPU. Per-run absolute rates remain history. Local
CPU runs prove the harness and its runtime, never GPU performance or a speed
claim. This supersedes every earlier host-choice question/candidate above.
Provider selection is undecided and outside this Task; no spending is authorized.

Reversible documentation choice: keep the existing `--record-reference`,
`--compare` and `--reference-root` flags and full-ladder recording prerequisite;
change the described baseline content to commit-and-interface metadata. This
uses the existing interaction without inventing a new command or file schema.
The implementation below now supplies that change. No new product question is required.


## Reference metadata implementation — 2026-11-11

Reversible choice: the baseline has exactly `commit` (full SHA) and
`suite_interface: 1`. A literal `SUITE_INTERFACE` declaration in the reference
module admits the worker protocol without importing or running that checkout.
No baseline was committed under the old Report representation, so it is removed
without a compatibility reader or migration. Historical run reports remain
readable evidence. The current comparison job supplies both sides' configuration;
recording on CPU does not restrict later jobs to that host, device or options.

Validation: `uv run --locked --extra dev --extra evaluation pytest tests/training/test_pipeline_speed.py -k 'comparison or reference or suite_interleaves or missing_or_failed' -q` — 39 passed, 9 deselected in 2.88 s. Focused Ruff, Python compilation, CLI help and diff checks passed. No benchmark workers, servers or watchers were launched; full-ladder runtime, baseline adoption and real-source interleaving remain for gate/measurement in `scratch/plan.md`. The fixtures do not establish device performance.
