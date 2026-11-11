# MTG-281 collection growth

Jack Heart, Task directive dated 2026-11-11: "do 1 and 2. You have $15".
The outcome is retained historical attribution followed by a matched CUDA
before/after measurement of the outcome-table repair. The $15 covers every
rental cost, including setup, failures and teardown. Frozen frontier runs,
the arena disk and EXP-46 remain outside the change.

## What exists

`experiments/self-play-collection-273.md` and seven manifest-published artifacts
retain 1,744 w64 (`59718fe34945`) and 932 w160 (`5b7d8fa7b38c`) updates.
Both show RSS growth and collection spikes after permanent-block endpoints.
At update 900, each has 81 permanent blocks and about 454 MB of headers; the
larger w64 final index reflects its longer history. The w160 flat-memory
counterexample fails, but different models, damping schedules, game lengths
and hosts still limit attribution. Exact historical manifest sizes/fold times,
CPU quotas/models and final game-error counts were unavailable at capture.
No Python-allocation trace isolates index memory from native/tensor memory.

`scripts/measure_collection_growth.py::capture` projects one block at a time
and rejects multiple writers/attempts; `_arrays` batches ranges to avoid a
remote request per metric. The normal SDK reader materializes omitted matchup
columns. `scripts/report_collection_growth.py::reading` reconstructs cumulative
header bytes; `fold_alignment` associates updates with endpoints, not measured
fold times. Exclusive snapshot creation and new report directories preserve
retained inputs. No production training implementation is replaced here.

The upstream read on 2026-11-11 (`git ls-remote origin refs/heads/main`) returned
`1659b0e290f55daa632ec8f869adff4a3059d79f`, already integrated. Its parent is
`b7c102f2c4266dab4be91b8b54901861745b9a3c`; those remain the comparison sources.
The pair's diff does not change native sources or dependency locks. That code
inspection does not replace admission of the prepared runtime bundles.

## Delete — do not maintain

No production implementation is replaced. Preserve the historical captures,
original recipe, original comparison envelope and MTG-273 reproduction. The
new preparation and paired reader extend the measurement; neither changes the
pinned workers. No predecessor code or data needs deletion.

Completed reductions: the paired reader's `_validate_arm`, `_pair_limitations`
and `_analyze` separate evidence admission from numerical interpretation.
`ArmAnalysis` keeps each arm's statistics, scores and limitations together;
consecutive update slices replace repeated masks, and standard `asdict`
replaces the custom statistics serializer. The preparation test reads the gzip
archive directly. No remaining deletion targets; published analysis bytes stay
frozen while the maintained reader preserves their calculations.

## Prepared comparison and remaining execution

Both 500-update TrainingJobs, their source hash bundles and compressed git
archives are frozen in `experiments/data/collection-growth-281/prepared/`.
Seed 21501, CUDA/L4, all declared capacities and damping horizon 2474 remain.
Native world, deck source and shape admission passed locally; remote CUDA
numerics and runtime admission remain per-arm gates. Exact source/analysis
bytes and the local admission receipt are retained with hashes.

Registered Task supply `mtg281-l4` binds startup 900s, delivery 600s, no added
setup seconds and one slot/machine. Each job requests 42,900s, leaving 300s
for teardown in the 12-hour cost envelope; watchdog 39,600s, finalization 600s,
with numerics/admission allowances outside the watchdog. Setup derives its
cutoff from the original rental deadline. The worker's existing stop/export
path and max_attempts=1 remain. Launch arms separately, serially; combining
both in a cohort would extend one rental and invalidate this deadline contract.
Provider deletion is best effort, not a guaranteed invoice bound.

The pre-execution reader freezes updates 51–500, endpoint windows 51–100 and
451–500, <=10% timing growth and <=1 GB RSS growth as practical flatness.
All three primaries must reproduce growth before an overall verdict; >=50%
reduction in every score qualifies partial flattening. Incomplete targets,
missing metrics and material host/work/error differences withhold verdict.
The report defines units and exact thresholds; the prepared artifacts retain
the implementation. These thresholds are engineering choices, not confidence
bounds or a later-regime claim.

At 2026-11-11 23:15:36 UTC, fresh availability for the configured Task supply
failed with `ProviderUnavailable: GET catalog/gpus: no reply`. The retained
`comparison-preflight.json` contains no run, rental or quote; new rental spend
remains $0. The earlier 22:42 refusal remains historical. This blocks spending,
not preparation, which is now complete.

Remaining:
1. Obtain fresh individual whole-rental quotes and capacity; account for disk,
   storage/transfer and teardown within both $7.20 reservations plus $0.60
   ancillary, cumulatively $15. Quotes reserve no capacity. Keep both arms'
   allocation reserved before the first launch; recheck before the second.
2. Execute each pinned job once, retaining every attempt. Stop a dud/slow arm
   without extending its bound. Confirm TrainingRun status and all 500 updates,
   not just launcher success. Verify the delivered output inventory and retain
   host, device, errors/recovery, billing and confirmed teardown.
3. Capture each new run, read the four paired inputs with
   `scripts/report_collection_comparison.py`, then publish measured evidence
   and extend the report. A residual-cause follow-up depends on this result;
   no further profiling belongs here.

No repaired-arm verdict exists. If growth fails to recur before the repair,
measurements are incomplete, or host/work differences prevent attribution,
report that limit. Even a flat repaired arm through 500 cannot certify the later
regime. If residual growth is observed, name the best-supported candidate and
file a follow-up; further profiling is outside this Task. Scientific acceptance
remains open independently of software publication or verification.

## Retained checks

`PYTHONPATH=. uv run --no-sync pytest tests/infra/test_collection_growth_capture.py tests/infra/test_experiment_data_untracked.py -q`
— 7 passed; Ruff passed. Offline regeneration in
`.runs/collection-growth/compress-report/` produced byte-identical `summary.json`,
`phases.png` and `growth.png`; existing evidence-directory output was refused.
All seven retained files matched the manifest. No rental or training ran.

Gate collection of `tests/infra/`, `tests/academy/test_clean_machine_verifier.py`
and `deploy/play/tests/` collected 161 tests, then failed on
`test_runtime_config.py` because the native extension was unavailable. Gate/CI
still owes collection with the built runtime; this diagnostic needs no native
imports. These checks are retained results, not rerun by reconciliation.

Implementation checks (2026-11-11):
`PYTHONPATH=. uv run --no-sync pytest tests/infra/test_collection_comparison_report.py -q`
— 10 passed; `tests/training/test_collection_comparison_preparation.py` — 1
passed; `tests/infra/test_experiment_data_untracked.py` — 1 passed. Ruff and
`git diff --check` passed; the paired reader imported with Torch/native imports
blocked. Twelve new preparation/preflight artifacts were published; all 19
historical and new files verified against the manifest. The CPython 3.13
native release wheel built and its matching extension was installed atomically.
The earlier missing-native gate collection is now runnable, but remains a gate
obligation; no broad suite, training or rented execution was run in this step.

Compression check (2026-11-11):
`PYTHONPATH=. uv run --no-sync pytest tests/infra/test_collection_comparison_report.py tests/training/test_collection_comparison_preparation.py tests/infra/test_collection_growth_capture.py tests/infra/test_experiment_data_untracked.py -q`
— 18 passed; Ruff and `git diff --check` passed. Eight generated comparison
reports match all pre-refactor values and limitations (ignoring issue order
from metric-set iteration); both prepared job declarations remain byte-identical.
All 19 published files still match the manifest. Evidence is under
`.runs/collection-growth/compress-{before,after}-tests/`; no paid execution ran.
