# MTG-281 collection growth

Jack Heart, Task directive dated 2026-11-11: "do 1 and 2. You have $15".
Retained historical attribution precedes a matched CUDA before/after measurement
of the outcome-table repair. The $15 covers all rental costs, including setup,
failures and teardown. Frozen frontier runs, the arena disk and EXP-46 are outside
this change. Scientific acceptance remains open independently of software delivery.

## Evidence and approach

`experiments/self-play-collection-273.md` owns reproduction, thresholds, paired
input mapping and failure recovery. Seven manifest-published historical artifacts
retain 1,744 w64 (`59718fe34945`) and 932 w160 (`5b7d8fa7b38c`) updates. Both show
RSS growth and collection spikes after permanent-block endpoints. At update 900,
each has 81 permanent blocks and about 454 MB of headers. The w160 flat-memory
counterexample fails; models, damping schedules, game lengths and hosts still
limit attribution. Exact historical manifest sizes/fold times, CPU quotas/models,
final game-error counts and Python-allocation traces were unavailable at capture.

`scripts/measure_collection_growth.py::capture` projects one block at a time,
rejects multiple writers/attempts and batches remote array ranges. The normal
SDK reader expands omitted matchup columns. `report_collection_growth.py::reading`
reconstructs header-byte totals; `fold_alignment` associates updates with block
endpoints, not measured fold times. New snapshot/report destinations protect
retained inputs. No production training path is replaced.

Both 500-update jobs, source bundles/archives, local world/shape admission and
pre-execution analysis are published under
`experiments/data/collection-growth-281/prepared/`. Before source is
`b7c102f2c4266dab4be91b8b54901861745b9a3c`; after is
`1659b0e290f55daa632ec8f869adff4a3059d79f`. Native sources and dependency locks
agree. Seed 21501, explicit CUDA/L4, declared capacities and damping horizon 2474
are frozen. Remote CUDA numerics and runtime admission remain per-arm gates.

`scripts/collection_comparison.py::Analysis` and
`scripts/report_collection_comparison.py::compare` own the frozen descriptive
criteria recorded in `scratch/questions.md`. All three baseline primaries must
grow; incomplete or materially mismatched evidence withholds the verdict. The
500-update horizon cannot certify the later 850–1700-update regime.

The 2026-11-11 reconciliation read (`git ls-remote origin refs/heads/main`)
again returned `830161795f3ae65563ae07128f14e90dfaf90235`, already integrated. Its
`docs/workflows.md` adds same-machine comparison guidance and separates software
delivery from scientific completion. The prepared separate serial L4 allocations
remain descriptive: matching `HostFacts` cannot establish identical physical
hosts or isolate contention. Material host differences withhold the verdict.

## Execution blocker and remaining work

Provider reads at 2026-11-11 22:42:24 and 23:15:36 UTC failed with
`ProviderUnavailable: GET catalog/gpus: no reply`. Published
`comparison-preflight.json` retains no run, rental or quote; new rental spend is $0.

At 23:36:06 UTC,
`doppler run --project manabot --config prd -- doppler run --project etude --config prd -- uv run --no-sync etude availability mtg281-l4 --timeout 30`
exited 1 with `etude: GET catalog/gpus: no reply`. An earlier invocation with only
`etude/prd` stopped at missing `R2_ACCESS_KEY_ID`; nested injection reached the
provider. Neither command requested a rental. No launch, quote or additional
spend occurred; published preflight stayed unchanged. Preparation is complete
and needs no repetition while availability and fresh quotes remain missing.
The provider read failed; it did not establish a capacity or price refusal.
The next missing input is a successful provider read and affordable quotes,
not another software change or preparation run.

The same bounded availability command at 2026-11-11 23:47:57 UTC exited 1
with `etude: GET catalog/gpus: no reply`. It requested no rental; no quote,
run or spending resulted. Published preparation and preflight bytes remain
unchanged. Implementation has no remaining code change; execution still needs
provider visibility and fresh affordable quotes.

1. Obtain fresh capacity and individual whole-rental quotes for both arms.
   Reserve at most $7.20 each plus $0.60 ancillary, cumulatively $15 including
   disk, storage/transfer and teardown. Quotes reserve no capacity; recheck before
   the second launch. Historical duration is no completion guarantee.
2. Publish/admit the pinned bundles and execute each once, serially as separate
   allocations. The registered `mtg281-l4` supply binds startup 900s, delivery
   600s, zero added setup and one slot/machine. Each job requests 42,900s with
   300s teardown allowance inside 12 hours; watchdog 39,600s, finalization 600s,
   numerics/admission allowances outside the watchdog and max_attempts=1.
   Setup derives its cutoff from the original deadline. Do not combine arms in
   a cohort that extends a rental. Stop a dud/slow arm without extending bounds.
3. Reconcile uncertain acquisition state before another acquisition; a failed
   status read is not a refused create. Retain every attempt, TrainingRun status,
   all 500 updates, delivered output inventory, host/device/provider machine
   identity, errors/recovery, billing and confirmed teardown. Deletion is best
   effort. Release ended reservations only against retained accounting; keep
   unknown/live liability reserved within the cumulative cap.
4. Capture each new metric run and analyze the four paired inputs with
   `scripts/report_collection_comparison.py`; publish measured evidence and
   extend the report. No repaired-arm verdict exists. Missing baseline growth,
   incomplete targets or attribution limits must remain explicit. If residual
   growth remains, name the best-supported candidate and file a follow-up;
   further profiling is outside this Task.

## Retained checks (2026-11-11)

- Latest blocker-note check: `git diff --check` passed; `lf context` reports
  scratch and Wave memory within both token and byte limits. No code changed.
- Shipping docs: `experiments/self-play-collection-273.md` now leads with
  retained-plot reproduction; `experiments/README.md` links directly to the
  distinct paired inputs. Existing commands, criteria and evidence are preserved.
  Both report CLIs' `--help` matched the documented options; `git diff --check`
  passed. No product question was exposed. `lf open --host desktop --file` for
  both docs reported "No Desktop Files view attached; nothing opened."
- Focused compression check:
  `PYTHONPATH=. uv run --no-sync pytest tests/infra/test_collection_comparison_report.py tests/training/test_collection_comparison_preparation.py tests/infra/test_collection_growth_capture.py tests/infra/test_experiment_data_untracked.py -q`
  — 18 passed; Ruff and `git diff --check` passed. Eight reports matched prior
  values/limitations (ignoring metric-set issue order); both prepared job
  declarations were byte-identical. Evidence:
  `.runs/collection-growth/compress-{before,after}-tests/`.
- Historical regeneration in `.runs/collection-growth/compress-report/` and
  `.runs/collection-growth/docs-check-20261111/report/` matched published
  `summary.json`, `phases.png` and `growth.png` byte-for-byte. Existing output
  was refused. All 19 published files matched the manifest. CLI help matched
  documentation; paired-reader import passed with Torch/native imports blocked.
- The CPython 3.13 native release wheel built and its matching extension was
  installed atomically. Gate collection initially found 161 tests then failed
  for missing native extension. After rebuild,
  `PYTHONPATH=. uv run --no-sync pytest --collect-only tests/infra/ tests/academy/test_clean_machine_verifier.py deploy/play/tests/ -q`
  collected 172 tests in 1.56s. This closes collection only; gate/CI still owns
  broader suite execution. No games, training or rentals ran in those checks.

Reconciliation and the latest blocker note change documentation only; existing
checks remain applicable. No paired execution occurred. Earlier preparation
and compression detail remains at
`c995c00e719398a0f328ee4f9b3cf1050acb6625:scratch/collection-growth.md`.

## Delete — do not maintain

No production path is replaced by these measurement tools. The redundant
capture subcommand was removed; do not restore a parallel CLI. Keep published
snapshots, frozen analysis, source archives and historical metric compatibility.
No further deletion is needed for the remaining execution work.
