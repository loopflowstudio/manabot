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

The latest 2026-11-12 UTC reconciliation read
(`git ls-remote origin refs/heads/main`) returned
`0b296e617a87017c4cca71227749eddac0a0481b`, already integrated through
`37ffe26657f394b10f69256555ff45b3c58e82e4`. Its hand snapshots, game-generation
tracking and reset changes alter five protected native source files. The
published preparation still passes all ten file digests and both arm bindings;
regenerating it from this newer runtime must reject, not refresh the frozen
sources. Preparation-compatible revision
`8037b78d4a6ff485c4ec50747a4e5072b2281f15` contains the scripts and has no
protected-source differences from the pinned arms.

`docs/workflows.md` requires same-machine software performance evidence and
separates delivery from scientific completion. The prepared separate serial L4
allocations remain descriptive: matching `HostFacts` cannot establish identical
physical hosts or isolate contention. Material host differences withhold the verdict.

## Execution blocker and remaining work

**Resolved at 2026-11-12 00:19 UTC (QA): the blocker was local TLS, not the
provider.** At 00:18:56 UTC the plain command above still failed in about 1 s.
`curl` of the exact URL (`https://api.runpod.io/v2/catalog/gpus?...`, same
headers) returned HTTP 200 with prices. The venv's python.org CPython 3.13
raised `SSLCertVerificationError ... unable to get local issuer certificate`.
Etude 0.2.15's `providers/runpod/http.py` maps any `OSError` to "no reply".
With `SSL_CERT_FILE=$(.venv/bin/python -I -c "import certifi;print(certifi.where())")`
prefixed to the same command, the read returned: `mtg281-l4  NVIDIA L4 with at
least 6 vCPUs, 48 GB at $0.59/hour (ceiling $0.60)  available  0 of 1 in use
rentable now  asked 2026-11-12 00:19:23 UTC`. No rental was requested. Launches
from this machine need the same `SSL_CERT_FILE` prefix. Etude's error wording
hides a local certificate failure; that is a possible Etude follow-up.
Proposed Wave memory correction (read-only here): the four "no reply" reads were
local certificate-verification failures, not provider outages.

The earlier record follows unchanged.

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
Paid execution needs a successful provider read and affordable quotes; repeating
preparation supplies neither.

The same bounded availability command at 2026-11-11 23:47:57 UTC exited 1
with `etude: GET catalog/gpus: no reply`. It requested no rental; no quote,
run or spending resulted. Published preparation and preflight bytes remain
unchanged. Provider visibility and fresh affordable quotes remain missing.
No provider read or rental was requested in this reconciliation.

The preparation test now separates synthetic declaration/budget admission from
unpatched current-runtime rejection and an injected world mismatch. The fixture
uses pinned source hashes and offline regime validation; real job bundles,
archives, observation shapes and output hashes remain checked. No games or
training use the fixture. Production source/world guards and published inputs
remain unchanged. The matching native extension was rebuilt and installed
atomically; this fixes test collection without making current sources compatible
with the frozen pair. The subsequent combined check passed 22 tests, including
the extracted rental-wrapper checks. The earlier feedback's fixture-repair and
extension-rebuild work is complete; gate still owns broader validation. No code
repair is currently needed before obtaining provider visibility and quotes.

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

- Rental-wrapper compression:
  `PYTHONPATH=. uv run --no-sync pytest tests/training/test_collection_comparison_preparation.py tests/infra/test_collection_comparison_report.py tests/infra/test_collection_growth_capture.py tests/infra/test_experiment_data_untracked.py -q --basetemp=.runs/collection-growth/compress-budget-after-tests`
  — 22 passed in 20.57s. Both generated job declarations match the pre-edit
  `compress-budget-before-tests/` bytes; all 19 published files match the manifest.
  Python compilation, changed-file Ruff check/format and `git diff --check` passed.
  No games, training, provider reads or rentals ran; the existing extension sufficed.
- Preparation fixture repair: CPython 3.13 release wheel rebuilt and matching
  extension installed atomically.
  `PYTHONPATH=. uv run --no-sync pytest tests/training/test_collection_comparison_preparation.py tests/infra/test_collection_comparison_report.py -q`
  — 13 passed in 16.70s, including unpatched current-runtime rejection.
  No game, training, provider read or rental ran.
- Post-sync reconciliation: `prepared_runs` verified ten digest-bound prepared
  files and both pinned arms. Comparing protected sources found five changed
  native files; the older preparation-compatible revision has zero differences.
  `PYTHONPATH=. uv run --no-sync pytest tests/training/test_collection_comparison_preparation.py -q`
  stopped during collection: `AttributeError: module 'manabot.managym' has no
  attribute 'HandSnapshot'`. It executed no test, game or rental. The local
  extension predated the merged Python API. The fixture-repair and combined
  checks above supersede this collection failure.
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

The fixture-repair and combined checks above close the post-sync preparation test
gap; earlier checks remain historical evidence. Reconciliation inspected the
fixture, unpatched rejection and production admission guards without repeating
tests. Upstream main remains the already-integrated `0b296e617`; no new runtime
change requires another extension rebuild. No paired execution occurred.
`git diff --check` passed; `lf context` kept scratch and Wave memory within both
limits. The complete Wave memory was read; it has no immediate child directories.
Earlier preparation and compression detail remains at
`c995c00e719398a0f328ee4f9b3cf1050acb6625:scratch/collection-growth.md`.

## Maintained boundaries

No production path is replaced by these measurement tools. Capture has one CLI
operation. Published snapshots, frozen analysis, source archives and historical
metric compatibility remain intact.
The declaration test's assumption that the live checkout equals the frozen
runtime is removed. Recipe, archive and budget assertions use the explicit
fixture, with separate real-runtime rejection coverage.

The rental shell wrapper now belongs to
`scripts/prepare_collection_comparison.py::_bound_rental`; its controlled-clock
test runs independently of source archives and native admission. This removes
the repeated shell checks nested inside each arm's archive test. Preparation
checks recipe/source admission and serialization; deadline arithmetic and
generated job bytes are unchanged. No further code deletion targets remain.

## Concept review (2026-11-11 America/Los_Angeles)

Outcome for loop-or-next: **nothing to change and no open product choice**.
Advance this review; scientific execution remains incomplete. No next implement
direction is needed. The retained provider failures require visibility and fresh
quotes, not another preparation cycle or a different experiment.

Keep the experience as “inspect retained growth, execute the frozen pair within
the allowance, then read a qualified comparison.” Preparation is an immutable
input record, not a live execution status. Rental state belongs to Etude;
scientific completion belongs to each delivered TrainingRun; the verdict derives
from bound captures and receipts. These distinctions preserve recovery and
accounting, so merging them into one success flag would remove useful meaning.
The existing usage guide and performance-review skill already express the
necessary boundaries; no replacement instructions are proposed.

The normal path consumes the published preparation, executes separately quoted
arms, captures each receipt's metrics_run and produces a new comparison report.
Failure walkthroughs retain the same contract: failed availability reads do not
invalidate preparation; uncertain acquisition requires reconciliation before
another acquisition; changed local native sources reject preparation replay;
partial delivery permits a new capture of the same run, never invented rows or
a replacement scientific attempt. Original identities, attempts, costs and
deadlines survive throughout.

Counterexamples checked against the implementation: a launcher can finish while
training pauses at 480 updates; a receipt can point at another metric run; the
baseline can remain flat; two rentals can share HostFacts without sharing a
physical host. The first three withhold or reject the conclusion. The last
limits even a complete comparison to the declared descriptive early-growth
scope, not same-machine software speed evidence. Keep that qualification next
to the result; neither a larger horizon nor a shared rental is selected here.

The review page is `scratch/concept-review.html`. What to show next is the actual
paired report with its limitations, delivered inventories and rental accounting
once execution is possible. This review ran no tests, provider reads or rentals
and changed no implementation or frozen evidence.
