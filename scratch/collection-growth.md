# MTG-281 collection growth

Jack Heart, Task directive dated 2026-11-11: "do 1 and 2. You have $15".
Retained historical attribution precedes a matched CUDA before/after measurement
of the outcome-table repair. The $15 covers all rental costs, including setup,
failures and teardown. Frozen frontier runs, the arena disk and EXP-46 are outside
this change. Scientific acceptance remains open independently of software delivery.

Current outcome (2026-11-12 UTC): **inconclusive**. The baseline stopped at
251/500 and its provider deletion is confirmed. Both fresh after-arm quote
attempts refused L4 capacity before submission. No active Task rental remains.
The execution evidence and qualified report are published; the paired scientific
acceptance is unmet.

## Open decision: remaining budget cannot buy the frozen pair

The iteration direction asks to "execute the pinned arms serially within the
cumulative $15 cap". About $3.78 is spent (whole-lifetime estimate), which
leaves about $11.22. The frozen envelope reserves $7.20 per arm plus $0.60
ancillary ($15.00 for two arms). Expected compute does not fit either: the
historical 500 updates took 10.05 h ($5.93 per arm, $11.86 per pair), and the
baseline's own pace (251 updates in 4.53 h) projects about $5.3 per arm before
growth. Only one frozen-envelope arm ($7.80) fits. The baseline also stalled
for an unknown reason, so a retry may stall again. Any of these choices needs
a person:

- (a) Run only the after arm (≤$7.80, total ≤$11.58). Under the frozen reader
  the verdict stays inconclusive, because the baseline has no 451–500
  endpoint. Comparing updates 51–251 would be a new, post-hoc endpoint.
- (b) Raise the cap to about $23 so the baseline can be retried and the after
  arm run, both under the frozen envelope.
- (c) Stop here and close the Task as inconclusive, with the published evidence.

Under the cumulative cap no paid launch is authorized until one is chosen.
Shrinking the per-arm reservation to fit would change the frozen preparation.

## Evidence and approach

`experiments/self-play-collection-273.md` owns reproduction, thresholds, paired
input mapping, execution record and failure recovery. Seven manifest-published
historical artifacts retain 1,744 w64 (`59718fe34945`) and 932 w160
(`5b7d8fa7b38c`) updates; both show RSS growth and collection spikes after
permanent-block endpoints. Exact historical manifest sizes/fold times, CPU
quotas/models, final game-error counts and Python-allocation traces were
unavailable at capture.

Both 500-update jobs, source bundles/archives, local world/shape admission and
pre-execution analysis are published under
`experiments/data/collection-growth-281/prepared/`. Before source is
`b7c102f2c4266dab4be91b8b54901861745b9a3c`; after is
`1659b0e290f55daa632ec8f869adff4a3059d79f`. Seed 21501, explicit CUDA/L4,
declared capacities and damping horizon 2474 are frozen.

Main's `0b296e617` (integrated through `37ffe2665`) changes five protected native
source files. The published preparation still passes all ten file digests and
both arm bindings; regenerating it from the newer runtime must reject.
Preparation-compatible revision `8037b78d4a6ff485c4ec50747a4e5072b2281f15`
has no protected-source differences from the pinned arms.

Separate serial L4 allocations are descriptive: matching `HostFacts` cannot
establish identical physical hosts or isolate contention
(`docs/workflows.md`). Material host differences withhold the verdict.

## Execution facts (2026-11-12 UTC)

Provider reads from this machine need
`SSL_CERT_FILE=$(.venv/bin/python -I -c "import certifi;print(certifi.where())")`:
the python.org CPython 3.13 venv fails RunPod TLS verification, and Etude
0.2.15's `providers/runpod/http.py` reports any `OSError` as "no reply". The
four pre-execution "no reply" reads (2026-11-11 22:42–23:48 UTC) were local
certificate failures, not provider outages (now in Wave memory);
possible Etude follow-up for the error wording.

Both fresh individual quotes were $7.0308333333 at $0.59/hour for 42,900 seconds.
Reservations were $7.20 per arm plus $0.60 ancillary, $15 total. Initial
inventory found no Task machine, Run or unsettled session; the two historical
frontier pods remained untouched.

Baseline `d053e8a1f225` (launch `15e782abd6e2`, attempt `5a910b054eaa`) acquired
`mtg281-l4-c7be7a664da0` / pod `9gt72h4djozdfz` at 00:30:51 UTC, passed CUDA
numerics, and stopped progressing at 251 updates around 05:02 UTC. The polling
batch did not escalate the unchanged clock until 06:50 UTC; that idle time is
included in cost. Direct reads found the main thread waiting and one busy thread;
they do not isolate a cause. Deletion at 06:55:24 UTC and provider absence are
retained. Whole-lifetime compute is estimated at $3.781877823; provider billing
is provisional (see `scratch/questions.md`). The original worker receipt still
says running at 251 beside the terminal launcher status.

After source `62f650e82e93` was published; its quote (~06:58 UTC) and a recheck
after evidence publication both refused L4 capacity. No after Run or rental
exists. Hardware, source, seed, target and thresholds stay frozen.

Real receipts exposed a reader bug: `metrics_run` is a worker-local path and
`actual_device` is `cuda:0`. `scripts/report_collection_comparison.py::_validate_arm`
now compares the path's final component and admits `cuda`/`cuda:0`;
`reader-erratum.json` binds old and corrected reader bytes. Immutable
preparation/analysis snapshots are untouched.

Operational receipts and recovery bytes: `.runs/collection-growth/execution-20261112/`.
Published: `experiments/data/collection-growth-281/execution-20261112/`.
The frozen paired reader cannot run without a genuine after capture/receipt;
the 451–500 endpoint is missing, so no flattening score or residual-cause
ranking is supported.

## Checks

- `PYTHONPATH=. uv run --no-sync pytest tests/training/test_collection_comparison_preparation.py tests/infra/test_collection_comparison_report.py tests/infra/test_collection_growth_capture.py tests/infra/test_experiment_data_untracked.py -q`
  — 22 passed (2026-11-11, before execution). Receipt-binding/device tests —
  5 passed; artifact tracking check — 1 passed (2026-11-12). Gate owns the rest.
- Historical report regeneration matched published `summary.json`, `phases.png`
  and `growth.png` byte-for-byte; all 19 preparation files matched the manifest.

Earlier blocker history, procedure, concept review and check detail:
`0287473ec:scratch/collection-growth.md`. Concept review page:
`scratch/concept-review.html` (no product change; show the actual paired report
with limitations, inventories and rental accounting once execution is possible).

- Implement (×5), realign (×3) and write-docs (×3) passes on 2026-11-12 found no Task
  comment or budget choice and main unchanged past base; RunPod listed only
  `etude-8a95fe6d04e9` (not an MTG-281 rental, untouched). No code or doc change.
- Compress 2026-11-12: `report_collection_growth.py` shares one phase/RSS series
  helper; regenerated `summary.json`, `phases.png`, `growth.png` byte-identical.
  Frozen comparison reader (`e9adf766…`, bound by `reader-erratum.json`) untouched.
  Second and third compress passes found no code change since `db3298be8`; nothing to reduce.

## Docs

User-facing docs: `experiments/self-play-collection-273.md` (report, reproduction,
execution record, frozen paired analysis) and `experiments/README.md` (Data
pointer). Reviewed 2026-11-12 against the execution facts above; they already
state the inconclusive outcome and need no edit until the open budget decision
is made.
