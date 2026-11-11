# Self-play metric indexing (MTG-273)

Self-play logging creates a history-dependent allocation and CPU cost in the
training process. Each deck pair, starting position and outcome became a metric
column. With 134 decks this permits 107,736 names, each repeating two deck hashes.
Etude 0.2.15's background writer repeatedly materializes their accumulated block
headers when refreshing its index. The collector's counter itself is drained
after each update; it is not a lifetime matchup history.

This reproduces a concrete amplification mechanism. It does **not** isolate the
entire reported width-64 CUDA slowdown or establish that every other allocation
is bounded. The Task's reported collection increase from about 41 to 107 seconds
and RSS increase from about 12.5 to 26 GB remain historical observations.

## Mechanism

`execution.py` passed `SelfPlayCounter.drain()` through `metric_row`, producing
`self_play_outcomes/<deck>/<opponent>/<position>/<outcome>` names. The SDK's
`Run._fold` invokes `train.manifest.write_manifest`. On permanent folds that
function describes every permanent block; on other folds it reads and validates
the entire permanent manifest merely to recover the latest values. This work
grows with accumulated metadata even when each training update does the same work.
It runs in a Python thread in the training process, outside the collector's
explicit bookkeeping timers. Competition for CPU and the interpreter can affect
other collection phases; their timers alone cannot identify that cause.

The frontier owner's retained report-generation note independently records a
1.303 GB permanent manifest expanding to about 14 GB RSS before the reader was
stopped. That observation is from
`manabot.c5fba4-frontier-self-play/scratch/frontier-evaluation-direction.md`
(2026-11-11); it concerns a report reader, not a profile of the training writer.
No historical run or stored artifact was changed here.

## Bounded reproduction and repair

```sh
uv run --locked --extra artifacts python scripts/diagnose_outcome_index.py
```

The fixture uses the installed SDK's real block and manifest code: eight tiny
updates, 16 matchups, both positions and three outcome counts. Each update is
one permanent block. It compares scalar metrics with an immutable JSON row
table referenced by one digest. No game, policy, optimizer, rental or artificial
resource pressure is involved. Temporary files are removed when it exits.

Observed locally on 2026-11-11, CPython 3.12 / Etude 0.2.15:

| Representation | Blocks | Manifest bytes | Refresh peak Python bytes | Refresh ms |
| --- | ---: | ---: | ---: | ---: |
| Scalar matchup columns | 1 | 62,645 | 292,885 | 1.57 |
| Scalar matchup columns | 4 | 202,533 | 1,093,536 | 4.82 |
| Scalar matchup columns | 8 | 389,180 | 2,160,340 | 8.27 |
| Row table reference | 1 | 762 | 12,812 | 0.67 |
| Row table reference | 4 | 2,902 | 28,349 | 0.97 |
| Row table reference | 8 | 5,757 | 49,433 | 1.23 |

Refresh means `write_manifest(..., permanent=False)`. Memory is tracemalloc's
peak Python allocation, not process RSS, native allocations or GPU memory.
Timings include tracing overhead and are descriptive, not CI thresholds. At
eight blocks, peak Python allocation falls about 44-fold. Scalar indexing's
dependence on history is present even with a fixed set of matchup names.

New training writes one digest-named table under the metric run's
`artifacts/self-play-outcomes/` and records only its reference. Tables retain
all deck, opponent, position and count values; storage grows with evidence, while
the writer only holds the current update's table. Both Etude carriers already
deliver immutable author artifacts. Offline readers select the winning update
and attempt before verifying and reading its table, then expose the existing
flat reporting projection. Historical metrics are read unchanged. Unreferenced
tables supply no results, and a missing or corrupt selected table fails visibly.

## Verification and remaining acceptance

The focused tests cover shipment and relocation, idempotent table writes,
missing/corrupt references, historical rows mixed with new tables, duplicate
updates, a resumed attempt with no logged rows, and exact outcome counts through
ordinary two-stage self-play and raw/EMA checkpoint reload. A deterministic
index test increases matchup diversity from 1 to 64 pairs at eight updates;
only the digest column enters the manifest. Byte counts replace timing assertions.

Full long-run flat RSS and collection time have **not** been demonstrated.
The SDK still indexes ordinary scalar history and table-reference labels; this
repair removes matchup-column amplification, not all history-dependent SDK work.
Any follow-up should profile an already-running trainer or use separately
authorized bounded execution, compare equal completed work, and retain host,
source, game-length and error/recovery differences. Do not restart or mutate the
frozen frontier runs to turn this software proof into experimental evidence.

## Retained frontier measurement (MTG-281, 2026-11-11)

**Both historical runs show growing host memory and collection spikes aligned
with permanent-block boundaries.** The width-160 run is not a flat-memory
control. This supports the index-refresh mechanism, but does not establish that
the outcome-table fix flattens training: the matched before/after runs have not
launched. The L4 provider availability read failed; new rental spending is $0
of the authorized $15. MTG-281 remains incomplete.

### Evidence and differences

The retained projection covers w64 updates 1–1744 (`59718fe34945`) and w160
updates 1–932 (`5b7d8fa7b38c`). Each has one writer and one launcher attempt,
with no resumed checkpoint. Both ran source
`6cd54d349ccbbaf1f5f2aa10831ccfb3ecf4653b`, seed 21501, explicit CUDA float32,
Torch 2.10.0+cu128, 256 streams, four producers and one Torch thread. They use
the same deck population, input capacities and game-error policy. Recovery
journaling is absent. W64 has width 64 / two attention layers / four heads;
w160 has width 160 / eight layers / eight heads. Their frozen damping horizons
also differ: 2474 versus 999 updates.

They are separate NVIDIA L4 rentals: `l4-ecca7abeb4df` (provider pod
`awcdavrycv1gvy`, writer host `88f68d3cefb7`) and `l4-8a95fe6d04e9`
(`k39bmw5981u6sv`, `99579aa47026`). Both allocations specify at least six vCPUs
and 48 GB host RAM. Recorded node-visible memory is 811.4 versus 540.7 GB and
kernel versions differ; these figures are **not** container allowances. Exact
CPU models/quotas and final game-error counts were unavailable: neither live
run had delivered its final `training/run.json`. One attempt does not mean zero
recovered game failures. No trainer, machine or original evidence was modified.

The reader selects the SDK's surviving blocks, checks contiguous sequence
coverage, validates selected array CRCs and preserves sparse update positions.
It reads one raw header at a time, omitting matchup and unrelated learning
metrics before constructing typed headers. It does not load either cumulative
manifest. Snapshots retain declarations, writer metadata, object versions,
header hashes, block coordinates and selected per-update values. Directed tests
cover sparse coordinates and damaged-array rejection.

At listing time, w64 had 158 permanent blocks with 888.91 MB of raw header
JSON; w160 had 84 with 470.89 MB. Their actual current permanent manifests were
1,428,999,177 and 765,386,130 bytes, respectively, according to object listings.
At **equal update 900**, both had 81 permanent blocks, with 454.52 versus
453.96 MB of headers. Thus block-count/index growth does not distinguish the
arms at equal updates. W64's larger final index primarily reflects its longer
update history.

Historical manifest versions and actual fold start/end timestamps were not
retained. The plots reconstruct cumulative permanent **header** bytes at each
block's final logged update. This excludes manifest overviews and latest-value
maps, and is a lower bound on eventual index size, not an exact historical file
size or fold timestamp. Live snapshots were taken separately and can lag the
launcher clocks.

### Collection, memory and work

Collection means include the expensive tail; medians describe typical updates.
RSS is decimal GB. Game length below means native decisions per completed game,
including decisions spent in any failed/replaced games; it is not turns per game.

| Updates | W64 collection mean / median (s) | W160 collection mean / median (s) | W64 / w160 median RSS (GB) | W64 / w160 decisions per game |
| --- | ---: | ---: | ---: | ---: |
| 1–50 | 37.87 / 36.03 | 90.95 / 86.42 | 12.25 / 11.73 | 164.73 / 150.38 |
| 400–450 | 48.77 / 41.75 | 97.20 / 88.93 | 17.88 / 18.03 | 153.32 / 142.52 |
| 550–600 | 58.94 / 47.00 | 107.32 / 98.38 | 20.52 / 19.99 | 160.03 / 149.63 |
| 850–900 | 79.84 / 60.23 | 119.52 / 99.07 | 19.23 / 19.85 | 160.84 / 142.80 |
| 1600–1700 | 104.15 / 78.58 | unreached | 25.09 / unreached | 145.20 / unreached |

Median completed games stay around 269–273 per update. Games do not get
systematically longer as collection slows. W160's longer model computation and
changing game lengths obscure a smaller proportional collection increase; they
do not make its memory flat. Over common updates 51–932, descriptive OLS slopes
per 100 updates are +4.37 / +1.35 collection seconds and +0.625 / +0.534 GB RSS
(w64 / w160). After dividing collection time by each update's collected learner-row
count, slopes are +0.119 / +0.116 seconds per 1,000 rows per 100 updates.
These are descriptive slopes, without independent-sample uncertainty or a
causal interpretation.

![Collection subphases over updates](data/collection-growth-281/phases.png)

The spikes have a strong index association. In common updates 300–932, the
57 updates immediately following a permanent block's final logged coordinate
average **165.18 s for w64 and 201.53 s for w160**. At offsets −3 through 0 and
+2 through +4, the corresponding offset-group means range 46.88–57.40 s and
98.23–103.42 s. All eight slowest w64 updates in the full snapshot are at offset
+1, taking 397.6–412.7 s. Most of their extra measured time falls inside
inference/selection, with some in observation preparation and recording.
Those scopes measure elapsed time, so they can include pauses caused by the
background writer; they do not isolate neural computation.

This is consistent with the SDK folding a block and refreshing its accumulated
index while the next collection runs. Permanent blocks grow regularly (about
11 updates each), rather than beginning only near updates 550 or 850. The
reported steps are not a unique onset of index growth. Host differences,
allocator behavior, game errors and scheduling remain possible contributors;
node-wide load readings alone cannot establish contention. There is no retained
Python-allocation trace with which to separate Python index memory from native
or tensor memory. No further profiling was performed.

![Memory, index growth and game lengths](data/collection-growth-281/growth.png)

### Matched comparison: bounded plan, not executed

A prospective **500-update** horizon catches the early memory increase and
45 permanent blocks: historical w64 RSS is 20.12 GB at update 500. It cannot
establish behavior in the later 850–1700-update regime. Keep the w64 recipe,
seed 21501, explicit CUDA and L4 machine type for both arms; retain the original
2474-step damping schedule when shortening the stopping horizon. The exact
sources are before `b7c102f2c4266dab4be91b8b54901861745b9a3c` and after
`1659b0e290f55daa632ec8f869adff4a3059d79f`. Source/native/content/dependency
identities must be checked together before launch. Neither arm resumes the
original run.

The original 500 updates took 10.05 hours from tracker start, suggesting
$11.86 for two equal-duration arms at the historical $0.59/hour rate, before
setup/teardown. Reserve both arms before starting: at most 12 whole-rental hours
and $7.20 each at a $0.60/hour ceiling, plus $0.60 ancillary reserve; no separate
pilot or automatic retry. The 900-second startup bound is inside that window.
Use ordinary TrainingJob execution and its stop/export controls, retain every
attempt and failure, and stop rather than extend an incomplete arm. This is a
budget envelope, not a fresh quote or a guarantee of completing 500 updates.

A 600-update pair already projects $14.46 before setup/teardown; 650 projects
$15.76 and 900 $23.88. Spending on those longer paired windows is not justified
under $15 without new measured evidence. No speedup of the repaired arm is
assumed to fund admission.

At 2026-11-11 22:42:24 UTC, the fresh configured L4 availability request failed
with `ProviderUnavailable: GET catalog/gpus: no reply`. No current price,
capacity, quote, new run or rental was obtained. The paid comparison therefore
remains blocked, and there is **no verdict of flat, partly flat or not flat**
after the repair. A remaining-cause follow-up depends on that comparison; this
historical evidence cannot rank `_slice_obs_tensors`, allocator/native memory
and `game_errors` as residual causes after an unexecuted fix.

### Reproduction and remaining acceptance

Evidence is under `experiments/data/collection-growth-281/`: `w64.json`,
`w160.json`, `provenance.json`, `summary.json`, `comparison-plan.json` and the two
figures. Fetch it through the manifest, then reproduce the figures without
network access or native/game imports:

```sh
uv sync --locked --no-install-project --extra artifacts --extra reports --extra dev
PYTHONPATH=. uv run --no-sync python -m manabot.data fetch experiments/data/collection-growth-281
PYTHONPATH=. uv run --no-sync python scripts/report_collection_growth.py \
  --data experiments/data/collection-growth-281 --out .runs/collection-growth/report
```

The output directory must be new. A fresh capture (with artifact credentials)
also refuses to overwrite an existing snapshot:

```sh
uv run --no-sync python scripts/measure_collection_growth.py \
  --source r2://manabot/jobs --run 59718fe34945 \
  --cache .runs/collection-growth/cache --out .runs/collection-growth/w64.json
```

Remaining acceptance is the bounded matched execution, equal-work slopes,
complete error/recovery and host receipts, whole-rental accounting and confirmed
teardown. The historical attribution is associative, and the unretained exact
manifest/fold history remains unavailable. The software checks and retrospective
plots do not complete the scientific Task.
