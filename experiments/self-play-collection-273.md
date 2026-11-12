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

2026-11-12: training now logs each update's rows with Etude 0.2.16's
`Run.log_table("self_play_outcomes", ...)` (stored as `artifacts/<digest>.json`)
and reads them through `RunData.read_table`; manabot's table writer and reader
are deleted. Runs carrying `self_play_outcomes/table` references are refused,
naming `bf4e280f7`. Etude 0.2.16 also replaced `write_manifest` with
`manifest.Index`, whose recent refresh no longer redescribes permanent blocks:
the diagnostic's refresh peak is now 35,666 bytes (scalar) and 6,408 bytes
(table) at 1, 4 and 8 blocks, while the permanent manifest still grows to
389,180 versus 5,709 bytes at eight blocks.
