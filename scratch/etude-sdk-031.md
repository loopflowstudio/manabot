# MTG-348: move to etude-sdk 0.3.1

## Problem

manabot pins etude-sdk 0.2.16. 0.3.0 and 0.3.1 rename the evaluation and
report vocabulary and change two database schemas, so experiment files,
reports, shared helpers and tests break on upgrade. The Task ends in one PR
that lands once green. No paid runs.

Brief, in its words: "Search the whole repository (experiments, reports,
scripts, tests, docs) for every use of a renamed or removed name. Check with the
repo's tests, type checker and lint, and build at least one experiment's report
(e.g. the TLA frontier experiment) to confirm it renders. Do not launch paid
runs."

## Demo

```bash
ETUDE_HOME=$(mktemp -d) doppler run -p manabot -c prd -- \
  uv run etude report experiments/tla_frontier.py --source r2://manabot/jobs --out /tmp/mtg348-tla
```

writes the TLA frontier page with its four eval charts drawn from the stored
1,120 matches, under etude-sdk 0.3.1.

## Findings (probed 2026-11-13, SDK 0.2.16 against 0.3.1 wheels)

The repo has no type checker configured (no mypy/pyright in `pyproject.toml`,
CI or docs). "Type checker" in the brief has nothing to run; ruff and the tests
are the checks. Do not add one in this Task.

### What breaks

Import probe over every tracked `.py` (497 `etude` imports): 18 fail on 0.3.1.
Test collection: 17 modules error before any change (16 from Etude, one from a
missing `llm` extra that is unrelated).

| Old | New | Where |
| --- | --- | --- |
| `eval.records.Matchup`, `Benchmark(matchups=)` | `Seating`, `seatings=` | `experiments/{elemental,learning_signal,precision_ladder,tla_frontier,tla_frontier_calibration,value_readout_252}.py`, `manabot/eval/{benchmark,duel}.py`, `tests/eval/{test_curriculum_report,test_learning_signal}.py` |
| `Checkpoint.clocks` | `.at` | `experiments/{elemental,learning_signal,precision_ladder,tla_frontier}.py`, `tests/eval/test_experiment_files.py`, `tests/training/test_value_readout_252.py` (`Checkpoint(clocks=)`) |
| `eval.tournament.curves` | `scores` (returns `Scores`, field `seating`) | `tests/training/test_etude_evaluation.py` |
| `run.listing.places(storage)` | `run_folders(storage.keys(""))` | `manabot/eval/card_query.py::_folder` |
| `report.Schedule`, `SchedulePoint` | `Planned`, `PlannedPoint` | `manabot/training/report.py::learning_schedule` |
| `report.HeadToHead` | removed | `experiments/elemental.py`, `experiments/two_seat.py` |
| `Curve(opponent=, subject=, inference=)` | removed | `manabot/training/report.py::{against_fixed,head_to_head}`, `experiments/reports/mtg183-elemental-eval.py` |
| `report.Earlier`, `Fraction` | moved to `eval.records` as seats of a `Seating`, need `inference` | `experiments/reports/mtg183-elemental-eval.py` |
| `Benchmark.matchups` (attribute) | `.seatings` | `experiments/curriculum_gpu.py:47`, `tests/eval/test_etude_player_migration.py:103`, `tests/eval/test_experiment_files.py:173` |
| `etude.experiment.TrainingRun(name, experiment, state, checkpoints)` | also requires `job`, `launch`, `line` | `tests/eval/test_experiment_files.py:145`, `tests/training/test_value_readout_252.py:293` |
| `RunData.tables`, `StoredBlock.table()`, dictionary block rows | `RunData.columns`, `StoredBlock.columns()`, `LoggedRow.values` / `.clocks` | `manabot/reports/training.py:186`, `tests/sim/test_tracking.py:110`, `tests/training/test_monitoring.py:64,171`, comment in `manabot/training/update_metrics.py:169` |

Imports do not show these. An empty statsdb dumped under both versions differs
in exactly these columns:

- `schedule.matchup` → `seating`: `experiments/closure_lambda_report.py:195-196`.
- `training_checkpoints.clocks` → `at`: `experiments/tla_frontier_analysis.py:488`.
- `players.at`, `at_conflict`, `line`, `line_conflict` → `seated_at`, `lines`,
  plus `names`; `seats` gains `name`; `matches` gains `newest`. No repo SQL reads
  the removed `players` columns.
- Rollup column `opponents` is `VARCHAR[]`, was `VARCHAR`:
  `manabot/eval/card_query.py:213` joins `o.player = r.opponents`.

A page build now reloads every imported module of the repository, not only
those under `experiments/` (`etude/page.py::_neighbours`; 141 modules for a
second file here). A folder report in one process still works: two files ran
in turn, `managym._managym` stayed the same object and `report()` built. In
tests it splits class identity: after a build,
`tests/eval/test_experiment_files.py::test_launched_runs_track_under_the_file_stem`
fails at `experiments/seat_comparison.py:68` (`assert isinstance(stage,
TrainSelfPlay)`), because `keep_imported_modules` restores saved modules but
leaves the ones first imported during the build, which hold the reloaded
classes. The fixture must restore repository entries in `sys.modules` exactly, while
leaving new dependency imports intact. Removing Torch lazy imports causes
duplicate cache-artifact registration in a later test (implementation probe).

First failures seen with the renames applied, one file per process:
`test_report_helpers` 1 of 4 (`Curve.opponent`), `test_experiment_files` 2 of
4 (above, `TrainingRun`), `test_etude_player_migration` 1 of 5
(`Benchmark.matchups`), `test_tracking` 1 of 4 (`RunData.tables`);
`test_experiment_report` 17 pass. The machine's load average was above 250,
and under it `tests/eval/test_learning_signal.py` and
`tests/training/test_monitoring.py` each hit the five-minute limit while
playing games or training, which ends the whole run without a summary; that
says nothing about Etude.

Unchanged and checked: `Result.tables`, `hold(...)` (no caller passes `share`),
`Job(until="updates=N")` (the string still validates into `Until`).

### The TLA frontier report

- It renders under 0.3.1: `etude.page.build("r2://manabot/jobs",
  experiments/tla_frontier.py)` returned a page with all four eval panels bound
  to selections, 0 failures, 0 absent columns, in 1,279 s.
- The time is the bucket, not the SDK: `r2://manabot/jobs` lists 95,368 keys in
  125 s and holds 6,476 runs, and a build reads each run's `run.json` in turn.
  0.2.16 did not finish the same build in 30 minutes. A build needs a 30-minute
  allowance and runs in the background.
- `eval(state)` runs during a report and calls `with_launcher_ends`, which asks
  the local launcher about run `5b7d8fa7b38c`. With a launcher that does not
  know the run the build stops: `etude: no run '5b7d8fa7b38c'`. The only record
  is `~/.etude/launcher.db`, schema 13, written by 0.2.15; 0.2.16 (14) and 0.3.1
  (17) both refuse it. So the report already cannot build with the locked SDK on
  main, and never will from a fresh launcher database.
  `experiments/tla-frontier-c5fba4-results.md:51` records the fact the launcher
  held: "machine lost; 931 updates stored, last checkpoint u925".
- The page notes "Tournament tla-frontier-c5fba4-paired-v1 is drawn as its eval
  runs recorded it: eval(state) now defines it otherwise (tournament_conflict)".
  Etude draws the recorded tournament; nothing is lost. Not chased further.
- Head-to-head charts change. A tag `Curve` in 0.3.1 draws a line per job, and
  both charts now hold both jobs' cells (equal-hours 11 + 11, head-to-head
  10 + 10), so Small's mirrored line appears beside Large's. 0.2.16 drew one
  line through `Curve(subject=large)`. A single line in 0.3.1 needs
  `Curve(matches=Matches(seating=...))` over a seating with an `Inference` seat;
  the frontier's seatings are fixed pairs (`called.checkpoint is None` in
  `etude/page.py::_points`), so they draw no seated points.

### Local launcher database

`~/.etude/launcher.db` is schema 13 and was being written during this probe
(WAL modified 10:09), so another Etude client is using it. 0.3.1 refuses it:
"was written by another version of etude; move or delete it to start again".
Every check here runs under a temporary `ETUDE_HOME`, which also keeps 0.3.1's
statsdb (schema 13, was 9) away from the kept files other checkouts read.

## Approach

One PR: bump, rename, repair the four places where meaning changed, and say
plainly what a report no longer draws.

1. **Pins.** `pyproject.toml` (three `>=` floors, the `==` pin) and `uv.lock`
   to 0.3.1; `docs/development.md:44` and `:580` say the new floor. Leave
   `RELEASE_NOTES.md` and the `--with "etude-sdk[s3,eval]==0.2.15"` lines in
   `experiments/cuda_precision.py` and `precision_ladder.py`: they record how
   past runs were launched.
2. **Renames** from the table above, plus the three SQL sites.
   `card_query.py:213` reads the one opponent of a two-player row
   (`r.opponents[1]`); keep the rollup's name, since its definition is Etude's
   and old blocks still read.
3. **Report helpers** (`manabot/training/report.py`). `against_fixed(title,
   tags)` and `head_to_head(title, tag)` lose `opponent` and `subject`, which
   `Curve` can no longer honour; the tags already select the matches. Their
   docstrings say what is drawn: `head_to_head` now draws both jobs, and a
   single line needs an `Aligned` seating drawn by its name. Keep the three
   names; they state the question. Update
   `tests/training/test_report_helpers.py` and `docs/experiments.md:253-305`.
4. **Frontier lost run.** `with_launcher_ends` reads `5b7d8fa7b38c` from a
   small frozen record in `experiments/tla_frontier.py` that cites the results
   table, before it asks the launcher. Other running runs still ask the
   launcher and still raise when it does not know them. Add the dated line to
   `experiments/tla-frontier-c5fba4-results.md`.
5. **Panels with no 0.3.1 form.**
   - `experiments/elemental.py`: `HeadToHead(matches=Matches(tournament=PAIRS))`
     becomes `Crosstable(matches=Matches(tournament=PAIRS))`; PAIRS holds
     consecutive and endpoint pairs within one line, which is what a crosstable
     shows.
   - `experiments/two_seat.py`: drop the cross-job `HeadToHead`; its docstring
     says the direct-games chart needs a seating the experiment never held.
   - `experiments/reports/mtg183-elemental-eval.py`: its matches carry no
     tournament or tag, so the "400 updates earlier" and "half its updates"
     curves cannot be selected. Keep `Curve` (the Opponent control picks the
     heuristic), `Crosstable` and `Anchors`; the docstring says which two
     charts went and why.
6. **Docs.** `docs/experiments.md`: helper table and example, "matchup" where
   it means Etude's record, and a short note that 0.3.1 refuses an older
   launcher database and that it is moved aside, never migrated.

## Docs written (2026-11-13, describe intended behavior)

- `docs/experiments.md`: seating vocabulary and `Checkpoint.at`; the launcher
  database refusal and `ETUDE_HOME`; report example and helper table with
  `against_fixed(title, tags)` and `head_to_head(title, tag)`; both-lines
  head-to-head; how a single line is drawn (`Matches(seating=)` with a relative
  seat); `Crosstable`; the frontier report command and its half hour.
- `docs/development.md`: `etude-sdk>=0.3.1` in both places, pointer to the
  launcher note.
- `experiments/README.md`: `two_seat` row loses "head to head", `elemental`
  row says crosstable.

The implementation must match the helper signatures the guide now shows.
`RELEASE_NOTES.md` and `release/` describe released versions and are unchanged.

## Alternatives

- **Re-seat the frontier's head-to-heads as `Aligned` seatings** to keep one
  line. Rejected: the experiment is finished and frozen at ab2b757d9, its
  analysis reads games by match key, equal-hours pairs w64's last checkpoint at
  or before w160l8's hours (no exact clock position), and a seating that
  resolved a different partner would leave gaps and make `etude eval` want new
  games.
- **Stay on 0.2.16 until Etude draws a fixed seating once.** Rejected: the
  brief asks for 0.3.1 now, and the locked SDK already cannot open the local
  launcher database.
- **Move `~/.etude/launcher.db` aside in this Task.** Not done: it is shared,
  in use, and holds the only launcher history of runs still referenced.
- **Add a type checker.** Out of scope; the brief assumed one exists.

## Open, for a person

- The mirrored Small line on the frontier's two head-to-head charts. Jack Heart
  on 2026-11-11 asked for "one chart per question, with a line per model and no
  mirrored head-to-head line" (memory `report-compares-jobs`). 0.3.1 cannot draw
  a fixed pair once. The fix belongs in Etude (a seated curve over fixed
  seatings, positioned at the first seat's checkpoint), then a manabot bump.
- When to move the shared launcher database aside. Until then `uv run etude`
  launch, status and eval commands refuse it from this checkout.

## Forbidden outcomes

- Compatibility shims that re-create `Matchup`, `HeadToHead` or
  `Curve(subject=)` in manabot.
- A fallback that hides an unknown running run from `with_launcher_ends`.
- Rewriting retained evidence, fixtures under `tests/fixtures/etude_sdk_025/`,
  frozen protocol documents or stored tables for the new names.
- Any `etude launch`, `queue` or launching `etude eval`.
- Running Etude 0.3.1 against `~/.etude`.

## Remaining work

Implementation and docs are in the working tree. The report helpers, SQL
reads, TrainingRun fixtures, repository-module cleanup, frozen frontier outcome
and historical panels use the surviving 0.3.1 APIs. The release validator and
its metadata fixtures require the new dependency floor too.

Implementation sanity checks passed; results are recorded below.
Gate owns the broader tests, isolated dependency check, whole-repository lint,
import probe and stored report builds listed next. Publication and landing
belong to the caller's later steps. No paid runs.

## Acceptance at gate

Run each with `ETUDE_HOME` set to a fresh temporary directory.

- Names gone: `git grep -nP "\bMatchup\b|HeadToHead|\bSchedulePoint\b|\.clocks\[|\bplaces\(" -- '*.py' ':!tests/fixtures'`
  prints only `manabot.sim.matchup` names (`AuthoredMatchup`, `DeckMatchup`).
- Imports: every `from etude... import` in tracked `.py` files resolves against
  the installed SDK (AST walk; 497 names, 0 missing).
- Collection: `uv run pytest --collect-only -q tests` errors only in
  `tests/llm/` for the missing `llm` extra.
- Tests: the Etude-facing files, one process each so no test nears five minutes
  on a loaded machine:
  `tests/training/{test_report_helpers,test_etude_evaluation,test_monitoring}.py`,
  `tests/eval/{test_experiment_files,test_curriculum_report,test_etude_player_migration,test_card_evaluation}.py`,
  `tests/sim/test_tracking.py`, `tests/infra/test_release.py`, plus
  `tests/eval/{test_learning_signal,test_mtg186_cards,test_elemental_long_eval,test_elemental_long_report}.py`,
  `tests/training/{test_experiment_report,test_etude_sdk,test_etude_job,test_etude_source,test_outcome_tables,test_value_readout_252,test_precision,test_head_to_head,test_elemental_closure_lambda,test_time_cohort}.py`;
  all pass. Then the isolated dependency check from AGENTS.md:
  `uv run --isolated --extra dev --extra artifacts pytest tests/training/test_artifact_storage.py tests/training/test_etude_sdk.py tests/training/test_etude_source.py tests/training/test_etude_job.py -q`.
- Lint: `uv run --extra dev ruff check . && uv run --extra dev ruff format --check .`
- Graphs: `uv run python scripts/check_loopflow_workflows.py`.
- Fast report: `uv run --extra evaluation etude report experiments/reports/mtg183-elemental-eval.py --source ~/.manabot/mtg183-eval-db/store --out <tmp>`
  writes a page (local store, seconds).
- Frontier report: the Demo command, in the background with a 30-minute
  allowance; a page is written and names no failure.

CI runs the suite; nothing here needs a display or a person.

## Done when

- `uv run etude report experiments/tla_frontier.py --source r2://manabot/jobs`
  draws the frontier page on 0.3.1 from any machine with R2 access, with no
  launcher database.
- No tracked Python imports a name 0.3.1 removed, and no SQL reads a renamed
  statsdb column.
- The Etude-facing tests and lint pass, CI is green, and the PR says which
  three report panels 0.3.1 no longer draws and that head-to-head charts show
  both jobs.

## Delete — do not maintain

Remove Etude's retired Matchup/HeadToHead/Schedule interfaces and Curve's
subject/opponent arguments, old checkpoint/rollup column reads, and the
probe-only lost-run shortcut. No aliases or replacement storage. Keep native
manabot matchup concepts, frozen launch commands and retained evidence. The
surviving paths are Seating, Planned, tagged Curves, Crosstable, RunData.columns
and the explicitly documented frozen lost-run outcome.

## Implementation progress (2026-11-13)

Finished the remaining consumer renames, report helpers and panels, SQL reads,
stored TrainingRun fixtures and module restoration. Lost-run outcome now cites
the results record; unknown active runs still require the launcher.

Focused checks (all with temporary ETUDE_HOME and the dev, evaluation,
reports and artifacts extras):

- `uv run --extra dev --extra evaluation --extra reports --extra artifacts pytest tests/training/test_report_helpers.py tests/eval/test_experiment_files.py tests/eval/test_card_evaluation.py -q`: 49 passed, 1 failed in 137 s. Removing all new modules duplicated Torch registration; cleanup now only removes repository modules outside hidden directories.
- `uv run --extra dev --extra evaluation --extra reports --extra artifacts pytest tests/training/test_report_helpers.py tests/eval/test_experiment_files.py tests/eval/test_card_evaluation.py tests/infra/test_release.py tests/sim/test_tracking.py tests/training/test_value_readout_252.py -q`: 106 passed, 2 failed in 251 s. Repository cleanup and the frozen endpoint regression passed. Failures exposed typed LoggedRow reads and a new assertion comparing different selection metadata. Fixed both, and the related StoredBlock.columns() consumer.
- `uv run --extra dev --extra evaluation --extra reports --extra artifacts pytest tests/eval/test_card_evaluation.py::test_runs_pool_into_their_experiment_and_filters_narrow_them tests/sim/test_tracking.py::test_dashboard_delivery_failure_retries_without_duplicate_rows tests/training/test_outcome_tables.py tests/training/test_monitoring.py::test_retained_rows_without_coordinates_stay_unavailable tests/training/test_monitoring.py::test_publisher_reads_etude_prefix_and_rejects_rewritten_history tests/training/test_monitoring.py::test_publisher_waits_for_evaluation_completion_without_losing_local_progress -q`: 10 passed in 21 s.
- `uv run --extra dev ruff check <changed Python files>`: passed; `uv run --extra dev ruff format <changed Python files>`: unchanged after the last edits.

The release validator and synthetic metadata fixtures now require 0.3.1;
released notes and historical launch commands retain their original pins.
Gate retains the wider acceptance checks above, including isolated dependency
validation and real report generation; no paid evaluation or training is needed.

## Compress (2026-11-13)

Diff is renames plus the planned cuts; no retired Etude name or shim survives.
Fixed the stale `tla_frontier.report()` docstring (equal-hours is drawn at each
job's own update, not w160l8's) and collapsed two calls. No behavior change.
`uv run --extra dev ruff check` and `ruff format --check` on both files: passed.

Left for a person: `against_earlier_self`, `against_fixed` and `head_to_head`
in `manabot/training/report.py` now build the same tagged `Curve` and have one
caller (the frontier). Kept per "Keep the three names; they state the
question"; folding them into plain `Curve`s would also drop three rows of the
`docs/experiments.md` helper table.
