# MTG-348: move to etude-sdk 0.3.x (lands on 0.3.2)

## Problem

manabot pinned etude-sdk 0.2.16. 0.3.0 and 0.3.1 rename the evaluation and
report vocabulary and change two database schemas, so experiment files,
reports, shared helpers and tests break on upgrade. The Task ends in one PR
that lands once green. No paid runs.

Brief, in its words: "Search the whole repository (experiments, reports,
scripts, tests, docs) for every use of a renamed or removed name. Check with the
repo's tests, type checker and lint, and build at least one experiment's report
(e.g. the TLA frontier experiment) to confirm it renders. Do not launch paid
runs."

Jack Heart, 2026-11-13, Task comment: "Fix etude first". Etude will make
`Curve(matches=Matches(seating=...))` draw a fixed-pair seating once, at its
first seat's checkpoint, and release 0.3.2. "Hold until 0.3.2 is out, then
move straight to 0.3.2 and draw the frontier head-to-head charts that way (no
mirrored Small line). Do not land on 0.3.1."

Jack Heart, 2026-11-13, later Task comment: "Launcher database, decided: do not
move ~/.etude/launcher.db. MTG-347 and MTG-324 are tracking live RunPod pods
through it with etude 0.2.15. Etude 0.3.2 (with #110) names its launcher file
by schema version (~/.etude/launcher-17.db), so 0.3.2 runs beside the old file
with nothing to move. Drop the "move it aside first" open question; after
bumping to 0.3.2, confirm `uv run etude` launch/status/report work from this
checkout with the old launcher.db still in place."

## State (2026-11-13, implementation pass)

- The 0.3.1 move is committed on the branch (aefeefcaf7 and after; main synced
  at 27cb65bba7): pins, renames, SQL reads, report helpers, the frozen lost-run
  record, the three reduced panels, docs. Its probe table and progress log are
  in git at fc4784d2e5 `scratch/etude-sdk-031.md`.
- Slice 1 is implemented and checked: six focused helper tests pass; changed
  Python files pass Ruff check and format check. The retained R2 page built
  against Etude #109 at ba00d4c in 790 seconds, in a temporary Etude home.
- The saved page's two head-to-head chart selections have 11 and 10 scored
  points, all on Large (`5b7d8fa7b38c`), each 16 matches / 64 games. There
  are no failures or missing-seating notes; only the expected recorded
  `tournament_conflict` note remains. HTML: `/tmp/mtg348-probe/seated.html`.
  The first probe assertion mistook panel indices for selection IDs; checking
  the saved page via each panel's `matches` selection passes without a rebuild.
- Newly checked equal-hours points: u600 0.500 [0.421875, 0.59375], u800
  0.53125 [0.453125, 0.609375], u925 0.53125 [0.4375, 0.609375]. Equal-update
  u800 is 0.65625 [0.546875, 0.765625]; the docs now use its actual hover.
- PyPI's JSON API still reports 0.3.1 after that build. Slice 2 remains blocked
  on 0.3.2. No launcher command was run against the real home, and no paid
  run was launched. The project pin and lock remain 0.3.1 until the release.

## Demo

```bash
doppler run -p manabot -c prd -- \
  uv run etude report experiments/tla_frontier.py --source r2://manabot/jobs --out /tmp/mtg348-tla
```

writes the TLA frontier page under etude-sdk 0.3.2, with `~/.etude/launcher.db`
left in place: four eval charts, and each Large-vs-Small chart one line, the
Large job's.

## Findings

Kept from the 0.3.1 probe (still constrain the work):

- No type checker is configured (`pyproject.toml`, CI, docs). Ruff and the
  tests are the checks; do not add one here.
- A frontier build takes 11 to 21 minutes: `r2://manabot/jobs` holds 6,476
  runs and a build reads each `run.json`. Run it in the background with a
  30-minute allowance.
- `eval(state)` runs during a report. `with_launcher_ends` takes lost run
  `5b7d8fa7b38c` from the frozen `LOST_RUNS` record, so a fresh launcher
  database builds the page.
- The page notes "Tournament tla-frontier-c5fba4-paired-v1 is drawn as its eval
  runs recorded it: eval(state) now defines it otherwise (tournament_conflict)".
  Etude draws the recorded tournament; nothing is lost.
- Page builds reload repository modules; the test fixture restores repository
  entries in `sys.modules` and keeps new dependency imports (AGENTS.md).
- Under high machine load `tests/eval/test_learning_signal.py` and
  `tests/training/test_monitoring.py` reached the five-minute limit; that says
  nothing about Etude. Run Etude-facing files one process each.

Probed 2026-11-13 against Etude main's #109 (`--with-editable`
`etude.fixed-pairs` at ba00d4c, temporary `ETUDE_HOME`, read-only):

- **A tag cannot be drawn seated.** `etude/page.py::_seated` draws a selection
  as seated only when every part is `Matches(seating=...)`. `Matches(tag=...)`
  stays the per-job view with the mirrored line, in 0.3.2 too. The one-line
  chart must name each seating.
- **It works by name.** With the two charts built from tuples of
  `Matches(seating=name)`, the frontier page built in 677 s with no failures.
  Every point was on line `5b7d8fa7b38c` (Large), 16 matches each, none for
  Small:
  - equal hours (8 of 11 names given): u25 0.422, u50 0.422, u75 0.484,
    u100 0.391, u150 0.453, u200 0.438, u300 0.438, u400 0.484.
  - equal updates (10): u25 0.500, u50 0.547, u75 0.516, u100 0.531,
    u150 0.609, u200 0.469, u300 0.625, u400 0.563, u600 0.625, u800 0.656.
- **A name nothing holds is a page note**, not an error: "A Curve panel names
  seating 'w160l8 u1200 vs w64 u1200', which nothing holds." Generating names
  from every protocol milestone gave six such notes. The names must be the
  recorded ones.
- **Recorded names** (read from the eval runs' `tournament.json` in R2; both
  `tla-frontier-c5fba4-paired-v1` and `-cuda-v1` hold each, and a seating with
  no tournament pools them, each match once):
  - tag `head-to-head`, 10: `w160l8 u{N} vs w64 u{N}` for N in 25, 50, 75,
    100, 150, 200, 300, 400, 600, 800.
  - tag `equal-hours`, 11: `w160l8 u{N} vs w64 u{M} at equal hours` for (N, M)
    in (25, 50), (50, 100), (75, 175), (100, 225), (150, 350), (200, 475),
    (300, 700), (400, 875), (600, 1200), (800, 1525), and
    `w160l8 final u925 vs w64 u1700 at equal hours`.
  - `w160l8 final vs w64 final` (tag `endpoint-head-to-head`) is in neither
    chart today and stays out.
- `report()` receives no state, so it cannot recompute the equal-hours
  partners (they come from checkpoint hours). The experiment is frozen at
  ab2b757d9 and finished, so the partners are a fact to record, like
  `LOST_RUNS`.

Launcher and kept files under 0.3.2 (read from #110's diff and
`etude/eval/query.py`; not yet run):

- #110 opens `$ETUDE_HOME/launcher-17.db` and never touches `launcher.db`.
  A new file starts empty: `etude status` and `etude runs` list nothing, and
  it knows no provider, storage or on-demand until a person adds them. The
  0.2.15 runs stay visible only to the 0.2.15 client.
- `statsdb` keeps one DuckDB file per experiment and store under
  `ETUDE_HOME/statsdb`, and deletes and rebuilds one written by another schema.
  `~/.etude/statsdb` does not exist today, so a report under the real home
  deletes nothing; the first frontier build there writes it.
- `etude launch --quote` prints machines and planned cost without renting.

## Approach

### Slice 1 (complete): one line per head-to-head chart

Needs only #109, which a locked 0.3.1 does not have. The code is valid on
0.3.1 (it draws no seated points for fixed pairs there), so tests pass on the
current lock and the charts are checked against Etude main.

1. `manabot/training/report.py::head_to_head(title, seatings)` takes the
   seating names and returns
   `Curve(title=title, matches=tuple(Matches(seating=name) for name in seatings))`.
   An empty sequence raises `ValueError`. Docstring: one line, the first
   seat's score where its checkpoint stood; the names are the ones `eval(state)`
   gave. `against_earlier_self` and `against_fixed` keep their tags.
2. `experiments/tla_frontier.py`:
   - Two name functions own the formats, used by `comparison_schedule` and by
     `report()`: equal updates `f"{LARGE} u{N} vs {SMALL} u{N}"`, equal hours
     `f"{LARGE} {point} vs {SMALL} u{M} at equal hours"`. The stored names do
     not change, so no match key moves and `etude eval` wants nothing new.
   - A frozen record beside `LOST_RUNS`, citing the stored tournaments: the 10
     equal-update milestones and the 11 equal-hours (point, partner) pairs
     listed above.
   - `report()` passes those names to `head_to_head`; drop the "Both jobs are
     drawn" comment and say in the docstring that each pair is drawn once, at
     Large's update.
3. Tests, in `tests/training/test_report_helpers.py`: `head_to_head` builds one
   `Matches(seating=...)` per name and no tag; the frontier `report()`'s two
   charts name exactly 11 and 10 seatings. No page build in tests.
4. Docs: written 2026-11-13, ahead of the code, in `docs/experiments.md`
   ("What an experiment is" launcher paragraph; "Writing a report" example,
   page description, helper row and the seating-name passage) and
   `docs/development.md` (launcher sentence, two `etude-sdk>=0.3.2` mentions).
   They describe the finished state, so the code follows them:
   - `head_to_head(title, seatings)`; empty raises `ValueError`.
   - The frontier's name functions are `equal_hours_seatings()` and
     `equal_update_seatings()`, and the frozen equal-update record is
     `EQUAL_UPDATES`; rename in the docs if the code picks other names.
   - Hover example `0.66 · 95% interval 0.53–0.77 · 64 games · 16 matches`:
     the 0.66 is the probed u800 point, the interval is illustrative. Replace
     both with the built page's values.
   - The docs say 0.3.2 while the lock holds 0.3.1 until slice 2. If 0.3.2's
     launcher file or schema number differs from #110 (`launcher-17.db`),
     correct the launcher paragraph.
   - The `--with "etude-sdk[s3,eval]==0.2.15" etude status` line is the form
     `experiments/cuda_precision.py` uses; slice 2's launcher check confirms
     it still reads `launcher.db`.
5. Dated line in `experiments/tla-frontier-c5fba4-results.md`: the report draws
   Large-vs-Small by seating from 2026-11-13; no game or score changed.

Check for this slice, before 0.3.2 exists:

```bash
ETUDE_HOME=$(mktemp -d) doppler run -p manabot -c prd -- \
  uv run --extra artifacts --extra evaluation --extra reports \
  --with-editable <etude checkout at main> python <probe that calls etude.page.build and prints page.points, page.notes>
```

Expected: equal-hours selection 11 points, equal-updates 10 points, every
point's line `5b7d8fa7b38c`, the values above for the points already seen, and
no note containing "which nothing holds".

### Slice 2 (blocked on the 0.3.2 release)

1. Pin 0.3.2 wherever 0.3.1 appears: `pyproject.toml` (three floors, the `==`
   pin), `uv.lock`, `docs/development.md`, `scripts/release/validate.py`,
   `tests/infra/test_release.py` fixtures. Read `release/v0.3.2/NOTES.md` for
   further breaks. Leave `RELEASE_NOTES.md` and the `==0.2.15` launch lines in
   `experiments/cuda_precision.py` and `precision_ladder.py`.
2. Confirm beside the old launcher file, with no `ETUDE_HOME` override:
   `uv run etude status`, `uv run etude runs`, then the Demo report, then
   `uv run etude launch experiments/tla_frontier.py --quote` if the new file
   has a provider (it will not until a person adds one; record what it says).
   `~/.etude/launcher.db` keeps its name and its 0.2.15 clients keep working;
   `launcher-17.db` appears beside it.
3. Gate acceptance below.

If 0.3.2 is not on PyPI when slice 1 is done, stop there and say so.

## Alternatives

- **Let a tag selection draw seated in Etude**, so the helper keeps
  `tag="equal-hours"`. Cleaner for an unfinished experiment, since the names
  would not be listed. Not taken: #109 is merged as the answer to "Fix etude
  first", a tag can mix relative and fixed seatings, and this experiment's
  names are final. Worth raising in Etude if a second experiment needs it.
- **Derive the equal-update names from `protocol.milestones`.** Six of sixteen
  were never played and each becomes a page note.
- **Re-seat the frontier as `Aligned` seatings.** Rejected earlier: frozen
  experiment, equal-hours partners have no exact clock position, and a new
  seating would make `etude eval` want new games.
- **Land on 0.3.1 with mirrored lines.** Overruled by the first comment above.
- **Add a type checker.** Out of scope; the brief assumed one exists.

## Open, for a person

- A 0.3.2 client starts with an empty `launcher-17.db`: launching from it
  needs providers, storages and on-demands added again. Not part of this Task.

## Forbidden outcomes

- Shims that re-create `Matchup`, `HeadToHead` or `Curve(subject=)` in manabot.
- A fallback that hides an unknown running run from `with_launcher_ends`.
- Renaming a stored seating, or anything that makes `etude eval` want games.
- Rewriting retained evidence, `tests/fixtures/etude_sdk_025/`, frozen protocol
  documents or stored tables for the new names.
- Any `etude launch` without `--quote`, any `queue`, any launching `etude eval`.
- Moving, renaming or opening `~/.etude/launcher.db`, or running 0.3.1 against
  `~/.etude`.
- Landing with a 0.3.1 pin, or with a git or path dependency on Etude.

## Delete — do not maintain

Etude's retired Matchup/HeadToHead/Schedule interfaces, `Curve`'s
subject/opponent arguments and old column reads are already gone; add no
aliases. In this slice `head_to_head(title, tag)` is replaced, not kept beside
the seated form, and the "line for each job" wording goes from its docstring,
`docs/experiments.md` and the frontier comment. Keep native manabot matchup
concepts, frozen launch commands and retained evidence.

## Acceptance at gate

After slice 2. Tests run with `ETUDE_HOME` set to a fresh temporary directory;
the launcher confirmation and Demo use the real home, per the second comment.

- Names gone: `git grep -nP "\bMatchup\b|HeadToHead|\bSchedulePoint\b|\.clocks\[|\bplaces\(" -- '*.py' ':!tests/fixtures'`
  prints four lines, all prose about manabot's own deck matchups:
  `manabot/sim/search/flat_mc.py:407`, `manabot/sim/serving_protocol.py:13`,
  `manabot/training/update_metrics.py:4`, `tests/training/test_matchups.py:1`.
- Pins: `git grep -n "0\.3\.1" -- pyproject.toml docs scripts tests/infra` prints
  nothing; `uv lock --check` passes.
- Imports: every `from etude... import` in tracked `.py` files resolves against
  the installed SDK (AST walk; 0 missing).
- Collection: `uv run pytest --collect-only -q tests` errors only in
  `tests/llm/` for the missing `llm` extra.
- Tests, one process each:
  `tests/training/{test_report_helpers,test_etude_evaluation,test_monitoring}.py`,
  `tests/eval/{test_experiment_files,test_curriculum_report,test_etude_player_migration,test_card_evaluation}.py`,
  `tests/sim/test_tracking.py`, `tests/infra/test_release.py`,
  `tests/eval/{test_learning_signal,test_mtg186_cards,test_elemental_long_eval,test_elemental_long_report}.py`,
  `tests/training/{test_experiment_report,test_etude_sdk,test_etude_job,test_etude_source,test_outcome_tables,test_value_readout_252,test_precision,test_head_to_head,test_elemental_closure_lambda,test_time_cohort}.py`;
  all pass. Then
  `uv run --isolated --extra dev --extra artifacts pytest tests/training/test_artifact_storage.py tests/training/test_etude_sdk.py tests/training/test_etude_source.py tests/training/test_etude_job.py -q`.
- Lint: `uv run --extra dev ruff check . && uv run --extra dev ruff format --check .`
- Graphs: `uv run python scripts/check_loopflow_workflows.py`.
- Fast report: `uv run --extra evaluation etude report experiments/reports/mtg183-elemental-eval.py --source ~/.manabot/mtg183-eval-db/store --out <tmp>`
  writes a page.
- Launcher: `uv run etude status` exits 0; afterwards `~/.etude/launcher.db`
  and `~/.etude/launcher-17.db` both exist.
- Frontier report: the Demo command, background, 30-minute allowance. A page
  is written; the slice 1 probe on the pinned SDK shows 11 and 10 points, all
  on `5b7d8fa7b38c`, and no "which nothing holds" note.

CI runs the suite; nothing here needs a display or a person.

## Done when

- The frontier page builds on etude-sdk 0.3.2 from a machine with R2 access,
  beside an untouched 0.2.15 `launcher.db`, and each Large-vs-Small chart shows
  one line.
- No tracked Python imports a name 0.3.x removed, and no SQL reads a renamed
  statsdb column.
- The Etude-facing tests and lint pass, CI is green, and the PR says which
  three report panels 0.3.x no longer draws (`elemental` head-to-head becomes a
  crosstable, `two_seat` loses its cross-job chart, `mtg183-elemental-eval`
  loses "400 updates earlier" and "half its updates").
