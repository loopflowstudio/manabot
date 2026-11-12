# MTG-274 plan

Built: `manabot/training/report.py` (`job_labels`, `against_earlier_self`,
`against_fixed`, `head_to_head`, `learning_schedule` → two `Schedule` panels
through the damping horizon). `experiments/tla_frontier.py` `report()` now
composes them (labels, four charts, PERF/ML with schedules, `unit="games"`).
Docs `docs/experiments.md` § "Writing a report" corrected to Etude 0.2.16
behavior (seed count appended; earlier-self mixes roles; no setup pooling;
unknown tags are page notes, not build errors).

Already in Etude 0.2.16 (EXP-44, #90): hidden Matches control when every panel
names matches, two-decimal tooltip with named interval, `Curve.subject`,
`Report.labels`, `Schedule`.

Etude Tasks filed (unstarted, unowned on 2026-11-12):
- EXP-52 (eb0db3) Pool one comparison's games across execution setups.
- EXP-53 (9f66c2) Curve scores each checkpoint against whichever earlier
  checkpoint of its line it met (the later-side-only self chart).
- EXP-55 `etude report` reads retained evidence without `eval(state)`
  re-deriving matchups that must agree with the current source digest
  (`records.py` `assemble` → `tournament_conflict`).

Docs pass 2: report command examples made true to what runs (same-revision
report; finished pages from their revision or scoped prefixes), EXP-55 filed.
Check: `pytest tests/training/test_report_helpers.py tests/eval/test_experiment_files.py` 8 passed (compress: dropped helper tag validation; Etude `Curve` already rejects empty matches).
Demo: rebuilt the frontier page from its retained runs (result below).

## Demo result (2026-11-12)

Rebuilt from the frontier's 2 training + 10 eval run prefixes only (scoped
storage, throwaway script outside the repo, empty `ETUDE_HOME`): 65 s, 0.76 GB
peak, 418 cells, no notes. Against `experiments/data/tla-frontier-c5fba4/report/page.json`:
- Model vs heuristic: all 26 points identical (value and 64 games).
- Large vs Small, fixed updates and fixed time: Large's line identical at
  every point; `subject` drops Small's mirrored line.
- Model vs itself at half training time: identical where a checkpoint was only
  the later side (u400+ for Small, u200+ for Large); differs at u75–300 (128
  games: both roles mixed) and adds earlier-partner-only points (u25, u50, …).
  Needs EXP-53. The previous per-size report() mixed roles the same way.

Found while building:
- `etude report experiments/tla_frontier.py` from this revision refuses the
  held tournament (`tournament_conflict`): `eval(state)` rebuilds matchups
  with today's manabot source digest (`_source()` hashes every
  `manabot/*.py`). The demo used a wrapper file with this `report()` and no
  `eval`. Also `~/.etude/launcher.db` is from another Etude version, and the
  unscoped `--source r2://manabot/jobs` scan ran >15 min, 3.9 GB.
- Steer (Etude #98, merged, unreleased): launch-name subjects/labels/schedules
  and Refresh of repository helper modules. Not needed by these helpers (job
  names equal launch names here); no Etude release cut. EXP-52 narrowed to
  eval-environment pooling, which #98 does not cover.

## Remaining

- Self chart later-side-only and setup pooling wait on EXP-53 / EXP-52;
  docs state today's behavior. Nothing else in the Task outcome is open.
- Rebuilding a finished experiment's page from a later revision waits on
  EXP-55. Docs (§ "Writing a report" and the command block at the top) now
  give the same-revision command and say a finished page is rebuilt from its
  revision or with scoped prefixes.

## Concept review (2026-11-12, headless)

The concepts hold: a report asks questions (earlier self, a fixed player,
head-to-head, schedule), each a helper returning a plain Etude panel, and jobs
get names a reader knows them by. `against_earlier_self` stays separate from
`against_fixed`: once EXP-53 lands it is the one that asks for the later side
only. `job_labels` naming layers relative to the first job is kept (docstring
states the order dependence; scratch/questions.md keeps the alternative).
Its two actions (docs command true to what runs; Etude Task for re-deriving
matchups) are done: docs pass 2 and EXP-55. No manabot `_source` workaround,
no edit to the frozen `experiments/tla_frontier.py` eval.

## Concept review 2 (2026-11-12, headless)

Concepts unchanged since review 1. One mismatch, settled by the design:
`docs/experiments.md` says the frontier page's four Evals charts each have a
line for Small and one for Large, but `head_to_head` draws only `subject`'s
line (its own table row says so; the demo confirmed Small's mirrored line is
dropped). Done: the frontier paragraph now says earlier-self and heuristic
charts have a line per size and the two Large vs Small charts show Large's
line only (above 0.5 means Large is stronger). No open product choice.

## Concept review 3 (2026-11-12, headless)

No change since review 2's fix: helpers, frontier `report()`, the docs
example and its paragraph and helper table agree (head-to-head draws
`subject` only; earlier-self mixes roles until EXP-53). No open product choice.
