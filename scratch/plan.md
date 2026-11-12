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
get names a reader knows them by. No helper hides a choice the experiment
should make. `against_earlier_self` and `against_fixed` are both thin over
`Curve`, but they stay separate: once EXP-53 lands, the self question is the one
that asks for the later side only.

Biggest mismatch: what a reader does with the docs. `docs/experiments.md`
§ "Writing a report" says `uv run etude report <experiment file> --source
r2://manabot/jobs --serve` draws the frontier page. From this revision the
frontier file is refused (`tournament_conflict`): `eval(state)` re-derives its
matchups with today's `_source()` digest. The unscoped source scan also ran
more than 15 min and used 3.9 GB. The demo only worked through a wrapper with no `eval` and
scoped run prefixes.

Next implement direction:
1. Make the docs paragraph true: give the command for a report whose
   evaluations come from the same revision, and say that a finished
   experiment's page is rebuilt from the revision that ran it, or with scoped
   prefixes, until reading held evidence stops re-deriving matchups. Leave the
   rest of the section as it is.
2. Reading retained evidence should not require `eval(state)` to agree with
   the current source. Per "Etude is ours", that gap belongs in Etude (the
   report path through `records.py` tournament check), not in a manabot
   `_source` workaround or an edit to the frozen frontier file. Loop-or-next or
   implement files it as an Etude Task beside EXP-52/EXP-53 (filed: EXP-55).

Kept as is: `job_labels` names layers relative to the first job. That
reproduces the requested "Small (width 64)" / "Large (width 160, 8 layers)",
and its docstring says the order matters (scratch/questions.md keeps the
alternative). No open product choice.
