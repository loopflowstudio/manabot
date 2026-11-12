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

Etude Tasks filed (pending Linear sync, in ../etude):
- eb0db3 Pool one comparison's games across execution setups in reports.
- 9f66c2 Curve opponent "whichever earlier checkpoint of its line it met".

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
  Needs Etude 9f66c2. The previous per-size report() mixed roles the same way.

Found while building:
- `etude report experiments/tla_frontier.py` from this revision refuses the
  held tournament (`tournament_conflict`): `eval(state)` rebuilds matchups
  with today's manabot source digest (`_source()` hashes every
  `manabot/*.py`). The demo used a wrapper file with this `report()` and no
  `eval`. Also `~/.etude/launcher.db` is from another Etude version, and the
  unscoped `--source r2://manabot/jobs` scan ran >15 min, 3.9 GB.
- Steer (Etude #98, merged, unreleased): launch-name subjects/labels/schedules
  and Refresh of repository helper modules. Not needed by these helpers (job
  names equal launch names here); no Etude release cut. eb0db3 narrowed to
  eval-environment pooling, which #98 does not cover.
