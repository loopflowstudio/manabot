# MTG-206 assumptions and dependencies — 2026-10-27

The Task directive (Jack Heart, 2026-10-27) is the accepted design; the plan
and remaining work are in `scratch/mtg206.md`. Inherited MTG-188 notes remain
at `37e5ccd8d65527f63486f1d9770af20f612fa8c8:scratch/questions.md`.

- **etude-sdk 0.2.10** (released 2026-10-27 via `lf release run`, etude #69)
  carries blocks (#63) and the batched `estimate` (#68); manabot pins it.
- **Isolated launcher home for the L4 smoke.** `~/.etude/launcher.db` is
  schema v10, held by the live MTG-183 controller's older etude; 0.2.10 needs
  v11. The smoke launched from `ETUDE_HOME=~/.etude-mtg206` with the same r2
  storage, runpod provider and `l4-etude` on-demand, leaving the shared one
  untouched.
- **`excluded` now counts player-games**, like `games`: the rollups hold per-seat
  rows, so a game excluded for both seats counts twice when no subject is chosen
  (it was distinct games). With a subject, the counts are unchanged.
- **`source_commit` added to both rollups' `by`** to keep `cards data --commit`;
  not in the directive's column list.
- **Historical per-file runs** (MTG-186 cards, the 0.2.5 fixtures, the MTG-183
  bucket copy) are readable only after `etude convert-eval`. Conversion of retained
  data is a separate explicit step; the MTG-186 docstring names the command.
- **elemental_long_rent.py / GPU `demo` command** live on the MTG-183 branch,
  not on main, so the L4 smoke uses `experiments/mtg206_cards.py` (MTG-186's
  evaluation infrastructure with the checkpoint-free `cards verify` schedule)
  instead of the demo's heuristic grids.
- **PR #389 local snapshot** (`d2f46c62c`, inspected 2026-10-27) changes
  `cards.py`'s verification source identity to `sim/anchors.py` and replaces
  subprocess environment dictionaries with `child_environment()` in
  `mtg186_cards.py` and `elemental_long_eval.py`, alongside imports. The changes
  are compatible in purpose with this Task's blobs/rollups port; the earlier
  claim that overlap was limited to imports was inaccurate.
- **Headless demo acceptance (2026-10-27).** The caller explicitly delegated
  the decision. Use the stated completion requirement, generation in seconds,
  rather than treating the 1.5-second prototype as an unstated hard cap. The
  recovered original cohort takes 16.01 s wall / 5.28 s CPU for the page;
  database catch-up adds 30.48 s wall in the separated measurement. All card
  metrics match prior results and the L4 audit replays 96/96. This supports
  advance, not a claim of human approval or a 1.5-second CLI. See
  [the demo record](mtg206-demo.md). No stored-data conversion was needed.
