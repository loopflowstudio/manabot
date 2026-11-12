diff --git a/.lf/skills/qa.md b/.lf/skills/qa.md
index 49b96058f..bf099843a 100644
--- a/.lf/skills/qa.md
+++ b/.lf/skills/qa.md
@@ -104,6 +104,19 @@ publish, land or choose the next Graph step from here.
   `--seed` venv and upgrade with plain `pip` (wheel paths plus `--find-links`
   and the CPU Torch index), since uv's one-step upgrade hides file-ownership
   problems that pip's install order exposes.
+- Model and training paths: train with `uv run manabot train --device cpu --steps 2
+  --out .runs/<name>`; the same command on the base commit must give an equal
+  `regime_digest` and bitwise-equal `self-play-raw.pt` tensors. For the base, use a
+  detached `git worktree add` in `/tmp`, copy in the `.so` when `managym/` is
+  unchanged, and run `PYTHONPATH=<worktree> .venv/bin/python -m manabot.cli ...`
+  from inside it. A checkpoint plays only its bound decks: drive
+  `manabot.sim.sim.run_simulation` with `MatchConfig` from the run's
+  `run.json` `regime.match`, and grep the log for `Error getting action`, because
+  the sim loop replaces a raising player's action with a random one.
+  `flat_mc.play_games` drops sideboards, so wrap its `MatchConfig` the same way.
+  Academy lists checkpoint opponents named in `ACADEMY_PLAY_CHECKPOINTS` (the
+  `MANABOT_RUNS_DIR` scan does not follow symlinks); choose one from the
+  `opponent-select` testid before New Game.
 - Training recipes: write a regime JSON from Python
   (`TrainingRegime(...).model_dump_json()`) and run
   `uv run manabot train --regime <file> --seed N --out /tmp/<dir>/<run>` with a
