# MTG-329 plan

Approach (task): prototype manabot against the ideal Etude API → review at
demo → build the Etude API (its own linked Task/PR) → land the manabot diff
that deletes `execution.py`.

## Status

- write-docs (2026-11-12): `docs/training.md`, `README.md`, `manabot/README.md`
  describe the end state.
- implement (2026-11-12): prototype built; see `scratch/etude-api.md` (the
  one-page review: files, the Etude gaps, parity). Docs corrected to the
  built signatures (loop body, `settings.update`, `Tracker.log_table`,
  checkpoint names, `belief_sampler.py`, `collect_belief(match=)`,
  `game-attempts/` ledgers, PPO schedule note).
- write-docs, second pass (2026-11-12): docs matched to the simplified loop
  (`final.pt` is replaced when a later `updates` target runs; `--steps` sets
  the target; default `--out` is `.runs/<recipe>-<seed>`; `Stopped` exits 1;
  `failures/` in the run directory; `save_final`). Executor wording removed
  from `README.md` (belief bullet), `manabot/README.md` (Etude jobs section,
  package layout), `docs/learning.md` (code table), `docs/engine.md` (deck
  source), `docs/development.md`, `experiments/README.md`. Changed doc paths:
  `docs/training.md`, `README.md`, `manabot/README.md`, `docs/learning.md`,
  `docs/engine.md`, `docs/experiments.md`, `docs/development.md`,
  `experiments/README.md`.
  Left for the land diff: `docs/experiments.md` "Declaring a comparison"
  (`TrainingRegime` resolution, `active_runtime`, `pause.request`) and the
  `experiment/` line of the package layout, until each scheduler piece is
  verified as covered by Etude. `docs/training-loop.md` is linked but arrives
  with PR #474 (open). Three frozen reports link the removed
  `training.md#regimes-and-stages` / `#freeze-step-targets` anchors
  (`experiments/ataraxos-technique-screen.md`, `ataraxos-transfer.md`,
  `training-regime-followups.md`); not rewritten.
- Check: `uv run --extra dev pytest tests/training/test_self_play_loop.py -q`
  → 5 passed (parity PPO + damped bitwise, resume, refusal, stop). CLI
  `--recipe belief` twice and a 4-stream Elemental run via the stub both ran.

- implement, second pass (2026-11-12): code checked against the second-pass
  docs; no behavior change needed (ruff format of `cli.py`, `self_play.py`;
  loop snippet's omission list now names the carried counts). Check:
  `uv run --extra dev pytest tests/training/test_self_play_loop.py -q` → 5
  passed (226 s total). CLI smoke `--recipe belief --out TMP`: run 36 s exit 0;
  rerun 6 s exit 0; `ETUDE_STOP_FILE` set with `--steps 2` → "stopped at update
  1; run again to resume", exit 1, `final.pt` removed; rerun reaches 2 and
  writes a new `final.pt`; `--seed 6` on the same out refuses ("written for
  another seed: 5, not 6"). See `questions.md` 10.
  Docs describe, and the prototype does not yet build (all land-diff work):
  `manabot train --list` and experiment recipes through `--recipe`;
  `--regime`/`--regime-name` refusing (they still execute);
  `sampled_self_play()` returning `SelfPlay` (it returns a `TrainingRegime`);
  `collect_selection(policy, out, population)`; `run.json`/`metrics/` at the
  top of `out` (the stub writes `out/etude/<run>/`); `etude continue` from
  `run.out`; `local` as a loop recipe.

- write-docs, third pass (2026-11-12): recipes in `docs/training.md` call
  `track` with no experiment (Etude refuses a name that differs from the
  launch); default `--out` stated for every recipe, `local` keeps a fresh
  timestamped directory; `manabot/README.md` loses its remaining executor
  wording (`checkpoint_updates` → `SelfPlay.milestones`, `TrainSelfPlay` →
  `SelfPlay`, `TrainingRun` admission, the resume snapshot);
  `experiments/README.md` no longer cites `TrainingJob.experiment`. Changed
  doc paths are the same eight. Docs now also describe, and the prototype does
  not build: recipes omitting `experiment=` with the CLI supplying it;
  baseline on the resumable default `--out` and `--seconds` as a saved stop.
  New: `questions.md` 11–13, `etude-api.md` gaps 6–7.

- implement, third pass (2026-11-12): recipes call `track` with no
  experiment; `manabot train --recipe` sets `ETUDE_EXPERIMENT` to the recipe's
  name beside `ETUDE_OUT`; the stub refuses a recipe called with neither.
  Check: `uv run manabot train --recipe belief --out TMP` → exit 0,
  `run.json` records `"experiment":"belief"`; a bare `belief(BeliefConfig())`
  refuses. The loop is unchanged, so its test was not rerun. Still docs-only
  (land diff): baseline on the resumable default `--out`, `--seconds` as a
  saved stop, and the earlier list above.

## Remaining work (after demo review, not before)

Etude Task: `Run.out`, `Run.stopping()`, call-again on `Stopped`,
`etude continue` from `run.out` (gaps 1–5 in `etude-api.md`).

Land diff (step 3), **Delete — do not maintain**:
- `manabot/training/execution.py` and its exclusive tests (`test_recovery`,
  `test_active_recovery`, executor paths of `test_regimes`, …), the
  `--regime`/`--regime-name`/`--resume-from`/`--source-repair-reason`/
  checkpoint-cadence CLI options (refuse naming the last revision).
- `training/state/recovery.py`, executor parts of `admission.py`,
  `infra/runtime.RunContext`, `experiment/{job,runner,evaluation_queue,
  monitor_worker,study}.py` as Etude covers them (verify each).
- `manabot/recipes/proposed_etude.py` (recipes import `etude.train.run`).
- `SelfPlay.regime()` (move `validate_regime`'s checks into `SelfPlay`) and the
  executor parity test.
- Must survive: `TrainingRegime`/`records` reading, `load_checkpoint`, saved
  keys, game-error ledger/replay, failure capture, paired deals, world-bound
  checkpoints, seeds.
- Move `local`/`baseline`/`batched_baseline` recipes and experiment files'
  `runs()` to `training(recipe, config=...)`; add `manabot train --list`.
