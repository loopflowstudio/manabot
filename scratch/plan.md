# MTG-329 plan

2026-11-12 naming direction, Jack Heart: "why is _verify_checkpoint private -- in etude i saw that we had instructions to keep functions private if they werent used outside, but i dont think thats necessary and find that to be distracting".

2026-11-12 recipe interface review, Jack Heart: "im not sure i liek this set u     self_play: SelfPlay = Field(default_factory=elemental_self_play)" and "i liekd it better when there were explicit eval(), train() and launch() functions". He clarified: "the idea was just to expose more primitives in etude to facilitate the ML OPs part of the train loop" and "but not to change the overall structure of hte recipe files". Preserve the existing recipe-file structure and explicit operations. Expose and use Etude primitives for the loop's ML operations; do not introduce a recipe framework or require the prototype's nested `ElementalConfig.self_play` structure. This narrows the prototype direction above; the proposed recipe restructuring is not the intended outcome.

Further clarification, Jack Heart: "well, to move the thigns that were in the execution loop itno things that are provided as general tools by etude" and "so they dont have to be in a 1000 line file". The intended simplification is extracting reusable ML operations from the large manabot executor into general Etude tools, then calling those tools from ordinary training code while preserving recipe-file structure. Merely adding conveniences around the existing executor would not achieve this outcome.

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

- functional pass (2026-11-12, in the Task conversation). Jack Heart: "the
  style in self_play.py still seems very Object ORiented and Frameworky. can
  we make it more functional?" `SelfPlay` is data only; its methods became
  functions (`to_regime`, `root_match`, `learner_digest`, `open_collector`,
  `is_checkpoint`); `Every`/`every()` became `checkpoint_every: int | None`;
  `LoopState`/`Counts` are frozen; `LoopState.advance`/`.behavior` became
  `counted`/`behavior`; one update is `step(state, ...) -> (state, row)`
  using `learn`. Check: `uv run --extra dev pytest
  tests/training/test_self_play_loop.py -q` → 5 passed (153 s), parity
  bitwise for PPO and damped. Docs updated (`docs/training.md`,
  `manabot/README.md`). Recipes' `try/except: run.finish("failed")` is still
  frameworky; not changed.

- Etude Task filed and started 2026-11-12: EXP-56 "Give a training recipe a run
  directory, a stop signal and a retry after a planned stop" (loopflow://etude/task/EXP-56),
  `code` graph; its brief carries gaps 1-8 of `etude-api.md`. Jack Heart:
  "ready to run an etude task?" / "open its task session in rex so i can
  review its work once its done".

- sync + write-docs, fourth pass (2026-11-13, clarified extraction scope).
  Main (`b531853a5`, with the torch/MLX backend move) is merged as `344b7d116`.
  Resolution: main's `backends/torch` `Learner` class is kept (it is the
  backend protocol MLX also implements), so the branch's `new_learner` /
  `learner_precision` are gone and callers use `Learner(...)` /
  `learner.forward_precision()`; updates take `torch_backend` first. The
  branch's functional collector (`open_collector`/`collect`/`close_collector`)
  is kept, with main's bucket-needs recording ported into it.
  `self_play.py` imports from `manabot.backends.torch.*` and is listed in
  `_TORCH_NOT_YET_MOVED`; `cli.py` uses `recipes.device.accelerator`.
  Checked after the merge: 173 passed across the merged learner, collector,
  update-backend, admission, artifact and boundary tests. Failing and not
  diagnosed to a cause: `tests/training/test_game_errors.py::
  test_resource_limit_receipt_replays_its_retained_encoding_boundary` (3
  params) returns `binding_mismatch`: the replayed decision frame has
  `schema_version` 8 and the fixture `tests/fixtures/tla-pilot-permanent-capacity.jsonl`
  has 7. `replay_attempt`'s comparison and the fixture are identical on main,
  where CI is green, so check whether CI runs this test before changing
  anything. `tests/training/test_self_play_loop.py` (executor parity) was not
  rerun after the merge. Pre-existing on the branch, untouched:
  `experiments/l4_step_profile.py`, `collection_tail_graphs.py` and
  `cuda_precision.py` still patch `SeatRoutedCollector`, which no longer exists.
  Docs now describe the preserved file structure. Changed doc paths:
  `docs/training.md`, `manabot/README.md`, `experiments/README.md`,
  `docs/learning.md`, `docs/training-loop.md` (the last two only for the merge).
  What the docs now say that the code does not yet do (implement step):
  - A recipe module exposes `train(config)` (`manabot.recipes.elemental.train`,
    `manabot.recipes.belief_sampler.train`), not `elemental(config)` /
    `belief(config)`. Configs hold only varying inputs (`seed`, `updates`,
    `games`, `validate_numerics`); the `SelfPlay` settings are built in the
    function body, not held as `ElementalConfig.self_play`.
  - `updates = int(run.until("updates", default=config.updates))`; imports are
    Etude's own (`etude.train.run.track`, `local_launch`, `Stopped`;
    `etude.train.records.Step`); `manabot/recipes/proposed_etude.py` is gone
    and the loop raises Etude's `Stopped`, not its own.
  - No `final.pt` and no `save_final`: the loop always writes and records the
    last update's checkpoint (tag `final`) and returns its path (EMA when
    kept). `save_checkpoint` never rewrites an existing file.
  - Belief outputs are named for the policy (`histories-<policy stem>.json`,
    `belief-<policy stem>.pt`).
  - A planned stop exits 75 from `manabot train` (was 1).
  - The local Etude record is under `OUT/etude/`.
  - `etude continue --run RUN --until updates=N`.
  - Experiment files: `runs()` returns `training(train, config=..., smoke=...)`.
  Still describing the executor, unchanged: `docs/experiments.md` "Declaring a
  comparison" (questions.md 13).
