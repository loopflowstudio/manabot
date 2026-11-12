# MTG-304 plan: training library pieces

## Layout (built)

- `manabot/rl/` — the generic package. It imports only torch and itself (`tests/test_package_boundaries.py`: a static AST check plus a fresh interpreter that refuses every other manabot module, managym, academy and etude).
  - `numerics.py`: `NumericalError`, `require`, `float64_operand`.
  - `returns.py`: `terminal_distances`, `reverse_waves`, `transition_gae`, `standardized_advantages`, `categorical_lambda_returns`.
  - `losses.py`: `clipped_surrogate`, `reverse_kl`, `squared_value_loss`, `categorical_value_loss`, `entropy`.
  - `selection.py`: `selected_rows`, `magnitude_quantile` (was `_magnitude_quantile`), `selected_moves`, `selection_mask`, `selected_mean`.
  - `step.py`: `StepRecorder` Protocol (`track`, `rejecting`, `sampling`, `before_clip`, `record_gradients(model, norm_before)`, `record_step`), `check_objective`, `gradient_step(..., recorder=None)` (a `None` recorder becomes the no-op `_Unrecorded`), `update_ema`.
- `manabot/training/updates/` (was `learning/`): an update learns from one batch. `damped_policy.py` (`damped_policy_loss`, `DampedTargets`, `_targets`, `update_damped_iteration`), `ppo.py` (`update_iteration`), and their pieces `learner`, `references`, `health` (`OptimizerHealth` implements `StepRecorder`), `selection` (`selection_diagnostics`, `value_scale`, `UpdateDiagnostics`).
- `manabot/training/loops/`: loops only; `minimal.py` is the example loop (its update written inline on purpose).
- Deprecated aliases (`manabot/infra/deprecation.py::moved`, removal 0.3.0): every 0.2 `learning.*` module stays as an alias module (`learner`, `references`, `health`, `selection`, `damped_policy`, `returns`, `objectives` with its warning dispatcher) pointing at `training.updates.*` or `manabot.rl.*`; `model.policy_distribution`, `model.numerics`. Covered by `test_learning_pieces.py::test_moved_names_resolve_through_deprecated_aliases`. Unreleased names (`loops.ppo`, `loops.damped_policy`, `rl.step`, `rl.losses`, `failure_tensors`, `minibatch_tensors`) moved without aliases.
- All call sites in manabot/, scripts/, tests/ import the new paths. `experiments/protocols/*.json` keep their old paths (frozen evidence). `paper/` paths updated.

## Docs

Naming steer cfe305b0 (final): update, loop and recipe are the only nouns; the guide opens with that sentence, then "write your own recipe or use one of ours, and use Etude to launch, track and evaluate it, or bring your own", then where to start; docs/training.md carries the same sentence. TrainingRegime and regime keys are not renamed here (MTG-303). The "what is specific to Magic" list is gone (steer 0f7227c7: "more of a guide than a sales pitch"); sections: the updates we maintain, the player's pieces, standard methods kept local (with what a deck builder reuses), write your own loop with `minimal.py` verbatim.

Deck builder: no design doc in this checkout; the steer text ("free-order picks until stop, a win/loss/draw outcome head at every prefix, entropy-to-go, PPO clip 0.2 plus reverse KL to the generating policy") is the source. Lacks: an entropy-to-go piece, its own masked pool+stop distribution, collection, loop.

## Steers (2026-11-12)

Full text: `.lf/prompts/20261112-024634-ZEiaX4-steer.md` (maintained loops under `loops/`), `20261112-024717-GXJU5d-steer.md` (generic vs role-specific), `20261112-025331-prJpdZ-steer.md`. The third, Jack Heart on "Keep the generic functions, but small and unglamorous": "this sound right". It asks for PyTorch built-ins only where behaviour stays identical and no RL library. `clip_grad_norm_` was already in use. `AveragedModel` was not adopted: it wraps the module and adds an `n_averaged` buffer, so EMA and buffer-copy semantics would not stay exact. `torch.distributions.Categorical.entropy` was not adopted either: it clamps `-inf` logits to the dtype minimum and runs a different reduction, so the floats may not match exactly.

## Proof

Parity script (outside the repo, `/tmp/mtg304parity/parity.py`): on one collected batch (2 streams, 6 decisions, hidden 16), run PPO at retained 0.5 with KLs on, then damped scalar and damped categorical at quantile 0.5 and iteration 1, each through the rule its learning record selects, and dump every diagnostic and the per-parameter sums. Pre-rebuild check history (412 passed across changed files) is in git before 9ae190e51; the current results are under "Per-file check pass" below.

## Compress (2026-11-12)

`updates/ppo.py` computes the value residual once and starts all three exposure counters in the diagnostics literal. Considered and kept: building PPO's mask with `selection_mask` would derive `selected` in sorted order, which changes the top-count minibatch order and breaks parity; Parity JSON byte-identical; focused suites 74 passed.

Second compress (rebuilt branch): `gradient_step` takes a no-op `_Unrecorded` recorder when none is passed, removing its `None` branches. Parity JSON byte-identical to `main_e5897.json`; pieces/minimal/health/boundaries 68 passed.

Third compress (after the naming pass): `updates.health.clip_fraction` replaces the clip-fraction expression both rules repeated; the damped rule starts its exposure counters in the diagnostics literal and reads `batch.values` once. Parity (`/tmp/mtg304parity/parity_new.py`, new import paths, `-W error::DeprecationWarning`) byte-identical to `main_e5897.json`; pieces/minimal/health/objectives/admission/boundaries 97 passed.

## Remaining

- PR #474 targets `main` at 4729e8d37 (naming pass, third compress and main 88b71387e merged); its description matches the `updates/` / `loops/` layout. CI was in progress at the realign check.
- Naming-pass checks: pieces/minimal/boundaries/objectives/credit 79 passed; admission/learner/health/mps/precision/ci-selection 124 passed; test_regimes 18 passed; test_game_errors 3 failed (`binding_mismatch`, already on main).
- Collection is the existing `SeatRoutedCollector` → `RolloutBatch`, documented as a piece in `docs/training-loop.md`; no new collection API.
- Outcome 4 (smaller executor) belongs to MTG-303: `execution.py` is 1,794 lines here against 1,784 on main (new import paths plus the rule choice).
- Open outside this Task: the `binding_mismatch` and packed-seed digest failures already on main (see the per-file check pass) need their own bug Tasks or fixes on main.

## Decisions kept

- `manabot.rl` is top-level, not under `manabot/training/`, because `manabot/training/__init__.py` imports `records` and `regime`.
- `legal_mask`/`policy_logs`/`check_*` stay in `manabot.model.policy_distribution` as the player's legality, per the steer, though they are pure tensor code.
- `damped_policy_loss` lives with its rule in `updates/damped_policy.py`: it reads `check_logs`/`check_probabilities`/`policy_logs`.
- `OptimizerHealth.check` before `gradient_step`'s own check is a timing boundary; both call `check_objective`.
- `failure_tensors`/`minibatch_tensors` are shared plumbing, undocumented in the guide.
- Example loop takes `model: ModelConfig | None` so the test can use a tiny model.

## Concept review, second pass (2026-11-12)

Nothing to change. Checked against the old tree (on the rebuilt branch the choice is in `execution.py`): `stages.py:564` chose `update_damped_iteration` for `DampedPolicyLearning`, else `updates.ppo.update_iteration` (typed `Learning` only); the deprecated dispatcher in `learning/objectives.py` warns and keeps the 0.2 behaviour. Considered and kept: renaming `updates.ppo.update_iteration` so it matches `update_damped_iteration`. The module path already names the rule, and the guide and tests use that name.

Third pass (rebuilt branch, after the no-op recorder): nothing to change. `minimal.py` still composes only `rl` pieces, the player's legality and `AdmittedRollout`, `SeatRoutedCollector`; the rule choice stays one branch in `execution.py`; no open product choice.

## Rebuilt on main without MTG-303 (2026-11-12)

Steer (Jack Heart, via comment 69b01701): "ok, lets do that. let me review a clean 304". This Task lands first; MTG-303 stacks on top. Outcome 4 (shrinking the executor) moves to MTG-303.

- Built from `main` e5897c7ab: this Task's own files applied from the old tip (old branch kept locally as `mtg304-pre-rebuild`, 364470932). Nothing from MTG-303: no `run.py`, `stages.py`, `two_models.py`; `records.py`, `regime.py`, `state/recovery.py` as on main.
- `execution.py` = main plus imports from the new paths and the rule choice at the `stage.trainable` branch (`update_damped_iteration` for `DampedPolicyLearning`, else `updates.ppo.update_iteration`). Tests that patched `execution.update_iteration` for damped regimes now patch `execution.update_damped_iteration`.
- Ported main's `AdmittedRollout` (#480) into `updates/ppo.py`, `updates/damped_policy.py` (`_targets` takes the admitted rollout) and `loops/minimal.py`. `tests/model/test_admission.py` (new on main) calls the rule its learning record selects.
- `experiments/l4_step_profile.py` keeps its old import paths (an experiment at its commit); the aliases resolve them.
- Parity: `/tmp/mtg304parity/parity.py` on main e5897c7ab vs this branch (with `-W error::DeprecationWarning`): JSON byte-identical (PPO, damped scalar, damped categorical; diagnostics and per-parameter sums).

## Docs (write-docs, rebuilt branch)

Changed doc paths: `docs/training-loop.md` (the guide; Play table now lists `AdmittedRollout`, example is `minimal.py` verbatim), `docs/learning.md` (code table), `docs/training.md` (pointer), `README.md` (docs table), `manabot/README.md` (intro, package layout), `manabot/model/training.md`, `AGENTS.md` (deprecated-alias mechanism).

## Per-file check pass (2026-11-12, after merging origin/main 92e476217)

Steer: no pytest run of 5 minutes or longer; run changed files first, name the slow test. Every `tests/training` and `tests/model` file ran alone under a 280 s alarm: none hung; 96 files took 1,233 s in total. Slowest were test_training_throughput 125 s, test_experiment_runner 117 s and test_training_throughput_sizes 81 s, so the earlier 50-minute run was one serial process over the whole set, not a hang. Pieces, loops and boundaries: 125 passed in 19 s.
- Fixed: `scripts/inspect_batch.py` passed a `RolloutBatch` to `_targets`, which takes an `AdmittedRollout` (broken on main too). test_inspect_batch_script passes.
- Already on main: test_game_errors `binding_mismatch` (3 failures); test_training_throughput::test_a_packed_seed_digests_as_it_does_solo, because checkpoint `source/etude_run` (added in #464) differs between the solo and packed ETUDE_RUN names and the frozen experiment ignores only `bc/run_id`.
- Flaky: test_attack_execution::test_selected_world_attack_cohort_replays_all_rungs, whose collector hits its 30 s training wall. It failed once and passed on the rerun, here (68 s) and on main (101 s).
- Need duckdb, which is not installed here: test_etude_evaluation and test_head_to_head.
