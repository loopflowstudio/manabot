# MTG-304 plan: training library pieces

## Layout (built)

- `manabot/rl/` — the generic package. It imports only torch and itself (`tests/test_package_boundaries.py`: a static AST check plus a fresh interpreter that refuses every other manabot module, managym, academy and etude).
  - `numerics.py`: `NumericalError`, `require`, `float64_operand`.
  - `returns.py`: `terminal_distances`, `reverse_waves`, `transition_gae`, `categorical_lambda_returns`.
  - `losses.py`: `clipped_surrogate`, `reverse_kl`, `squared_value_loss`, `categorical_value_loss`, `entropy`.
  - `selection.py`: `selected_rows`, `magnitude_quantile` (was `_magnitude_quantile`), `selected_moves`, `selection_mask`, `selected_mean`.
  - `step.py`: `StepRecorder` Protocol (`track`, `rejecting`, `sampling`, `before_clip`, `record_gradients(model, norm_before)`, `record_step`), `check_objective`, `gradient_step(..., recorder=None)`, `update_ema`.
- `manabot/training/learning/` — the player package: `learner`, `references`, `health` (`OptimizerHealth` implements `StepRecorder`; `before_clip` stores `critic_gradient_norms`), `selection` (`selection_diagnostics`, `value_scale`, `UpdateDiagnostics`).
- `manabot/training/loops/` — `damped_policy.py` (`damped_policy_loss`, `DampedTargets`, `_targets`, `update_damped_iteration`), `ppo.py` (`update_iteration`), `minimal.py` (example).
- Deprecated aliases, via `manabot/infra/deprecation.py::moved` (lazy module `__getattr__`, warns with the new path and removal in 0.3.0): `learning.objectives` (`update_iteration`, `update_ema`), `learning.damped_policy` (rule, loss, `DampedTargets`, `categorical_lambda_returns`), `learning.returns`, `learning.selection` (the four selection functions), `model.policy_distribution` (`NumericalError`, `require`, `clipped_surrogate`, `reverse_kl`), `model.numerics` (`float64_operand`). `test_learning_pieces.py::test_moved_names_resolve_through_deprecated_aliases` covers each. Unreleased names (`step.py`, `losses.py`, `check_objective`, `critic_gradient_norms`) moved without aliases.
- All call sites in manabot/, scripts/, tests/ import the new paths. `experiments/protocols/*.json` keep their old paths (frozen evidence). `paper/` paths updated.

## Docs

`docs/training-loop.md` follows the three steers: what is specific to Magic first (engine/observations/card tokens, legal-action policy, two-seat complete-game credit, WDL head and damped rule, collector, our loops), then the maintained loops, the player's pieces, one short "Standard methods, kept local" section on `manabot.rl` with "What a deck builder reuses", then "Write your own learner" with `minimal.py` verbatim. learning.md code table, README docs table, manabot/README (intro + package layout) updated.

Deck builder: no design doc in this checkout; the steer text ("free-order picks until stop, a win/loss/draw outcome head at every prefix, entropy-to-go, PPO clip 0.2 plus reverse KL to the generating policy") is the source. Lacks: an entropy-to-go piece, its own masked pool+stop distribution, collection, loop.

## Steers (2026-11-12)

Full text: `.lf/prompts/20261112-024634-ZEiaX4-steer.md` (maintained loops under `loops/`), `20261112-024717-GXJU5d-steer.md` (generic vs role-specific), `20261112-025331-prJpdZ-steer.md`. The third, Jack Heart on "Keep the generic functions, but small and unglamorous": "this sound right". It asks for PyTorch built-ins only where behaviour stays identical and no RL library. `clip_grad_norm_` was already in use. `AveragedModel` was not adopted: it wraps the module and adds an `n_averaged` buffer, so EMA and buffer-copy semantics would not stay exact. `torch.distributions.Categorical.entropy` was not adopted either: it clamps `-inf` logits to the dtype minimum and runs a different reduction, so the floats may not match exactly.

## Proof

Parity script (outside the repo, `/tmp/mtg304parity/parity.py`): on one collected batch (2 streams, 6 decisions, hidden 16), run `update_iteration` with PPO at retained 0.5 and KLs on, then damped scalar and damped categorical at quantile 0.5 and iteration 1, and dump every diagnostic and the per-parameter sums. Before the split and after it (run with `-W error::DeprecationWarning`), the JSON is byte-identical.

Checks this pass: test_learning_pieces + test_minimal_loop + test_numerical_health + test_package_boundaries: 66 passed.

## Checks (after merge)

- Parity JSON (`/tmp/mtg304parity/parity.py`, now calling the rule the record selects, `-W error::DeprecationWarning`): byte-identical to the pre-split dump.
- Stage-selects-rule pass: focused + mps/recovery/active_training/credit_assignment: 147 passed; precision, two_seat_learning, phase_timing, regimes, adaptive_two_seat, rl_objectives, game_errors, naming_transitions: 101 passed, 3 failed (the known `binding_mismatch` below); `scripts/inspect_batch.py --updates 1` runs.
- Every test file changed since aeb9b8cf0 except `tests/eval/test_tla_frontier.py` (needs duckdb, not installed): 412 passed, 3 failed — the known `test_game_errors.py::test_resource_limit_receipt_replays_its_retained_encoding_boundary[*]` `binding_mismatch`, present at d2cde1117 and outside this change. Gate decides between a bug Task and a fix on main.

## Compress (2026-11-12)

`loops/ppo.py` computes the value residual once and starts all three exposure counters in the diagnostics literal. Considered and kept: building PPO's mask with `selection_mask` would derive `selected` in sorted order, which changes the top-count minibatch order and breaks parity; Parity JSON byte-identical; focused suites 74 passed.

Second compress (rebuilt branch): `gradient_step` takes a no-op `_Unrecorded` recorder when none is passed, removing its `None` branches. Parity JSON byte-identical to `main_e5897.json`; pieces/minimal/health/boundaries 68 passed.

## Remaining

- PR #474 against `main` with this diff alone; force-push this branch; retitle and rewrite the body for this Task only.
- Collection stays as-is (`SeatRoutedCollector`); no new collection API.

## Decisions kept

- `manabot.rl` is top-level, not under `manabot/training/`, because `manabot/training/__init__.py` imports `records` and `regime`.
- `legal_mask`/`policy_logs`/`check_*` stay in `manabot.model.policy_distribution` as the player's legality, per the steer, though they are pure tensor code.
- `damped_policy_loss` lives with its rule in `loops/damped_policy.py`: it reads `check_logs`/`check_probabilities`/`policy_logs`.
- `OptimizerHealth.check` before `gradient_step`'s own check is a timing boundary; both call `check_objective`.
- `failure_tensors`/`minibatch_tensors` are shared plumbing, undocumented in the guide.
- Example loop takes `model: ModelConfig | None` so the test can use a tiny model.

## Concept review, second pass (2026-11-12)

Nothing to change. Checked against the old tree (on the rebuilt branch the choice is in `execution.py`): `stages.py:564` chose `update_damped_iteration` for `DampedPolicyLearning`, else `loops.ppo.update_iteration` (typed `Learning` only); the deprecated dispatcher in `learning/objectives.py` warns and keeps the 0.2 behaviour. Considered and kept: renaming `loops.ppo.update_iteration` so it matches `update_damped_iteration`. The module path already names the rule, and the guide and tests use that name.

## Rebuilt on main without MTG-303 (2026-11-12)

Steer (Jack Heart, via comment 69b01701): "ok, lets do that. let me review a clean 304". This Task lands first; MTG-303 stacks on top. Outcome 4 (shrinking the executor) moves to MTG-303.

- Built from `main` e5897c7ab: this Task's own files applied from the old tip (old branch kept locally as `mtg304-pre-rebuild`, 364470932). Nothing from MTG-303: no `run.py`, `stages.py`, `two_models.py`; `records.py`, `regime.py`, `state/recovery.py` as on main.
- `execution.py` = main plus imports from the new paths and the rule choice at the `stage.trainable` branch (`update_damped_iteration` for `DampedPolicyLearning`, else `loops.ppo.update_iteration`). Tests that patched `execution.update_iteration` for damped regimes now patch `execution.update_damped_iteration`.
- Ported main's `AdmittedRollout` (#480) into `loops/ppo.py`, `loops/damped_policy.py` (`_targets` takes the admitted rollout) and `loops/minimal.py`. `tests/model/test_admission.py` (new on main) calls the rule its learning record selects.
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
