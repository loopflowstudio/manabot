# MTG-329: Etude train API rebuild

## Direction and scope

`scratch/design-train-api.md` is the current design. It replaces the earlier
`state.pt` cutover, whose implementation is still in this checkout. A recipe
remains ordinary `train(config)` over a visible loop. Etude owns seeded runs,
resume/save/steps, stopping, retries and checkpoint publication; Manabot owns
learning, world-bound policies and game-error evidence. Pydantic is required.
The affected user docs already describe the new API; preserve those edits.

Jack Heart, 2026-11-13: "you can run both. i dont care that manabot cant be test. just make sure to clarify it should build what ican then stop".
The implementation boundary is: build independent changes, then stop at the
missing EXP-56 calls; no local stub and no delivery. EXP-56 owns the SDK and
its naming migration. The full design conversation and earlier review history
are preserved at `bd14801a72d48ad38eeca18d08fc1c9e7f99612a:scratch/` (plan, positioning,
cutover, ontology, etude-api, functional-audit, etude-readme-draft, with exact
remarks and contrary findings). That history selects no work; Etude's
README/ontology proposals belong to EXP-56. No Wave is selected.

## Independent CLI cut — 2026-11-13

`manabot train` now exits 2 naming `uv run etude run FILE` and `--resume RUN`.
Old flags also receive that guidance. Removed its recipe discovery, config
reflection, automatic device selection, environment/deadline management and
local_launch handling. The local recipe's CLI-only 120-second constant is
gone. CI and release smoke callers now use Etude; the installed smoke writes
an ordinary recipe file importing the installed library. SDK-dependent smoke
execution remains unproved. This is an incomplete rebuild, not delivery.

## Delete — do not maintain

Already deleted in earlier commits: training/execution.py;
training/state/{recovery,learning_state,admission}.py;
experiment/{job,runner,evaluation_queue,monitor_worker,study}.py;
recipes/proposed_etude.py; RunContext; TrainingRecipe; obsolete scheduler,
recovery and stage execution tests. Keep retained record/report readers.

Deleted in the CLI cut: RECIPES, recipe_names, experiment_recipe, the old
train_command body, recipes/device.py and its boundary allowlist entry;
exclusive CLI tests from test_recipe_etude, test_regimes and test_mps_training.
Their surviving non-CLI coverage remains. test_train_command covers migration.

Still to delete with the SDK cut: Tracker/untracked path; saved LoopState;
load_state/save_state, setting_differences/StateMismatch and format/world
resume guards; manual stop/resume branches and SelfPlay.save_every;
recipe config seeds and local_launch uses; belief's cached pipeline; benchmark
Rows tracker. Their exclusive tests/fixtures go in that same cut, with behavior
coverage on the real surviving SDK path. No compatibility loading for state.pt.

Must survive: raw/Adam/EMA/counts, schedules independent of stop targets,
checkpoint_source, immutable world-bound policies, game-error ledgers and
replays, bounded numerical incidents, paired deals and training-seed evidence.
Frozen TrainingRegime/TrainingRun records retain serialized keys; execution
refuses with pre-cutover main f4e1822faf46bbc9cdc43909293f8cbc8de4a300.
Older collector snapshots use 439379869. Never rewrite frozen evidence.

## Remaining implementation

1. EXP-56 must provide run.seed/resume/steps/save/input, Pydantic track config,
   checkpoint folders and start= jobs, and etude run/--resume. The available
   SDK source remains 105abdd; none of resume/steps/save/input or run.seed is
   implemented there (source inspection this pass). Pin a real release and
   refresh requirements, runtime pins, lock and consumer checks together.
2. Rebuild self_play and minimal, configs/recipes and experiment seeds against
   the actual API. Split belief_policy and belief_sampler into linked jobs.
   Migrate baseline evaluation, benchmark and remaining local_launch callers.
   Keep all user docs consistent; questions.md lists unresolved contracts.
3. Real fresh/stop/resume/continuation and immutable-policy checks, seed-197
   learner parity, installed consumers, launch restore/retry and service
   continuation. Do not count the older state.pt evidence as new-API proof.
4. Gate affected checks once. Resolve/report the frozen replay schema mismatch
   (fixture 7 vs current 8) without regenerating it. Demo recovery losses:
   no current-game resume, exact uninterrupted equivalence or in-run source
   repair. Frozen staged/adaptive/frozen-opponent launchers remain retired.

Validation for this pass is recorded in implementation.md. No paid run,
publication, landing, or background service is part of this pass.
