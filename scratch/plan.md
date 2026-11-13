# MTG-329: Etude train API rebuild

## Direction and scope

`scratch/design-train-api.md` is the current design. It replaces the earlier
`state.pt` cutover, whose implementation is still in this checkout. A recipe
remains ordinary `train(config)` over a visible loop. Etude owns seeded runs,
resume/save/steps, stopping, retries and checkpoint publication; Manabot owns
learning, world-bound policies and game-error evidence. Pydantic is required.
The affected user docs already describe the new API; preserve those edits.

Jack Heart, 2026-11-13: "you can run both. i dont care that manabot cant be test. just make sure to clarify it should build what ican then stop".
Latest direction, 2026-11-13: "i still think we should restart mtg-329 to match the proposed API,
and just have it wait after its done writing to block then". Build the full
client cut against the proposed API; no local stub and no delivery. EXP-56 owns the SDK and
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

Deleted in the proposed-API cut: Tracker/untracked path, saved LoopState,
load_state/save_state, setting_differences/StateMismatch and resume guards,
manual stop/resume branches, SelfPlay.save_every, recipe seeds, local_launch
callers, cached belief pipeline and benchmark Rows. Their exclusive fixtures
are replaced by real SDK save/resume, immutable-policy and failure tests.

Must survive: raw/Adam/EMA/counts, schedules independent of stop targets,
checkpoint_source, immutable world-bound policies, game-error ledgers and
replays, bounded numerical incidents, paired deals and training-seed evidence.
Frozen TrainingRegime/TrainingRun records retain serialized keys; execution
refuses with pre-cutover main f4e1822faf46bbc9cdc43909293f8cbc8de4a300.
Older collector snapshots use 439379869. Never rewrite frozen evidence.

## Remaining implementation

1. Rebase onto MTG-348's Etude naming migration, then pin EXP-56's released
   SDK with requirements, runtime pins, lock and consumer checks together.
   No source override or compatibility shim is committed. EXP-56 at dda6b17
   now implements run.seed/resume/steps/save/input and snapshot-pointer saves;
   its CLI, job seeds and start=Job contracts remain unfinished. A real-source
   import stops at Manabot's `etude.run.records` (removed in EXP-56); the
   prerequisite naming migration belongs to MTG-348.
2. Proposed-API client cut is written: self_play and minimal, Pydantic counts,
   seed-free configs, linked belief jobs, baseline policy evidence, Etude CLI
   benchmark and nightly/release callers. User docs and surviving tests move
   with the cut. This is unvalidated execution code until dependency work lands.
3. Real fresh/stop/resume/continuation and immutable-policy checks, seed-197
   learner parity, installed consumers, launch restore/retry and service
   continuation. Do not count the older state.pt evidence as new-API proof.
4. Gate affected checks once. Resolve/report the frozen replay schema mismatch
   (fixture 7 vs current 8) without regenerating it. Demo recovery losses:
   no current-game resume, exact uninterrupted equivalence or in-run source
   repair. Frozen staged/adaptive/frozen-opponent launchers remain retired.

User docs for the rebuild, written against the proposed API and unrun:
`docs/training.md` (run, write a recipe, stages, remote, checkpoint cadence,
recovery), `docs/training-loop.md` (minimal loop), `docs/experiments.md`,
`README.md`, `manabot/README.md`, `.lf/skills/qa.md`. The 2026-11-13 docs pass
moved them to the save-folder layout, in-place `Counts`, job-and-seed run ids
and `--resume RUN --until`; questions.md lists what that took from EXP-56's
draft. The implementation follows these docs: `step` updates `counts` in
place, and the smoke callers read `track/local/smoke-seed-0/`.

Validation for this pass is recorded in implementation.md. No paid run,
publication, landing, or background service is part of this pass.

Historical ontology and Etude README proposals: `bd14801a72:scratch/ontology.md` and
`bd14801a72:scratch/etude-readme-draft.md`; these belong to EXP-56.
