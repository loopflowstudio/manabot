# MTG-329 implementation

Design: design-train-api.md; historical plan: 382c653ec8:scratch/plan.md.
Etude 0.4.0 now replaces the development source override; all requirements,
release checks and lock use the release. No publication or landing in this pass.

## Delete — do not maintain

Deleted: training/execution.py; recovery/admission/learning_state; experiment
job/runner/study/evaluation queue/monitor worker; TrainingRecipe/RunContext;
Tracker, LoopState, load/save_state, StateMismatch, resume guards, local_launch,
CLI recipe discovery/device selection and their exclusive tests. No replacement
executor, shim or SDK stub. Retain record/report readers and frozen bytes.
Retired regimes name f4e1822faf46bbc9cdc43909293f8cbc8de4a300; old state.pt folders
name f577362135562e569033fe33e48c7ef0411a829d. No old-folder loading path.

Compression removes the backend capture's regime-to-loop adapter: capture takes
SelfPlay and an Etude Run. Frozen MLX/MPS capture commands refuse at the same
revision boundary; their saved-data analysis survives. Artifact cache validation
has one owner (fetch_artifact); dashboard locking has no recovery mode.

Keep raw/Adam/EMA/counts, fixed schedules, world-bound immutable policies,
checkpoint_source, game-error/replay evidence, numerical incidents, paired deals.
Local start= carriage is still absent in Etude 0.4.0's local._prepare; do not
build a Manabot fallback. Gate owns broader checks; demo owns recovery tradeoffs
and automatic dependency review. Launched retry/restore and continue are unproved.
Frozen replay schema 7 versus runtime 8 remains contrary evidence, never regenerate it.
Checks and historical observations: implementation.md.
