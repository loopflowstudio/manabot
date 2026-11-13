# MTG-329 cutover: implementation pass (2026-11-13)

## Current code

The executor, its recovery/portable-stage admission and local scheduling workers
are deleted. Local, baseline, Elemental and belief use ordinary `train(config)`
functions over the self-play loop and Etude's real run API. Elemental's existing
`runs()` calls `etude.jobs.training`; batched baseline exposes `train(config)`.
The CLI lists explicit functions without importing experiment evaluation code.

The loop validates SelfPlay directly, saves before publishing immutable
checkpoints and returns EMA when configured. Etude derives the finished run's
final checkpoint from its last recorded checkpoint.
State resumes with Adam, EMA and counts; native game/RNG streams restart at the
saved update. It raises `etude.stopping.Stopped`. CLI deadlines are ISO-8601;
`--updates` sets Etude's target, preserving the recipe horizon. Belief filenames
include the policy checkpoint stem. Selection collection retains and verifies
replay journals and checks cached policy/population and collection-input identity.
Numerical failures retain bounded tensors, Adam/EMA and source diagnostics in
unique private incident directories, without replacing the last valid state.
Capture errors are attached to the original numerical error.

The nightly baseline was an additional live consumer. It now calls explicit
`train` then `evaluate`, preserving raw initial/final paired deals, native choice
inspection, exposure totals, learning.json and the training-seed aggregate. The
old and new report input paths share the same paired calculation, inspection and
renderer. MLX pilot capture instruments the surviving loop.

Frozen regime launch/continuation functions refuse with revision
`f4e1822faf46bbc9cdc43909293f8cbc8de4a300`. This includes seat comparison,
TLA frontier/vocabulary, value-readout segments, old Elemental continuations,
learning-signal training and the old CUDA/profile capture drivers. Their data,
evaluation and report functions remain. They were not translated to the new
loop: adaptive/frozen-opponent/staged semantics are not new loop capabilities.

## Delete — do not maintain

Deleted: training/execution.py; training/state/{recovery,learning_state,admission}.py;
experiment/{job,runner,evaluation_queue,monitor_worker,study}.py;
recipes/proposed_etude.py; RunContext; TrainingRecipe; SelfPlay.to_regime/save_final.
Exclusive scheduler/recovery tests and fixtures went with those paths. Mixed
model, collector, record and evaluation tests retain their non-executor checks;
shared policy fixtures no longer import recovery tests.

Retained: TrainingRegime/TrainingRun and comparison snapshots; checkpoint loading,
world binding, original serialized keys, game-error ledgers/failure evidence,
paired deals and training seeds. Monitoring values and historical checkpoint
projections moved into existing record/evaluation modules; no scheduler remains.
Rust/uv machine setup remains Manabot-specific. Historical regime tags remain.

## Earlier focused validation

Full initial cutover and compress transcripts remain at
`5da3574799fb39c8525c2e937beff50583be6741:scratch/implementation.md`.
Focused loop, recipe, transport, selection/cache, failure capture, report,
CI-selection and boundary checks passed with the real EXP-56 SDK source
override. A stale local native extension was rebuilt later (see below).
The post-merge parity evidence follows.

SDK release/pin, isolated installed consumers, launcher output restore/retry,
service continuation and recovery review remain delivery obligations in
plan.md. Three retained game-error replay fixtures still fail schema binding
(7 versus 8); their frozen bytes were not rewritten. No paid launch ran.

## Fresh-run parity after the backend merge — 2026-11-13

`uv run --no-sync --with /Users/jack/src/etude.exp-56-give-a-training python scratch/parity.py`: passed all four raw/EMA comparisons. Each of PPO and damped policy matched all 87 state-dict tensors bitwise, both in the final exported checkpoint and in state.pt. No production-code fix or numerical difference was needed.

Reference: unmodified git archive of main `6ce618b8adacee3a4894d05572399853680bb9fe`, including the backend merge and retired executor. Candidate: this checkout at `eb040c04f` plus the existing documentation-only self_play edits. Separate processes used Torch 2.15.0, CPython 3.13, CPU float32, one thread, the same native extension (SHA256 `fb6e3de8f371472ea88ef6a895466e2746676c52cef4c5829291acb938f310a4`). The reference archive was admitted through its hashed source manifest; no old module was restored or adapted in production. Temporary archives and outputs were removed after comparison.

Both used seed 197, gray-ogre match, hidden_dim 16, two attention heads, two streams, 16 transitions per stream as a floor, both learning seats, complete games and two updates. PPO used one epoch/minibatch and EMA 0.9; damped policy used its defaults (EMA 0.999). The reference used iteration_fraction scheduling. The candidate additionally exported checkpoints at updates 0 and 1; final raw/EMA at update 2 still matched. Each completed four games and 357 learner/environment decisions, with 23 idle stream steps. PPO counted two optimizer updates/179 exposures; damped policy 61/90.

The initial harness invocation failed before learning because EXP-56 removed track(resumed_at=...). The unchanged reference therefore used released SDK 0.2.16 from the root environment; the candidate used real EXP-56 source `105abdd323016b7f7baaaf57aaaa7f5d896c71bf` (whose metadata still says 0.2.16). The corrected invocation above passed. This is an execution extraction comparison, not an installed-consumer proof.

Intentional differences remain in persistence and tracking: update-named immutable policies and state.pt replace stage exports/recovery snapshots; checkpoint metadata and container bytes differ, while model tensors match. Native/RNG streams restart on resume; this fresh-run proof does not restore exact interrupted-run equivalence. Frozen staged/adaptive/frozen-opponent execution still refuses. No GPU, provider retry or service continuation claim follows from this CPU fixture.

## Earlier review validation

Full concept-review 1–5 and compress transcripts remain at
`82991cf850976b388cb05ba80d189c1c7f2560a5:scratch/implementation.md`.
They include focused checks for typed configs, PPO horizon, monitoring rows,
benchmark migration, retained records, boundary/CI selection and frozen
refusals. Their code decisions remain in plan.md. The latest parity and
milestone-recovery proof follows; release, installed-consumer, launcher/service
and retained replay obligations remain open.

## Concept review 9 recovery repair — 2026-11-13

Entry repairs any saved grid milestone before collector creation. Tracked
recovery covers loss before policy publication and after recording, checking
one record at the saved updates/time coordinate and unchanged earlier bytes.
The guide and surviving conditional-search import guard match the change.

With `uv run --no-sync --with /Users/jack/src/etude.exp-56-give-a-training`:
- `pytest tests/training/test_recipe_etude.py::test_resume_repairs_a_saved_milestone_before_training_on -q`: 2 passed (2.86s). Initial fixture run omitted required clock meanings; corrected.
- `pytest tests/training/test_self_play_loop.py::test_a_rerun_resumes_and_a_finished_loop_returns_at_once tests/training/test_self_play_loop.py::test_numerical_failure_retains_diagnostics_without_replacing_valid_state tests/sim/test_conditional_search.py::test_import_does_not_reach_training_only_dependencies -q`: 4 passed in the initial combined invocation (4.87s including the two clock-fixture failures).
- `python scratch/parity.py`: PPO/damped raw/EMA each match all 87 tensors bitwise in policy checkpoints and state.pt against main `6ce618b8a`, native SHA256 `a7de0ce0…`. Counts remain four games, 357 decisions/transitions, 23 idle steps; optimizer updates/exposures PPO 2/179, damped 60/90. Moving initial publication ahead of collector creation changes no fresh-run weights.

Compress: guide and comment wording only; same override,
`pytest tests/training/test_recipe_etude.py tests/training/test_self_play_loop.py tests/sim/test_conditional_search.py::test_import_does_not_reach_training_only_dependencies -q`: 17 passed (23.07s).
Ruff check/format and `git diff --check` pass. No provider launch, dependency pin, installed-consumer proof or
background service was started. Remaining delivery obligations are unchanged.

## Concept review 10 CLI naming — 2026-11-13

`uv run --no-sync --with /Users/jack/src/etude.exp-56-give-a-training pytest tests/training/test_recipe_etude.py::test_cli_function_recipes_track_under_the_module_in_distinct_directories -q`: failed before the fix with EtudeError unsafe_run_or_experiment_name; passed after it (0.70s). Two explicit functions record finished runs under the module name in separate default directories. Ruff and git diff --check pass. No training, provider launch or background service ran. Installed-wheel smoke remains gated on SDK release/pin; the other delivery obligations remain.
Compress: nothing to reduce in this one-line fix, its test or the guide sentence; no edits, no rerun.

## Concept review 11 recovery guidance — 2026-11-13

Both recovery sections now scope fixed source revisions to launched runs and
describe local resume after a code change. Checkpoint revision/run identity
and Etude attempt/resume records are distinct in the recovery and counters
text. Checked against checkpoint_source, load_state and EXP-56's Attempt and
first-row resumed_at handling. `git diff --check` passed; `lf context` is within
both limits after consolidating reviews 6–7 with a git reference. Prose only;
no build, test or parity rerun. Later delivery obligations remain unchanged.
Compress: the game-error section's repeated source-repair paragraph now links to the one statement, keeping only that a local rerun does not clear a threshold stop. Claims rechecked against `load_state` (world, seed, learner digest; `source_commit` saved, not compared) and `checkpoint_source`. `git diff --check` passed; prose only, no rerun.

## QA follow-through — 2026-11-13

CLI interruption guidance and guide corrections are implemented (plan.md).
`uv run --no-sync --with /Users/jack/src/etude.exp-56-give-a-training pytest tests/training/test_recipe_etude.py::test_cli_interrupt_names_the_saved_output_and_a_rerun_resumes -q`: 1 passed (8.93s).
The test injects KeyboardInterrupt during learning, checks exit 130 and the
resume message, verifies earlier state/policies byte-for-byte, then resumes
with real Etude tracking to update 2. No learner calculation changed.

The same override's `read_job_file` accepted the guide's Elemental arguments
(`--revision HEAD --updates 3200 --hours 11.9 --usd-per-run 9`, with HEAD
resolved first). This constructs jobs only; no smoke environment or launch ran.
Ruff check/format on both changed Python files and `git diff --check` pass.

Boundary probe: two independent temporary Etude runs log one update and finish;
a second track block raises ValueError or KeyboardInterrupt without logging.
Each changes finished to failed, retaining reached updates=1. Source inspection
at EXP-56 `105abdd` identifies Run.__init__'s eager attempt record and
Run.__exit__/finish's unconditional failure state write. `Attempt.first_row`
is initialized True, so it is not evidence that an attempt logged a row;
the probe made no log calls on either second attempt. No SDK fix is claimed.
PR #103 remains open, committed requirements/lock remain 0.2.16, and released
consumer/launcher/retry/service obligations remain pending. No server, watcher
or paid run was started. Earlier review transcripts were consolidated to their
git reference.
Compress: nothing to reduce in the interrupt handler, its test or the guide
corrections; no code edits, no rerun. `git diff --check` passed.
