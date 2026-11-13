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
`f09f5c2919b8b1a5cf6d72c4f4f4721dac68f653`. This includes seat comparison,
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

## Evidence and remaining delivery dependency

- Etude PR #103 (EXP-56) was open; local source revision `105abdd` provides the
  API. Installed/PyPI SDK 0.2.16 lacks local_launch. No dependency pin or lock was
  falsely advanced and no path to another checkout was committed. Tests use
  `uv run --no-sync --with /Users/jack/src/etude.exp-56-give-a-training ...`.
- `uv run --no-sync --with /Users/jack/src/etude.exp-56-give-a-training pytest tests/training/test_self_play_loop.py -q`: 3 passed (7.06s), resume,
  unchanged earlier checkpoint bytes, refusal and saved planned stop.
- `uv run --no-sync --with /Users/jack/src/etude.exp-56-give-a-training pytest tests/training/test_recipe_etude.py -q` before adding transport/selection:
  3 passed (20.13s), CLI fresh/stop/resume/rerun/extension and belief provenance.
- Targeted source, pretrained, builder, recipe, package-boundary and CI inventory
  checks: 72 passed, two stale inventory references failed; repaired; the two
  reruns plus transport/CLI and learning-regression checks passed (8, 5.38s).
- Real SDK `etude.call` invokes LocalConfig and exits 75 with state.pt and an
  Etude stopped attempt. Source provenance reads Etude's commit in an extracted
  source tree without .git.
- New nightly recipe report proof: 1 passed (77.05s), one CPU update, two paired
  deals (four legs each), replayed evaluation, native corpus and policy inspection.
  This is software proof, not learning or strength evidence.
- Changed test paths collected: 388 tests, no collection error (before the last
  report/selection additions). No Rust rebuild or remote paid run was needed.

- Final focused checks with the same SDK override: selection replay/cache and
  world-bound deck admission, 2 passed (4.61s); numerical incident and capture
  failure, 2 passed (3.26s). Discovery without DuckDB also passed after baseline
  evaluation was wired. Ruff checked/formatted 85 changed/new Python files;
  git diff --check passed.
- Rechecked Etude PR #103 at handoff: OPEN, mergedAt null. No published pin
  was available to replace the source override.
- The earlier main-sync replay mismatch (three game-error fixture variants,
  schema 7 versus 8) remains recorded in plan.md for gate; frozen evidence was
  not rewritten. The post-merge parity proof below supersedes the earlier
  prototype-only evidence.

Before publishing: release EXP-56's API, raise all Etude requirements and the
training-runtime pin together, refresh uv.lock, run the isolated dependency and
installed-wheel checks, then gate the affected suites. Actual launcher courier
restore/retry and service-backed `etude continue` have not been exercised from
Manabot; local target carriage and the real function caller are covered.
Publication/landing remain outside this implement step.

## Compress validation — 2026-11-13

`uv run --no-sync --with /Users/jack/src/etude.exp-56-give-a-training pytest tests/training/test_self_play_loop.py tests/training/test_recipe_etude.py tests/training/test_attack_execution.py tests/eval/test_elemental_long.py tests/training/test_value_readout_252.py -q`: 23 passed; five evaluation checks refused an untracked source file (`identity_path_without_files`). Marked the three new production files intent-to-add so Etude's git-based identity includes them. Reran `tests/training/test_value_readout_252.py tests/eval/test_experiment_files.py`: 15 passed (11.53s). `tests/infra/test_ci_selection.py`: 53 passed (0.55s). Changed Python files pass Ruff and compileall; `git diff --check` passes. No provider launch or installed-consumer proof was performed. Deletion rationale is in plan.md; the SDK release and frozen replay mismatch remain gate items.

## Fresh-run parity after the backend merge — 2026-11-13

`uv run --no-sync --with /Users/jack/src/etude.exp-56-give-a-training python scratch/parity.py`: passed all four raw/EMA comparisons. Each of PPO and damped policy matched all 87 state-dict tensors bitwise, both in the final exported checkpoint and in state.pt. No production-code fix or numerical difference was needed.

Reference: unmodified git archive of main `6ce618b8adacee3a4894d05572399853680bb9fe`, including the backend merge and retired executor. Candidate: this checkout at `eb040c04f` plus the existing documentation-only self_play edits. Separate processes used Torch 2.15.0, CPython 3.13, CPU float32, one thread, the same native extension (SHA256 `fb6e3de8f371472ea88ef6a895466e2746676c52cef4c5829291acb938f310a4`). The reference archive was admitted through its hashed source manifest; no old module was restored or adapted in production. Temporary archives and outputs were removed after comparison.

Both used seed 197, gray-ogre match, hidden_dim 16, two attention heads, two streams, 16 transitions per stream as a floor, both learning seats, complete games and two updates. PPO used one epoch/minibatch and EMA 0.9; damped policy used its defaults (EMA 0.999). The reference used iteration_fraction scheduling. The candidate additionally exported checkpoints at updates 0 and 1; final raw/EMA at update 2 still matched. Each completed four games and 357 learner/environment decisions, with 23 idle stream steps. PPO counted two optimizer updates/179 exposures; damped policy 61/90.

The initial harness invocation failed before learning because EXP-56 removed track(resumed_at=...). The unchanged reference therefore used released SDK 0.2.16 from the root environment; the candidate used real EXP-56 source `105abdd323016b7f7baaaf57aaaa7f5d896c71bf` (whose metadata still says 0.2.16). The corrected invocation above passed. This is an execution extraction comparison, not an installed-consumer proof.

Intentional differences remain in persistence and tracking: update-named immutable policies and state.pt replace stage exports/recovery snapshots; checkpoint metadata and container bytes differ, while model tensors match. Native/RNG streams restart on resume; this fresh-run proof does not restore exact interrupted-run equivalence. Frozen staged/adaptive/frozen-opponent execution still refuses. No GPU, provider retry or service continuation claim follows from this CPU fixture.

## Compress follow-through — 2026-11-13

Removed unused scheduler-only `MonitoringBudget.protocols_for`, `Hardware.admit`
and `HardwareInventory.select`; retained their schemas/validators and offline
readers. README now describes separate baseline evaluation and the actual update
clock. Historical stub/audit notes now point to their exact git versions.

`uv run --no-sync --with /Users/jack/src/etude.exp-56-give-a-training --with nbclient --with nbformat --with matplotlib --with ipykernel pytest tests/training/test_experiments.py tests/training/test_experiment_report.py tests/eval/test_experiment_files.py -q`: 34 passed, seven report failures solely for missing nbconvert. Same invocation with `--with nbconvert pytest tests/training/test_experiment_report.py --lf -q`: seven passed (17.25s). Earlier collection attempts lacked nbclient, then matplotlib; temporary uv overlays supplied dependencies without changing the project environment. Ruff, compileall and diff whitespace checks passed. No learner code changed, so the recorded post-merge parity proof was not rerun.

## Concept-review implementation — 2026-11-13

Moved retained experiment records and live learning types to their defining
modules, preserving schemas. Resume diagnostics now retain learner settings
and report nested field differences; machine-only changes remain resumable.
CLI --updates carries the target, --steps warns with removal 0.3.0, and both
flags together refuse. Current docs use --updates and learning.py.

With the same real SDK source override, focused self-play refusal, CLI
stop/resume/extension/alias, regime and package-boundary checks: 25 passed,
one stale boundary import expectation failed. Updated that expectation.
Then boundary rerun, config-without-updates, experiment records, CI selection
and release checks: 114 passed (2.11s). Ruff and diff whitespace checks pass.
No learner calculation changed; parity was not rerun. Installed-wheel proof,
SDK release/pin, recovery review and the retained replay mismatch remain gate
or delivery obligations. No service or background process was started.

## Compress after concept implementation — 2026-11-13

Removed the unused selection deadline callback and no-op adapter; centralized
learner-setting serialization; clarified loop helper names and recipe clock
labels. Required resume, selection and failure-capture behavior is unchanged.

`uv run --no-sync --with /Users/jack/src/etude.exp-56-give-a-training pytest tests/training/test_self_play_loop.py tests/training/test_recipe_etude.py tests/training/test_automatic_game_capture.py::test_semantic_training_retains_rejected_command_and_offer_frame -q`: 12 passed (20.72s). Changed production files pass Ruff and compileall; `git diff --check` passes. `lf context` is within both scratch limits. No background service or provider launch was started; delivery obligations above remain.

## Fixed PPO horizon — 2026-11-13

PPO now reads a recipe-owned `Learning.horizon`, validated by SelfPlay,
serialized in learner settings and named in resume refusals. Belief supplies
its own update count; the builder requires it. Unset horizons remain absent
from retained learning records. Removed target arguments from step/learn and
the unusable implicit SelfPlay learning default. Docs already describe this
contract. The parity harness now imports historical types from their original
module in the reference archive and supplies horizon=2 only to the candidate.

With `uv run --no-sync --with /Users/jack/src/etude.exp-56-give-a-training`:
- `pytest tests/training/test_self_play_loop.py tests/training/test_recipe_etude.py tests/training/test_pretrained.py -q`: 13 passed (24.65s), including raw/EMA first-update equality across launch targets, horizon refusal, planned stop/resume, immutable checkpoints and belief extension.
- `python scratch/parity.py`: PPO and damped-policy raw/EMA each match all 87 tensors bitwise against main `6ce618b8a`, in policies and state.pt; same native SHA256 and counts as the earlier proof.
- `pytest tests/training/test_builders.py::test_current_regime_files_round_trip_to_their_bytes tests/training/test_builders.py::test_checkpoint_builders_chain_stages_with_independent_settings tests/training/test_regimes.py::test_frozen_regimes_read_their_pack_selectors_as_world -q`: 3 passed (1.86s).
Changed Python files pass Ruff format/check; git diff --check passes. No
service, watcher, provider launch, dependency pin or publication was started.
Release/installed-consumer and launcher/service proofs, full gate and the
retained replay mismatch remain delivery obligations; ready for demo.

## Restored monitoring and benchmark — 2026-11-13

Review 4 is implemented. Collector/learner profiler deltas populate historical
phase keys; update rows sample RSS/load, CUDA process peaks and current MPS
allocation. Counts persists active completed-update seconds; all current
recipes declare the Time clock, including batched baseline. Belief sampler
checkpoints inherit the policy's complete clock coordinate. The benchmark runs
SelfPlay builders untracked and writes rows/summary, replacing the retired
regime/TrainingRun contract. Frozen throughput launch/drive/phase entry points
refuse; their exclusive execution tests are removed. Frozen refusal has one
owner in regime.py. Guides, panels and workflow skills describe these contracts.

With `uv run --no-sync --with /Users/jack/src/etude.exp-56-give-a-training`:
- `pytest tests/training/test_self_play_loop.py tests/training/test_recipe_etude.py tests/training/test_phase_timing.py tests/training/test_training_throughput.py::test_frozen_throughput_refuses_before_creating_output tests/training/test_game_errors.py -q`: 49 passed, three previously recorded retained replay binding_mismatch failures (45.68s). A missing belief checkpoint clock warning was fixed afterward.
- `pytest tests/training/test_recipe_etude.py::test_belief_extension_has_new_policy_histories_and_sampler tests/training/test_attack_execution.py tests/test_package_boundaries.py tests/training/test_report_helpers.py -q`: 22 passed, no warning (22.60s); sampler recording is now asserted.
- `python scratch/parity.py`: PPO/damped raw/EMA each match all 87 tensors bitwise in checkpoints and state.pt against main `6ce618b8a`. Both processes use the rebuilt native extension SHA256 `a7de0ce098207e8a2a3ac16b59bda44e0ab3abafc729c434b1121aeb6dba56e7`. Counts: four games, 357 decisions/transitions, 23 idle steps; PPO optimizer updates/exposures 2/179, damped 60/90. This supersedes the earlier native-binary proof for this pass.
- Actual benchmark CLI default Elemental builder, two streams/one update: JSON phase split and retained state/policies verified; temporary outputs removed.

Initial recipe checks hit a stale local extension missing pooled_mana_classes.
Rebuilt with the root maturin command and atomically installed the matching
cp313 wheel extension; both recipe failures disappeared. Frozen replay bytes
were unchanged. Workflow compilation passed; changed Python Ruff checks and
formatting and git diff whitespace checks pass. No provider launch, release,
pin, installed-consumer proof or service continuation was performed. No
background service or watcher was started. Delivery obligations in plan.md
remain.

## Compress after review 4 — 2026-11-13

`step` reads phase seconds and active time from the profilers only (the
overriding perf_counter pair and `import time` are gone); `counted` takes the
update's seconds. The benchmark summary reads `training_seconds`. The CLI
names `FROZEN_EXECUTION_REVISION` directly instead of catching the refusal;
the throughput driver's internal `run_phase` refusal is dropped (its callers
refuse). The clock rule moved to AGENTS.md's Etude training records section;
workflows no longer promises call counts from phase rows or host facts from
new runs.

With the same SDK override: self-play loop, phase timing, recipe/Etude,
frozen throughput refusal, CLI retired-override and attack refusal tests: 22
passed (48.45s). `python scratch/parity.py`: PPO/damped raw/EMA match all 87
tensors bitwise against main `6ce618b8a`, native SHA256 `a7de0ce0…`, same
counts. Ruff check/format and `git diff --check` pass.
