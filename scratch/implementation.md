# MTG-329 cutover: implementation pass (2026-11-13)

## Current code

The executor, its recovery/portable-stage admission and local scheduling workers
are deleted. Local, baseline, Elemental and belief use ordinary `train(config)`
functions over the self-play loop and Etude's real run API. Elemental's existing
`runs()` calls `etude.jobs.training`; batched baseline exposes `train(config)`.
The CLI lists explicit functions without importing experiment evaluation code.

The loop validates SelfPlay directly, saves before publishing immutable
checkpoints, returns EMA when configured, and records the endpoint as `final`.
State resumes with Adam, EMA and counts; native game/RNG streams restart at the
saved update. It raises `etude.stopping.Stopped`. CLI deadlines are ISO-8601;
`--steps` sets Etude's target, preserving the recipe horizon. Belief filenames
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
  not rewritten. Post-merge executor parity was not re-proved before deleting
  the executor; earlier parity evidence is historical.

Before publishing: release EXP-56's API, raise all Etude requirements and the
training-runtime pin together, refresh uv.lock, run the isolated dependency and
installed-wheel checks, then gate the affected suites. Actual launcher courier
restore/retry and service-backed `etude continue` have not been exercised from
Manabot; local target carriage and the real function caller are covered.
Publication/landing remain outside this implement step.

## Compress validation — 2026-11-13

`uv run --no-sync --with /Users/jack/src/etude.exp-56-give-a-training pytest tests/training/test_self_play_loop.py tests/training/test_recipe_etude.py tests/training/test_attack_execution.py tests/eval/test_elemental_long.py tests/training/test_value_readout_252.py -q`: 23 passed; five evaluation checks refused an untracked source file (`identity_path_without_files`). Marked the three new production files intent-to-add so Etude's git-based identity includes them. Reran `tests/training/test_value_readout_252.py tests/eval/test_experiment_files.py`: 15 passed (11.53s). `tests/infra/test_ci_selection.py`: 53 passed (0.55s). Changed Python files pass Ruff and compileall; `git diff --check` passes. No provider launch or installed-consumer proof was performed. Deletion rationale is in plan.md; the SDK release and frozen replay mismatch remain gate items.
