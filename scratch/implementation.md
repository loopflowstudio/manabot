# MTG-329 evidence

Earlier transcripts and contrary findings: 382c653ec8:scratch/implementation.md.
The earlier state.pt parity proof is not proof of the revised save API.

2026-11-14: uv lock and locked sync install Etude 5708e14 from the development
source override. Focused loop/minimal/recipe/report/SDK/source/CLI run: 36 passed,
2 failed (34.35s): guide/source drift and eager simulation import defeating
SDK-free migration guidance. Both fixed. Follow-up guide/CLI/recipe/experiment
discovery: 11 passed (5.48s). Existing Etude compatibility deprecations remain.

Pinned API imports resolve throughout non-fixture Python sources after following
removed modules and RunRequest. Report admission selects the producer schema
before validation; malformed untagged retained runs no longer enter via the
permissive Etude RunRecord. Frozen evidence was not rewritten.

- `uv run --no-sync python scratch/parity.py`: PPO and damped-policy raw/EMA,
  87 tensors each, bitwise equal to retired executor 6ce618b8adacee3a4894d05572399853680bb9fe,
  including the new saved learner/EMA. Seed 197, two updates/two streams,
  16-transition floor, four games/357 decisions. PPO: 2 optimizer updates/179
  exposures; damped: 52/79. Reference used real etude-sdk 0.2.16 in a temporary
  target; current used pinned 5708e14. Native SHA256:
  ee798c176871b6a5cb6af0fba49a50a8b4e18104fabbafe9fd74fe4794f567e5.
- Controlled real SIGINT and SIGTERM at the first completed update, through
  `etude train`: each exited 75 with saved clocks/counts at 1; `--resume`
  finished at 3, rows [1,2,3], two writers. No timing or load injection.
- Real `etude train belief_policy.py`, then `belief_sampler.py` under a fresh
  ETUDE_HOME: policy succeeded; sampler failed `input not given: 'start'`.
  Local `_prepare` omits job inputs/environment. No workaround or SDK edit.
  `runs()` now declares a completed checkpoint, consistent with landed start=;
  its input digest/owner and recipe given-input provenance are covered, but
  automatic chaining and actual local input carriage remain unmet.
- Empty `/tmp/mtg329-deps` locked dev+artifacts environment: artifact storage,
  SDK, source and recipe files, 14 passed (15.92s). Used uv sync excluding
  managym install and uv run --no-sync; native came from the existing checkout,
  so this isolates Python extras, not the engine build.
- Report admission (new and historical Etude), release, CI selection and
  Torch-free evaluation-child checks: 94 passed (2.18s).
- Installed smoke outside the checkout: CPython 3.13.15 and 3.12.12 passed,
  pure Manabot wheel rebuilt with release/prepare.py; existing paired Managym
  0.2.3 cp313/cp312 macOS wheels from /tmp/mtg352-release-wheels; Torch 2.15.0,
  real pinned Etude. No Rust rebuild in this pass. Release smoke initially
  exposed an overbroad command replacement (`uv train`); corrected to uv run
  with etude train before both passing smokes.
- Tiny benchmark consumer proof: seed 197, one Gray Ogre update, real local
  Etude rows, two games. Generated typed runs() declares the seed. Nightly
  caller follows that declaration/address shape; its full workload was not run.
- Etude import inventory resolves; changed guide examples/YAML/Python heredocs
  parse; minimal guide matches source; Ruff check/format, workflow validator
  and git diff --check pass.

No paid/provider/service run, launch retry/restore, continue proof, publication
or landing. Gate's broader suites remain for gate. The known frozen replay
schema 7 versus current 8 mismatch remains unmodified.

2026-11-14 compress: PyPI 0.4.0 resolved after `uv lock --refresh-package
etude-sdk --upgrade-package etude-sdk` (the first cached lookup saw only 0.3.2).
Locked sync passed. Removed the regime capture adapter and frozen MLX/MPS
capture bodies; current capture takes SelfPlay + Run. Focused capture/artifact/
outcome/release/recipe/loop check: 55 passed, 4 failed (97.61s). All four were
outcome tests using removed SDK APIs. Their cutover exposed first-row
supersession: new writers replace the boundary; legacy checkpoint cuts keep it.
Updated reader/tests: `uv run --no-sync pytest tests/training/test_outcome_tables.py
-q`: 5 passed (3.17s). Capture replay reproduced every minibatch loss.
0.4.0 local._prepare still has no job-input delivery. No launched proof attempted.
Report/discovery/CI follow-up: `uv run --no-sync pytest
tests/training/test_learning_dashboard.py tests/training/test_experiment_report.py
tests/infra/test_ci_selection.py tests/eval/test_experiment_files.py -q`: 82 passed
(81.19s); compatibility deprecations remain. Dashboard/simulation check: 7 passed,
2 failed (22.06s), both stale checkpoint-layout assumptions in test_tracking;
cut tests over to the public checkpoint reader. Publisher retries now explicitly
resume and put summaries after settled rows so first-row cuts keep the prefix.
`uv run --no-sync pytest tests/sim/test_tracking.py -q`: 4 passed (12.09s).
Ruff check/format (11 changed Python files), `uv lock --check` and
`git diff --check` pass. No remaining test process or server from this pass.
