# MTG-329 evidence

## Retained report fixes — 2026-11-17

- Local smokes now log explicit text kind/recipe labels alongside existing
  updates. The report reads those columns, preferring declaration fields.
  Config bytes and resume equality are unchanged; old unlabeled runs remain
  unclassified. First focused check exposed that Etude `RunData.points` reads
  only numeric metrics; the reader now decodes only the matching text column.
- Empty attempts no longer erase earlier observed work and duration. They
  stay in membership with a warning that their time is unmeasured. Fixtures
  cover silence before, between and after observed attempts, a wholly silent
  run, and the actual loop's saved-state inspection attempt.
- `uv run --extra dev --extra artifacts pytest tests/training/test_time_cohort.py tests/training/test_recipe_etude.py tests/training/test_self_play_loop.py::test_resume_keeps_adam_ema_counts_and_immutable_policies -q`:
  14 passed (9.08s). The local recipe test uses renamed experiment/run names,
  checks generated report selection through its actual JavaScript, and proves
  a smoke can still be explicitly included. Declaration kind/job precedence
  is covered separately. Ruff check/format and `git diff --check` pass.
- Fresh isolated `ETUDE_HOME`, actual `uv run etude train
  manabot/recipes/local.py --config '{"updates":1}' --json` then
  `uv run --extra reports manabot training-report --source <home>/track --out
  <home>/report.html`: both exit 0; generated JSON names
  `kind=smoke`, `recipe=manabot.recipes.local`, `updates=1`. No servers,
  watchers or rentals started. GPU measurements remain unrun for gate/CI.

## Earlier evidence (full text: ce688b24d3:scratch/implementation.md)

- 2026-11-16 evaluation errors: injected native AgentError/PyO3-shaped panic
  retain score-null games and continue the group through Etude's real game
  process; Python errors propagate from play and replay; all-invalid Elemental
  callbacks fail with retained rows/journals (24 passed). No real Rust panic.
- 2026-11-16 local recipe: `uv run etude train manabot/recipes/local.py --json`
  on a fresh ETUDE_HOME finished; final EMA policy
  `track/local/out/policies/00000002-ema.safetensors`. QA fourth pass
  (scratch/qa.html): evaluation, training and workflows guides run as written.
- 2026-11-17 L4: RunPod $0.69/hour, Vast and Lium, none available at
  14:45-14:50 UTC; nothing rented. The shared `~/.etude/launcher.sqlite` needs
  `provider_refusal_lookup` and was left unchanged; the isolated launcher is
  `/tmp/mtg329-performance-etude` (Doppler etude/prd + manabot/prd).
- 2026-11-17 regime-reader deletion: 21 affected test files, 187 passed, 3
  skipped; report importers 119 passed; tla_frontier w64 and w160l8 settings
  dumps equal the record-derived ones. Whole-tree collection has one error
  that is on main too: tests/test_policy_container_support.py imports
  `manabot.policy.checkpoint.require_supported` (it lives in
  `manabot.backends.mlx.checkpoint`); not in the CI lists, not changed.
  `scripts/check_loopflow_workflows.py` asserts in `_coverage` (graph step
  ids) with no `.lf` or script change here. `test_cuda_semantic_policy_admission`
  was validated on CPU only.

## Learning comparison method — 2026-11-17, before runs

Exploratory question: does moving execution change uninterrupted learning?
Compare origin/main 3afd92d8f3 with this branch, fresh local baseline runs,
CPU float32, seed 197, three updates, two streams and 16 transitions (complete
actual games, both seats), identical horizon-3 damped-policy schedule and
width-64/depth-2 FiLM model. Compare every raw and EMA tensor, including the
initial policy if available. A bitwise match settles only this short
uninterrupted case; real divergence triggers the planned strength comparison.
The main source is a temporary git archive, not another worktree. Use the same
compiled native extension (engine implementation unchanged), Torch and host;
main keeps its pinned Etude SDK. No runtime/RNG recovery equivalence is claimed.
CPU cost: no rental. Combined remote measurement cap remains USD 10; no remote
measurement has run. Resume strength and CUDA speed/admission remain separate.

## Report cleanup and CPU comparison — 2026-11-17

- Deleted the unreachable flamegraph, phase table and per-run synthetic profile
  hierarchy, plus elapsed/phase/cost payload fields. Removed notebook extra,
  nbformat, matplotlib and their now-exclusive lock dependencies. Updated both
  integration workflows, release install, development guide and AGENTS/CLAUDE
  command together. Historical install commands in the saved-regime guide
  still name the historical revision's notebook extra.
- `uv run --locked --extra dev --extra artifacts pytest tests/training/test_time_cohort.py -q`:
  8 passed (0.26s), after deleting one assertion on the removed phase fields.
  `uv run --isolated --locked --extra dev --extra artifacts --extra reports --extra evaluation pytest tests/training/test_time_cohort.py -q`:
  8 passed (0.81s), with the reduced dependency graph. Ruff check and format
  check pass for the three affected Python files.
- CPU method above completed against main 3afd92d8f3 and branch 9421c945d9,
  CPython 3.13.15, Torch 2.15.0, NumPy 2.4.2, native world w17. All 117
  initial raw tensors match bitwise. After three updates, all 117 raw and
  all 117 EMA tensors match bitwise (zero unequal; maximum absolute error 0).
  Both sides: 6 complete games, 1,421 environment decisions/learner transitions,
  268 optimizer updates. Branch cumulative optimizer updates: 89, 148, 268.
  Step 2 is unnecessary for this short uninterrupted case under the approved
  method; no broader strength or resume equivalence is implied.
- Main used its unchanged `local_smoke(steps=3, seconds=180, device='cpu')`
  through `execute_regime`; branch used `LocalConfig(updates=3)` through the
  real `etude train` CLI, declared seed 197. Main's SDK was its d60236d pin;
  the archived source was verified with its supported `.manabot-source.json`
  bundle. The shared native implementation has no main/branch diff. Outputs
  and comparison JSON remain in `/tmp/mtg329-learning`, CLI scripts/logs in
  `/tmp/mtg329-learning-{main,branch}*`; these temporary paths are not archival
  evidence. The numbers and method above are the retained result.
- Broader GPU capacity check, 2026-11-17 15:25:25 UTC, isolated
  `ETUDE_HOME=/tmp/mtg329-performance-etude`, Doppler etude/prd + manabot/prd:
  `uv run --extra artifacts etude availability mtg329-speed-4090{,-vast,-lium} --timeout 30`
  (three separate commands). RunPod NVIDIA GeForce RTX 4090: $0.89/hour,
  none available. Vast RTX 4090: available, 24 vCPUs/64 GB,
  $0.42702738444444444/hour. Lium: `reply is missing valid required fields`;
  capacity is unknown, not absent. Each template has a $1/hour ceiling.
  No rent/launch was issued; no paid cost or machine to terminate. The
  same-GPU speed/CUDA/resume workload remains to be prepared and run under
  the total USD 10 cap; this capacity check is not a failed GPU measurement.
- Required isolated dependency integration check:
  `uv run --isolated --locked --extra dev --extra artifacts pytest tests/training/test_artifact_storage.py tests/training/test_etude_sdk.py tests/training/test_etude_source.py tests/training/test_recipe_etude.py -q`:
  12 passed (15.73s). `node --check manabot/reports/time_cohort.js` and
  `git diff --check` pass. `lf context` retains default targets (16k tokens,
  131072 bytes), reports no usage entries; protected scratch notes untouched.
  Final isolated `etude machines` lists only owned local; no watcher/server
  was started. Gate, publication, CUDA/resume and nightly remain unrun.

## GPU attempts — 2026-11-17

Method, machine ids, exact events and launcher estimates:
experiments/reports/etude-execution.md. Summary and blocker: plan.md, Remaining.

- Harness preflight: actual `etude train` at seed 197, CPU, two updates,
  stop-file request after update 1: exit 75; `--resume`: exit 0. Reader returns
  updates [1, 2], final 1,012 decisions. Production managed callback on an
  existing checkpoint: 2 deals, 16/16 valid/replayed games, 645 CPU forwards,
  max batch 1. A harness check, not CUDA or main/branch performance. The main
  archive's resolved Elemental config validated against its historical schema.
- After compress: `python -m experiments.etude_execution settings` wrote 3,218
  bytes; `etude train experiments/etude_execution.py --config
  '{"workload":"baseline","device":"cpu","updates":1}' --json`: finished, one
  timed save (0.034 s).
- Vast 4090: SSH, upload and native compilation, then offline; no scored
  workload; termination confirmed 15:53 UTC. Lium 4090 (16:01 UTC): created,
  unreadable by the pinned SDK, deleted directly and confirmed absent by
  16:03 UTC; `etude terminate` refused (`Lium create already attempted;
  awaiting reconciliation`). No server, watcher or rented machine remains.

## Released SDK and capacity retry — 2026-11-17

- Replaced Etude's git source with PyPI 0.4.1 in requirements, lock, consumer
  release installs, metadata validation and docs. Release staging still named
  deleted checkpoint_evaluation.py; its assertion now checks baseline.py.
- `uv run --isolated --locked --extra dev --extra artifacts pytest tests/training/test_artifact_storage.py tests/training/test_etude_sdk.py tests/training/test_etude_source.py tests/training/test_recipe_etude.py tests/infra/test_release.py -q`:
  50 passed, one stale deleted-module assertion failed (25.90s). After that
  assertion's fix, isolated `pytest tests/infra/test_release.py -q`: 39 passed
  (0.28s). All 12 dependency integration checks passed in the first command.
  Ruff check/format for the changed Python files and `git diff --check` pass.
- Fresh launcher `/tmp/mtg329-runpod-vast-etude`, released 0.4.1: all eight
  RunPod/Vast templates report no capacity at 16:09 UTC. No rental, cost or
  GPU result. Final inventory: local only, zero sessions/rentals/runs; no
  server/watcher started. Full method and capacity table: experiment report.
  Retry still needs rentable RunPod/Vast capacity under USD 9. Gate and
  publication remain later steps.

## Optional report labels — 2026-11-17

`uv run --extra dev --extra artifacts pytest tests/training/test_time_cohort.py -q -k 'baseline_recipe_without_labels or local_recipe_is_a_named_smoke'`:
2 passed, 9 deselected (7.11s). Real baseline training writes neither label
column and reads as unclassified; local's explicit smoke labels still exclude
it by default. GPU speed and resume measurements were not run.
