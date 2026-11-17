# MTG-329 evidence

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

## GPU attempt — 2026-11-17

- Prepared `experiments/etude_execution.py` and its report before remote scored
  work: alternating main/branch Elemental runs, normal 256 streams, real save
  timings, a six-update full/stopped/resumed baseline and managed evaluation
  with actual device/batch logging. Main archive's resolved Elemental config
  validated against its historical schema; this adds no live regime reader.
- Harness preflight: actual `etude train` at seed 197, CPU, two updates,
  stop-file request after update 1: exit 75; `--resume`: exit 0. Reader returns
  updates [1, 2], final 1,012 decisions. Production managed callback on an
  existing checkpoint: 2 deals, 16/16 valid/replayed games, 645 CPU forwards,
  max batch 1. This is a harness check, not CUDA or main/branch performance.
- Vast 4090 reached SSH, source upload and native compilation, then provider
  status became offline. No scored GPU workload completed. Explicit termination
  confirmed 15:53 UTC by Etude's deleted listing and Vast API absence. Fallback
  RunPod A6000 query: no capacity. Method, machine id, exact events and the
  launcher's single retained cost estimate: experiments/reports/etude-execution.md.
- Ruff check/format and `git diff --check` pass. `lf context` keeps default
  targets and no usage entries. No server/watcher or rented machine remains.
  CUDA speed/batching and GPU resume-strength remain unmeasured; gate, nightly
  and publication remain the caller's steps.

## Compress — 2026-11-17

- `experiments/etude_execution.py`: dropped the re-validation of settings that
  are already a `SelfPlay`, a repeated `CheckpointRef` import and three
  function-local `import os`. AGENTS.md states the `__main__` rule once, in
  the existing callback paragraph. Ruff check/format pass.
  `python -m experiments.etude_execution settings` wrote 3,218 bytes;
  `etude train experiments/etude_execution.py --config
  '{"workload":"baseline","device":"cpu","updates":1}' --json`: finished, one
  timed save (0.034 s). No GPU was rented in this step.
