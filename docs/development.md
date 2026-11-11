# Development

How to develop in this repository: commands, required checks, CI, releases,
the clean-machine play proof, and where retained evidence lives. Conventions
are in [AGENTS.md](../AGENTS.md); test-resource rules in [TESTING.md](../TESTING.md).

- [Environment and commands](#environment-and-commands)
- [Validation by change area](#validation-by-change-area)
- [CI](#ci)
- [Training and evaluation speed](#training-and-evaluation-speed)
- [Releases](#releases)
- [Clean-machine play proof](#clean-machine-play-proof)
- [Retained evidence and archived corpora](#retained-evidence-and-archived-corpora)

## Environment and commands

From the repository root:

```bash
uv sync --locked --python 3.13 --extra dev --extra artifacts
uv run manabot --help
```

This prepares the development environment and displays the CLI commands. The
root project is both that environment and the single `manabot` distribution.
Development uses ordinary CPython 3.13; consumer compatibility includes 3.12.
Keep shared Python source compatible with 3.12. Free-threaded builds and newer
interpreters are outside the supported contract.

Every Python command runs through uv (`uv run ...`). `uv sync` builds the bundled extension; CI and existing scripts
also use `uv run --extra play maturin develop --manifest-path managym/Cargo.toml`
(add `--features python`, and `--release` for timing-sensitive use). After
changing Rust, rebuild as described in AGENTS.md and replace the `.so`
atomically.

- `./scripts/play` installs the locked play runtime and starts backend and
  frontend; it is the only clean-machine-certified play command.
  `uv run scripts/play.py` works in an already prepared environment but cannot
  diagnose a missing `uv`. Development flags: `--port`, `--frontend-port`,
  `--no-frontend`, `--demo learn`.
- `uv run manabot train --out .runs/<name>` runs a training smoke.

Public dependency requirements are `etude-sdk>=0.2.15` (also for its `eval` and
`s3` extras) and `torch>=2.10.0`. All extras, including `llm`, inherit the core
requirements. A downstream project resolves those with its own dependencies. The repository's
`uv.lock` records exact versions, and its dedicated training and play runtime
groups retain exact Torch/backend choices. Those development pins are not
constraints exported to consumers.

To update the locked Etude version without raising the public minimum:

```bash
uv lock --upgrade-package etude-sdk
uv sync --locked --python 3.13 --extra dev --extra artifacts
```

Commit the lock change before preparing a new experiment. When raising a public
minimum, update every requirement for that dependency and commit the metadata
with the lock.
Training setup uses `uv sync --locked`; it never resolves a fresh branch head.
Retained experiments keep their original source and lock, including the v7
representation study's historical Git source pin. Never downgrade the current
launcher to reproduce a historical experiment; use its isolated frozen source.

## Validation by change area

Polish and validate only the current branch scope. **NEVER intentionally induce
load for testing**; follow [TESTING.md](../TESTING.md) for resources and
subprocesses, including benchmarks, debugging and demos.

**Rust managym workspace** (`managym/Cargo.toml`, `managym/src/**`,
`managym/state/**`, `managym/cards/**`):

```bash
cd managym
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo nextest run   # or cargo test
```

The dev profile builds at opt-level 1, keeping debug assertions and overflow
checks while running the suite about 3x faster; release-only passes are not
enough. `cargo nextest run` runs every binary's tests in parallel, where
`cargo test` runs binaries one after another; `--profile ci` adds CI's 5-minute
per-test limit (`managym/.config/nextest.toml`). Use `-j 2` and
`--test-threads 2` on shared hosts, and select a binary while iterating
(`cargo nextest run -E 'binary(rules_tests)'`). Keep one Cargo operation per
build directory. `--all-features` builds the PyO3 bindings; if the default `python3`
is newer than PyO3 supports, set `PYO3_PYTHON=../.venv/bin/python`. A failure of
`cargo test --all-features` for missing Python dev/link symbols is environment
setup, not engine correctness. Record unavailable tooling in the gate summary.

**Authored rules, worlds and setups.** Run the complete debug Rust suite,
including semantic-kernel conformance, and focused browser interactions. Also:

- `uv run --extra dev pytest tests/semantic/test_elemental_world.py -q`
  for Elemental roster or training-admission changes; its PPO smoke reloads the
  final checkpoint written by the recipe's `execute_regime` after tracking closes.
- `uv run pytest tests/academy/test_choice_navigation.py -q`: browser coverage
  replays only two of its retained positions. Bind refreshed positions to a new
  world file and retain prior evidence.
- A world or manifest change also needs `tests/academy/test_learn_setup.py`
  (its Learn fixture pins manifest and state digests) and
  `uv run --extra dev pytest tests/model/ -q` (checkpoint loading and
  historical-world rejection). Tests constructing a current checkpoint use the
  runtime world version; incompatible checkpoints keep explicit historical
  versions. AGENTS.md lists the arena-admission and monitoring suite for world
  or observation-shape changes.
- Trigger-order browser cases need the test-only ASGI root in
  `academy/ui/playwright.release.config.ts`. After building the frontend, run
  `npm --prefix academy/ui run test:e2e -- --config playwright.release.config.ts trigger-order.spec.ts`
  with isolated `ACADEMY_RELEASE_API_PORT` and `ACADEMY_RELEASE_FRONTEND_PORT`.

**Training recovery.** Resume admission or game-error recovery:
`uv run --extra dev --extra artifacts pytest tests/training/test_recovery.py tests/training/test_game_error_recovery.py -q`.
Observation-input or model-cost changes add
`uv run --extra dev pytest tests/training/test_active_training.py -q`; its
endpoint checks use injected durations, so never mix host elapsed time into an
exact-update-count assertion.
Changes to observation encoding or action selection also require
`uv run --extra dev pytest tests/env/test_observation.py tests/training/test_action_cutoff.py tests/verify/test_native_buffer_capacities.py -q`.
Collector or batch-closure changes also require
`uv run --extra dev --extra notebook --extra artifacts --extra reports pytest tests/training/test_batch_closure.py tests/model/test_layer_input.py tests/semantic/test_elemental_world.py -q`.
Fixtures asserting exact rollout shapes or transition counts must explicitly
select `batch_closure="exact-quota"` as well as one learning seat; whole-game
closure treats the transition quota as a minimum.
Run the existing encoder tests alongside cutoff/parity fixtures so callers of
changed private helpers are checked too. Action overflow is checked at the
public encoder boundary, before the retained action list reaches row encoding.

**Local-update search.** Run `tests/sim/test_local_update.py`,
`tests/sim/test_local_update_sampling.py` and
`tests/sim/test_search_command_evidence.py` together. The core integration job
includes all three: collection alone cannot check receipt replay, terminal
frontiers, deadlines or atomic Command rejection. When games are restricted to
a remote host, defer the game-based cases to that job explicitly.
Inspect collector fixtures before selecting laptop-safe model tests:
`SeatRoutedCollector` defaults to `batch_closure="complete-games"`, so even
`collect(model, 1)` can play a complete game.

**Tracking and offline reports.** Checkpoint imports or managed evaluation
adapters:
`uv run pytest tests/eval/test_local_evaluation.py::test_cli_completes_without_training_cloud_or_callback -q`
(a fresh process rejecting managed evaluation imports; adapter-only tests cannot
establish that boundary). Tracking or report discovery:
`uv run pytest tests/training/test_experiment_report.py tests/training/test_experiment_runner.py -q`;
the live tiny comparison writes both TrainingRun exports and Etude tracker
declarations named `run.json`.

**Python package and CLI moves.** Experiment imports or recipe discovery:
`uv run --extra dev pytest tests/training/test_regimes.py -q` (listing recipes
must not import DuckDB). Adding, renaming or removing an Etude experiment also
requires updating its exact discovery inventory and running
`uv run --extra dev --extra artifacts --extra evaluation pytest tests/eval/test_experiment_files.py -q`.
When integrating a frozen pilot with a changed world or tensor schema, run its
focused tests too. Synthetic scheduling/analysis fixtures must pin both identity
and schema; test real runtime rejection separately, and keep those fixtures off
replay, encoding and training paths. Never rewrite frozen protocols to fit the
current runtime.
Checkpoint-time scheduling changes also require
`uv run --extra dev --extra artifacts --extra evaluation pytest tests/eval/test_tla_frontier.py -q`.
Store synthetic checkpoints with an injected wall clock, and cover both relative
run speeds; filesystem timing must not decide the expected matchup set.
When moving modules, tests or entry points, collect the
exact paths in `.github/workflows/manabot-integration.yml` and run
`uv run --extra dev pytest tests/infra/test_ci_selection.py tests/infra/test_release.py -q`.
Training CLI changes need the installed-wheel smoke from
[Build and verify](#build-and-verify); a source-tree smoke cannot verify the
installed entry point.

**Loopflow workflows.** With `lf` installed, run
`uv run python scripts/check_loopflow_workflows.py` for graph/skill changes.
It uses real compilation/discovery and checks iteration, delivery and experiment
review boundaries without launching agents, games or training. The
[work boundaries](workflows.md) document defaults and migration limits.

**New Python test modules.** Basenames must be unique across unpackaged test
directories. Collect new modules with the existing tests of their CI invocation
(`uv run pytest --collect-only`); isolated file runs miss name collisions.

**Play layout.** Run `combat-presentation.spec.ts` for board/sidebar geometry
and narration controls, and `npm --prefix academy/ui run test:e2e:release`.
Settle narration through `finishBoardNarration` before captures; reduced motion
does not stop semantic beat timers.

**Shared game history.** Exercise `game-history.spec.ts` against an isolated
built SPA/ASGI instance
([Academy isolated browser validation](../academy/README.md#isolated-browser-validation)),
including empty results, visible request errors and retry, shared/private
replay and feedback. Permission or replay-projection changes add the relevant
history, replay and shared-table cases; match terminal traces by the live
table's `attempt_id` (replays omit the deal seed). Treat failures in frozen
advice evidence as gate blockers; never regenerate fixtures or relax historical
checkpoint validation to pass.

**Website** (`website/**`): from `website/`, run
`uv run --extra test pytest tests/ -q`. Keep coverage for
parseable blog-only sitemap XML, the root redirect, and the plain-text
`/healthz` Fly probe.

**Clean-machine startup.** Run `tests/academy/test_play_launcher.py` and
`tests/academy/test_clean_machine_verifier.py`, then `./scripts/verify-clean-machine`
under its [empty-artifact precondition](#reference-profile), preserving and
restoring local build artifacts and using unused ports. Keep the 90,000 ms
playable budget, offline reload and session-identity assertions. A local macOS
pass does not certify the Ubuntu CI profile. Startup ordering is a contract:
proof-runner startup overlaps installation once npm is ready; frontend startup
and its first request overlap a cold native build; failure or timeout of either
service reaps the other and the browser runner. Exercise orchestration with
native preparation initially unavailable, since pack validation imports the
engine and a warm launch cannot expose serialized startup.

## CI

`.github/workflows/ci.yml` runs on pull requests, pushes to main and manual
dispatch, and always reports the required aggregate **Tests Result**. A newer
PR revision cancels its obsolete run; main and manual runs do not cancel.

| Job | Runs |
|---|---|
| Rust Tests | debug `cargo nextest run --profile ci` in `managym/`, including `semantic_conformance_tests`; JUnit timings uploaded |
| Python Unit Tests | `tests/infra/`, clean-machine verifier, `deploy/play/tests/`, website |
| Integration | `manabot-integration.yml` (`core`, `evaluation` and `long` groups on separate runners; Perf report proof and training smoke in `evaluation`) |
| Academy Integration | `academy-integration.yml` (world bindings, Elemental admission and play) |
| Protocol v1 Conformance | Rust, Python and TypeScript experience protocol; launcher contracts |
| Semantic Kernel Conformance | bounded valid-action fuzzing only, over `semantic-kernel-w17-v2` |
| Academy UI | `npm test`, `npm run check`, `npm run test:e2e:release` |
| Clean-machine Play | `./scripts/verify-clean-machine` |
| Engine Budget | `engine-budget.yml` (release `bench_search` and `bench_steps`, candidate against the reference commit of `managym/perf/baseline.json` on one runner) |

Semantic-kernel conformance is recorded as manifest seeds plus a per-case
`trace_sha256` in `evidence.json` ([conformance](../conformance/README.md)). A
drifted case is written to `target/semantic-conformance/<corpus>/`, which Rust
Tests uploads on failure; fuzz failures keep exact seeds under
`target/semantic-conformance/failures/`. Rust Tests runs whenever the fuzzing
job does.

**Selection.** Detect Changes pipes the NUL-delimited diff (`--no-renames`, so
both sides of a rename) into `.github/ci/select-tests.sh`. Manual dispatch, an
unknown base, an empty diff or an unknown path selects everything; a dispatch
with `measure` selects only Engine Budget. Tests Result
fails if detection failed, a selected job did not succeed, or an unselected job
neither skipped nor succeeded.

| Change | Selected jobs |
|---|---|
| Native engine, `content/`, `conformance/`, shared Python/Cargo dependencies, CI selector or `ci.yml` | All |
| `managym/tests/` | Rust, semantic conformance |
| `managym/.config/nextest.toml` | Rust |
| `protocol/` | Rust, protocol, UI, semantic conformance |
| Curated packs and UI fixtures | All but Python unit |
| UI dependency, build config, scripts, static assets, card art | UI, clean-machine |
| Other Academy UI | UI |
| Academy server, presentation, demo, pack or villain modules, `academy/fixtures/` | Academy integration, protocol, UI |
| Other `academy/`, `tests/academy/` | Academy integration, protocol |
| Launcher, `scripts/verify-clean-machine`, `deploy/play/` | Python unit, protocol, clean-machine |
| `manabot/env/`, world and flat-search/structured-action modules | Integration, Academy integration, protocol, UI |
| Other `manabot/`, training/eval/sim tests, `scripts/`, `experiments/`, `ops/`, benchmark docs, `docs/experiment-metrics.md` | Integration |
| `manabot/infra/` | Python unit, integration |
| `engine-budget.yml`, `managym/examples/bench_*.rs`, `managym/examples/curated/`, `managym/perf/` | Engine budget (`engine-budget.yml` also Python workflow regressions; `bench_search.rs` also Rust digest tests; `scripts/engine_budget.py` also Python unit) |
| `website/`, `tests/infra/`, rules audit | Python unit |
| Docs, Markdown, `.lf/`, `paper/`, `scratch/`, other workflows | Detection and aggregate only |

Runtime inputs such as experiment Markdown and benchmark data are not prose-only.
`tests/infra/test_ci_selection.py` covers these boundaries and the aggregate;
update both when adding a package or consumer.

**Other workflows.** `macos-nightly.yml` reruns both integration suites on
`macos-latest` nightly. `baseline.yml` trains the CPU test variant nightly on
seeds 197, 281 and 419 and combines them with `manabot.verify.learning_regression`.
`card-evaluation.yml` (manual) verifies 1024 native games with
`manabot.eval.cards verify` (see [evaluation](evaluation.md)).
`release.yml` is described next.

## Training and evaluation speed

Measure the training and evaluation pipelines on this machine, across the fixed
`w64`, `w128l4`, `w160l8` and `a384l8` model ladder:

```sh
uv run --locked --extra evaluation python scripts/bench_training_update.py \
  --suite --device cpu --out .runs/pipeline-speed-001
```

Run from the repository root with the pinned Python environment and a release
native extension built from the same source (see [environment setup](#environment-and-commands)).
The command prints one training row and one evaluation row per size and retains
the underlying receipts in the new output directory. It runs locally without
cloud credentials, a queue or a rental. Use it when changing collection,
encoding, model inference, learning or shared evaluation serving; it is not a
CI timing gate.

The suite uses the current-world Elemental recipe with its own benchmark regime
identity, explicit v6 input (512 local facts), 256 streams and float32 precision.
Training goes through the real
`execute_regime` stage. Evaluation plays checkpoint against checkpoint through
shared inference, using two reproducibly initialized policies, deal seed 101 and
four BG/GR deck-and-seat legs (`elemental_bg` / `elemental_gr`). The allocation
admits four active games and at most 32 inference rows. This measures serving
and gameplay throughput, not playing strength. World, observation configuration,
model shapes, seeds, checkpoint identities and execution limits stay in the
receipt. Historical experiment declarations and evidence are unchanged.

The default is three training updates per size, with the first excluded from
the steady-state reading. Evaluation uses the suite's fixed deal set. The whole
invocation has a 3,600-second deadline, including startup and cleanup; this is a
limit, not an expected duration. The report records actual elapsed time and
completed rows. `--seconds` changes that limit, and `--updates` changes the
training count. A slow machine may need a larger explicit budget to complete
the ladder. A partial ladder is not a complete suite proof. The process tree has a
16 GiB RSS ceiling; `--memory-mib` changes it explicitly. Subset checks use
`--size w64` and/or `--workload training` (repeat either flag for multiple rows);
`--streams` remains available. Subsets cannot replace a full-ladder comparison.

`report.json` retains every attempted row, its source/diff identity, measured
host, elapsed time, cleanup time and any failure. Each row directory retains
`worker.log` and its ordinary pipeline receipts. Training adds `training.json`
and `updates.json`; evaluation keeps `panel/evaluation.json`, shared-serving
performance, Commands and replay evidence. Failed runs stop the ladder and
retain completed rows without assigning rates to missing rows.

Read the rows as follows:

| Reading | Meaning |
| --- | --- |
| Learner decisions/s; seconds/update | Training work over collection plus learning time, excluding the warmup update. Setup, export and total invocation time remain separate. |
| Native stepping; encoding; inference; learning | The real stage's timing breakdown. Parent and child clocks overlap; do not sum both. Other collection work remains visible. |
| Inference calls/update; mean rows/call | Whole-stage policy calls divided by updates, and `environment_decisions / inference_calls`, including warmup, for both-seat self-play. |
| Idle stream fraction | `idle_stream_steps / (inference_calls × streams)` over the whole stage, including warmup. |
| Evaluation games/s | Completed, replay-verified games over the evaluation interval, including serving startup and replay. Checkpoint preparation is separate; replay seconds and serving startup remain in the underlying receipts. |

Complete-game batch closure explains why training calls often contain fewer
rows than the configured stream count. Once a stream satisfies its collection
quota and finishes its game, it idles until the remaining streams finish. The
policy sees only active streams, so batches shrink toward the longest game's
tail. Mean batch size and idle fraction expose this cost; increasing a serving
batch limit does not remove it.

Add `--hourly-price 0.59` to express learner decisions and evaluation games per
dollar at that supplied USD/hour rate: `rate_per_second × 3600 / hourly_price`.
Training is a steady-state estimate; evaluation uses its measured allocation
interval including startup and replay. Both omit rental setup, idle rental time,
failed attempts and billing adjustments and are never a rental bill.

### Compare against a reference commit

Run from the candidate checkout, with a separate reference checkout on the same
machine:

```sh
uv run --locked --extra evaluation python scripts/bench_training_update.py \
  --suite --device cpu --compare manabot/perf/baseline.json \
  --reference-root /path/to/reference-checkout --out .runs/pipeline-compare-001
```

`manabot/perf/baseline.json` contains `commit` (the full source SHA) and
`suite_interface` (currently `1`).
There is no designated reference host. Absolute rates and machine details live
in each run's report as history; they do not bind the baseline to the machine
or device that recorded it. Only ratios from freshly measured reference and
candidate runs in the same job can support a performance comparison.

Both source checkouts must be clean, with their own locked environments and
matching release native extensions. The reference checkout must match the
baseline commit and declare the matching `SUITE_INTERFACE` in
`scripts/bench_pipeline.py`. The command creates no
worktrees. Recording and comparison cover all four sizes and every workload
supported by the selected device; a subset remains diagnostic. Invalid reference
metadata or an incompatible checkout rejects before workers start and retains
the admission error. Correct the cause and retry with a new output directory.

The comparison runs both sources interleaved in one job, with three repetitions,
and reports minimum durations. Each minimum names its sample; phase timings come
from that sample rather than independently selected minima. Use matching suite
options for both sources, including updates, streams and memory allowance.
Comparison has the same total lifetime limit; six interleaved readings may need
a larger explicit `--seconds` budget. The report retains every attempt, world,
input configuration, device, precision, workload and behavior digest, plus
`TrainingRun.host` facts, machine identity, Python and Torch versions. Host
admission compares the fresh samples within this job, not an earlier recording.

Training behavior hashes the final policy tensor contents and work counters;
evaluation hashes initialized tensor contents, retained Commands, state transitions
and outcomes, excluding clocks and arrival order. A digest match is a conservative
comparison check, not a proof of learning equivalence.

Compare only matching workloads, host identities, Python and Torch versions,
devices and precision with matching behavior digests. An identity or behavior
mismatch is reported as incomparable, not a performance regression or improvement. A changed world or
recipe needs a new reference; retain the previous evidence. To propose a new
reference, run the complete suite on clean source with
`--record-reference manabot/perf/baseline.json`. Review and commit the resulting
commit-and-interface metadata; keep the measured rates in the run report. A
partial or failed run cannot replace the baseline, and `--compare` never adopts
one automatically. Timing agreement does not establish learning equivalence
for a change to numerical computation.

CPU, MPS and CUDA are separate readings. Training on an available Apple GPU
uses `--device mps --workload training`. Shared evaluation currently admits only
CPU and CUDA; a full MPS suite fails explicitly without substituting CPU. Local
CPU runs are free harness checks: a complete eight-row ladder establishes
coverage and actual runtime, not GPU performance. The CPU comparison above
checks the interleaving and admission workflow without making a speed claim.

On an already provisioned CUDA host, prepare both clean checkouts and their
environments on that host. The named CUDA comparison check is:

```sh
uv run --locked --extra evaluation python scripts/bench_training_update.py \
  --suite --device cuda --compare manabot/perf/baseline.json \
  --reference-root /path/to/reference-checkout --out .runs/pipeline-cuda-compare-001
```

On a rented GPU, run both sources on the same pod in that one job. Matching GPU
labels on separate rentals do not establish a comparison, and earlier absolute
rates are never its denominator. The command provisions nothing; arranging or
paying for a host is separate. Do not introduce artificial contention to test a
shared host.

An unavailable device, admission failure, capacity overflow, timeout or failed
replay retains a failed or incomplete receipt and exits nonzero; missing rows
never become zero-speed measurements. Resolve the reported cause and rerun into
a fresh directory. Do not silently reduce capacities, change precision or
substitute another device to complete a comparison.

For attribution of a single recipe, omit `--suite`: the existing
`--recipe`, `--streams`, `--updates`, `--json` and `--out` workflow remains
available. CUDA profiler counts (`--cuda-counts`) perturb timing and do not
belong in an ordinary suite comparison.

Workload rows are registered in `scripts/bench_pipeline.py`. A runner returns a
measured population and interval, workload/behavior identities and receipt paths;
the shared coordinator owns bounds, repetition, comparison and price conversion.
Belief or deck-building training can add a runner to that registry.

## Engine performance budget

`Engine Budget` builds two release benchmarks on a GitHub-hosted runner of
their own and compares a change with the reference commit named in
`managym/perf/baseline.json`:

- `managym/examples/bench_search.rs`: Search-64 replies (16 worlds x 4
  rollouts) on the curated UR Lessons versus GW Allies matchup, seeds 1-3,
  40 decisions each. Budget lines: worst reply, summed total, and the search
  digest of each seed. The search digest covers every action score in legal
  order, with each decision framed by its action count, simulation count and
  chosen index. Scores use exact IEEE-754 bits in little-endian order; the
  integer digest is the first 64 bits of BLAKE3. A changed losing score fails
  even when the winner and simulation count stay the same.
- `managym/examples/bench_steps.rs`: headless steps per second, 512 seeded
  games of the same matchup with both seats uniformly random over the legal
  actions and native trivial-decision skipping. Budget lines: steps per
  second, step count and the digest over every decision.

GitHub-hosted `ubuntu-24.04` runners land on different CPU generations, and
the same engine measured 6.8k steps/s on an AMD EPYC 7763 and 10.3k on an
Intel Xeon 6973P-C (2026-11-02), so wall time is never compared across runs.
The job builds both commits with the change's benchmarks before timing, then
runs candidate/reference/reference/candidate on that machine. It compares the
median duration of the two samples per engine (including each seed's worst
reply; step throughput is the inverse median time per step) and fails the PR
when the candidate is more than 10% slower than the reference on
any timing or either run's digests differ from the committed ones (two runs
of one engine on separate EPYC 7763 machines agreed within 2.5%). So a
speedup that changes search results fails until the reference moves, and so
does a baseline file whose digests no longer describe its reference commit.
The figures in the file are the reference's own recording, for orientation
only.

**Moving the reference.** Every run is a recording of its candidate: the
`engine-budget` artifact holds the candidate's output and a `measurement.json`
naming the candidate commit, date, run and runner CPU. Raw samples and
`sampling.json` retain both orders, CPU time, context switches, and host load /
Linux CPU-pressure counters. These can show scheduling noise; without historical
telemetry they cannot establish the cause of an earlier failure. There are no
adaptive retries. A failed reference build still records the candidate before
failing the check. A PR adopts its own run as the reference, saying why in
the commit, for one of three reasons: a digest moved (a rules change reaching
these decks, or a change to the benchmarks or the curated lists); a measured
cost is accepted; or a speedup is kept, so later slowdowns are budgeted
against the faster engine rather than allowed back up to the old reference.
The PR body cites the delta from the run it adopts; the recording commit's
own run then compares the same engine with that reference and passes. When
digests match but timing fails, inspect the retained samples and scheduling
evidence before attributing the failure to runner noise. An unexplained failure
remains a failure; do not retry until green or widen the tolerance.

```bash
gh run download <run-id> --name engine-budget --dir .cache/engine-budget
uv run scripts/engine_budget.py record --reason "<why the reference moves>"
```

The reference commit must stay reachable from origin: on a pull request the
job checks out and records the PR head, never GitHub's ephemeral merge
commit, and a PR head stays fetchable after a squash merge. The benchmarks
use only the stable `Env` surface (`new`, `reset`, `step`, `action_count`,
`is_game_over`, `winner_index`, `flat_mc_scores`); a change to that surface
that the reference commit no longer compiles with still records the
candidate despite the failed reference build and moves the reference in the
same PR like any other move. For a commit that has no PR, the `measure`
dispatch input on `ci.yml` selects only Engine Budget, benchmarks that commit
with the dispatched ref's benchmarks and uploads the same artifact without
comparing (`gh workflow run ci.yml --ref <branch> -f measure=<commit>`).

**Current reference.** Commit `ea316ff14` is adopted from
[PR #415 CI run 41711692234](https://github.com/loopflowstudio/manabot/actions/runs/41711692234)
(2026-11-03 UTC, AMD EPYC 7763). MTG-239 gives Accumulate Wisdom explicit
bottom-order decisions and makes Rallier evaluate Suki's power in the library.
All three cards are in the unchanged benchmark decks; these rules changes
affect both played decisions and search rollouts. All three search digests and
the step digest changed, with 101,650 steps instead of 99,923 across 512 games
(none capped). The previous reference reproduced its committed digests.

| Metric | Previous reference `1616b210f` | Adopted candidate |
|---|---:|---:|
| Worst search reply | 5.280 s | 4.986 s |
| Total search | 177.796 s | 171.233 s |
| Step throughput | 12,874.6/s | 12,891.2/s |

These are the two measurements from that one runner. Timing checks passed;
changed gameplay prevents treating the delta as equivalent-workload speedup
evidence. The rules change moves the reference while preserving the **10%
tolerance**. The [MTG-238 investigation record](../.lf/wave/engine-performance/research/engine-speed-2026-11.md)
retains the preceding engine improvements and their evidence.

**Previous reference: derived-state caching.** Commit `4b5e9c0e4` was adopted from
[CI run 41555845634](https://github.com/loopflowstudio/manabot/actions/runs/41555845634)
(2026-11-03 UTC, Intel Xeon 6973P-C), retaining merged main `51fbacbbf`'s
derived-state revisions, indexed effect/trigger records and cache-carrying
clones. Its native source matches that main commit. On the same runner:

| Metric | Previous reference `88dc8c1ad` | Adopted candidate | Improvement |
|---|---:|---:|---:|
| Worst search reply | 10.226 s | 7.132 s | 30.26% less time |
| Total search | 290.711 s | 208.251 s | 28.36% less time |
| Step throughput | 8,833.0/s | 13,943.8/s | 36.65% less time per step |

All full-score/step digests and counts match. Delegated demo judgment applied
the retained-speedup rule so future regressions are budgeted against this
faster engine. Adoption preserves the **10% tolerance**; it is not shipping
approval. These figures compare engines on one runner, not against the older
reference's recorded absolute timings below.

**Previous reference and accepted cost.** Commit `88dc8c1ad` was adopted from
[CI run 41552162812](https://github.com/loopflowstudio/manabot/actions/runs/41552162812)
(2026-11-03 UTC, AMD EPYC 7763): 6,104.0 steps/s, 14.857 s worst search
reply and 433.796 s total. It establishes the initial budget at merged main
`a8abf01b1`, including decklist disclosure (`230b2f563`) and T5 token copies
and effective battlefield characteristics (`a8abf01b1`). On that same runner,
the older `e5615edcc` reference measured 6,774.5 steps/s: the candidate used
**10.98% more time per step** (9.90% fewer steps/s), correctly failing the
10% budget. Search timings and all full-score/step digests and counts passed.
Delegated demo judgment accepted this measured cost of the already-merged
engine; the source comparison does not isolate its cause. The measured head
`88dc8c1ad` has no native source diff from that main commit. Adoption preserves
the **10% tolerance** and the failed run as evidence; it is not shipping approval.

Measured once on 2026-11-02 with the original search digest (steps/s; worst Search-64
reply; total over 3 seeds x 40 decisions), each on its own runner:

| Commit | CPU | steps/s | worst reply | total |
|---|---|---:|---:|---:|
| T4 main `4c8324a59` (original reference) | AMD EPYC 7763 | 6819.5 | 13.807 s | 408.9 s |
| T5 head `836042c07` | Intel Xeon 6973P-C | 10285.9 | 9.976 s | 273.9 s |
| T6 head `2c1ba169f` | AMD EPYC 9V74 | 8883.1 | 10.651 s | 310.9 s |

The three rows are on three CPU models, so they rank the runners, not the
commits. All three reproduced the original digests, but that search digest
covered only simulation counts and winning actions, not every action score.
These historical timings do not prove full-score equality. The committed
baseline is recorded separately with the strengthened digest. Run the benchmarks
locally only as a smoke check:
`cargo run --release --example bench_search --manifest-path managym/Cargo.toml`
(and `bench_steps`). Never add load to make a measurement move
([TESTING.md](../TESTING.md)).

## Releases

Manabot ships as one PyPI distribution, `manabot`, containing the Python library
and the native engine at `manabot.managym`; `import managym` is a compatibility
alias. Versions are `X.Y.Z`, tagged `vX.Y.Z`, shared by Python metadata and the
bundled crate; the first public release is
[0.2.0](https://pypi.org/project/manabot/0.2.0/). Package versions do not replace
world, content, observation or checkpoint identities, and an upgrade does not
make old checkpoints compatible. Releases are research releases, not
certification of playing strength or rules coverage.

### Install

Install a locally built compatibility wheel in a fresh directory outside this checkout:

```bash
mkdir manabot-example
cd manabot-example
uv venv --python 3.13
uv pip install --python .venv/bin/python --torch-backend cpu "/path/to/manabot-<version>-cp313-cp313-<platform>.whl"
uv run --no-project --python .venv/bin/python manabot --help
```

Replace `<version>` and `<platform>` with the values in the built wheel filename.
This installs the library and bundled engine into `.venv`; the last command
lists CLI commands without starting training. No repository dependency groups
are installed. For CPython 3.12, create the environment with
`--python 3.12` and use the matching `cp312` wheel. These examples use local
artifacts because the compatibility changes have not been published. The
published 0.2.0 release is 3.12-only; install `'manabot==0.2.0'` in a 3.12
environment when reproducing that release.

For Linux NVIDIA training use `--torch-backend cu128`; always select the backend
explicitly. CI validates CPU, not GPU, execution.

Supported: ordinary CPython 3.12 and 3.13 (no free-threaded ABI) on Linux x86-64 with glibc
2.28+ (CPU or CUDA 12.8 Torch) and Apple Silicon macOS 14+ (CPU or MPS Torch).
MPS is admitted for float32 self-play training with Torch 2.10. Baseline requires
CUDA or Apple MPS; local selects the available accelerator (CUDA first) or CPU.
Evaluation and search run on CPU for this workflow. GPU-less tests use
`baseline_test()`, which shares baseline construction and changes only the device.
No speed or strength claim is implied. Windows,
Intel Mac, Linux ARM and musl are not supplied or tested.

Wheels contain the compiled engine; the sdist needs Rust and a C linker
(`--only-binary :all:` refuses source builds). Both forms declare the same
runtime requirements, including `torch>=2.10.0` and `etude-sdk>=0.2.15`.
These minimums allow newer versions to resolve; they do not certify every future
dependency release. Keep your project's lockfile for reproducibility. If
resolution fails, check for an incompatible Python version, an older exact
Torch/Etude pin in your project, or a missing wheel on the selected backend.
Do not bypass dependency checks with `--no-deps`.

Notebook file generation remains a core dependency because ordinary experiment
reporting creates an editable notebook. Pandas is no longer a direct requirement;
no library entry point consumes it.

Choose extras for the features you use:

| Extra | Use |
|---|---|
| `academy` | Academy backend; use `./scripts/play` from a checkout for the browser |
| `evaluation` | Etude evaluation and plots |
| `reports` | Reports and Etude S3 report sources |
| `artifacts` | Artifact-store access |
| `notebook` | Jupyter and notebook execution |
| `llm` | Local transformer inference |
| `openrouter` | OpenRouter inference |

For notebook work, append `[notebook]` to the quoted wheel path in the install
command. In a checkout, use `uv sync --locked --extra notebook`. If a command reports
a missing optional dependency, install its named extra in the same environment
and retry. Development groups are not installed as part of a downstream package.

`--regime-name`, experiment drivers and data-ledger tools need a
source checkout. Never change a running experiment's pinned source to install a
release.

### Build and verify

From a clean checkout, with a fresh staging path and empty output directory:

```bash
uv run --no-project python scripts/release/check.py
uv run --no-project python scripts/release/prepare.py .runs/release/staging
uv build --no-sources .runs/release/staging --out-dir dist
uv run --no-project python scripts/release/validate.py dist
uv run --no-project --with twine twine check --strict dist/*
uv run --no-project --with pytest pytest --confcutdir=tests/infra tests/infra/test_release.py -q
```

Staging copies tracked files, refuses to overwrite a prior build and excludes
experiment drivers and build products; published artifacts must use it. Staged
metadata must preserve the source project's public minimum requirements rather
than replacing them with development pins. CI builds one source archive for all
four wheels, so checkout files cannot hide missing content. The complete release inventory is
four wheels (3.12 and 3.13 on each supported platform) and one source archive.

Clean-install proof, using only the wheel matching this interpreter and host:

```bash
uv venv --python 3.13 /tmp/manabot-release-venv
uv pip install --python /tmp/manabot-release-venv/bin/python --only-binary :all: \
  --torch-backend cpu "$PWD/dist/manabot-<version>-cp313-cp313-<platform>.whl"
cp scripts/release/smoke.py /tmp/manabot-release-smoke.py
cd /tmp
export PATH="/tmp/manabot-release-venv/bin:$PATH"
unset PYTHONPATH
uv run --no-project --python /tmp/manabot-release-venv/bin/python manabot-release-smoke.py
```

The smoke checks one shared engine behind both import paths, catalog binding,
gradients and a tiny CPU training update. Repeat in a separate environment with
3.12 and its wheel. `release.yml` covers both interpreters on Linux and macOS,
then installs and imports the optional extras. Consumer checks run outside the
checkout with default dependency groups disabled, including the minimum
Torch/Etude versions and available newer versions where practical. Record exact
versions tested; successful resolution alone is not a runtime compatibility
check.

### Publish

`release.yml` builds and validates relevant PRs, manual runs and `v*` tags; only
tags publish. Tag builds reject inconsistent versions and commits not on main.
A release is complete only when its PyPI artifacts and matching GitHub Release
both exist. The GitHub entry uses the same `vX.Y.Z` tag and that version's
checked-in `release/vX.Y.Z/NOTES.md`, with a link to that version on PyPI.
Only the PyPI job gets OIDC `id-token: write`; only the GitHub Release job gets
`contents: write`. No PyPI token is stored, and uploads carry attestations.
The trusted publisher is project `manabot`, owner `loopflowstudio`, repository
`manabot`, workflow `release.yml`, environment `pypi`. The pending publisher
was used successfully on 2026-10-22; the GitHub `pypi` environment restricts
tags to `v*`. Recheck its required reviewers and trusted publisher settings before
tagging.

Per release: update `pyproject.toml`, Cargo.toml/Cargo.lock, `uv.lock`, examples
and write `release/vX.Y.Z/NOTES.md`; merge the reviewed release PR; confirm the workflow
is green; push `vX.Y.Z` at that main commit; review the `pypi` job's artifacts
and approve it. A PR alone does not authorize a tag or upload.

If PyPI succeeds but GitHub completion fails, retry the failed completion job
for the same tag. It creates or verifies the matching entry without uploading
again. An entry with matching tag and notes makes the retry a no-op. Retrying
the PyPI job checks the complete inventory and SHA-256 digests against retained
build artifacts and skips upload only when they match. Rebuilding can change
archive bytes; retry failed jobs with the original artifacts instead of rebuilding
a published version. Partial or different publication fails for inspection.
Never overwrite PyPI files, delete a release or move a used tag to repair it.
Afterwards, install from PyPI on both supported platforms and interpreters and
confirm the version, attestations and GitHub Release link.

0.2.0 was published on 2026-10-22 from `v0.2.0` at
`29ab4fa0ddcc759ddb2b699ce3d08ccfe51ad757`
([successful workflow run](https://github.com/loopflowstudio/manabot/actions/runs/39790120844)).
Its installed-wheel smoke and optional API/report dependency checks passed on
Linux and macOS with CPython 3.12. The PyPI trusted-publisher identity was
verified against the repository; PyPI holds two wheels and the sdist. The full
record, with SHA-256 digests, is in the
[pinned release notes](https://github.com/loopflowstudio/manabot/blob/a74f31828/docs/releases.md#020-publication-record).
That workflow published only to PyPI and did not create a GitHub Release.

## Clean-machine play proof

From a fresh checkout, `./scripts/play` syncs the locked play runtime, builds the
CPython 3.13 extension, runs `npm ci`, starts FastAPI and Vite, validates the
installed curated pack and prints an `ACADEMY_PLAY_READY` JSON record and the
URL. `./scripts/verify-clean-machine` proves this once: it runs the launcher,
runs `academy/ui/e2e/clean-machine.spec.ts` against it, writes
`.cache/clean-machine/proof.json` and tears the process group down.

### Reference profile

The current reference profile is GitHub-hosted `ubuntu-24.04` x86_64 with
uv 0.9.22 and CPython 3.13, Node 22, stable Rust and native build tools, and
Playwright 1.61.1's Chromium. Historical receipts retain CPython 3.12; they do
not certify the updated profile. Each receipt records the observed OS, CPU,
memory and tool versions. Before timing, the checkout must not contain `.venv`,
`.cache`, `academy/ui/node_modules`, generated
Svelte/Vite/build/test output, `managym/target`, a built `_managym*.so`, GUI trace
JSON or a prior proof browser profile; no uv, npm or Cargo cache is restored.

### Time to playable

The clock starts just before `./scripts/play` is spawned and stops when a fresh
Chromium context shows a connected WebSocket, the default Search 64 opponent,
`UR Lessons vs GW Allies`, an enabled authoritative legal action, and pack
treatments for every visible card with no fallback. The budget is 90,000 ms. It
was 60,000 ms until 2026-09-30; the release native build alone had reached about
52 s on GitHub's runner, so the budget became 90 s. The receipt records
`elapsed_to_playable_ms`, `budget_ms` and `within_budget`; CI reports the time
and warns, but does not fail, when over budget, because runner variance decides
it. The proof fails if the launch is not ready within 600 s or any playable or
offline check fails. The launcher emits `ACADEMY_PLAY_PHASE` records and streams
installer and compiler output live, so a timeout leaves diagnosable logs.

### Offline reload

After readiness, Playwright denies every non-loopback HTTP/WebSocket request,
takes a legal action, records the session and a canonical visible-state
signature, reloads, and requires the same session, pack, matchup, state and
pack-only treatments, then a second action. Installation itself may use package
registries. The pack authority is
`academy/ui/src/lib/packs/tla-ur-lessons-vs-gw-allies/v2/manifest.json`; launcher
and receipt derive `{id, version, manifest_sha256}` from its installed bytes. The
launcher never fetches card data or art.

### Diagnostics and scope

Controlled failures print one `ACADEMY_PLAY_ERROR` JSON record and exit
nonzero, with a stable code: `prerequisite.{uv,node,npm,rust,cargo}`,
`lock.{python,cargo,frontend}`, `python.version`,
`pack.{missing,invalid,notice}` (before any service starts),
`native.{build,import}`, `frontend.install`, `port.in_use`,
`{backend,frontend}.start`, `ready.timeout` or `launcher.internal`.

The certified slice is one local session, one Chromium context, UR Lessons
against GW Allies, Search 64 and the installed pack; the minimal environment
also plays random and passive opponents. Checkpoint opponents need the full
development installation. This is not an installer, deployment artifact or
Windows certification.

## Retained evidence and archived corpora

The repository keeps current code, contracts and offline regression fixtures.
Experiment results and historical corpora live in the content-addressed artifact
store (`manabot/data.py`); git tracks only `experiments/data.manifest.json`, with
each path's digest and size. Generated agent and runtime output (`.lf/logs/`,
`.lf/tmp/`, native `.dSYM`) stays local and ignored.

Fetch exact original bytes (fetch never overwrites a differing local file):

```bash
uv run --extra artifacts python -m manabot.data fetch experiments/data/archive
# Or one corpus:
uv run --extra artifacts python -m manabot.data fetch experiments/data/archive/conformance/semantic-kernel-w12-v1
```

An archived file lives at `experiments/data/archive/<original repository path>`
and keeps its original source identity. Old worlds run from the revision that
recorded them: to reproduce an old experiment, check out its recorded source and
restore the files at their original paths. Fetching does not admit archived
bytes under the current runtime.

Task afbd6a (2026-10-30) archived byte-for-byte, with a verified empty-cache
restore: `authored-match-v1`, `authored-match-w11-v1`,
`authored-match-parity-v1`, `authored-match-parity-w11-v1`,
`public-commitment-parity-v1`, `semantic-kernel-w12-v1` through
`semantic-kernel-w15-v1`, and the w4 Learn cohort `learn-lesson-w4-v3` with its
wrapper (`experiments/data/archive/scripts/verify-learn-lesson`). The w1–w11 card
sources and the Python card compiler are in git history. The earlier archive
holds older semantic-kernel corpora, superseded authored-match receipts, older
Learn cohorts, Jeong replays, Elemental w10 evidence and structural-kata inputs.

Retained in source for current consumers: the four w17 corpora in
[`conformance/`](../conformance/README.md); `academy/learn_lesson_evidence.py`
(its `register`/`run`/`verify` modes record new Learn cohorts); `.lf/config.yaml`,
`.lf/wave/` (formerly top-level `wave/`) and referenced `.lf/chapters/` planning
evidence; protocol and model fixtures, the rules
audit index and the Comprehensive Rules PDF.

Frozen receipts and protocols keep the field names and source paths they were
recorded with (for example arena-named fields, now owned by `sim/`, `eval/` and
`experiment/`); recover retired code from git history. The retired-surface
inventory, migration counts and superseded procedures are in the pinned history:
[CI](https://github.com/loopflowstudio/manabot/blob/a74f31828/docs/ci.md),
[releases](https://github.com/loopflowstudio/manabot/blob/a74f31828/docs/releases.md),
[validation](https://github.com/loopflowstudio/manabot/blob/a74f31828/docs/development/validation.md),
[repository archive](https://github.com/loopflowstudio/manabot/blob/a74f31828/docs/repository-archive.md),
[clean-machine play](https://github.com/loopflowstudio/manabot/blob/a74f31828/docs/clean-machine-play.md).

### Bounded layer/input attribution

Keep the default `bench_steps` (512 curated games) and `bench_search` (seeds
1–3, 40 decisions each) separate from added positions; their `--layers` option
runs the overlapping-anthem fixture on its own. Search receipts retain every
reply sample, legal-action count, simulation/cap count and the full BLAKE3 score
digest beside the budget's historical 64-bit digest. `bench_steps --tail-samples`
and the dependency fixture retain the first 128 step durations in execution
order: sample maxima, not population percentiles. The default Engine Budget path
is unchanged.

**Dependency fixture.** For a genuine same-layer dependency:

```bash
cargo build --release -j 2 --manifest-path managym/Cargo.toml --features benchmark --example bench_layers
managym/target/release/examples/bench_layers [--allocations | --probe-only | --payment-only --allocations]
```

It asserts Moon/Urborg's dependency, then plays 16 bounded games and eight
Search-64 replies. `--probe-only` runs eight isolated pool writes and eight taps
on cloned warmed boards; `--payment-only` runs eight complete Cadets activations
from a floating pool and eight from two land sources on the dependency board.
Receipts retain complete legal action sets, selected actions, mana/life/tap
resources, stack and semantic digests around each action. Setup, forking and
serialization are outside the measured step. The fixture changes two physical
cards locally, never the public catalog or frozen worlds. The `benchmark` feature
adds inclusive, overlapping board/dependency/clone timers; never sum them or
compare a counting run with an ordinary timing run, and measure allocation
traffic in the separate `--allocations` run. Engines that fail the asserted
dependency are not correctness-equivalent timing comparators.

**Policy input cost.** `scripts/bench_layer_input.py` runs the baseline recipe's
width-64, depth-2 model on CPU float32 (one thread, batch 16, two warmups, eight
measured calls), without optimizer updates. Run each case sequentially, in a
fresh process, on an otherwise idle runner:

```bash
for state in ordinary shared; do for version in 6 7 8; do
  PYTHONPATH=. uv run --no-sync python scripts/bench_layer_input.py --version $version --state $state
done; done
PYTHONPATH=. uv run --no-sync python scripts/bench_layer_input.py --version 8 --state program
```

`shared` is a resolved Drill activation representable by all three inputs;
`program` adds Kellan's live grant and captured combat trigger. Receipts include
each sample, model/input configuration, native/Python encoding parity, live
detail/fact counts, observation, batch and parameter bytes, and process peak RSS
(including imports and warmups). They establish neither accelerator performance
nor learning efficacy. `--inventory` reports parity, occupancy and storage
without policy execution or timings, for busy hosts.

**Matched comparison.** `layer-risk.yml` runs `scripts/measure_layer_risk.py`
on PRs touching its inputs (or via `gh workflow run layer-risk.yml --ref
<branch>`) and uploads a `layer-risk` artifact, including partial evidence from
failed runs. One GitHub-hosted VM builds and measures sequentially: the original
TLA workloads at the Engine Budget reference, pre-mitigation `fb3555ffb` and the
candidate; dependency workloads, payments and the seven policy cases at the
latter two. The historical engine receives only the
`managym/perf/layers-attribution.patch` counters/fixture access, current
benchmark fixtures and the candidate's card definitions (so state digests compare
engines over identical content), never a cache or model mitigation. Receipts
record exact revisions, harness hashes, dirty source diff, CPU/dependency
metadata, raw samples, full digests/counts and memory. A resource, action or
digest mismatch between pre-mitigation and candidate fails the comparison; a
reference-world mismatch is recorded, never labeled a speedup. Each policy case
must retain eight finite positive samples per operation with matching settings;
missing, mislabeled or inventory-only receipts cannot satisfy timing comparison.
The workflow never updates a baseline or accepts a regression; the 10% Engine
Budget still governs.

For payment attribution on a busy host, omit timing workloads and use a fresh
output directory:

```bash
uv run --no-sync python scripts/measure_layer_risk.py prepare --out .cache/layer-payment
uv run --no-sync python scripts/measure_layer_risk.py native --out .cache/layer-payment --leg before --payment-only
uv run --no-sync python scripts/measure_layer_risk.py native --out .cache/layer-payment --leg after --payment-only
uv run --no-sync python scripts/measure_layer_risk.py compare --out .cache/layer-payment --payment-only
```

Allocation traffic and operation counts stay interpretable under contention;
elapsed values do not. `model --inventory` builds each revision's wheel and checks
all seven cases without inference.
