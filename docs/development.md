# Development

How to develop in this repository: commands, required checks, CI, releases,
the clean-machine play proof, and where retained evidence lives. Conventions
are in [AGENTS.md](../AGENTS.md); test-resource rules in [TESTING.md](../TESTING.md).

- [Environment and commands](#environment-and-commands)
- [Validation by change area](#validation-by-change-area)
- [CI](#ci)
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
root project owns that environment and the `manabot` distribution; the uv
workspace installs its matching `managym` member from this source tree.
Development uses ordinary CPython 3.13; consumer compatibility includes 3.12.
Keep shared Python source compatible with 3.12. Free-threaded builds and newer
interpreters are outside the supported contract.

Every Python command runs through uv (`uv run ...`). `uv sync` builds the local
engine extension; CI and existing scripts
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

The `manabot` public dependency requirements are `etude-sdk>=0.2.16` (also for
its `eval` and `s3` extras) and `torch>=2.15.0`. All extras, including `llm`, inherit the core
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

**Engine cost on large boards.** Random play and the Engine Budget decks
rarely build big boards, so a change to mana payment, legal-action, target or
combat enumeration, layers, triggers or observation encoding also runs the
[stress states](#stress-states) it could make expensive, before and after, on
one machine.

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
Changes to retained execution or its record vocabulary also require
`uv run --extra dev pytest tests/model/test_record_coverage.py -q` after rebuilding
the native extension. This bounded sweep checks that emitted execution tokens
have catalog rows, including decisions reached through changed game paths.

**Local-update search.** Run `tests/sim/test_local_update.py`,
`tests/sim/test_local_update_sampling.py` and
`tests/sim/test_search_command_evidence.py` together. The model integration group
includes all three: collection alone cannot check receipt replay, terminal
frontiers, deadlines or atomic Command rejection. When games are restricted to
a remote host, defer the game-based cases to that job explicitly.
`tests/sim/test_local_update.py` checks the defaults, compact rollout evidence
and native frontier encoding, against full evidence and the Python per-branch
encoder: the same roots and seeds must choose identical Commands and write
identical tensors. It also checks that v3 receipts and registrations omitting
`rollout_evidence` still read as full evidence. A change to
`managym.encode_observations_into` also runs `tests/env/test_observation.py`
and the debug native suite.
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
Store synthetic checkpoints for checkpoint-time scheduling tests with an injected
wall clock, and cover both relative run speeds; filesystem timing must not decide
the expected matchup set.
When moving modules, tests or entry points, collect the
exact paths in `.github/workflows/manabot-integration.yml` and run
`uv run --extra dev pytest tests/infra/test_ci_selection.py tests/infra/test_release.py -q`.
Package initializer changes also require
`uv run --extra dev --extra artifacts --extra evaluation pytest tests/test_package_boundaries.py tests/eval/test_inference_receipts.py -q`;
offline receipt reports must work without probing the native engine, even to
diagnose an installation problem.
Training CLI changes need the installed-wheel smoke from
[Build and verify](#build-and-verify); a source-tree smoke cannot verify the
installed entry point.

Check changed Python modules and type stubs with
`uv run --extra dev ruff check --fix <paths>` and
`uv run --extra dev ruff format <paths>`. Include moved `.pyi` files: the
compatibility stub is part of the public package surface.

**Policy backends.** Shared damped-update or backend changes need
`uv run --extra dev pytest tests/training/test_update_backend_protocols.py tests/training/test_update_backends.py tests/model/test_torch_backend_protocol.py tests/model/test_torch_container_load.py tests/model/test_mlx_backend.py tests/test_package_boundaries.py -q`.
The installed-backend tests run the real update, force gradient clipping,
compare every post-Adam weight and reject malformed raw rollout rows. MLX
cases skip where it is unavailable; run them on Apple Silicon with the `mlx`
extra before claiming MLX parity. For a changed forward or parameter mapping,
also run the [captured parity check](../manabot/backends/README.md#parity-check)
against a capture whose world binding still matches; it gates the optimizer
step as well as forward, loss and gradients.

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
| Integration | `manabot-integration.yml` (one runner per matrix group; Perf report proof and training smoke in `evaluation`) |
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

**Job time.** Pull-request jobs have an 8-minute budget on the ordinary
Ubuntu runner, measured from job start to job end, so installation and
the extension build count against it. Read a job's time from the run:

```console
$ gh run view <run-id> --json jobs \
    --jq '.jobs[] | select(.conclusion == "success") | "\(((.completedAt | fromdate) - (.startedAt | fromdate)) / 60 | floor) min  \(.name)"' | sort -rn | head -4
15 min  Engine Budget / Engine Budget
6 min  Rust Tests
6 min  Integration / Integration (ubuntu-latest, training)
6 min  Integration / Integration (ubuntu-latest, experiment)
```

Integration and Academy Integration install every extra they use in one
`uv sync` that leaves `managym` out, build the extension once with
`maturin develop`, and run with `UV_NO_SYNC=1` so no later `uv run` compiles the
engine a second time. Each Integration group runs one pytest worker per logical core
(`pytest -n logical`; [TESTING.md](../TESTING.md) says what that asks of a test)
and prints its 20 slowest tests (`--durations=20`) at the end of its test step.

Integration and Python unit install the locked CPU Torch with
`--no-default-groups --group play-runtime` because hosted Ubuntu runners have no GPU.

When a job passes 8 minutes, in order of preference:

1. Make its slowest tests cheaper (a smaller fixture, fewer updates or games,
   a shared setup) while keeping what they assert.
2. Cut the shared installation and build time.
3. Move test files to a faster group, or add a group to the matrix, balanced by
   the measured times; `tests/infra/test_ci_selection.py` checks that every
   path a group names exists.

Never skip, deselect or delete a test to fit, and never raise the five-minute
single-test limit ([TESTING.md](../TESTING.md)). Groups select directories,
files and individual tests. If a new file is not covered by a directory
selection, add it to the group with the most room.
Engine Budget is the exception: it builds and measures the change and its
reference on one runner and keeps that shape, whatever it takes. The macOS
nightly reuses the same groups without this limit.

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

**Current reference: MTG-297 mana spending.** Commit `d53220d36` is adopted
from [dedicated run 43417765062](https://github.com/loopflowstudio/manabot/actions/runs/43417765062)
(2026-11-13 UTC, AMD EPYC 7763). Native unit/class spending adds decisions to
these decks: all three search digests and the step digest change, with 102,209
steps instead of 101,650 across 512 games. The previous reference reproduces
all committed digests. Every timing threshold passes; both samples agree
closely and user CPU tracks wall time. Changed gameplay means these are
budget measurements of different workloads, not an isolated algorithmic
speedup. The 10% tolerance is unchanged.

| Metric | Previous reference `7ef6ce588` | Adopted candidate |
|---|---:|---:|
| Worst search reply | 4.928 s | 3.611 s |
| Total search | 169.080 s | 125.545 s |
| Step throughput | 11,760.9/s | 13,728.0/s |

**Previous reference: MTG-286 TLE rules.** Commit `7ef6ce588` was adopted from
[PR #469 CI run 43221435403](https://github.com/loopflowstudio/manabot/actions/runs/43221435403)
(2026-11-12 UTC, AMD EPYC 7763) as an accepted cost. MTG-286's TLE rules
additions (mana payment, casting, proposed events and triggers) leave every
search digest, the step digest and the 101,650-step count unchanged, but slow
headless steps by 9.2%. Both candidate step samples (8.79 s, 8.64 s) were slower
than both reference samples (7.98 s, 7.84 s), with user CPU time matching wall
time, so the gap is engine cost rather than scheduling. Search stayed within
budget.

| Metric | Previous reference `ea316ff14` | Adopted candidate |
|---|---:|---:|
| Worst search reply | 4.913 s | 4.944 s |
| Total search | 167.486 s | 169.186 s |
| Step throughput | 12,880.0/s | 11,693.6/s |

**Previous reference: MTG-239 rules.** Commit `ea316ff14` was adopted from
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

### Stress states

Engine Budget plays two curated decks, where no payment plan passes 12 search
calls, and random play rarely builds a board with many permanents. Cost that
grows with the size or shape of a board is measured separately, on constructed
states, by one harness that runs each state through every engine layer:

```bash
cargo run --release --features benchmark --example bench_states \
  --manifest-path managym/Cargo.toml -- diverse-mana
```

```json
{"state":"diverse-mana","size":8,"checkpoint":"priority","layer":"actions","work":{"plans":31,"calls":212,"states":154},"ms":0.41}
```

The line above illustrates the JSON-lines format, not a measurement. Each row
is one state, size, decision checkpoint and layer. Read down a layer's rows:
work that grows much faster than size is a cell to investigate. Without a name
the harness runs every state. Defaults are 4, 8 and 16 copies; `restricted-filter`
uses 2, 3 and 4 because its nested-filter search already grows sharply there.
`recorded` has one size. `--sizes 8,16` overrides constructed-state sizes
(1–64), `--layers actions,step` selects layers, and `--list` names the cards and
known expensive shapes. Do not increase a known expensive cell just to make it
run longer.

**States.** Each uses defined cards or registered tokens and is deterministic.
`--list` names every state with its cards and known growth; the list lives in
the harness, not here. They come in three families: many copies of one thing
(`duplicate-lands`, `duplicate-creatures`, `duplicate-tokens`), many different
things (`diverse-mana`, `restricted-filter`, `diverse-statics`, `attachments`,
`mixed`) and one mechanic at size (`herbalist`, `paid-filter`, `counted-tap`,
`waterbend-x`, `triggers`, `zones`, `choices`, `names`). Combat states put
creatures on both sides and measure the pass that resolves damage; payment
states isolate one player. `recorded` is the checkpoint after the 426 reported
actions of `managym/tests/fixtures/payment_search/manabot_slow_step.py`, kept
byte for byte: a board that real play reached.

**Layers.** Every checkpoint runs through the same list. Action steps with no
applicable action and characteristics reads without permanents print no row.
Preparation reaches combat, cast choices, trigger ordering, a deep stack and
naming through rules actions; it is outside the measured region. The recorded
closed-decklist checkpoint reports search as unavailable: determinization
requires open decklists, and the harness preserves the recording's disclosure.

| Layer | Measures | Work reported |
|---|---|---|
| `actions` | computing the legal action space at priority | payment plans, calls, states; actions offered |
| `step:cast`, `step:activate`, `step:attack`, `step:block`, `step:damage`, `step:choice` | one representative action of each kind, to the next decision | the same, plus trigger sources and matches |
| `characteristics` | cold `Game::characteristics` for one permanent then `characteristics_many` for the board | board rebuilds, views, dependency probes |
| `triggers` | matching one controlled entry event against the indexed trigger sources | sources examined, triggers found |
| `observe` | the native observation encoder | visible cards, permanents and legal actions (allocated capacities are explicit) |
| `vector` | a one-slot `VectorEnv` step into buffers | as `step` |
| `clone` | cloning the game with its caches | separately measured allocation traffic in bytes |
| `search` | one `flat_mc_scores` reply (4 worlds × 1 rollout, 4-step horizon per action) | simulations and cap hits |

*Work* counts are deterministic for the same features and build profile
(debug invariant probes perform additional work): compare them
between two commits anywhere. A payment *plan* is one payment request, a *call*
is one search entry, and a *state* is a call that passes the bounds and memo and
expands successors; the other counters are described in `benchmark_layers.rs`
and `benchmark_payments.rs`. They exist only in `benchmark` builds, never in
wheels. *ms* is the median of repeated runs and is comparable only between two
builds on one machine. The closing digest folds every state's legal actions and
first payment witnesses, so an optimization that changes behavior shows as a
changed digest beside its faster numbers.

The Python-side layers read the same states (the Rust harness prints their
recipes with `--recipes`) and report time only:

```bash
uv run --extra play python scripts/bench_states.py diverse-mana
```

It adds `encode-py` (the Python policy-input encoder) and `text` (the LLM text
renderer) time rows. `--recipes` exports the exact native observation and
atomic decision capture beside each recipe, so both consumers see the same
checkpoint, including separately serialized continuation frames via
`Observation.from_json(..., continuation_json=...)`. Text capture is outside the Python renderer timing. Academy frame
construction requires a live Academy session and is not measured here.
`--recipes-file <path>` reuses an export; `--layers encode-py` selects one
consumer. The script uses the installed extension, so rebuild after Rust changes.
A renderer stage it cannot represent prints `unavailable`, an audit finding
without a timing. Invalid captures still fail the run.
Both encoders use input v10 with capacities expanded to hold the complete state;
neither truncates actions or facts.

**Using it.** When a change adds or alters payment, choice, targeting,
enumeration or continuous-effect behavior, run the states whose shape it could
make expensive on the base commit and on the change, and put both tables in
the PR. A cell that grows much faster than its size is fixed there; when the fix is larger than the change, the state's `--list`
line names the growth so the next reader finds it measured rather than
rediscovers it.
Preserve legal actions, first payment witnesses and the `bench_steps` and
`bench_search` digests, and validate in debug: debug builds recompute cached
views and assert they match.

A mechanic that no state reaches gets a state: add a recipe to
`managym/examples/bench_states.rs` from defined cards, with the shape that
makes it expensive. A large constructed board is a workload, not induced load
([TESTING.md](../TESTING.md)); keep sizes small enough that the whole harness
finishes in about a minute, and stop a state at the size where its cost is
already clear.

The harness is not a CI job and asserts no ceiling. Nothing fails when a
number grows; the table is read by whoever changes the code under it.
`bench_payment` (`replay <script>` for per-step payment counts, `boards` for
its source mixes) and `bench_layers` remain for attributing one layer in
detail once a state has pointed at it.

## Releases

One release contains two PyPI distributions at the same `X.Y.Z` version:
`managym`, the native engine and its Python API, and `manabot`, the Python
training library and clients. `manabot` requires `managym==X.Y.Z`.
`import managym` owns the engine; `manabot.managym` remains a compatibility
alias when `manabot` is installed. One `vX.Y.Z` tag builds and publishes both;
neither package has an independent release cadence. The Python projects and
Rust workspace crates share that version, checked before building. The first
public release, before this split, is
[0.2.0](https://pypi.org/project/manabot/0.2.0/). Package versions do not replace
world, content, observation or checkpoint identities, and an upgrade does not
make old checkpoints compatible. Releases are research releases, not
certification of playing strength or rules coverage.

### Install

For an unpublished candidate, install the local wheels using
[Build and verify](#build-and-verify). The PyPI commands below apply after the
paired release is published.

For the rules engine alone, follow [installing managym](engine-python.md).
For training, install `manabot` in a fresh directory outside this checkout:

```bash
mkdir manabot-example
cd manabot-example
uv venv --python 3.13
uv pip install --python .venv/bin/python --only-binary :all: --torch-backend cpu manabot
uv run --no-project --python .venv/bin/python manabot --help
```

This installs the library and its exact-version engine dependency into `.venv`;
the last command lists CLI commands without starting training. No repository dependency groups
are installed. For CPython 3.12, create the environment with
`--python 3.12`; installation selects the matching engine wheel. Pin both
distributions to the same published version in reproducible projects. See the
[0.2.2 release notes](../release/v0.2.2/NOTES.md) for upgrading from the bundled
0.2.0/0.2.1 wheels, and the [0.2.1 release notes](../release/v0.2.1/NOTES.md)
for API migration guidance.
The historical 0.2.0 release is 3.12-only; install `'manabot==0.2.0'` in a 3.12
environment when reproducing that release.

For Linux NVIDIA training use `--torch-backend cu132`; always select the backend
explicitly. Torch 2.15 publishes no CUDA 12 wheels, so the CUDA 13.2 build needs
an NVIDIA driver from the 580 series or newer and a Turing or newer GPU
(compute capability 7.5+). CI validates CPU, not GPU, execution.

Supported: ordinary CPython 3.12 and 3.13 (no free-threaded ABI) on Linux x86-64 with glibc
2.28+ (CPU or CUDA 13.2 Torch) and Apple Silicon macOS 14+ (CPU or MPS Torch).
MPS is admitted, on explicit request, for float32 self-play training with Torch 2.15 or
newer. Baseline requires CUDA or an explicit `--device mps`; local selects CUDA when
available and CPU otherwise.
Evaluation and search run on CPU for this workflow. GPU-less tests use
`baseline_test()`, which shares baseline construction and changes only the device.
No speed or strength claim is implied. Windows,
Intel Mac, Linux ARM and musl are not supplied or tested.

The platform-specific `managym` wheels contain the compiled engine; its sdist
needs Rust and a C linker (`--only-binary :all:` refuses source builds).
`manabot` is a pure-Python wheel and gets the engine through its exact dependency.
The engine's runtime requirements are NumPy and Pydantic. `manabot` retains its
training requirements, including `torch>=2.15.0` and `etude-sdk>=0.2.16`.
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

For notebook work, use `'manabot[notebook]'` in the install command and retain
the resolved versions in your lockfile.
In a checkout, use `uv sync --locked --extra notebook`. If a command reports
a missing optional dependency, install its named extra in the same environment
and retry. Development groups are not installed as part of a downstream package.

`--regime-name`, experiment drivers and data-ledger tools need a
source checkout. Never change a running experiment's pinned source to install a
release.

### Build and verify

Before publishing a release PR, update both Python project versions, all three
Rust crate versions and the exact `managym` dependency together. Refresh
`uv.lock` with `uv lock` and `managym/Cargo.lock` with
`cargo metadata --manifest-path managym/Cargo.toml --format-version 1`, then run
the release metadata check and `tests/infra/test_release.py` command below.
These checks apply even when the release changes only versions and notes.

From a clean checkout, with a fresh staging path and empty output directory:

```bash
uv run --no-project python scripts/release/check.py
uv run --no-project python scripts/release/prepare.py .runs/release/staging
uv build --no-sources .runs/release/staging/managym --out-dir dist
uv build --no-sources .runs/release/staging --out-dir dist
uv run --no-project python scripts/release/validate.py dist
uv run --no-project --with twine twine check --strict dist/*
uv run --no-project --with pytest pytest --confcutdir=tests/infra tests/infra/test_release.py -q
```

Staging copies tracked files, refuses to overwrite a prior build and excludes
experiment drivers and build products; published artifacts must use it. The
engine staging project places its crate and shared content beneath one root,
so the source archive preserves Rust asset paths without the checkout. Staged
metadata must preserve the source project's public minimum requirements rather
than replacing them with development pins. CI builds each distribution's wheels
from its source archive, so checkout files cannot hide missing content. The
complete release inventory is seven files at one version: four `managym` wheels
(3.12 and 3.13 on each supported platform), one pure-Python `manabot` wheel and
two source archives. Validate both inventories and the exact `manabot`
dependency before either upload.

Clean-install proof, replacing the filename placeholders with built artifacts:

```bash
uv venv --python 3.13 /tmp/manabot-release-venv
uv pip install --python /tmp/manabot-release-venv/bin/python --only-binary :all: \
  "$PWD/dist/managym-<version>-cp313-cp313-<platform>.whl"
```

From outside the checkout, run the [engine example](engine-python.md) with
that environment and confirm Torch, Gymnasium and Etude were not installed.
CI additionally completes a game in the engine-only environment, without a
Rust toolchain. Then, from the build directory, add training with both wheel
paths so resolution uses the candidate engine rather than PyPI:

```bash
uv pip install --python /tmp/manabot-release-venv/bin/python --only-binary :all: \
  --torch-backend cpu "$PWD/dist/managym-<version>-cp313-cp313-<platform>.whl" \
  "$PWD/dist/manabot-<version>-py3-none-any.whl"
cp scripts/release/smoke.py /tmp/manabot-release-smoke.py
cd /tmp
export PATH="/tmp/manabot-release-venv/bin:$PATH"
unset PYTHONPATH
uv run --no-project --python /tmp/manabot-release-venv/bin/python manabot-release-smoke.py
```

The training smoke checks one shared engine behind both import paths, catalog
binding, gradients and a tiny CPU training update. Repeat in a separate environment with
3.12 and its wheel. `release.yml` covers both interpreters on Linux and macOS,
then installs and imports the optional extras. Consumer checks run outside the
checkout with default dependency groups disabled, including the minimum
Torch/Etude versions and available newer versions where practical. Record exact
versions tested; successful resolution alone is not a runtime compatibility
check.

### Publish

`release.yml` builds and validates relevant PRs, manual runs and `v*` tags; only
tags publish. Tag builds reject inconsistent versions and commits not on main.
A release is complete only when both packages' full PyPI inventories and the
matching GitHub Release exist. The GitHub entry uses the same `vX.Y.Z` tag and
that version's checked-in `release/vX.Y.Z/NOTES.md`, with links to both packages
at that version on PyPI.
Only the PyPI job gets OIDC `id-token: write`; only the GitHub Release job gets
`contents: write`. No PyPI token is stored, and uploads carry attestations.
Before the first paired release, the PyPI account owner configures two
publishers with these exact fields:

| Field | Engine | Training |
|---|---|---|
| PyPI project | `managym` | `manabot` |
| Repository owner | `loopflowstudio` | `loopflowstudio` |
| Repository name | `manabot` | `manabot` |
| Workflow filename | `release.yml` | `release.yml` |
| Environment | `pypi` | `pypi` |

For `managym`, use account **Publishing** to add a pending publisher if the
project does not exist. This does not reserve the name; the first successful
upload creates the project. See [PyPI's pending publisher instructions](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/).
For the existing `manabot` project, verify its publisher under **Manage →
Publishing**, following [PyPI's existing-project instructions](https://docs.pypi.org/trusted-publishers/adding-a-publisher/).
The GitHub `pypi` environment restricts tags to `v*`; verify its required
reviewers and both publisher identities before tagging. Configuring a publisher
does not authorize a release.

Per release:

1. Align both Python project versions, the Rust crate versions and lockfiles,
   and the exact `manabot` requirement on `managym`. Write one
   `release/vX.Y.Z/NOTES.md` covering both packages and update install examples
   that pin versions. Merge the reviewed release PR and confirm its checks.
2. With explicit publication authorization, tag that main commit `vX.Y.Z`.
   Review the retained seven-file inventory and the engine-only and combined
   consumer checks before approving the `pypi` deployment.
3. Publish `managym` first and confirm its full inventory, then publish
   `manabot` from the same run. Both builds, version checks and validations
   must pass before either upload begins. If either package cannot pass those
   checks, publish neither.
4. Verify both PyPI inventories, attestations and the GitHub Release. Install
   from PyPI in clean environments on both platforms and interpreters: first
   the engine alone, then the training package at the same version.

A PR alone does not authorize a tag or upload. Uploads to two projects are not
one atomic operation: a failure after the engine upload can leave an incomplete
paired release. Do not report success or advance one package to another version
to hide that partial state.

If PyPI succeeds but GitHub completion fails, retry the failed completion job
for the same tag. It creates or verifies the matching entry without uploading
again. An entry with matching tag and notes makes the retry a no-op. Retrying
the PyPI job checks each project's inventory and SHA-256 digests against retained
build artifacts and skips an already complete package only when they match.
If `managym` completed and `manabot` has no files yet, verify the engine's
digests, then resume the missing training upload with the original artifacts.
Partial files within either package, or different bytes, stop for inspection.
Rebuilding can change archive bytes; retry failed jobs with the original artifacts instead of rebuilding
a published version.
Never overwrite PyPI files, delete a release or move a used tag to repair it.

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
