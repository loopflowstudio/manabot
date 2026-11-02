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

The root project is both the development environment and the single `manabot`
distribution. Python is CPython 3.12; every Python command runs through uv
(`uv run ...`). `uv sync` builds the bundled extension; CI and existing scripts
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

## Validation by change area

Polish and validate only the current branch scope. **NEVER intentionally induce
load for testing**; follow [TESTING.md](../TESTING.md) for resources and
subprocesses, including benchmarks, debugging and demos.

**Rust managym** (`managym/Cargo.toml`, `managym/src/**`):

```bash
cd managym
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test
```

CI tests in debug with `CARGO_PROFILE_DEV_OPT_LEVEL=1`, which keeps debug
assertions and overflow checks; match it locally (release-only passes are not
enough) and use `-j 2` on shared hosts. Keep one Cargo operation per build
directory. `--all-features` builds the PyO3 bindings; if the default `python3`
is newer than PyO3 supports, set `PYO3_PYTHON=../.venv/bin/python`. A failure of
`cargo test --all-features` for missing Python dev/link symbols is environment
setup, not engine correctness. Record unavailable tooling in the gate summary.

**Authored rules, worlds and setups.** Run the complete debug Rust suite,
including semantic-kernel conformance, and focused browser interactions. Also:

- `uv run --python 3.12 --extra dev pytest tests/semantic/test_elemental_world.py -q`
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
must not import DuckDB). When moving modules, tests or entry points, collect the
exact paths in `.github/workflows/manabot-integration.yml` and run
`uv run --extra dev pytest tests/infra/test_ci_selection.py tests/infra/test_release.py -q`.
Training CLI changes need the installed-wheel smoke from
[Build and verify](#build-and-verify); a source-tree smoke cannot verify the
installed entry point.

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
`uv run --python 3.12 --extra test pytest tests/ -q`. Keep coverage for
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
| Rust Tests | debug `cargo test` in `managym/`, including `semantic_conformance_tests` |
| Python Unit Tests | `tests/infra/`, clean-machine verifier, `deploy/play/tests/`, website |
| Integration | `manabot-integration.yml` (`main` and `long` groups, Perf report proof, training smoke) |
| Academy Integration | `academy-integration.yml` (world bindings, Elemental admission and play) |
| Protocol v1 Conformance | Rust, Python and TypeScript experience protocol; launcher contracts |
| Semantic Kernel Conformance | bounded valid-action fuzzing only, over `semantic-kernel-w17-v1` |
| Academy UI | `npm test`, `npm run check`, `npm run test:e2e:release` |
| Clean-machine Play | `./scripts/verify-clean-machine` |

Semantic-kernel conformance is recorded as manifest seeds plus a per-case
`trace_sha256` in `evidence.json` ([conformance](../conformance/README.md)). A
drifted case is written to `target/semantic-conformance/<corpus>/`, which Rust
Tests uploads on failure; fuzz failures keep exact seeds under
`target/semantic-conformance/failures/`. Rust Tests runs whenever the fuzzing
job does.

**Selection.** Detect Changes pipes the NUL-delimited diff (`--no-renames`, so
both sides of a rename) into `.github/ci/select-tests.sh`. Manual dispatch, an
unknown base, an empty diff or an unknown path selects everything. Tests Result
fails if detection failed, a selected job did not succeed, or an unselected job
neither skipped nor succeeded.

| Change | Selected jobs |
|---|---|
| Native engine, `content/`, `conformance/`, shared Python/Cargo dependencies, CI selector or `ci.yml` | All |
| `managym/tests/` | Rust, semantic conformance |
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
| `website/`, `tests/infra/`, rules audit | Python unit |
| Docs, Markdown, `.lf/`, `paper/`, `scratch/`, other workflows | Detection and aggregate only |

Runtime inputs such as experiment Markdown and benchmark data are not prose-only.
`tests/infra/test_ci_selection.py` covers these boundaries and the aggregate;
update both when adding a package or consumer.

**Other workflows.** `macos-nightly.yml` reruns both integration suites on
`macos-latest` nightly. `baseline.yml` trains the baseline recipe nightly on
seeds 197, 281 and 419 and combines them with `manabot.verify.learning_regression`.
`card-evaluation.yml` (manual) verifies 1024 native games with
`manabot.eval.cards verify` (see [evaluation](evaluation.md)).
`release.yml` is described next.

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

```bash
uv venv --python 3.12
uv pip install --python .venv/bin/python --torch-backend cpu 'manabot==0.2.0'
uv run --no-project manabot --help
```

For Linux NVIDIA training use `--torch-backend cu128`; always select the backend
explicitly. CI validates CPU, not GPU, execution.

Supported: CPython 3.12 (no free-threaded ABI) on Linux x86-64 with glibc
2.28+ (CPU or CUDA 12.8 Torch) and Apple Silicon macOS 14+ (CPU Torch). Windows,
Intel Mac, Linux ARM and musl are not supplied or tested.

Wheels contain the compiled engine; the sdist needs Rust and a C linker
(`--only-binary :all:` refuses source builds). Extras: `academy` (backend only;
use `./scripts/play` for the browser), `evaluation`, `reports`, `artifacts`,
`notebook`. `--regime-name`, experiment drivers and data-ledger tools need a
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

Staging copies tracked files, refuses to overwrite a prior build, excludes
experiment drivers and build products, and adds the runtime requirement from
`[tool.manabot.release]`; published artifacts must use it. Wheels are built from
the sdist, so checkout files cannot hide missing content. Clean-install proof:

```bash
uv venv --python 3.12 /tmp/manabot-release-venv
uv pip install --python /tmp/manabot-release-venv/bin/python --only-binary :all: \
  --torch-backend cpu "$PWD"/dist/manabot-*.whl
cp scripts/release/smoke.py /tmp/manabot-release-smoke.py
cd /tmp
export PATH="/tmp/manabot-release-venv/bin:$PATH"
unset PYTHONPATH
uv run --no-project --python /tmp/manabot-release-venv/bin/python manabot-release-smoke.py
```

The smoke checks one shared engine behind both import paths, catalog binding,
gradients and a tiny CPU training update. `release.yml` repeats it on Linux and
macOS, then installs and imports the optional extras.

### Publish

`release.yml` builds and validates relevant PRs, manual runs and `v*` tags; only
tags publish. Tag builds reject inconsistent versions, commits not on main and
already-published versions. Only the publish job gets OIDC `id-token: write`;
no PyPI token is stored, and uploads carry attestations. The trusted publisher
is project `manabot`, owner `loopflowstudio`, repository `manabot`, workflow
`release.yml`, environment `pypi`. The pending publisher submitted by Jack Heart
was used successfully on 2026-10-22; the GitHub `pypi` environment restricts
deployment to `v*` tags and requires review. Verify these settings before
tagging.

Per release: update `pyproject.toml`, Cargo.toml/Cargo.lock, `uv.lock`, examples
and notes; merge the reviewed release PR; confirm the workflow is green; push
`vX.Y.Z` at that main commit; review the `pypi` job's artifacts and approve it.
A PR alone does not authorize a tag or upload. `skip-existing` is disabled: after
an interrupted upload, inspect what reached PyPI and prepare a new patch version
if needed; never retry blindly, delete a release or move a used tag. Afterwards,
install from PyPI on both targets and confirm version and attestations.

0.2.0 was published on 2026-10-22 from `v0.2.0` at
`29ab4fa0ddcc759ddb2b699ce3d08ccfe51ad757`
([PR #351](https://github.com/loopflowstudio/manabot/pull/351)); attestations
verified against the repository; PyPI holds two wheels and the sdist. The full
record, with SHA-256 digests, is in the
[pinned release notes](https://github.com/loopflowstudio/manabot/blob/a74f31828/docs/releases.md#020-publication-record).

## Clean-machine play proof

From a fresh checkout, `./scripts/play` syncs the locked play runtime, builds the
CPython 3.12 extension, runs `npm ci`, starts FastAPI and Vite, validates the
installed curated pack and prints an `ACADEMY_PLAY_READY` JSON record and the
URL. `./scripts/verify-clean-machine` proves this once: it runs the launcher,
runs `academy/ui/e2e/clean-machine.spec.ts` against it, writes
`.cache/clean-machine/proof.json` and tears the process group down.

### Reference profile

GitHub-hosted `ubuntu-24.04` x86_64 with uv 0.9.22 and CPython 3.12, Node 22,
stable Rust and native build tools, and Playwright 1.61.1's Chromium. The receipt
records the observed OS, CPU, memory and tool versions. Before timing, the
checkout must not contain `.venv`, `.cache`, `academy/ui/node_modules`, generated
Svelte/Vite/build/test output, `managym/target`, a built `_managym*.so`, GUI trace
JSON or a prior proof browser profile; no uv, npm or Cargo cache is restored.

### Time to playable

The clock starts just before `./scripts/play` is spawned and stops when a fresh
Chromium context shows a connected WebSocket, the default Search 64 opponent,
`UR Lessons vs GW Allies`, an enabled authoritative legal action, and pack
treatments for every visible card with no fallback. The budget is 90,000 ms. It
was 60,000 ms until 2026-09-30; the release native build alone had reached about
52 s on GitHub's runner, so Jack Heart accepted 90 s. The receipt records
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
