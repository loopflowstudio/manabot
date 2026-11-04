# Publication verification — 5da5f4

The accepted publication scope is the explicit v7 shared-effect repair. Main
`3b122de9078c985f846a0f3c5a3a42ebca3b1182` is the actual base and was integrated
by `lf sync`. Fresh defaults, frozen v1–v6 inputs and scientific protocols remain
unchanged. This step publishes for review; it does not merge or complete the Task.

## Checks and repairs

- Complete debug Rust workspace: 909 passed, 2 ignored, no failures, with
  `CARGO_PROFILE_DEV_OPT_LEVEL=1 cargo test -j 2`.
- Broad Python selection: 454 passed, 1 skipped, 2 stale text-fixture failures.
  This includes all model tests, transport/parity, active training, experiment
  discovery, package boundaries, artifact tracking, and the required current-world
  arena/admission/monitoring selection. The text failures expected the old field
  order and injected legacy effects without supplying the new shared authority.
  Updated fixtures retain exact assertions and cover both current and historical
  captures. Full text/CI-selection/release recheck: 107 passed.
- Added shared-effect transport, source-duration parity and study-summary tests
  to the existing integration CI command; they previously were not selected.
  That exact CI invocation collects 953 tests without collisions. An earlier
  partial collection lacked the llm extra; installing the declared extras fixed it.
- Clippy all targets/features passes with warnings denied. Initial interpreter
  lookup required the absolute pinned interpreter path. Replaced constant-sized
  chunk iteration with typed array chunks as required by Clippy, and formatted
  the two native test files missed by earlier focused checks.
- Rebuilt the native release wheel from the root and atomically replaced the
  cp312 extension. Post-rebuild transport/model checks: 23 passed, including
  native scalar/vector/Python parity, retained 140/270 cases, gradients,
  checkpoint reload and ordinary learner/export.
- Ruff passes for the new modules and changed text tests; `git diff --check`
  passes. All four focused debug native shared-effect tests also pass after the
  Clippy-only edit.

Logs live under `.runs/representation/publish-*`. Existing published engineering
receipts retain their original source hashes; these local checks do not rewrite
them. No new paid compute was launched.

## Review limits

Hosted CI, including same-runner Engine Budget, remains pending until publication.
Publication does not establish performance-budget acceptance or authorize a merge
with failing checks. No v7 CUDA run, replicated learning comparison, strength
claim, universal capacity guarantee, memory implementation or semantic timestamp
coalescing is included. The pre-existing keyword-order defect remains documented
in `scratch/questions.md`. V7 has smaller active effect payloads for the retained
cases but larger padded buffers; do not claim a memory or throughput win.

## Gate review after demonstration — 2026-11-04

Comparison remains base `3b122de90` through published head `91635f583`, plus
the untracked demonstration note. No production source changed after publication.
The branch review found one test weakness: Ty Lee's expiry assertion compared a
public object ID with an internal permanent index. It now requires the captured
complete effect payload (including timestamp) to disappear. This also catches a
stale record whose departed source no longer has a current public object join.

Validation plan, derived before execution:

- `managym/src/agent/{observation,observation_encoder,shared_effects}.rs`,
  Python scalar/vector bindings and native observability/rules tests reach debug
  observation, continuous-effect, card and transport coverage. Retain the prior
  full debug result and current-head hosted Rust/Engine Budget passes. Rerun only
  the modified Ty Lee test; run Rust formatting and Clippy after that test.
- `manabot/env/{observation,shared_effects}.py`, config, model encoders,
  rollout/collection/search, representation fixtures and their Python tests reach
  parity, gradients, permutations, checkpoint reload, learner/export, frozen
  admission and world/monitoring checks. Their content is unchanged; retain the
  454-pass initial selection (with both text failures repaired in the 107-pass
  recheck), 23-pass rebuilt transport/model proof and 23-pass demo. Hosted main
  integration's first test step also passed on `91635f583`.
- `experiments/state_representation.py` and its tests reach discovery and
  scientific cohort validation; the manifest reaches artifact tracking. Retain
  their passing publication checks. No paid experiment or evidence rewrite.
- `.github/workflows/manabot-integration.yml` reaches CI selection and combined
  test collection; retain the 953-test collection and CI-selection/release pass.
  The repository's changed-aware runner is `.github/ci/select-tests.sh`, a CI job
  classifier rather than a local result cache. Use it to verify job coverage;
  hosted CI owns the complete matrix. No new frontend behavior needs UI proof.
- README, package READMEs, architecture HTML, agent guidance and scratch notes
  need contract review and `git diff --check`. Retained calibration and Wave
  memory are evidence; this gate does not modify them.

The native release wheel is rebuilt from the root and installed atomically
as required by AGENTS.md. Production sources are unchanged, but the rebuilt
binary hash differs, so rerun the 23 transport/model checks once against the
installed binary. Hosted checks apply to the published
head until delivery pushes this test correction; they are not claimed for a
future commit.

Final gate results:

- `git diff --no-renames --name-only -z 3b122de90 | bash .github/ci/select-tests.sh` — selects all nine affected CI categories; the current workflow covers them.
- `CARGO_PROFILE_DEV_OPT_LEVEL=1 cargo test -j 2 --manifest-path managym/Cargo.toml -p managym-cards --test cards ty_lee_chi_blocker_locks_a_creature_down_while_you_control_her` — 1 passed, 511 filtered; final assertion checked (`gate-ty-lee-final.log`).
- `cargo fmt --manifest-path managym/Cargo.toml --check` — passed.
- `PYO3_PYTHON=/Users/jack/src/manabot.5da5f4-design-effective-state/.venv/bin/python CARGO_PROFILE_DEV_OPT_LEVEL=1 cargo clippy -j 2 --manifest-path managym/Cargo.toml --all-targets --all-features -- -D warnings` — passed (`gate-clippy-final.log`).
- `CARGO_BUILD_JOBS=2 uv run --extra play maturin build --release -i .venv/bin/python -m managym/Cargo.toml -o managym/target/wheels` — passed in 34.94 seconds; cp312 extension atomically replaced, SHA256 `e0ac3baebbabdbcde1a0f6f0dc02a317a29af2ca20f89d4c1675de5a04ac34e6`.
- `uv run --extra dev --extra artifacts pytest tests/verify/test_shared_effect_transport.py tests/model/test_shared_effects.py -q` — 23 passed in 5.33 seconds against the rebuilt extension (`gate-rebuilt-transport.log`).
- `git diff --check` — passed.
- `gh pr view 419 --json headRefOid,statusCheckRollup` — published head `91635f583` has no failing checks; main integration remains in progress in its two-seat/Etude evaluation step, after the main Python integration step passed. The remaining hosted job is deferred to CI/delivery, not marked passed.

Gate is complete locally. Delivery owns publishing the test correction and
waiting for required CI on that head before landing PR 419. No check timed out
in this gate; no local server, training process or paid allocation remains.
Learning-quality and v7 CUDA claims remain outside this acceptance.
