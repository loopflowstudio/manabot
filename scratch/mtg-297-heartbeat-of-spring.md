# MTG-297: Heartbeat, Animal Attendant, Toph, Uncle's Musings

Contract and completed deletion list: `tle-production-design.md`. Verbatim
review answers and coordination: `heartbeat-design.md`. Explicit limitations
and fixture choices: `questions.md`. Prior implementation, QA, corpus and
verification receipts are preserved at
`39784f2c3:scratch/mtg-297-heartbeat-of-spring.md`; older implementation detail
is at `8d2721ab9:scratch/mtg-297-heartbeat-of-spring.md`.

## Outcome

All four cards are defined with Scryfall ids, directed behavior, observability
and learning tokens. Production and spending are implemented on MTG-242's
`ManaRestriction` / `SpendingContext`. Badgermole uses the production transaction;
Henge/Citadel retain their source-owned suffix behavior. Native spending preserves
color, restriction, rider, expiry and saved callers; receipts exclude nested
activation fees. Existing fact kinds 11/14 project pools, offers and receipts in
v9/v10. No new input shape, model kind or automatic payment preference was added.

The corpus admits 998/1,000 events, including all three Heartbeat events. Only
the two Lita events fail; `docs/engine.md` matches. Frozen evidence is unchanged.
Main's MTG-242 pool types, #504 search bounds, #506 sharding, CPU CI selection,
v0.2.3 preparation and `6ce618b8a` payment fast paths are incorporated.
The latest merge is `b274d4c5f`; `8888428c4` checkpoints the subsequent
saved-caller simplification. Focused validation covers both revisions below.
The unrelated CI receipts at
`3cb9cc70a:scratch/ci-under-8-minutes.md` do not certify this mana change.
The superseded intermediate payment diagnostic is retained at
`3cb9cc70a:scratch/payment-fixture-investigation.md`. No Wave is selected.

Earlier stable affected debug coverage passed 643 tests; Clippy, native/Python
shared input, current-world arena/monitoring, model/belief, protocol and frontend
checks are recorded in the archived plan. Headless Academy QA confirmed spending,
reconnect, Musings and both-seat Heartbeat play (`scratch/qa.html`). Automatic
production over-tapping also occurs on main and was filed as MTG-342. No human
demo or approval occurred. The fixed multi-type-source and tap-predicate-snapshot
limitations remain explicit in `questions.md`.

## Autoreset investigation (2026-11-12)

Publication had stopped after two vector fixtures exceeded 300 seconds. Jack
Heart's steer, verbatim:

> For the two vector_env timeouts: first time those two tests on origin/main vs this branch in debug, one at a time (no added load), and record both numbers. If the branch is materially slower, find and fix the engine cost (likely spending-choice search per step) rather than splitting the fixtures; if main is equally slow, split them without weakening contracts. Also run the Engine Budget comparison before publishing.

Compared unchanged fixtures at branch `39784f2c3` and `origin/main` `054e987a0`,
sequentially with `cargo nextest run --profile ci --build-jobs 2 --test-threads 1
--lib -E 'test(=<exact name>)'` and the appropriate manifest. Main was exported
to `/tmp/mtg297-main-baseline`; no new Task or managed checkout was created.
Compilation is excluded from these test durations.

| Fixture | Main | Branch |
| --- | ---: | ---: |
| `masked_mixed_setups_preserve_scalar_autoreset` | 11.571s pass | initially 300.052s timeout; uninstrumented final run 5.528s pass |
| `restored_game_seeds_preserve_future_autoreset_stride` | 10.958s pass | 10.834s pass |

Logs: `/tmp/mtg297-{main,branch}-{masked,restored}.log`; final masked timing:
`/tmp/mtg297-branch-masked-final-timing.log`, run
`abd52d9d-20e5-47ce-b812-58818739468c` (whole command 6.11s wall, 4.37s user,
0.23s system).

The apparent masked regression did not survive execution without suspension.
A diagnostic process was observed in state `T` (stopped), CPU time 4.17s after
2m30s elapsed. Its sampled stacks were stationary, first in payment search,
then layer/subtype lookup. A temporary 1,000-state search guard never fired;
that run passed after resuming its owned process. The guard was removed entirely.
A subsequent **uninstrumented** process was also stopped (`T`, 0.68s CPU after
42s elapsed), then passed after resume. Several other owned tests in the full
run were similarly stopped and resumed. The source of these external stops is
unknown; those elapsed times are not engine performance measurements. No load
was added beyond the sequential builds/tests. Both original fixtures, all
1,000 steps, their assertions and the five-minute watchdog remain unchanged.

## Completed fixture repairs and debug coverage before the latest merge

The complete debug run at unchanged runtime `39784f2c3` exposed 37 more funded
card fixtures outside the earlier 643-test selection:

```sh
cargo nextest run --manifest-path managym/Cargo.toml --profile ci --build-jobs 2 --test-threads 2
```

Run `e1333716-1ad5-4db2-8522-638c98993a81`: **1,182 passed, 37 failed, 2 skipped**,
821.085s; `/tmp/mtg297-final-debug.log`. No timeout. The original vector fixtures
passed in 12.194s / 9.740s, and semantic conformance passed in 168.283s.

Nineteen set-test files now select native payments explicitly before inspecting
committed costs, cast triggers or post-cast hand counts. Kellan preserves red
for its second upgrade; Sailor completes its activation before its removal
response. Abandon Attachments chooses blue for its hybrid pip, then ordinary
colorless for its generic pip. The existing `spend_named` / `spend_colorless`
helpers suffice. No assertion, production code, vector fixture, timeout, or
shared casting/priority/target helper changed.

The exact failing selection passed 36/37; after the explicit hybrid choice,
the prowess witness passed (`175352c5-2bf5-4361-ac41-559790daa727`). Final
**70/70 tests in all 19 affected set modules passed** in 1.448s, run
`e1ecb506-8f53-410b-9151-078c5dcbadbc`,
`/tmp/mtg297-affected-card-sets.log`. This includes every repaired failure.
Combined with the full run's unchanged passing tests, all 1,219 selected tests
had passing debug evidence before the subsequent runtime changes. This is
two-run coverage, not a claimed single all-green final full-suite run; only
test fixtures changed between the runs.

- Workspace `cargo fmt --all --check` and `git diff --check` passed.
- `cargo clippy --manifest-path managym/Cargo.toml --jobs 2 --all-targets --all-features -- -D warnings` passed; `/tmp/mtg297-fixture-clippy.log`.
- Root `uv run --extra play maturin build --release -i .venv/bin/python -m managym/Cargo.toml -o managym/target/wheels` passed. Installed the matching cp313 arm64 extension atomically and verified its bytes against the wheel; `/tmp/mtg297-fixture-wheel.log`.

## Engine Budget (local paired run, 2026-11-12)

Both release builds succeeded, using the candidate's benchmark sources against
pinned reference `7ef6ce5884782000eb8f14934dd8c2bd48a05387`. Candidate runtime is
unchanged from `39784f2c3`; the uncommitted delta at measurement was fixtures and
scratch only. Both binaries ran on the same host, in
candidate/reference/reference/candidate order. No own
build or test ran alongside them. Native cp313 import also passed after rebuild.

```sh
uv run --no-project --python 3.13 python scripts/engine_budget.py sample --candidate managym/target/release/examples --reference /tmp/mtg297-budget-reference/managym/target/release/examples --artifact .cache/engine-budget-mtg297-autoreset
uv run --no-project --python 3.13 python scripts/engine_budget.py compare --artifact .cache/engine-budget-mtg297-autoreset --reference .cache/engine-budget-mtg297-autoreset/reference
```

Sampling completed; comparison **failed seven checks**. All reference digests
match the committed baseline. Candidate search digests differ for all three
seeds; step count changes from 101,650 to 102,209 and its digest changes.
Median worst search reply is 12.975s versus reference 6.844s; search total
238.018s versus 96.808s, both over the unchanged 10% tolerance. The median
step-throughput check passes (9,753.8/s versus 4,864.8/s).

| Pass | Steps wall / user CPU (s) | Search wall / user CPU (s) |
| --- | ---: | ---: |
| 0 candidate | 17.761 / 8.046 | 396.741 / 155.886 |
| 1 reference | 38.912 / 7.360 | 118.404 / 82.047 |
| 2 reference | 3.574 / 3.374 | 76.282 / 75.805 |
| 3 candidate | 4.054 / 3.833 | 79.947 / 79.356 |

The same binaries vary substantially within this run. Candidate search has
330,981 involuntary switches in its first sample versus 33,807 in its second;
reference steps vary more than tenfold in elapsed time. This evidence prevents
attributing the full median timing gap to engine work alone, but **does not
turn the failed check into a pass**. No adaptive retry, tolerance change,
performance acceptance or reference update was made.

Raw samples, aggregation, CPU/scheduling data, provenance and comparison:
`.cache/engine-budget-mtg297-autoreset/`; log `/tmp/mtg297-budget-sample.log`.
The local recording is not a hosted PR artifact for baseline adoption.

## Compression (2026-11-12, after `3cb9cc70a`)

`PaymentModel::witness` now answers a fully pooled final payment with
`Mana::can_pay`: no saved caller consumes its hypothetical remainder. Earlier
payments still search unit assignments, and execution still offers the native
spending choices. The zero-state regression covers hybrid backtracking, rider
and combat mana under all source policies. The existing class-bound witness now
has two obligations so it continues exercising that bound. Foretell reuses
`spend_named` instead of duplicating offer selection.

- `cargo nextest run --manifest-path managym/Cargo.toml --profile ci --build-jobs 2 --test-threads 2 -E 'test(flow::mana::tests::) or test(tla::sozins_comet_foretell_spending_choices_preserve_color_and_private_identity) or test(chk::) or test(tle::) or test(roe::)'` — **79 passed**, 7.339s; `/tmp/mtg297-compress-debug.log`.
- Separate `--lib --test-threads 1 -E 'test(=<exact name>)'` runs — original masked autoreset **passed, 6.274s**; original restored-seed stride **passed, 4.040s**. Both retain 1,000 steps and the CI watchdog; `/tmp/mtg297-compress-{masked,restored}.log`.
- `cargo clippy --manifest-path managym/Cargo.toml --jobs 2 --all-targets --all-features -- -D warnings` — passed; `/tmp/mtg297-compress-clippy.log`.
- Root `uv run --extra play maturin build --release -i .venv/bin/python -m managym/Cargo.toml -o managym/target/wheels` — passed; matching cp313 arm64 extension installed atomically, bytes checked against the wheel, native import passed; `/tmp/mtg297-compress-wheel.log`.
- `uv run --extra dev pytest tests/semantic/test_pooled_mana_classes_input.py -q` — **4 passed**, 0.54s; `/tmp/mtg297-compress-parity.log`.
- Workspace `cargo fmt --all --check` and `git diff --check` — passed. `lf context` — scratch below both configured limits.

Removed the superseded diagnostic and unrelated inherited CI plan from active
scratch; their exact history references are above. No human demo occurred.

## Upstream reconciliation at `b274d4c5f`

The merge adds `6ce618b8a`'s direct fixed-source payment calculation and shared
source scans to the spending redesign. In `managym/src/flow/mana.rs`,
`PaymentModel::direct` admits one mana cost, optionally followed by a generic
waterbend split of the same purpose. Searched abilities, counted taps, pending
outputs and the automatic-production prefix retain the finite-resource search.
`plan` and `payable` compare direct answers with that search in debug builds;
`PaymentScan` shares sources only within its immutable game borrow and checks
reused sources in debug builds. The merged direct witness ends at
`PaymentStep::Spend`, preserving native unit selection rather than spending a
hypothetical remainder. The fully pooled final-cost shortcut remains in
`PaymentModel::witness`.

The same merge incorporates `ef28f3a58`'s training/backend reorganization;
model and checkpoint checks must use the current tree's paths. It does not
change either autoreset fixture, the watchdog or the Engine Budget baseline.
The earlier timeout feedback is superseded by the completed investigation and
passing original fixtures above; it is not evidence that they still need a
split. The pre-merge 79-test selection, two autoreset runs and Python parity
receipt do not validate the newly combined direct/search paths.

## Focused merged-runtime validation (2026-11-12)

Runtime revision `b274d4c5f05fcb696abbe4921b78d2e10bb27590`; only scratch
notes differed during these checks. No runtime or fixture repair was needed.

- `cargo nextest run --manifest-path managym/Cargo.toml --profile ci --build-jobs 2 --test-threads 2 -E 'test(flow::mana::tests::) or test(chk::) or test(tle::) or test(cr_106_mana::) or test(waterbend) or test(nested_mana) or test(combat_mana) or test(badgermole) or test(henge) or test(citadel)'` — **106 passed**, 10.525s; run `457ba04f-b434-4ca6-8aed-ac1ad6ff9e63`, `/tmp/mtg297-merged-debug.log`. Covers direct/search answers and witnesses, shared-source lifetime, nested payments and fees, rider/expiry choices, waterbend and production/suffix behavior.
- `cargo nextest run --manifest-path managym/Cargo.toml --profile ci --build-jobs 2 --test-threads 1 --lib -E 'test(=agent::vector_env::tests::masked_mixed_setups_preserve_scalar_autoreset)'` — **passed, 37.746s**; run `9c6741eb-7c61-4795-94a8-f6e90ddf781a`, `/tmp/mtg297-merged-masked.log`.
- The same command selecting `agent::vector_env::tests::restored_game_seeds_preserve_future_autoreset_stride` — **passed, 15.766s**; run `33d7534e-032d-43eb-8353-432d60c3fd27`, `/tmp/mtg297-merged-restored.log`. Both original fixtures retain 1,000 steps and the unchanged CI watchdog; these durations are correctness receipts, not a paired Engine Budget result.
- `CARGO_BUILD_JOBS=2 uv run --extra play maturin build --release -i .venv/bin/python -m managym/Cargo.toml -o managym/target/wheels` — **passed**, 1m52s; `/tmp/mtg297-merged-wheel.log`. Installed the matching cp313 arm64 extension atomically, verified its bytes against the wheel (SHA256 `98868c0bcc32a0edb1809985db3228ce5425a72b8771fb00e37650b7d103f756`), and imported the native w17 runtime successfully.
- `uv run --extra dev pytest tests/semantic/test_pooled_mana_classes_input.py tests/semantic/test_restricted_mana_input.py tests/verify/test_observation_parity.py -q` — **33 passed**, 161.68s; `/tmp/mtg297-merged-parity.log`. Includes v9/v10 scalar/vector/text spending and receipt parity, restriction rejection, supplied capacities and the existing 200-game native/reference encoder comparison.
- `git diff --check` — passed. `lf context` — scratch remains below both configured limits. No code, fixture, timeout, baseline or frozen evidence changed in this pass.

## Compression of saved-caller ownership (2026-11-12)

Built on `b274d4c5f` and now committed in `8888428c4`,
`managym/src/flow/mana.rs` blob
`3c3ed8afab464a9e2327fbcab743eb02b7ba6cba`:
`complete_mana_upfront` consumes the saved caller directly, removing the
temporary game-state restoration, second decision extraction and infallible
`Result`. The source reference is captured once for both output and suffix.
`mana_spend_choices` asks `PaymentModel::payable` when no witness is consumed.
The production transaction, choices and serialized state retain their contract.
The surrounding callers, fixtures, projections and docs need no further cut.

- `cargo nextest run --manifest-path managym/Cargo.toml --profile ci --build-jobs 2 --test-threads 2 -E 'test(flow::mana::tests::) or test(chk::) or test(tle::) or test(roe::) or test(cr_106_mana::) or test(waterbend) or test(nested_mana) or test(combat_mana) or test(badgermole) or test(henge) or test(citadel) or test(cost_effect_completion_retains_output)'` — **121 passed**, 32.672s; run `86947d90-fc79-4e60-bcf3-8ba61eab26bf`, `/tmp/mtg297-compress-caller-debug.log`.
- The same command with `--lib --test-threads 1 -E 'test(=agent::vector_env::tests::masked_mixed_setups_preserve_scalar_autoreset) or test(=agent::vector_env::tests::restored_game_seeds_preserve_future_autoreset_stride)'` — **2 passed**, 83.652s / 105.006s; run `51d90605-ff6e-4c7e-acab-8518a7f81627`, `/tmp/mtg297-compress-caller-autoreset.log`. Original 1,000-step fixtures and watchdog intact. During the first fixture, `ps` showed the owned test running with 3.56s CPU at 66s elapsed; these are correctness receipts, not Engine Budget measurements.
- `cargo clippy --manifest-path managym/Cargo.toml --jobs 2 --all-targets --all-features -- -D warnings` — **passed**; `/tmp/mtg297-compress-caller-clippy.log`.
- `CARGO_BUILD_JOBS=2 uv run --extra play maturin build --release -i .venv/bin/python -m managym/Cargo.toml -o managym/target/wheels` — **passed**, completed 2026-11-13; `/tmp/mtg297-compress-caller-wheel.log`. Matching cp313 arm64 extension installed atomically, bytes checked against the wheel (SHA256 `7189ed0589ef2bdaf7857655c1900be5a4234190e37861190e695915348dceeb`), native import passed.
- `uv run --extra dev pytest tests/semantic/test_pooled_mana_classes_input.py -q` — **4 passed**, 16.22s; `/tmp/mtg297-compress-caller-parity.log` (2026-11-13). Covers native/reference/vector/text and Academy spending/receipt projections in v9/v10.
- Workspace `cargo fmt --all --check` and `git diff --check` — **passed**. `lf context` — scratch below both configured limits. Full debug/conformance and dedicated-runner Engine Budget remain gate obligations.

## Delivery status

Current gate evidence follows. No product decision is needed for correctness
validation. Performance acceptance and the full CI matrix remain distinct from
focused local passes. No human demo, landing or Task completion is implied.

## Gate plan (2026-11-13, `d53220d36`)

Review covers the branch diff against origin/main, with no initial uncommitted
work. The CI selector (`git diff --name-only -z origin/main...HEAD | bash
.github/ci/select-tests.sh`) selects every CI job for native changes; it has no
per-test plan or content reuse. This gate uses a focused local selection rather
than running every package, as the gate instruction requires:

- Debug: changed flow/agent modules, mana arithmetic, changed card-set modules,
  observability, CR 106/603/608, state hashing and semantic conformance. Existing
  current-runtime receipts remain valid for unchanged code; CI owns the rest of
  the full native matrix.
- Python: the five required current-world arena/monitoring files, model world,
  policy-input rejection and catalog admission, Academy Learn/navigation and
  changed protocol/text/belief/parity fixtures.
- Frontend: PlayerArea rendered tests, Svelte checks and build. Reuse the current
  runtime headless browser proof in `scratch/qa.html`; no UI inputs changed.
- Static: workspace Rust format and Clippy, Ruff on changed Python/stub files,
  schema generation consistency and diff whitespace.
- Performance: dedicated Engine Budget workflow 43417765062, dispatched against
  the already-published current HEAD. Wait for its result; do not reinterpret
  the earlier local failure or adopt a measured cost without acceptance.

The matching extension build for runtime blob `3c3ed8afa` is reused; this gate
has not changed native source. No new rules or product decision is assumed.

### Gate results

Runtime and fixture sources stayed unchanged throughout this gate. The diff
review found no further implementation repair: the accepted production timing,
source joins, explicit spending, receipt ownership and projection contract are
present. Docs already describe the behavior. Existing current-runtime browser
QA is reused; no manual demo is claimed.

- `cargo nextest run --manifest-path managym/Cargo.toml --profile ci --build-jobs 2 --test-threads 2 -E 'test(flow::) or test(agent::) or test(mana::) or binary(semantic_conformance_tests) or binary(match_state_hash_tests) or test(cr_106_mana::) or test(cr_603_triggers::) or test(cr_608_resolution::) or test(chk::) or test(tle::) or test(tla::) or test(observability::) or test(afr::) or test(blb::) or test(bro::) or test(cmr::) or test(fdn::) or test(ice::) or test(iko::) or test(kld::) or test(ktk::) or test(lea::) or test(m12::) or test(m15::) or test(m20::) or test(mmq::) or test(por::) or test(psal::) or test(roe::) or test(stx::) or test(war::) or test(zen::)'` — **870 passed**, 208.412s; run `9b05743d-ac37-4d66-b30d-cff720b02444`, `/tmp/mtg297-gate-debug.log`; remaining full native matrix deferred to CI.
- `cargo nextest run --manifest-path managym/Cargo.toml --profile ci --build-jobs 2 --test-threads 2 -E 'binary(experience_protocol_tests) or binary(canonical_replay_tests) or binary(study_protocol_tests) or test(checked_in)'` — **16 passed**, including all five Rust-generated schemas; `/tmp/mtg297-gate-protocol.log`.
- `uv run --extra dev --extra notebook --extra artifacts --extra reports pytest tests/semantic/test_elemental_world.py tests/eval/test_evaluation_models.py tests/eval/test_mirror_admission.py tests/training/test_monitor_evaluation.py tests/model/test_choice_separability.py tests/model/test_world.py tests/model/test_policy_input_rejection.py tests/model/test_policy_catalog.py tests/academy/test_learn_setup.py tests/academy/test_choice_navigation.py tests/academy/test_experience_protocol.py tests/semantic/test_text_observation.py tests/semantic/test_pooled_mana_classes_input.py tests/belief/test_state.py -q` — **224 passed**, 104.21s; `/tmp/mtg297-gate-python.log`; unrelated integration/model suites deferred to CI.
- `cargo fmt --manifest-path managym/Cargo.toml --all` — passed, no changes.
- `cargo clippy --manifest-path managym/Cargo.toml --jobs 2 --all-targets --all-features -- -D warnings` — passed; `/tmp/mtg297-gate-clippy.log`.
- `git diff --name-only -z origin/main...HEAD -- '*.py' '*.pyi' | xargs -0 uv run --extra dev ruff check --fix` and the same selection with `ruff format` — passed, 11 files unchanged.
- `npm --prefix academy/ui test -- --run src/lib/components/PlayerArea.svelte.test.ts src/lib/protocol-conformance.test.ts` — **10 passed**; `/tmp/mtg297-gate-frontend-test.log`.
- `npm --prefix academy/ui run check` — passed, zero errors/warnings; `/tmp/mtg297-gate-frontend-check.log`.
- `npm --prefix academy/ui run build` — passed; `/tmp/mtg297-gate-frontend-build.log`.
- `uv run --extra play python` comparison of `advice_schema()` against `protocol/advice-v1.schema.json` — exact match; `git diff --check` — passed.

No test timed out. Semantic conformance reproduced in 62.843s; the unchanged
masked and restored autoreset fixtures passed in 4.253s and 4.232s. Three owned
test processes were observed in OS state `T` and resumed with SIGCONT, without
changing the watchdog. The repeatedly stopped uniform-roots fixture had used
12.33s CPU at 165s elapsed and ultimately passed at 170.623s. These are
correctness receipts, not performance measurements; the source of the external
stops remains unknown. No induced load or extra test workers were used.

### Dedicated Engine Budget and adoption

`gh workflow run engine-budget.yml --ref jack/mtg-297-define-heartbeat-of-spring-with-the-tle-mana`
completed as [run 43417765062](https://github.com/loopflowstudio/manabot/actions/runs/43417765062),
measuring exact HEAD `d53220d36` against `7ef6ce588` on one AMD EPYC 7763 runner.
Both builds and all samples completed. The comparison failed **only five
intentional digest/count checks**; every timing threshold passed. All old
reference digests reproduced. Native spending choices change all three search
digests, the step digest and the step count (101,650 → 102,209).

| Metric | Reference | Candidate | Change |
| --- | ---: | ---: | ---: |
| Worst reply | 4.928s | 3.611s | −26.7% |
| Total search | 169.080s | 125.545s | −25.7% |
| Steps per second | 11,760.9 | 13,728.0 | +16.7% |

Candidate search samples were 125.475/125.672s wall and 125.452/125.636s user
CPU; reference samples were 169.914/168.356s wall and 169.852/168.292s user CPU.
Candidate step samples were 7.429/7.529s versus reference 8.732/8.631s. The
samples are consistent; the earlier local noisy failure does not describe this
current-runtime dedicated measurement. Because gameplay changed, these deltas
are not an isolated algorithmic speedup or evidence of playing strength.

- `gh run download 43417765062 --name engine-budget --dir .cache/engine-budget-mtg297-gate` — complete artifact retained with raw samples, telemetry and provenance.
- `uv run --no-project python scripts/engine_budget.py record --artifact .cache/engine-budget-mtg297-gate --reason '<native spending changes decision digests; all paired timing checks pass; preserve 10% tolerance>'` — adopted the exact recording in `managym/perf/baseline.json`; the full reason is in that file. `docs/development.md` records the same-run comparison.

This is the documented deliberate-digest adoption, with no cost requiring
acceptance and no adaptive retry. The original workflow remains failed against
its old baseline; delivery CI must compare the recording commit against the
new reference and run the deferred full matrix. Local gate found no remaining
implementation blocker. No runtime source changed, so its passing checks are
reused after the baseline/docs-only update. No servers, tests or workflow
watchers remain running from this gate.

- After adopting the baseline, `uv run --extra dev pytest tests/infra/test_engine_budget.py -q` — **19 passed**, 12.94s; `/tmp/mtg297-gate-budget-tests.log`. `git diff --check` passed; `lf context` remains below both scratch limits.
