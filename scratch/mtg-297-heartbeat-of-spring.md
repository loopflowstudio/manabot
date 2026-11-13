# MTG-297: Heartbeat, Animal Attendant, Toph, Uncle's Musings

Contract and completed deletion list: `tle-production-design.md`; verbatim
review and coordination: `heartbeat-design.md`; limitations: `questions.md`.
Earlier implementation and check details are retained at
`8d2721ab9:scratch/mtg-297-heartbeat-of-spring.md`.

## Approach and remaining work

`flow/mana.rs` owns production, payment search and native spending. Capture
`TappedForMana` at tap; select live trigger sources at production. Unit choices
preserve color, restriction, rider, expiry and every saved caller. `ManaSpent`
excludes nested activation fees and reaches `ZoneEntry` and the entering
permanent's own triggers. Existing fact kinds 11/14 carry pools, offers and
receipts in v9/v10; no input shape or model kind was added. The public pool
projection is `pooled_mana_classes`, with absent qualifiers omitted.

The automatic first-remainder payment path is deleted. The surviving search
emits `PaymentStep::Spend` only at the zero-cost transition and pools the whole
current cost before unit spending. Upstream source-equivalence memoization and
supply bounds are retained; `PaymentSearchState::usable` counts allowed rider
and combat classes. `cards/tests/cards/tle.rs` and `roe.rs` exercise receipt,
restriction and expiry choices with Command replay, stale rejection and forks;
`tests/semantic/test_pooled_mana_classes_input.py` checks shared encoding.

The earlier search regression visited 1,001 states, 1,000 already unable to
meet a saved counted tap (run `877a3189-d326-44e7-a45a-7f0aa7f63fbd`). Rejecting
those states before exploring earlier payments repaired it without changing
the 1,000-state witness bound. Details and earlier validation are in the
revision above. Main's MTG-242 pool types, #504 search bounds, #506 sharding and
v0.2.3 preparation are incorporated; no dependency wait or Wave remains.

Compression stops feasibility after the first valid ordering when all usable
units must be consumed; the receipt and remaining pool are identical for all
such orders. Explicit fixture spending shares one offer selector and checks
`Decision::ManaSpend`, avoiding duplicate label-selection loops. Priority and
mode helpers continue to expose pending choices. No deletion target is open.

## Debug fixture repair (2026-11-12)

Follow-up compression reuses `support/inspect.rs::permanent_named` / `card_of`
for the two Heartbeat source lookups and `support/play.rs::spend_named` for
Prism's expiry choice. The native `ManaSpend` assertion in that shared helper
prevents a color predicate from choosing production instead. Existing source,
pool, Command and outcome assertions remain. `state/mana.rs` now explains
restriction, rider and expiry as independent class properties. No deletion
target remains open.

An intermediate retry of the original 75 failures passed 63 and failed 12
(run `4dce981b-45a6-4dba-8cbc-6c6cd1f21b2d`,
`/tmp/mtg297-compress-fixtures.log`). Fixture edits continued during this pass;
the later 643-test repair result below supersedes that diagnostic.

The previous publication block's 75 failures are resolved. They were fixture
assumptions about automatic payment, not engine regressions. Fixtures now
complete native spending before priority, stack inspection, hand snapshots or
pool replacement. Ordinary colorless pays generic costs, hybrid pips choose
stated colors, and Azula's Bolt chooses combat red while retaining regular red.
The repeated Sailor fixture preserves blue for all three draws. Lesson casting
finishes before its separate resolution pool is installed; its nested-payment
and decline assertions remain unchanged.

`support/play.rs` owns the shared `spend_named` offer selector and explicit
`spend_colorless` driver. Casting, priority, mode and target helpers retain
their behavior; no native decision is suppressed. The failure fixes require no
engine behavior change. The deletion list remains complete; no compatibility
path was introduced. Additional simplification stops feasibility search after
the first feasible unit when the entire usable pool must be consumed, and the
shared spending helpers assert the native `ManaSpend` decision directly.
The combined source is covered by the final debug run below.

- Fixture repair before the later simplifications: **643 passed, 492 skipped** in
  37.453s, run `b5b039a8-a079-4a97-93cf-02cbca7c67a1`, log
  `/tmp/mtg297-repaired-native.log`.
- Diagnostic progression: 454 selected tests had 442 passes and 12 failures
  (`c8897094-5c31-4a0d-9704-caa63e322bfb`); fixing explicit hybrids and later
  payments left one unpaid Jet-fixture Lightning Strike
  (`f42d14eb-9fcc-405d-8cfe-39877b04888c`, 11/12). The final run above includes
  that repair and every originally failing test.

- Combined source, same affected debug command: **643 passed, 492 skipped**
  in 186.728s (one slow conformance case, 181.573s), run
  `82b74871-d7da-499e-8a32-8f254dfc3a12`, log
  `/tmp/mtg297-combined-native.log`. The Rust diff stayed unchanged throughout
  this run. The original 75 failures and subsequent simplifications are covered.

- `PYO3_PYTHON="$PWD/.venv/bin/python" cargo clippy --manifest-path managym/Cargo.toml --all-targets --all-features -j 2 -- -D warnings`
  — passed on the combined source (`/tmp/mtg297-combined-clippy.log`).
- Workspace Rust formatting and diff whitespace pass. The release build,
  atomic cp313 installation and four shared-input checks are recorded below.

Remaining: the caller's review and delivery steps. Headless runs provide no personal demo approval.

## Prior publication check (2026-11-12)

Focused TLE fixture repair: Kindly Customer and Toucan-Puffin explicitly spend
ordinary colorless mana before taking the post-cast hand snapshot or passing
priority to the entry trigger. Their original outcome assertions are unchanged.
`cargo nextest run --manifest-path managym/Cargo.toml --profile ci --build-jobs 2 --test-threads 2 --test cards -E 'test(=tle::tle_baseline_kindly_customer_draws_on_entry) | test(=tle::tle_baseline_toucan_puffin_pumps_only_your_creature)'`
— **2 passed**, run `6a12cdc1-ed51-45ca-adf0-1468642d0433`.

`lf sync` completed cleanly at `3ee971e99`. Publication was blocked: the broader
affected-file debug run had **568 passed, 75 failed, 492 skipped** (65.362s;
nextest run `9644df6c-0cbb-4ba7-b5e3-30fb71cb76f7`). Earlier narrow passes do
not establish readiness. No PR was published, auto-merge enabled or Task completed.

Reproduce with:

```sh
cargo nextest run --manifest-path managym/Cargo.toml --profile ci --build-jobs 2 --test-threads 2 --lib --test cards --test rules_tests --test match_state_hash_tests --test semantic_conformance_tests -E 'test(mana::tests) | test(chk::) | test(tle::) | test(roe::) | test(tla::) | test(observability::) | test(cr_106_mana) | test(cr_603_triggers) | test(cr_608_resolution) | test(cost_effect_completion_retains) | binary(match_state_hash_tests) | binary(semantic_conformance_tests)'
```

The prior log is `/tmp/mtg297-publish-native.log`. Many failures hit
`support/play.rs:46`, "priority pass action should be available": fixtures
passed priority while a native mana-spending choice was pending. Examples:
`observability::a_waiting_delayed_trigger_is_observed_until_it_fires_or_expires`,
`tla::aang_airbends_another_permanent_and_gains_lifelink_from_lessons`, and
`tla::committed_optional_payment_resumes_after_two_nested_mana_decisions`.
Other failures included
`tla::hybrid_draw_payoffs_use_draw_count_and_trigger_on_the_second_draw_only`
(unwrap at `tla.rs:7844`) and the TLE Kindly Customer/Toucan Puffin baseline
witnesses. The fixture repair above completes those payments while preserving
all card outcome assertions and native decisions.
Clippy and remaining broad gate checks were not run in that publication pass.

The current-world evidence gap is repaired in this checkout. The original
Academy check had **3 failed, 5 passed**: stale authority/parity state hashes
and Learn's manifest mismatch. `refresh_academy_fixtures.py` replayed all
existing Commands without label changes (Learn 10; precision 21;
precision-mana 54; four navigation positions; trigger-order root).
The authority receipt, authored parity, public-commitment parity, semantic
kernel w17-v2 evidence and only current-v3 belief receipt pins were refreshed.
Historical world evidence and old belief receipts are unchanged.

Verification after refresh:

- Academy authority/parity, Learn setup and choice navigation: **29 passed**.
  Authored parity: 135 Commands, 136 checkpoints, 816 viewer checks, no
  divergence; stale-object and stale-revision rejection retained.
- Public-commitment writer: 135 Commands, 66 commitments, zero gaps.
- Semantic conformance: 14 cases / 2,715 Commands / 15 Phase rows recorded;
  checked evidence reproduction passes in the debug run above.
- Observation, action cutoff, native buffer capacities and active training:
  **118 passed**.
- Elemental world, evaluation models, mirror admission, monitor evaluation,
  choice separability and belief state: **77 passed, 1 stale belief pin failed**.
  After refreshing only current-v3 pins, belief state plus model world:
  **32 passed**, including independent scalar marginal accumulation.
- Frontend PlayerArea, protocol conformance and decision labels: **14 passed**.
- Changed-Python Ruff, Rust formatting, workflow validation and diff whitespace
  pass. No Rust implementation changed during this publication check; the
  already rebuilt extension was used.

MTG-213 corpus bytes fetched from the artifact cache and re-admitted from
original `Decklist.source` / `filename`, preserving historical records:
**998/1,000 admit**. Heartbeat events `af3d52fd84ed4bfab9e13dd56354de5a`,
`55481e295a8d4f28ac505c982cd9af31`, `d88c59ee998e456cad266dbebe4c0852` all admit.
Only Lita events `2937d550424b4ec1b44c9fc30c85c418` and
`ed932f4c0dad48e3b868517f69696c4c` fail. `docs/engine.md`'s counts are confirmed.
Release to Memory and Lita remain outside this Task. Production limitations
remain in `questions.md`; no product scope was reduced.

## Earlier check receipts

Pooled-class, search-repair and pre-merge commands and results are retained at
`8d2721ab9:scratch/mtg-297-heartbeat-of-spring.md`, including the 18 shared-input
checks, 14 text checks and frontend/schema passes. They establish those earlier
sources, not the final compression diff.

## Current compression checks (2026-11-12)

- `cargo nextest run --manifest-path managym/Cargo.toml --profile ci --build-jobs 2 --test-threads 2 --test cards -E 'test(chk::) | test(roe::prism_pool_alternatives)'`
  — **8 passed**, run `7764da1a-350a-477d-89d9-3c651e44f327`;
  `/tmp/mtg297-compress-focused.log`.
- `lf context` initially reported **8,807 / 12,000 tokens** and
  **36,688 / 98,304 bytes**. Rechecks after editing failed with
  `Task "task_e698a90fb9d9456f9495f0ca7497fb1e" is not registered`.
  Scratch Markdown was 27,395 bytes before this receipt; limits are unchanged.
- `CARGO_BUILD_JOBS=2 uv run --no-sync --extra play maturin build --release -i .venv/bin/python -m managym/Cargo.toml -o managym/target/wheels`
  — passed in the concurrent repair run (`/tmp/mtg297-repair-extension.log`).
  Verified the installed cp313 Darwin extension matches the rebuilt wheel bytes.
- `uv run --no-sync --extra dev --extra play pytest tests/semantic/test_pooled_mana_classes_input.py -q`
  — **4 passed** against that extension (`/tmp/mtg297-compress-input.log`).

- Repeated the affected debug command above after compression:
  **643 passed, 492 skipped**, run `bbdbab2a-e47a-4e13-8bf3-c6effbf6be3b`,
  `/tmp/mtg297-compress-verified.log` (186.972s; conformance 182.313s).
  Hashes of all eight changed Rust files stayed fixed through this run.
  The initial compression diagnostic saw **632 passed, 11 failed** while
  fixture repairs were still arriving; that diagnostic log was subsequently
  replaced. This was intermediate evidence, superseded by the stable pass.
- Independently verified the rebuilt cp313 wheel postdates changed runtime
  sources and its extension bytes match the installed module. The shared-input
  parity log above covers that extension; no further runtime edit followed.
- `cargo fmt --manifest-path managym/Cargo.toml --all --check` and
  `git diff --check` pass. A later `lf context` succeeds: **6,612 / 12,000
  tokens, 28,571 / 98,304 bytes** before this receipt; neither limit changed.

- The independent final debug selection also passed: **643 passed, 492 skipped**,
  run `da6e9270-c440-4849-8560-4cd363887cd9`,
  `/tmp/mtg297-compress-native.log` (189.825s; conformance 184.694s).
  Final `lf context` retry again reports the Task as unregistered; the
  successful size receipt above remains the latest available measurement.
- `PYO3_PYTHON="$PWD/.venv/bin/python" cargo clippy --manifest-path managym/Cargo.toml --all-targets --all-features -j 2 -- -D warnings`
  — passed (`/tmp/mtg297-compress-clippy.log`). Compression leaves the diff for
  the caller; no publication or personal demo approval is inferred.
