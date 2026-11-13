# MTG-297: Heartbeat, Animal Attendant, Toph, Uncle's Musings

Contract: `tle-production-design.md`; verbatim review and coordination:
`heartbeat-design.md`. Earlier inventory: `2af8bc650:scratch/mtg-297-heartbeat-of-spring.md`.

## Implemented approach

Production stays in `flow/mana.rs`: capture `TappedForMana` at tap, select live
trigger sources at production, then resolve independent bonuses. Spending uses
`Mana::spending_choices` with saved `SpendCompletion`s in `flow/decision.rs`;
unit choices preserve color, restriction, rider, expiry and every saved caller.
The receipt excludes nested activation fees and reaches `ZoneEntry` and the
entering permanent's own triggers. Existing fact kinds 11/14 carry the pools,
offers and receipts in v9/v10; no input shape or model kind was added.

The old `Mana::payment_options` / `Game::spend_mana` first-remainder path is
gone. Search completes both zero costs and a cost's last unit through the
same zero-cost transition. That transition
alone emits the Spend boundary; a unit is explored only when the remaining
current cost is already pooled. Payment completion types live with their saved
decisions, and owned resolution effects move directly into the frame.
`Game::pooled_mana` uses `Mana::add_combat`, and `spend_mana_unit` debits the
selected live pool. `Mana::spending_choices` computes the next pip once, then
enumerates eligible classes and colors in the same order. Tap candidate scans
are iterators; checking a saved counted-tap cost stops after enough candidates
are found, without allocating their identities. The saved-cost pruning and
original search-state bounds remain intact.

`cards/tests/cards/tle.rs` witnesses rider selection, different converge
receipts from the same pool, additional cast costs and nested filter fees.
`support/inspect.rs::assert_choice_contract` checks each offer's Command replay,
stale rejection and fork isolation; TLE spending and ROE expiry cases call it.
`tests/semantic/test_pooled_mana_classes_input.py` covers shared v9/v10 encoding,
text and Academy projection.

The unpublished pool projection is now `pooled_mana_classes` with native
`PooledManaClassData` and PyO3 `PooledManaClass`, including its Python stub.
Absent restrictions/effects are optional and omitted in JSON. Text states
amounts before qualifiers; Academy falls back to aggregate combat expiry only
without class rows. `PooledManaView`, token programs and buffer layouts remain
unchanged. Focused checks pass below; gate's acceptance remains open.

`state/card.rs::TriggeredManaOutput::alternatives` checks whether either
restricted or unrestricted mana contains a type, without summing their u8
amounts. The type is present even when those amounts total 256; the bounded
CR 106.12a witness covers each type, restricted-only output and empty output.

## Upstream reconciliation

Main through `7cce98a00` was merged as `08164e6ef`. It includes #504's
source-equivalence memoization and total/color
bounds in `flow/mana.rs::PaymentSearch` / `PaymentModel`, and #506's CI
sharding. That separate work's note is retained at
`08164e6ef:scratch/ci-under-8-minutes.md`, outside this Task's active context.
MTG-242 is incorporated; no dependency wait or Wave remains.

The later merge `0c2a366e6` incorporates main `b53ebf6f4` and the paired
v0.2.3 release preparation (`e82a20033`). Compared with the repaired checkpoint
`fffbe7816`, it changes release notes, package/crate versions, locks and release
validation guidance; mana source, card definitions and focused witnesses are
unchanged. The earlier search repair/compression results below remain evidence
for that source. The later pooled-class implementation and compression rebuilt
the extension on this merged checkout and passed the focused checks recorded
below; that rebuild is complete. Gate's broader current-world checks remain
open. Remote main still resolves to `e82a20033` at this reconciliation; its
release changes are already incorporated.

The new bound's `PaymentSearchState::usable` still called the removed
restriction-only pool API; the first realign debug build failed there. It now
counts `Mana::classes()` allowed by each pending purpose, including rider and
combat classes. Two upstream assertions still destructured the removed
`PaymentStep::Spend(remainder)`; they now check the Spend boundary. A directed
bound witness requires rider blue plus combat red to pay {U}{R}. Source
canonicalization leaves pool classes intact; the bounds relax supply only for
feasibility and do not choose the live spending unit.

## Resolved search regression

The bounded-state regression is diagnosed: the diagnostic run reached 1,001
visited states, of which 1,000 already lacked enough candidates for a saved
counted tap (run `877a3189-d326-44e7-a45a-7f0aa7f63fbd`). The first fixed
production tapped land 0; checking the later tap only when it reached the front
let unit spending explore doomed branches. Search now rejects any state whose
saved tap lacks present, untapped candidates. These resources only decrease,
so this removes no successful witness and chooses no spending class.
The 1,000-state limit and all source, class/expiry and production semantics
remain unchanged. The focused debug search/card witnesses and rebuilt
shared-observation parity passed in the runs below. The earlier iteration
feedback's 15/16 result predates this repair; it is not the current unresolved
state. No known implementation blocker remains.

## Remaining acceptance

The preceding iteration's pooled-class implementation direction is complete
in `fcc7a514c`: code inspection confirms the renamed native/Python surfaces,
optional omitted qualifiers, amount-first text, Academy's expiry fallback and
all six schemas. The old names and obsolete comment have no remaining source
consumers. The recorded 11 debug and 18 Python passes include the final
compression; the earlier 3 parity passes are not the latest evidence. No
runtime change or test rerun was needed for this reconciliation.

Gate still needs the affected suites, current-world authored parity/authority,
fixture binding and belief receipts, world-admission checks, and the three
Heartbeat events among the first 1,000 MTG-213 corpus decks. Keep
`docs/engine.md` counts aligned: its 998/1,000 claim still needs confirmation
against this catalog. No production/spending behavior is deferred.

The earlier merge retained main's w17 fixtures, parity hashes and belief
receipts. Neither parent's evidence describes the combined catalog: refresh
current-world evidence only, preserving frozen evidence. The relevant checks
include `tests/academy/test_authored_match_parity.py`,
`test_authored_match_authority.py`, `test_learn_setup.py`, the authored-match
receipt and fixture generators, and the world-admission/monitoring files in
AGENTS.md. Focused debug card tests pass with the combined search; CI owns
the full debug suite and Engine Budget.

The affected-check list also includes the observation/action-selection checks
in `docs/development.md#validation-by-change-area`:
`tests/env/test_observation.py`, `tests/training/test_action_cutoff.py`,
`tests/verify/test_native_buffer_capacities.py` and
`tests/training/test_active_training.py`. The focused presentation checks below
do not substitute for those checks or current-world admission.

Release to Memory and Lita are outside this Task; the token representation
prerequisite landed but their card definitions are absent. Production limits
remain in `questions.md`; none is newly approved out of acceptance. No new
product decision is identified.

The earlier `scripts/refresh_card_fixture.py` run dropped existing double-faced
entries, so only the four new entries were added to the Scryfall fixture.

## Pooled-class presentation checks (2026-11-12)

- Follow-up compression: `cargo nextest run --manifest-path managym/Cargo.toml --profile ci --build-jobs 2 --test-threads 2 --test cards --test rules_tests -E 'test(chk::) | test(cr_106_12a_produced_types) | test(observability::pooled_mana) | test(tle::animal_attendant_mana_in_the_combat_pool)'`
  — **11 passed** (run `455597ca-896b-454a-b6a9-c4a20e7c5eb2`), including
  the type-presence overflow witness. Changed Python/stub formatting also passes.
  Follow-up `CARGO_BUILD_JOBS=2 uv run --no-sync --extra play maturin build --release -i .venv/bin/python -m managym/Cargo.toml -o managym/target/wheels`
  passed; matching cp313 Darwin extension installed atomically.
  `uv run --no-sync --extra dev --extra play pytest tests/semantic/test_pooled_mana_classes_input.py tests/academy/test_experience_protocol.py -q`
  — **18 passed** against that extension. Rust formatting and `git diff --check`
  passed; gate's broader checks remain open.
- Focused debug `cargo nextest run --manifest-path managym/Cargo.toml --profile ci --build-jobs 2 --test-threads 2 --test cards -E 'test(observability::pooled_mana) | test(tle::animal_attendant_mana_in_the_combat_pool)'`
  — **3 passed** (run `54d1296b-ab60-42c3-b5fe-8be083ac9d53`).
- Academy `npm run check` — zero errors/warnings; `PlayerArea.svelte.test.ts`
  — **5 passed**, including qualifier rendering and aggregate expiry fallback.
  Protocol conformance initially failed on the old schema's field name;
  after regenerating all six schemas, the focused
  `npm run test -- src/lib/protocol-conformance.test.ts` retry — **5 passed**.
- `CARGO_BUILD_JOBS=2 uv run --no-sync --extra play maturin build --release -i .venv/bin/python -m managym/Cargo.toml -o managym/target/wheels`
  — passed; the matching cp313 Darwin extension was installed atomically.
- `uv run --no-sync --extra dev --extra play pytest tests/semantic/test_pooled_mana_classes_input.py tests/academy/test_experience_protocol.py -q`
  — **18 passed**, covering native/reference v9/v10 scalar/vector parity,
  omitted qualifiers, text, Academy projection and protocol schema shapes.
- `uv run --no-sync --extra dev --extra play pytest tests/semantic/test_text_observation.py -q -k 'pooled_mana or every_native_decision_code or every_observed_field'`
  — **14 passed**. Changed Python/stub Ruff checks, Rust formatting and
  `git diff --check` passed. All six generated schemas use the new field and
  optional qualifiers. `lf context` remains under both scratch limits.

## Compression checks (2026-11-12)

`cargo nextest run --manifest-path managym/Cargo.toml --profile ci --build-jobs 2 --test-threads 2 --lib --test cards --test rules_tests -E 'test(mana::tests) | test(chk::) | test(tle::animal_attendant) | test(tle::toph_) | test(tle::uncles_) | test(merchant_) | test(roe::prism_) | test(roe::filtering_) | test(the_great_henge) | test(tarnished_citadel) | test(badgermole_cub_earthbends) | test(waterbend_and_mana_taps) | test(sozins_comet_foretell_spending) | test(observability::cast_receipt) | test(observability::spend_effect_mana) | test(observability::pooled_mana_expiry) | test(cr_106_mana) | test(cost_effect_completion_retains)'`
— **83 passed, 4 failed** (run `39f6cf0b-d5eb-49de-826f-e7bc9da3d8c6`).
All search witnesses passed, including the unchanged 1,000-state bound.
Three Unlucky Cabbage Merchant tests mistook Food-fee spending for the search
mode; the waterbend/bonus witness tried to pass priority during cast spending.
Their fixtures now pool exactly the irrelevant costs: {2} for Food, {3}{U}
for Waterbending Lesson. Their original behavior assertions remain.

The four-test retry passed the three Merchant cases (run
`996521fb-39d9-4b21-a339-93cfde818ae1`); the Lesson fixture's initial {1}{U}
was insufficient. Corrected to its printed {3}{U},
`cargo nextest run --manifest-path managym/Cargo.toml --profile ci --build-jobs 2 --test-threads 2 --test cards -E 'test(waterbend_and_mana_taps)'`
— **1 passed** (run `e62834b8-74b3-466f-b7ed-6517e283aace`).
All 87 selected witnesses pass across these runs.

- `CARGO_BUILD_JOBS=2 uv run --extra play maturin build --release -i .venv/bin/python -m managym/Cargo.toml -o managym/target/wheels`
  — passed; matching cp313 Darwin extension replaced atomically.
- `uv run --extra dev --extra play pytest tests/semantic/test_spend_effect_mana_input.py -q`
  — **3 passed**, native/reference scalar/vector v9/v10, text and Academy.
- Rust formatting and `git diff --check` passed. Removing the inherited CI
  note keeps scratch under both configured limits; `lf context` confirms it.

## Search repair checks (2026-11-12)

- `cargo nextest run --manifest-path managym/Cargo.toml --profile ci --build-jobs 2 --test-threads 2 --lib -E 'test(payment_search_preserves_witnesses_and_counted_tap_membership)'`
  — **1 passed**, original bounds unchanged (run `9a44b2ad-3283-4238-ae60-9ca09d41d962`).
- Focused debug `--lib --test cards --test rules_tests`, remaining mana search,
  CHK, TLE spending/receipt/Merchant, Prism/nested fees, Henge/Citadel,
  Badgermole, expiry and CR 106 — **70 passed, 1 failed** (run
  `91448ebf-3050-41cf-a8da-3b484f1e31c2`). The failure was Badgermole's
  old fixture passing priority with the new generic spending choice pending.
  It now explicitly spends colorless, checks the remaining mixed pool and
  checks the Command contract before preserving all earthbend/bonus assertions.
- Focused `--test cards -E 'test(tla::badgermole_cub_earthbends_and_adds_green_when_a_creature_taps_for_mana)'`
  — **1 passed** (run `37d528ba-5758-4377-b45f-459ad73f6612`).
  All 72 distinct focused search/card/rules witnesses pass across these runs.
- `CARGO_BUILD_JOBS=2 uv run --extra play maturin build --release -i .venv/bin/python -m managym/Cargo.toml -o managym/target/wheels`
  — passed; matching cp313 extension installed atomically from the wheel.
- `uv run --extra dev --extra play pytest tests/semantic/test_spend_effect_mana_input.py -q`
  — **3 passed**, native/reference scalar/vector v9/v10, text and Academy.
- Rust formatting and `git diff --check` passed; `lf context` remains within
  both scratch limits.

The pre-repair realign run had 15/16 mana tests passing, plus 3/3 shared-input
checks; it did not resolve this regression. The diagnosed failing reproduction
is recorded above rather than treating it as successful validation.

## Run evidence before merge `08164e6ef`

These results establish the pre-merge implementation only. The new search
bounds and source grouping were not present in these runs.

The first compression debug build found two remaining `Mana::pay` calls in
`tests/rules/cr_106_mana.rs`; both witnesses now use the surviving unit API.

Compression check: `cargo nextest run --manifest-path managym/Cargo.toml
--profile ci --build-jobs 2 --test-threads 2 --lib --test cards --test rules_tests`
with the mana, CHK/TLE, Prism, Henge/Citadel, foretell, receipt/expiry,
completion and decision-code filters — **57 passed** (run
`d6d6fff3-5628-4083-bf65-edf94c276580`).

- Compression rebuild: `CARGO_BUILD_JOBS=2 uv run --extra play maturin build
  --release -i .venv/bin/python -m managym/Cargo.toml -o managym/target/wheels`
  — passed; matching cp313 Darwin extension replaced atomically.
- `uv run --extra dev --extra play pytest tests/semantic/test_spend_effect_mana_input.py -q`
  — **3 passed**, covering native/reference scalar/vector v9/v10, text and Academy projection.
- `cargo fmt --manifest-path managym/Cargo.toml --all --check` and
  `git diff --check` — passed.

The following passes preceded this compression pass.

Earlier receipts, including pre-merge checks, are retained at
`58c12b399:scratch/mtg-297-heartbeat-of-spring.md`; they do not validate this cut.

- `cargo nextest run --manifest-path managym/Cargo.toml --profile ci --build-jobs 2 --test-threads 2 --test cards -E 'test(chk::) | test(tle::animal_attendant) | test(tle::toph_) | test(tle::uncles_) | test(roe::prism_) | test(tla::sozins_comet_foretell_spending) | test(the_great_henge) | test(tarnished_citadel) | test(observability::cast_receipt) | test(observability::spend_effect_mana) | test(observability::pooled_mana_expiry)'` — 37 passed on the combined source.
- Focused `--lib --test cards` debug runs also passed the state mana tests,
  decision-code bound, nested Command fork/restore and cost-effect completion
  cases. The full-fee witness caught and fixed a partial-production edge case.
- `CARGO_BUILD_JOBS=2 uv run --extra play maturin build --release -i .venv/bin/python -m managym/Cargo.toml -o managym/target/wheels` — passed; matching cp313 Darwin extension installed atomically.
- `uv run --extra dev --extra play pytest tests/semantic/test_spend_effect_mana_input.py -q` — 3 passed; the extended text-submission witness passed again with `-k spending_choices`. Covers v9/v10 native/reference scalar and batched encoding, text, Academy projection and runtime token vocabulary. Two missing text-renderer mappings (stage and obligation 31) were found and fixed.
- `tests/academy/test_experience_protocol.py` — 14 passed. All six embedding schemas regenerated from their authorities; no historical fixtures rewritten.
- `uv run --extra dev --extra play pytest tests/semantic/test_text_observation.py -q -k 'every_native_decision_code or every_observed_field'` — 13 passed.
- Academy `npm run check` — zero errors/warnings; `npm run test -- src/lib/protocol-conformance.test.ts src/lib/decision-labels.test.ts` — 9 passed.
- Rust formatting, changed-Python Ruff checks and `git diff --check` — passed.
