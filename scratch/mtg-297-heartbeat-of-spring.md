# MTG-297: Heartbeat, Animal Attendant, Toph, Uncle's Musings

Contract: `tle-production-design.md`; verbatim review and coordination:
`heartbeat-design.md`. Earlier inventory: `2af8bc650:scratch/mtg-297-heartbeat-of-spring.md`.

## Approach and remaining work

Production stays in `flow/mana.rs`: capture `TappedForMana` at tap, select live
trigger sources at production, then resolve independent bonuses. Spending uses
`Mana::spending_choices` with saved `SpendCompletion`s in `flow/decision.rs`;
unit choices preserve color, restriction, rider, expiry and every saved caller.
The receipt excludes nested activation fees and reaches `ZoneEntry` and the
entering permanent's own triggers. Existing fact kinds 11/14 carry the pools,
offers and receipts in v9/v10; no input shape or model kind was added.

Compression removes the unused `Mana::set_class` remainder-writing API and
single-use `ManaClass::effect_applies` wrapper. Search completes both zero costs
and a cost's last unit through the same zero-cost transition. That transition
alone emits the Spend boundary; a unit is explored only when the remaining
current cost is already pooled. Payment completion types live with their saved
decisions, and owned resolution effects move directly into the frame.
The CR 106 color/restriction matrix now spends explicit offered units; the
hybrid feasibility test no longer substitutes draining a pool for payment.

Gate still needs the affected suites, current-world authored parity/authority,
fixture binding and belief receipts, world-admission checks, and the three
Heartbeat events among the first 1,000 MTG-213 corpus decks. Keep
`docs/engine.md` counts aligned. No production/spending behavior is deferred.

Main `285cd49ce` was merged as `a4768f624`; its w17 fixtures, parity hashes and
belief receipts were retained. Neither parent's evidence describes the combined
catalog: refresh current-world evidence only, preserving frozen evidence.
MTG-242 is incorporated; no dependency wait or Wave remains. Release to Memory
and Lita are outside this Task; their token prerequisite landed but their card
definitions are absent. Production limitations remain in `questions.md`.

The earlier `scripts/refresh_card_fixture.py` run dropped existing double-faced
entries, so only the four new entries were added to the Scryfall fixture.

## Run evidence

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
