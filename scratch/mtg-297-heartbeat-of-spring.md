# MTG-297: Heartbeat, Animal Attendant, Toph, Uncle's Musings

Contract: `tle-production-design.md`; review answers and MTG-242 coordination:
`heartbeat-design.md`. Implementation inventory before compression:
`2af8bc650:scratch/mtg-297-heartbeat-of-spring.md`.

## Current approach

- Production authority is `flow/mana.rs`: tap captures `TappedForMana`, output
  selects live `production_bonuses`, and `resolve_mana_bonuses` resolves or
  returns the next choice. `ManaProduction` retains source/bonus identity.
  The planner includes the same bonus alternatives; tap-time firing is deleted.
- Spending extends MTG-242's types with `ManaClass { restriction, effect }` and
  `ManaSpent`. The receipt follows payment → stack → resolution → entering
  permanent → its own trigger; `ZoneEntry` applies Attendant's counters.
  `EffectValue::{ManaSpentToCast, ColorsSpentToCast}` serves Toph/converge.
- Projection uses kind-11 `PooledMana` programs and kind-14 `mana spent`
  execution occurrences, with native/Python/text/token parity. No new layout,
  model kind or version branch. Only rule-relevant receipts are projected.
- The four definitions and Scryfall fixture entries are present.
  `scripts/refresh_card_fixture.py` dropped existing double-faced entries in
  the earlier run, so only the four new entries were added.

## Remaining / Delete — do not maintain

**Spending choices remain required.** Replace automatic first-remainder and
class/expiry selection in `Mana::pay` / `Game::spend_mana` /
`spend_mana_to_remainder` with native choices whenever receipts or remainders
differ. Do not refactor the automatic policy while waiting for this cut.
Today converge gets the colors automatic tapping/spending chooses, and a
player cannot hold Attendant mana back from a qualifying creature.

Remaining witnesses: countered then reanimated creatures receive no counter
or X; a filter's activation fee is excluded from the outer cast receipt.
The combat-pool Attendant witness is already present in
`cards/tests/cards/tle.rs::animal_attendant_mana_in_the_combat_pool_keeps_its_spend_effect`.
Academy's `experience.rs` player view still omits the spend-effect rider.

Production's obsolete bonus path and restriction-only pool equivalence are
already deleted; the detailed deletion list is in the contract.

## Compression

Return the next decision directly from the infallible bonus resolver; both a
bonus choice and its caller use the same suspension path. `advance_mana_upfront`
only advances upfront payments, so a bonus choice stays pending. Keep
trigger-enumeration helpers private to `flow/mana.rs`. Reuse `mana_description`
for receipt amounts; build the receipt relevance index in catalog order.
No serialized game state, choice order or projected bytes change.

## Run evidence

- Production before the main merge: cards 694, rules_tests 169, engine lib
  197 passed in debug; first 1,000 corpus decks: 998 admit, two failures name Lita.
- After merging MTG-242 and adding spending: debug cards 707, engine lib 197,
  rules_tests 170, match_state_hash 11, search_state_contract 6,
  canonical_replay 6, experience_protocol 5, conformance 2, content_pack 3,
  state 15 passed. Python academy parity/authority/fixture, semantic text/mana
  input, env observation/layout, belief state and authored matchup passed after
  re-recording w17 evidence (authored-match receipt 66d51605…,
  terminal state db1a0bbb…, belief v3 receipts).

For PR review: trigger timing follows CR 106.12a over the original proposal;
existing fact/frame joins replace its proposed v10 block and new kinds.
Spending choices remain unfinished.

Compression checks (2026-11-12):
- `cargo fmt --manifest-path managym/Cargo.toml --all --check` — passed.
- `cargo nextest run --manifest-path managym/Cargo.toml --profile ci --build-jobs 2 --test-threads 2 --lib --test cards -E 'test(chk::) | test(tle::) | test(observability::cast_receipt) | test(observability::spend_effect_mana) | test(agent::continuation::) | test(flow::mana::) | test(cost_effect_completion_retains_output) | test(the_great_henge) | test(tarnished_citadel)'` — 53 passed on the final Rust source.
- `PYO3_PYTHON="$PWD/.venv/bin/python" cargo clippy --manifest-path managym/Cargo.toml --all-targets --all-features -j 2 -- -D warnings` — passed.
- `CARGO_BUILD_JOBS=2 uv run --extra play maturin build --release -i .venv/bin/python -m managym/Cargo.toml -o managym/target/wheels` — passed; matching cp313 Darwin extension installed atomically.
- `uv run --extra dev pytest tests/semantic/test_spend_effect_mana_input.py -q` — 2 passed against the rebuilt extension (native/reference encoding and text).
