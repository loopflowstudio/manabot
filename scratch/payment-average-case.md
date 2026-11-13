# MTG-339 payment average case: working notes

Outcome and constraints are the Task brief. Baseline: `scratch/payment-baseline/README.md`.

## Built (committed on the branch; main at 054e987a0 merged)

Nothing upstream since e82a20033 touched `managym/src`, `managym/perf` or
`scripts/engine_budget.py`; the Engine Budget reference is still 7ef6ce588.

- `managym/src/flow/mana.rs`: `PaymentModel::direct` names the direct class
  (one mana cost, optional same-purpose generic waterbend split, no searched
  ability; `FixedOnly` single cost on any board; never `AutomaticPrefix`).
  `direct_payable` answers by one sum (`direct_margin`); `direct_witness`
  rebuilds the search's first path in one pass. `plan` and `payable` use them
  and `debug_assert_eq!` against `search`.
- `waterbend_taps_keeping_payable`: the one waterbend-after-taps question; one
  model per decision, each candidate reserved and restored (callers in
  `action.rs`, `decision.rs`, `play.rs`). Its debug oracle is the old fork
  calculation.
- `PaymentScan` (`Game::payment_scan`, `Game.payment_sources`): the three
  legal-action computations (`compute_player_actions`,
  `pending_choice_action_space`, `suspended_decision_action_space`) build a
  player's sources once; reservations, sacrifice and life stay per question;
  debug asserts every reuse equals a rebuild. A clone starts empty.
- `managym/README.md` payment paragraphs describe the class as built.

## Checks run

- Debug, assertion live: `cargo nextest run --workspace -E 'test(/waterbend|mana|payment|filter|convoke|treasure|restricted/)'`
  130 passed (before the scan test below). `flow::mana::tests::direct_payment_*`:
  20,000 seeded random boards equal the search (622 with a waterbend tap);
  modal land and filter stay with the search under `All`.
- `a_payment_scan_keeps_sources_until_its_last_guard_ends` (added at realign):
  sources kept only inside a scan, across a nested guard, never in a clone.
  Passed in debug with the three `direct_payment_*` tests.
- Workspace clippy `--all-targets` clean and `--features benchmark --examples`
  checked before the scan test was added; not rerun since.
- Release, one run each, digests unchanged (raw in `scratch/payment-after/`):

| | baseline median | after (1 run) |
|---|---|---|
| bench_steps instructions retired | 62.47 G | 49.20 G (-21%); 49.16 G on the final code |
| bench_steps user s | 3.565 | 2.94 |
| bench_search instructions retired | 1314.96 G | 946.03 G (-28%) |
| bench_search user s | 74.10 | 55.36 |

  The `scratch/payment-after/` files (19:26-19:27) predate the last `mana.rs`
  edit (19:33, the `DirectTap`/`direct_margin` simplification). Only
  `bench_steps` was rerun on the final code; `bench_search` has no run of it.

## Remaining

- Gate: full debug suites (the assertions run in every game they play), clippy
  and fmt on the final tree.
- Measure: both benchmarks on the final code, interleaved against
  `scratch/payment-baseline/bin/*.e82a20033` with the baseline's run counts
  (10 steps, 5 search); report instructions retired and user seconds, and what
  the curated decks cannot show (no modal lands, filters or chosen outputs).
- After CI's Engine Budget run: `scripts/engine_budget.py record --reason`
  adopting the PR's own run.
- No test counts source builds per legal-action computation; the scan test
  covers the sharing mechanism and the debug equality assertion its soundness.
- Not built, by choice: a revision-keyed cache across computations (see
  `scratch/questions.md`).
