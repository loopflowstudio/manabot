# MTG-339 payment average case: working notes

## Built (uncommitted, on e82a20033)

- `managym/src/flow/mana.rs`: `PaymentModel::direct` names the direct class
  (one mana cost, optional same-purpose generic waterbend split, no searched
  ability; `FixedOnly` single cost on any board; never `AutomaticPrefix`).
  `direct_payable` answers by one sum; `direct_witness` rebuilds the search's
  first path in one pass. `plan` and new `payable` use them and
  `debug_assert_eq!` against `search`.
- `waterbend_taps_keeping_payable`: the one waterbend-after-taps question; one
  model per decision, each candidate reserved and restored (callers in
  `action.rs`, `decision.rs`, `play.rs`).
- Compress: `DirectTap` and `direct_margin` replace the tuple helper and the
  duplicated totals; `waterbend_payment_after_taps` and its model helper are
  gone. After it: focused debug tests 130 passed, clippy clean, `bench_steps`
  digest unchanged at 49.16 G instructions (1 run).
- `PaymentScan` (`Game::payment_scan`, `Game.payment_sources`): the three
  legal-action computations build a player's sources once; debug asserts every
  reuse equals a rebuild.
- `managym/README.md` payment paragraphs describe the class as built,
  restricted fixed outputs included.

## Checks run

- `cargo nextest run --workspace -E 'test(/waterbend|mana|payment|filter|convoke|treasure|restricted/)'`
  (debug, assertion live): 130 passed. New tests: `flow::mana::tests::direct_payment_*`
  (20,000 seeded random boards equal the search, 622 with a waterbend tap).
- Workspace clippy `--all-targets` clean; `--features benchmark --examples` checks.
- One release run each, digests unchanged (raw in `scratch/payment-after/`):

| | baseline median | after (1 run) |
|---|---|---|
| bench_steps instructions retired | 62.47 G | 49.20 G (-21%) |
| bench_steps user s | 3.565 | 2.94 |
| bench_search instructions retired | 1314.96 G | 946.03 G (-28%) |
| bench_search user s | 74.10 | 55.36 |

## Remaining

- Gate: full debug suites (the assertion runs in every game they play).
- Measure: repeat interleaved against `scratch/payment-baseline/bin/*.e82a20033`
  with the baseline's run counts; one run each is not a median.
- After CI's Engine Budget run: `scripts/engine_budget.py record --reason`.
- No test counts model builds per legal-action computation; sharing is covered
  by the debug equality assertion only.
