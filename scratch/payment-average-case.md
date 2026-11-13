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
  checked before the scan test was added; after it, `cargo clippy -p managym
  --all-targets` clean.
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

## Concept review (2026-11-12): no design change

- `PaymentScan` stays separate from `BoardScan` although both count live
  guards and clone to zero. `payment_model` and `payment_sources` each take
  their own `BoardScan`, so a merged guard would be live for every lone
  question and would store sources it then drops; telling an enclosing
  computation from that inner scan is the second concept again. `BoardScan`
  is also free in release and would stop being so.
- `direct` / `payable` / `plan` and `waterbend_taps_keeping_payable` name what
  they answer; the search remains the one definition of a legal payment.
- No open product choice. The items in `scratch/questions.md` are engineering
  notes, none changes what the engine offers or taps.

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

## QA (2026-11-12, scratch/qa.html)

- 2,400 seeded random games (authored + TLA pool corpus) give identical
  observation digests on main's and this engine; 179 of them replayed with
  debug assertions live, no panic.
- Final code, interleaved with baseline binaries: bench_steps 49.11-49.16 G
  vs 62.43-62.51 G (3 runs each); bench_search 944.75 G / 53.11 user s vs
  1314.50 G / 74.17 (1 run each). Digests unchanged. Raw files are in ignored
  `.runs/qa339/bench/`.
- On main too: pool games seed 340067 (`not enough automatic mana`) and
  340431 (`no legal targets`) end in PyAgentError; no bug Task filed yet.

## Published (2026-11-12): PR #509

- Synced onto main at ef28f3a58 (no change under `managym/`). On that tree:
  `cargo fmt --all -- --check` clean, `cargo clippy --workspace --all-targets
  -- -D warnings` clean, `cargo nextest run --workspace --profile ci` (debug)
  1195 passed, 2 skipped.
- Still open: adopt the PR's own Engine Budget run as the reference after CI;
  `scratch/` (24 MB of baseline binaries and samples) is on the branch until
  the pre-land clear.

## Gate (2026-11-12, second pass)

- `managym/` was byte-identical to the tree of the published gate above, so
  its debug suite and clippy results stand and were not rerun.
- One edit: the `benchmark_payments` module doc now says a directly answered
  `payable` question is not counted as a plan (comment only).
- `cargo fmt --manifest-path managym/Cargo.toml --all -- --check` clean;
  `cargo check -p managym --features benchmark --examples` builds;
  `uv run python scripts/check_loopflow_workflows.py` valid (`.lf/skills/qa.md`
  changed).
- CI on PR #509 was pending at this pass (Rust Tests, Engine Budget); nothing
  is deferred to it beyond the Engine Budget reference adoption.
