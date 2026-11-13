# MTG-337 payment search guard

Build the ordinary debug CI regression from the unchanged reporter script
(SHA-256 `42c6bec5f6a94fbb189ef40a92c7f197489e75756d3e813eaff063bd996596fa`).
Keep all 426 actions, both closed 40-card decks and the seed. Compare scalar
and one-slot isolated-vector observations and decisions at every boundary.
Count calls with the existing payment instrumentation and stop each plan at
1,001 calls. Include unpayable Herbalist, Airship and Prism boards at 6 and
12 Forests, requiring equal expanded state counts.

## Delete — do not maintain

No production behavior is replaced. Keep the existing benchmark counters as
one owner; extend their build availability to debug rather than introducing
a second counter. Preserve exact search order, witnesses and legal actions.
Keep counts, current-plan metadata and the scoped limit in one thread-local
`benchmark_payments::Counters`: named fields replace the positional tuple,
and each call updates and checks its count under one borrow. No deletion
targets remain.
The inherited CI-under-eight-minutes notes concern PR #506, not this Task;
history is in commit `60664d149`.

## Remaining

Gate owns full affected checks and the three temporary search-protection mutations required by the Task
(whole-payment bound, interchangeable-source memo, color-set bound); never
commit the mutations. Include their failure evidence in the eventual PR.

## Implemented check

`cargo test --manifest-path managym/Cargo.toml -j 2 --test payment_search_tests -- --nocapture`
— 2 passed in 2.57 s in debug (2026-11-12). Scalar + vector together:
4,468 plans, 20,758 calls, 16,060 states, maximum 180 calls per plan,
21 sources. All 426 actions and every observation/decision agreed.
The brief's earlier measurement was 169 calls per plan; the unchanged
fixture on this checkout now measures 180. The limit remains 1,000.
Herbalist/Airship/Prism expanded 4/7/3 states at both 6 and 12 Forests.

Release CPython 3.13 wheel build passed; the matching extension was installed
atomically and `uv run --no-sync` native import passed. Rust formatting and
`git diff --check` passed. No search-protection mutation was run at implement.

Compression (2026-11-12): the same debug test command passed 2 tests in 2.72 s,
with all counts unchanged. Fixture readers use delimiters directly instead
of offset arithmetic; developer guidance keeps the contract and commands,
with historical counts retained here.
Release CPython 3.13 wheel rebuild, atomic extension install, native import,
Rust format check and `git diff --check` also passed after compression.

`cargo run --manifest-path managym/Cargo.toml -j 2 --features benchmark --example bench_payment -- replay managym/tests/fixtures/payment_search/manabot_slow_step.py` — completed 426 actions; digest `16547847303527648701`, 2,234 plans / 10,379 calls / 8,030 states, maximum 180 calls.
