# Slow payment recovery — 2026-11-12

Session `session_6f1af8add1a543e894df5268ae665776` is implementing and validating
the payment bound in `managym/src/flow/mana.rs` in this checkout. Original report
and standalone repro are retained unchanged alongside this note.

The saved PyPI 0.2.1 run completed with a 33.65 s final step. The initial run
labelled 0.2.2 imported this checkout instead; its output is now retained as
`local-environment-aborted.log` (step 242: 43.57 s, subsequently SIGTERM).
Do not treat it as a released 0.2.2 measurement.

Sampling that local process with symbols at 15:46:25 showed all 398 samples
inside `PaymentModel::witness`, reached through `legal_activation_x_values`
and `waterbend_payment_possible`. Full sample: `current-sample.txt`.

The usable-mana upper bound summed fixed mana, waterbending and each tapping
mana activation on one source, despite all consuming the same untapped
resource. The fix counts the maximum usable output of those exclusive choices,
while continuing to sum non-tapping activations optimistically. It bounds
the entire pending payment using mana spendable on any pending obligation,
including restricted mana exactly once. Checking only the first obligation was
insufficient: that intermediate candidate still took 137.38 s on step 426
(`fixed-reporter-replay.log`). The final patch only tightens the whole-payment
bound; the existing per-cost color/restriction bound is unchanged. No source
grouping or action reordering is needed.

## Recovery checks

The release extension built from the fixed checkout completes the unchanged
426-action reporter script with `slowest step: 0.00 s`
(`whole-payment-replay.log`). The diagnostic harness `verify_recovery.py`
repeats the closed-deck replay through scalar `Env.step` and a one-slot
`VectorEnv.step_isolated_into_buffers`, asserting identical full decision
frames at all 427 boundaries. First precise run: final scalar step 1.817 ms,
final vector step 1.388 ms; maxima 1.817 ms and 1.570 ms respectively.
Decision-frame SHA-256:
`cb3af2765e5032acec1a2a8f9fb79ffbcc44b39a310b52169ab1980cb46938a1`.

Step 426 opens `X = 1` through `X = 20` choices in this checkout, rather than
mana-color choices. This agrees with the sampled activation-X call path.
The full final decision frame is in `verified-recovery.log`.

Flat search on the original closed-deck setup correctly refuses with
`decklist closed: player 1 cannot sample the opponent's hidden cards`. The
search proof replays the same 425 actions with open decklists and asserts the
same root offers before calling `flat_mc_scores(1, 1, 42, 32)`. It completes in
32.0 ms, with eight simulations and seven depth caps, without mutating the
root. This is a bounded search-path proof, not playing-strength evidence.
Successful combined receipt: `verified-recovery-open-search.log`.

Five focused debug module tests pass, including early impossible-payment
rejection measured by visited states, source-order witness preservation,
tap-then-sacrifice, multiple restricted purposes and a restricted-mana filter.
Eighty existing debug rules/card tests pass, covering mana, waterbend, nested
Prism payments, Herbalist, Waterskin and counted tap costs
(`debug-payment-tests.log`). Workspace Clippy passes all targets and features
with warnings denied (`clippy.log`).

Final minimal patch: the unchanged reporter script again reports `0.00 s`.
The precise closed scalar/vector proof records a 1.313 ms final scalar step,
1.338 ms final vector step and maxima of 1.313 / 1.645 ms. The full decision
digest is unchanged. Bounded open-deck search takes 30.2 ms with identical
scores, eight simulations and seven depth caps (`final-recovery.log`).
The combined final debug selection passes all 90 tests, including the above
checks and five additional existing engine mana tests (`final-debug-tests.log`).
Three additional counted-tap regressions pass: repeat activation with four
Foods, only one activation with three, and no reuse of an already reserved
Food (`repeated-taps-tests.log`). Final workspace formatting and Clippy with
all targets/features and warnings denied pass (`final-clippy.log`).

Final release benchmarks preserve the reference behavior in
`managym/perf/baseline.json`: 512 games, 101,650 steps and step digest
`15843333459759757527`; Search-64 digests for seeds 1/2/3 are
`13733780288109658817`, `5337676601429031145`, `10035331835025021780`.
Receipts: `final-bench_steps.jsonl`, `final-bench_search.jsonl`. These local
runs verify behavior, not CI's speed ratio against a reference built and run
on the same runner. All subprocesses launched for this recovery have exited;
no server or watcher was started.

## Remaining limits

The fix is local source, not a published PyPI release or a verification of
Ryan's live Jace process. Exact search remains combinatorial for other
resource interactions; this fix removes the false resource bound responsible
for the reported large-board stalls, not every possible worst case. Original
release evidence and the reporter's script remain unchanged.
