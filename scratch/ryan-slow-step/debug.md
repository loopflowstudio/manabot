# Mana payment search — root cause and remaining bounds

2026-11-12, MTG-334 debug step, continuing the recovery in `unbreak.md`.

## Cause

The retained samples put the stall inside payment DFS, not board view creation.
Every payment state stored a separate resource bit for every permanent. Even
with memoization, equivalent subsets of identical lands were different keys:
there are 2^N such subsets. A loose feasibility bound lets an impossible payment
visit them. Searching successive automatic-prefix budgets repeats this problem.

The release and the #469 rewrite loosened the bound differently. Source review
of `v0.2.2:managym/src/flow/mana.rs` shows that its whole-obligation bound used
`Mana::total`, including restricted output unavailable to the pending payment.
Its per-cost spendability check only examined the current obligation. A zero or
small first obligation could therefore leave an impossible later waterbend debt
inside the search. #469 (`306612865`) kept a whole-payment raw-mana bound,
but changed both that bound and the first-cost spendability bound to sum fixed
mana, waterbend and tapping abilities that consume the same tap. It also
stopped subtracting activation costs. Spend-purpose filtering remained confined
to the first-cost check. The richer presence/tap/limited-use model is necessary for counted taps and tapping before
sacrifice; reverting it would lose those semantics.

The incoming debug brief reports: “an isolated build at 306612865 (PR #469,
TLE cards, which rewrote ~700 lines of flow/mana.rs) takes 35.4 s at replay step
242 and 16.1 s at step 243, against 0.68 s / 0.34 s on PyPI 0.2.2.” Those
measurements were supplied to this step, not rerun here. Source comparison
confirms the relevant replacement; PR #496 is not the origin of that bound.

The prior recovery fix addresses both spendability and shared tap capacity
across all pending obligations. This step adds symmetry to the visited key,
removing the identity-subset explosion when that optimistic bound is still
loose. The DFS state, transitions, action order and returned witness keep their
original concrete identities. Only states with identical production, activation
costs/outputs, restrictions, waterbend eligibility and membership in every
possible counted-tap set share keys. Presence, taps and limited uses are sorted
together, never independently. Future activation tap sets and saved obligations
both participate in the equivalence check.

## Controlled evidence

Small fixtures compare complete first witnesses against the original
identity-keyed DFS; no artificial load or wall-clock assertions are used.
Before canonicalization, eight identical lands with a seven-tap prefix budget
for an eight-mana cost visited 255 states; afterward it visits eight. A pending
four-tap cost competing with a five-mana debt on eight producers drops from 93
to four states. A single modal source before two blue obligations with eight
Forests drops from 1,792 to 63 states; a fixed-only blue probe drops from 256
to nine. Two seedless blue filters alongside six Forests drop from 256 to 21.
Raw initial output: `debug-symmetry-tests.log`.

The regression fixtures also run with 24 resources, failing immediately if
their state budgets are exceeded. Coverage includes shared counted taps across
two repeatable activations, restricted modal output, nested filters, automatic
prefixes, future/saved tap candidate membership, limited uses, life costs,
presence, tapping followed by sacrifice, and 240 small-board witness comparisons.
The existing reporter replay covers the activation-X range and all intervening
legal decision frames.

## Final validation

The final source passes 113 focused debug tests: 110 payment/mana/waterbend/
Prism/Herbalist/Waterskin/counting cases and three existing repeated-Merchant
regressions. Workspace formatting and Clippy pass all targets/features with
warnings denied. Receipts: `debug-final-tests.log`,
`debug-final-merchant-tests.log`, `debug-final-clippy.log`.

The rebuilt CPython 3.13 extension runs the unchanged 426-action reporter with
`slowest step: 0.00 s`. Precise scalar/vector final steps are 1.399 / 1.404 ms;
maxima are 1.399 / 1.588 ms. All 427 decision boundaries match between scalar
and vector, retaining SHA-256
`cb3af2765e5032acec1a2a8f9fb79ffbcc44b39a310b52169ab1980cb46938a1`.
The separate open-deck bounded flat search takes 26.7 ms and retains its scores,
eight simulations, seven depth caps and unchanged root. Receipts:
`debug-final-reporter.log`, `debug-final-recovery.log`.

Release benchmarks retain all action/score digests and counts against the prior
recovery receipts and `managym/perf/baseline.json`: 512 games, 101,650 steps,
step digest `15843333459759757527`; search seed digests
`13733780288109658817`, `5337676601429031145`, `10035331835025021780`.
Receipts: `debug-final-bench_steps.jsonl`, `debug-final-bench_search.jsonl`.
These verify behavior; CI must still compare speed against its reference on
one runner. No local speed-ratio claim is made. All owned subprocesses exited;
no server or watcher was started.

## Remaining limits

This is exact, finite search, not a universal polynomial-time solver. Many
distinct production/restriction classes, overlapping counted-tap candidate sets,
multiple pending activation stacks, and concrete colored pool remainders can
still produce combinatorial work. No production cutoff returns false legality,
changes witnesses, or truncates choices. The new bounds specifically remove
identity-subset growth for interchangeable resources; they do not claim every
large board is cheap. Saved source snapshots may conservatively keep otherwise
similar abilities in separate groups.

No PyPI release or live Jace process is changed by this debug step.
