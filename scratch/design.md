# Payment search optimisation: working design

Target: `PaymentModel::witness` (`managym/src/flow/mana.rs`). Measurements:
`scratch/measure.html`. Budget ends about 17:10 PST, 2026-11-12.

## Hypothesis 1 (next): whole-payment bound per colour set

**Mechanism.** The constructed boards are infeasible only by colour: {U}{R}
is owed and one source can make U *or* R. The per-cost bound adds the union of
a modal source's outputs (`upper.max_with`), so that source appears to cover
both pips; the whole-payment bound compares totals only. Nothing rejects the
root, and the DFS then proves infeasibility over every subset of Forests.

**Change.** Generalise the existing whole-payment bound (`total < owed`) from
"all mana" to every set S of the colours owed. For each non-empty subset S of
the distinct coloured kinds owed across all pending `Pay` obligations:

- `owed(S)` = coloured pips whose colour is in S, plus hybrid pips with both
  colours in S, summed over pending `Pay` obligations.
- `supply(S)` = pool mana of colours in S (restricted mana counted when any
  pending cost allows it, as `usable_total` does now)
  + per present source: `max(fixed in S, tapping abilities' uses × best single
  output choice in S)` if untapped, plus non-tapping abilities' `uses × best
  single choice in S`
  + best choice in S of each pending `Output`.
- Return `None` before `visited.insert` when `supply(S) < owed(S)`.

This is the same tap-exclusive accounting the total bound already uses, with
one choice per use instead of the union. It is Hall's condition, so for boards
of fixed sources and one-mana modal sources without activation costs it is
exact: infeasible payments are rejected with no expanded state. It only
removes states that cannot succeed, so the first witness, legal actions and
the bench digests are unchanged. Run it only when two or more distinct colours
are owed (one colour: the existing bounds suffice for the first cost; include
singletons when several `Pay` obligations are pending), after the cheap total
test, so the 11-12-call plans of `bench_steps`/`bench_search` pay nearly nothing.
Keep the existing per-cost `can_pay` check; it still owns restriction detail.

**Expected counts** (`bench_payment boards 14`, `boards 12`):

| Workload | Now (states) | Expected |
| --- | --- | --- |
| herbalist-unpayable-14 | 409,604 | constant in N, under 20 |
| dual-unpayable-14 | 81,927 | constant in N, under 20 |
| filter-unpayable-12 | 32,765 | constant in N, under 20 |
| replay | 9,651 calls, max 449/plan | same or lower; digest 16547847303527648701 |
| bench_steps / bench_search | max 11-12 calls/plan | unchanged counts and digests |

Refuted if any unpayable board still grows with N, or a digest moves.

**Tests** (visited-state bounds, no wall time): N Forests + one U/R modal
source owing {U}{R}, at N = 8 and 24, expands 0 states; the same with a
{1}-cost any-colour filter; [Pay U, Pay U] with one any-colour source; a
payable {U}{G} board still returns the same first witness. Compare feasibility
with the unbounded search on small boards in debug.

**Check while implementing.** Confirm nothing lets mana be spent as another
colour during payment (`Mana::usable_total` / `can_pay`), that `cost[6]` is
generic and index 5 colourless (treat {C} as a kind like a colour), and that
waterbend taps pay generic only, so they stay out of coloured supply.

## Result (2026-11-12, after merging the canonical memo 5e8eebbd3)

Implemented as `PaymentModel::colors_coverable`. Receipts: `scratch/payment-after/`.

| Workload | Baseline states | After |
| --- | --- | --- |
| herbalist-unpayable-14 | 409,604 | 4 (same at N = 6..14) |
| dual-unpayable-14 | 81,927 | 7 (same at N = 6..14) |
| filter-unpayable-14 / -12 | 131,069 / 32,765 | 3 (same at N = 6..14) |
| replay | 7,243 states, max 449 calls/plan | 7,161 states, max 169; digest unchanged |
| bench_steps | 793,333 states | 791,410; digest 15843333459759757527 |

Check: 135 debug mana/waterbend/payment tests, workspace clippy, fmt, reporter
replay SHA cb3af276..., bench_steps and three bench_search digests all pass.
Local wall time was not comparable (load average over 300); CI's Engine Budget
decides speed.

## What still searches exponentially

- An unpayable cost that passes every colour set, over sources that are
  pairwise not interchangeable: N lands each making differently restricted
  green, an Island and a {U}-cost any-colour filter, owing {U}{R}, expand
  2^(N+1) states (128 / 8,192 / 524,288 at N = 6 / 12 / 18). The memo only
  merges identical sources; the bound counts the Island's U once for the
  filter's cost and once for the pip.
- The same shape over identical Forests is linear (14 / 26 / 38 / 50 states).
- Feasible payments whose first DFS branches fail late, and overlapping
  counted-tap sets, are unmeasured here.

## Ruled out for today

- **Branch only on the first untapped member of a group.** With the canonical
  memo, later members already hit the memo on entry; this would only save
  calls (a factor of about N), not states, and it risks the witness order.
  Worth doing after the memo lands if calls per plan remain high.
- **Separate exact polynomial solver.** The Hall bound is that check for the
  fixed-and-one-mana-modal case; a second code path would need its own
  equivalence proof against the DFS inside the budget.
