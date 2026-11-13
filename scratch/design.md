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

## What this does not fix

- Infeasible payments that pass Hall still search exponentially in this
  checkout: a multi-mana modal output (Herbalist's two Lesson mana when the
  spell is a Lesson and the pips need UUR), and a filter whose cost consumes
  the unit the bound also counts (Prism + one Forest owing {U}{G}).
- Feasible payments whose first DFS branches fail late.
- Both need the canonical memo over interchangeable sources. That is being
  written in `/Users/jack/src/manabot.ryan-bug` (uncommitted `PaymentSearch`,
  same file, touches `witness`'s `visited` parameter and `search`). Do not
  duplicate it. This change touches only the bound block before
  `visited.insert`, so the two merge with one conflict line at most.

## Ruled out for today

- **Branch only on the first untapped member of a group.** With the canonical
  memo, later members already hit the memo on entry; this would only save
  calls (a factor of about N), not states, and it risks the witness order.
  Worth doing after the memo lands if calls per plan remain high.
- **Separate exact polynomial solver.** The Hall bound is that check for the
  fixed-and-one-mana-modal case; a second code path would need its own
  equivalence proof against the DFS inside the budget.
