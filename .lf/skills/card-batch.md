---
requires: a card or mechanic cohort and the interactions it must support
produces: a slice-ordered implementation plan in `scratch/plan.md`
action_style: procedural
---
Plan a card batch in `scratch/plan.md`: what each card needs, which shared mechanism it rides on, and the order to build them in.

1. Read the cohort, its interaction examples and the set ledger. Check each
   card against the native rules types (`managym/state/`) and catalog
   (`managym/cards/`), not the ledger's tags. Sort the batch into cards that
   compose from today's types, cards waiting on a small shared primitive, and
   cards waiting on an authority another Task owns. Mark the last group
   pending in the ledger with the missing authority; write no one-off
   shortcuts for them.
2. Cut the buildable cards into slices by mechanism, not by card. A slice is
   one shared primitive with every card in the batch it unlocks, or 10–30
   cards that already compose. One card is a slice only when it alone
   exercises a new primitive. List the slices in build order, riskiest
   primitive first, so each pass builds a whole slice without re-deriving the
   next one.
3. Give each slice its checks: the focused `cargo test` filter, the
   constructed interactions that show its cards working together, and the
   native, Python, LLM-text and learning-token observations it changes.
   Offers stay complete; replay, forks and viewer privacy hold; frozen input
   versions don't change. When a slice changes payment, choice, targeting,
   enumeration or continuous effects, name the board shape that would make it
   expensive (which cards, many copies or many different ones) so measurement
   runs it as a [stress state](../../docs/development.md#stress-states).
4. List the obsolete symbols each slice deletes. A new capability needs no
   invented deletion.
5. Record the author's synergies and interaction checks in `scratch/plan.md`
   for integrated validation. Card authors don't choose experiment decks or
   launch training; known bugs stay scoped defects, not support gates.

Run again on an existing plan, keep finished slices and change only what the
code now contradicts. A cross-Task handoff names a committed revision and
paths, never just a path in another checkout.
