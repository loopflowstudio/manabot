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
2. Design any new primitive cleanly on the first pass so later review does not
   have to rewrite it: small orthogonal pieces (selectors, conditions,
   triggers, effects, replacements) that compose—reshaping cramped existing
   types when needed—instead of specialized variants that fuse multiple axes
   for one card. State direct Comprehensive Rules ownership and wire/observation
   shapes up front rather than derived flags or card-specific fields.
3. Cut the buildable cards into mechanism-ordered slices sized for roughly
   30–50 minutes of implementation each (one shared primitive with every card
   in the batch it unlocks, or 10–30 cards that already compose). Avoid tiny
   one-card slices (unless one card alone exercises a new primitive) and avoid
   oversized multi-primitive slices that saturate context. List the slices in
   build order, riskiest primitive first.
4. Give each slice its focused `cargo test` filter, the constructed
   interactions that show its cards working together, and any native, Python,
   LLM-text or learning-token surfaces it changes. Offers stay complete;
   replay, forks and viewer privacy hold; frozen input versions don't change.
   Name the large-board shapes that could make the new cards expensive (which
   new cards, many copies or many distinct choices) so `card-perf` runs them as
   [stress states](../../docs/development.md#stress-states).
5. List obsolete symbols each slice deletes (no invented deletion for a pure
   addition) and record the author's synergies and interaction checks in
   `scratch/plan.md`. Card authors don't choose experiment decks or launch
   training; known bugs stay scoped defects, not support gates.

Run again on an existing plan, keep finished slices and change only what the
code now contradicts. A cross-Task handoff names a committed revision and
paths, never just a path in another checkout.
