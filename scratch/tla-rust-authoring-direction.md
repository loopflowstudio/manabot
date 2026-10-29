# Rust card authoring — accepted direction, 2026-10-28

Jack Heart clarified in the primary Task conversation that cards should be clear
code in Rust set files, with minimal configuration, rather than the current
semantic JSON authoring arrangement. Separate scripts directories may help build
and generate files; generated files should be checked in.

Implementation direction: make typed Rust set definitions the single authored
card authority. Generate the representations needed by learning and Academy from
that authority, retain generated artifacts in git, and check their freshness.
Remove the competing manually authored semantic definition/configuration path
instead of adding another source. Keep ordinary decklist data separate from card
behavior. Preserve the independent rules witnesses, explicit unsupported-card
eligibility, shared pool interface, historical evidence and one native runtime.

This supersedes the JSON-as-authority implementation choice in the earlier
handoff. It does not change the one-PR publication/demo boundary, authorize merge,
or transfer decklist/admission ownership from f871a8. Reconcile the current design
and implementation before publication; do not publish the superseded authoring
architecture as the accepted result.

## Direct Rust cards and worlds — Jack Heart, 2026-10-28 (accepted)

Supersedes the Rust → JSON → Python compiler → IR → Rust pipeline kept in PR394.
Accepted, in Jack Heart's words condensed:
- The actual Rust card code runs as directly as possible. No official JSON/YAML
  card representation; no Rust → JSON → Python → IR → Rust round trip.
- Scripts may generate Rust code; keep tools that help maintenance, but the
  resulting codebase must be clear and simple.
- `cardsets/` should be card set files, organized and de-duplicated; remove as
  much junk as possible around card sets, card files and card authoring.
- Worlds: the codebase has one current world, so any running engine knows its
  rules world. A world encodes which cards it supports, with a check of whether a
  card/deck is supported. There should be some way to run an old world; rules
  backwards compatibility is kept only where feasible (Jack Heart is unsure it is
  worth it).
- All of this lands in this Task's PR394; the commit may be large and combine
  concerns.

Constraints still in force: frozen experiment evidence, checkpoints and receipts
are never rewritten; D1/f871a8 owns decklist parsing/admission and depends on the
`content_pool` API; independent rules witnesses and explicit unsupported-card
eligibility remain. No merge authorized.

Open questions (asked 2026-10-28, unanswered; operator defaults applied, see
questions.md): old worlds run from their git revision; learning tokens may change
with the world.

Later accepted refinements (2026-10-29, Jack Heart): every card in its
first-printing set file, support by definition instead of qualification, one
world pool with set filters. Recorded in
[892c47-author-tla-cards.md](892c47-author-tla-cards.md).
