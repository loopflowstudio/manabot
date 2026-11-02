# PR 1 (T1, 892c47) brief — summary for stacked work

Full text: `b48ebae67:scratch/tla-content-handoff.md` (accepted by Jack Heart,
2026-10-28; Rust authoring revision in
[tla-rust-authoring-direction.md](tla-rust-authoring-direction.md)). Jack Heart's
2026-10-29 direction replaced its qualification model: support is definition in a
set file, one world pool of every defined card, set filters for TLA decks.

Still live for every TLA PR:

- Seven TLA PRs plus one decklist-import PR. Implement, test and publish one
  cohesive PR, then stop for Jack Heart's review: no auto-merge, landing, paid
  compute, training or follow-up Task launches.
- One typed authoring source lowered to the native runtime and learning
  projection; no browser rules, runtime Oracle parser, fallback registry or
  parallel current-world executors. Registration or generated tests never
  certify a card; independent expected rules outcomes cover each clause.
- Allocation of all 286 cards lives in `docs/rules/tla/allocation.jsonl`;
  research under `scratch/reference/full-tla-final/`.
- Decklist Task f871a8 owns parsing, deck admission and Academy/training input;
  content definitions, IR/runtime lowering and content versioning stay here.
  World w16 and learning schema 8 stay unreleased; frozen evidence is not rewritten.
- Exercise Command, replay/fork, observations and current-world admission where
  affected; keep choices uncapped and viewer-safe.
