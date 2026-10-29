# Rust card authoring — accepted direction (Jack Heart, 2026-10-28/29)

Full text: `48f14c9a9:scratch/tla-rust-authoring-direction.md`.

- Cards are clear Rust in their first-printing set file and run directly: no
  JSON/YAML card representation and no Rust → JSON → Python → IR → Rust round
  trip. Scripts may generate checked-in Rust; `cardsets/` stays de-duplicated.
- One current world, which knows its supported cards and checks cards/decks.
  Old worlds run from their git revision (operator default; see questions.md).
- 2026-10-29 refinements: support by definition, one world pool with set
  filters ([892c47-author-tla-cards.md](892c47-author-tla-cards.md)); later
  steers replace the authoring layer with engine-type definitions carrying
  Scryfall ids (T1's Remaining).
- Still in force: frozen evidence, checkpoints and receipts are never
  rewritten; D1/f871a8 owns decklist parsing/admission over `content_pool`;
  independent rules witnesses; no merge authorized.
