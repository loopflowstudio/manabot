# Task 30df0c — catalog fact/effect ids (2026-11-06)

Inherited MTG-243 notes are at `c59aeee0e:scratch/questions.md`; that Task
merged in `8e5d152ab` and confers nothing here.

Material implementation choices (reversible, for review):
- Fact and effect ids are hashes of names, not (kind, value) tokens. Native
  `learning_record_names` (`agent/record_vocabulary.rs`) enumerates every
  name the encoders hash, in three namespaces (token, set name, combat
  restriction), each with card text's token for the same concept. Encoders are
  unchanged. Enumeration needed `Deserialize` on `CardPredicate`/`CardType`
  (to describe catalog `AdditionalCombatPhase` restrictions), a public
  subtype iterator and `MODIFICATION_NAMES` (debug-asserted against writes).
- Set names (definition names and `SetCharacteristics.named`) and combat
  restrictions get appended rows; a set name does not read the named
  definition's program row (name ≠ rules).
- Ordinal ids (used-ability index, transform/continuation/combat frame + 1)
  are a numeric column (/128), not vocabulary rows.
- Captured program Integer/Boolean tokens read one appended row per kind plus
  the existing exact numeric projection; other program tokens must be card
  tokens or appended record tokens, else rejected.
- Serialization: `record_vocabulary` defaults to `catalog` and is omitted when
  catalog, so regime bytes are unchanged; checkpoints write it explicitly and
  `ModelConfig.from_checkpoint` reads omission as hashed. Architecture identity
  names catalog only where it applies (v8+ with a world). No v8/v9 regime is
  frozen in the repository; one saved elsewhere before this change would now
  read catalog.

Evidence: 200 seeded random-play games (Elemental roster plus random world
decks covering all 338 eligible cards) mapped every emitted id after adding
set-characteristic names; ad-hoc script, not retained.

Parameters (total; v9): w64 L2 h4 1,958,402 → 412,482; w160 L8 h8 9,470,402 →
5,605,602 (v8: 1,936,194 → 390,274; 9,414,882 → 5,550,082).
