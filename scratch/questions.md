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
  `ModelConfig.from_checkpoint` reads omission as hashed on v8+ with a world. Architecture identity
  names catalog only where it applies (v8+ with a world). No v8/v9 regime is
  frozen in the repository; one saved elsewhere before this change would now
  read catalog.

Evidence: 200 seeded random-play games (Elemental roster plus random world
decks covering all 338 eligible cards) mapped every emitted id after adding
set-characteristic names; ad-hoc script, not retained.

Parameters (total; v9): w64 L2 h4 1,958,402 → 412,482; w160 L8 h8 9,470,402 →
5,605,602 (v8: 1,936,194 → 390,274; 9,414,882 → 5,550,082).

Status (2026-11-06): implementation covers every acceptance item in code and
`tests/model/test_record_vocabulary.py` (one row per record name, shared
printed/granted row, gradients from fact and effect paths, unknown-id
rejection, checkpoint hashed fallback (v8+ world only), architecture identity); encoders are
untouched, so parity rests on the existing parity suites. Remaining: the
model/observation validation in `docs/development.md` and the CLAUDE.md
current-world arena admission suite have not been recorded on this branch.
The all-emitted-ids sweep above is unretained; the retained tests check the
native name list and real captured programs, not a played-game sweep.

Check: `uv run --extra dev pytest tests/model/test_record_vocabulary.py -q` — 8 passed (2026-11-06, after the from_checkpoint fix); broader validation deferred to gate.

## Concept review (2026-11-06) — resolved

The review found that omission read "catalog" in regimes but "hashed" in
`from_checkpoint`, so every pre-branch checkpoint (even v6, identical layouts)
failed the raw `agent.config != regime.agent` checks (admission.py,
learning_state.py, execution.py). Fixed: `ModelConfig.from_checkpoint(payload,
policy_input_version)` resolves omission to hashed only for v8+ with a world,
else the fresh default; `load_checkpoint` passes the saved input version.
Regression tests cover a pre-field v6 checkpoint equal to its regime and
v8/v9 world omission reading hashed. Kept: a pre-branch v8/v9 checkpoint loads
hashed and is rejected loudly against a catalog regime; none is frozen here.

No open product choice.

Sync 2026-11-06: main `dc8512bec` carries the merged b015d5 (training
update time) and earlier MTG-243 notes in this file, retained at
`dc8512bec:scratch/questions.md`; they confer nothing on this Task.

## Concept review, second pass (2026-11-06) — nothing to change

Against the implemented fix: `load_checkpoint` is the only reader that builds
a model from `agent_hypers` (`model/catalog.py` reads the shape label only);
architecture identity omits `hashed` and names `catalog` only for v8+ with a
world, so historical identities are stable; `with_*` builders round-trip an
explicit `hashed`. No v8/v9 regime or configuration is committed outside
tests. No open product choice; next proof remains the docs/development.md
model/observation validation and the arena admission suite at gate.
