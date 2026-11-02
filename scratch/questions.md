# MTG-214 Wave rename assumptions — 2026-10-29

- `.lf/wave/rules` → `.lf/wave/engine`: lf's `engine` Wave owns the Rules
  chapter Projects and the same Linear initiative. Added the engine Wave id
  (`74d9efbc…`) to its GOAL frontmatter, matching training's explicit id.
- Academy and eval have empty lf goals and no checked-in files; none created.
  Only training gets `config.yaml` (`workflow: iterate`), as directed.
- The top-level `wave/` tree (old `pm:` charters, numbered plans) is historical
  and cited by frozen experiments and code comments; left unrenamed, as are
  `.lf/chapters/` records and wave body prose.
- Live doc pointers moved to `.lf/wave/{engine,training}`. ARCHITECTURE now
  assigns evaluation to Eval (previously Intelligence); reversible.
- Eval split (Jack Heart's steer, 2026-10-29): MTG-212 and MTG-188 memory sections
  moved verbatim to `.lf/wave/eval/MEMORY.md` with cross-pointers; MTG-175 moved
  to eval in lf. No eval GOAL.md until Jack sets an objective.

## Concept review (2026-10-29)

- Resolved in `4dc25ad61`: `wave/README.md` opens with a note marking `wave/`
  as the historical tree and naming the current Waves (academy, engine, eval,
  training). Other `wave/intelligence` / `wave/rules` references cite existing
  historical files and stay.
- Not blocking, outside this Task: training's objective ends with "demonstrate
  a repeatable challenge to the User under a protocol fixed before scoring",
  which reads like evaluation. Moving it to an eval objective is Jack's choice;
  this Task changes no objective.
- Second pass after implement: no further change; no live `.lf/wave/{intelligence,rules}`
  pointers remain.
# Questions and assumptions

Notes from the Tasks this branch builds on (TLA T2–T4 syncs, afbd6a, the
characteristics view, MTG-234) are main's and live at
`4c8324a59:scratch/questions.md`; earlier inheritance is pointed to from there.

## MTG-238 (catalog-scale types) — 2026-11-02, headless

- Started on main `4c8324a59` although the Task says to start after MTG-225
  (T6) publishes: T6 was still in `implement` with no PR when this Task was
  launched, and at realign and the third implement that day MTG-225, MTG-235
  (#412) and MTG-236 (#411) were open on that base, each editing
  `managym/src/state/*` that this branch moved to `managym/state/src/`.
  `tla.rs` moved to `managym/cards/src/tla.rs` with its body otherwise
  untouched so git rename detection carries T6's card edits across. The
  "stack on the effect-vocabulary Task" constraint could not be met without a
  published branch; Jack Heart's steer (2026-11-02) holds landing until
  MTG-225 publishes, then sync.
- `CreatureWithNamesakes`, `Card::creature_with_namesakes` and the Airship
  definition are T5's (MTG-224, PR #405 onto main), merged into T6's branch;
  they become `TypeGrant { This, [Creature], ControlsAtLeast { 8, namesake } }`
  at the T6 sync. `CardPredicate.namesake` was not added early: its only
  reader arrives then, and a field nothing reads is a setup nothing uses.
- "State hashes unchanged" held for counters (serialization byte-identical)
  but not for animation: `Permanent.animated` was serialized on every
  permanent, so removing it changes every canonical encoding. Per the
  contract in `state/hash.rs` the version is 12; a dead serialized `animated`
  key was rejected. MTG-236's hash.rs edit is not a version bump and T6's new
  hashed `copy` field is `skip_serializing_if`, so no second bump collides.
- Jack Heart approved (steer, 2026-11-02) carrying the conditional
  `TypeGrant`, the `BecomeArtifactCreature` → `BecomeCreature` fold and the
  keyword-list serialization as one content-digest-moving batch with one
  re-record. Every corpus embedding state or content digests was re-recorded
  once: semantic-kernel, the authored-match receipt (needs the RUL-9
  artifact-store measurement) and both parity receipts, the five Academy
  fixtures via `scripts/refresh_academy_fixtures.py`, the presentation pins in
  `tests/academy/test_authored_match_parity.py` and the belief v3 pins.
  WORLDS.md had claimed the receipts were unchanged; corrected at realign.
- The steer's "learning schema 9 -> 10" repeats the concept review's number;
  the schema was already 10 before this branch (T4) and the schema file
  (kinds, roles, dtypes) is unchanged by the keyword list, since tokens derive
  from the definitions. No bump was made; a bump as a token-shape marker is a
  one-line change (`learning_schema.json`) at the sync if Jack Heart wants it.
- Deleting `CounterThreshold` converted Zhao's `StaticLandType.while_counters`
  to `condition: Condition::SourceCountersAtLeast`, the same "as long as"
  shape as the type grant. War Balloon's grant lists `[Creature]` only (a
  Vehicle is a printed artifact), as crew's `BecomeCreature` does.
  `permanent_is_animated` reads the permanent's own `This`-scoped Creature
  grant with a recorded-loss check only (the full loss check lives in the
  static index, which asks animation while it builds); predicate matching
  asks animation only for type-reading predicates, so name and counter
  questions never recurse. The definitions guard gained `Place::SourceStatic`
  for a type grant's or land-type overwrite's own condition; a scoped static
  (`PtBuff`) still rejects source questions.
- "Others become fact rows" has no instance: all 14 keywords are evergreen, so
  no `FactKind` was added. The exhaustive `Keyword` match in
  `observability.rs` forces the first non-evergreen keyword to name its fact
  row instead of columns. `PyKeywords`' Python API change is a follow-on Task;
  the Task outcome names the Rust types only.
- The decision capture's public definition (`ContentPack::public_definition`)
  keeps the observation's keyword object (`KeywordData`), as card and
  permanent rows do, rather than the definition's keyword list: the capture
  contract (`DECISION_CAPTURE_VERSION` 4, the retained w15 capture fixture)
  is an observation surface and does not follow the content encoding.
- `player.rs` and `deck.rs` stay in the engine crate (`managym::state`):
  `Deck` is a `#[pyclass]` whose `#[pymethods]` must be defined in the same
  crate, and nothing in the rules vocabulary reads them. The `managym-cards`
  tests dev-depend on the engine (a Cargo-permitted cycle) and mount
  `managym/tests/support` by `#[path]` so each primitive stays defined once.
- `tests/academy/test_ur_lessons_increment.py` asserted a 37-wide permanent
  row, stale since T4 added the dealt-damage column on main; corrected to 38
  here (one line, suite otherwise green). Not a rule of this Task.
- Compress (2026-11-02, after the batch): `Keyword` keeps its serde derive
  beside `name()` rather than serializing through `name()`, because the
  learning-token serializer reads unit variants as `Keyword::X` variant
  tokens and a string would move every card's tokens; the observability test
  ties the two spellings. Historical records (`docs/research/semantic-kernel.md`,
  WORLDS.md's w16 section, `experiments/card-conformance-audit.md`) keep the
  pre-move paths; live pointers were updated.
