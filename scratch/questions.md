# Questions and assumptions

Notes from the Tasks this branch builds on (TLA T2–T4 syncs, afbd6a, the
characteristics view, MTG-234) are main's and live at
`4c8324a59:scratch/questions.md`; earlier inheritance is pointed to from there.

## MTG-238 (catalog-scale types) — 2026-11-02, headless

- Started on main `4c8324a59` although the Task says to start after MTG-225
  (T6) publishes: T6 was still in `implement` with no PR when this Task was
  launched. T6's `StaticAbility::CreatureWithNamesakes` therefore does not
  exist here; converting it to a `Count(predicate) >= n` condition is deferred
  to the sync onto T6. T6's new `Keywords { x: true, ..Default::default() }`
  literals and `.keywords.x` reads convert mechanically at that sync
  (`Keywords::of(&[Keyword::X])`, `.keywords.has(Keyword::X)`); `tla.rs` moved
  to `managym/cards/src/tla.rs` with its body otherwise untouched so git
  rename detection carries T6's card edits across.
- "State hashes unchanged" held for counters and keywords (serialization
  byte-identical, content digest unchanged) but not for the animation change:
  `Permanent.animated` was serialized on every permanent, so removing it
  changes every canonical encoding. Per the contract in `state/hash.rs` the
  version is now 12 and every corpus embedding state digests was re-recorded
  (semantic-kernel, the authored-match receipt and both parity receipts, the
  five Academy fixtures via the new `scripts/refresh_academy_fixtures.py`, the
  presentation pins in `tests/academy/test_authored_match_parity.py`); the
  content digest did not move, so no card test or token evidence changed. The
  alternative, a dead serialized `animated` key, was rejected.
- `Modification::BecomeArtifactCreature` (crew) was kept beside the new
  `BecomeCreature` because folding the two into one general type-adding record
  would rewrite six card definitions and
  move the w17 content digest, which re-records every corpus including the
  authored-match receipt that needs the RUL-9 artifact-store measurement. Fold
  it at the next content-digest-moving change.
- "Others become fact rows" has no instance: all 14 keywords are evergreen, so
  no `FactKind` was added for non-evergreen keywords. The exhaustive `Keyword`
  match in `observability.rs` forces the first non-evergreen keyword to name
  its fact row instead of columns.
- `player.rs` and `deck.rs` stay in the engine crate (`managym::state`) rather
  than `managym-state`: `Deck` is a `#[pyclass]` whose `#[pymethods]` must be
  defined in the same crate, and nothing in the rules vocabulary reads them.
- The `managym-cards` tests dev-depend on the engine (a Cargo-permitted
  dev-dependency cycle) and mount `managym/tests/support` by `#[path]` so each
  scenario primitive stays defined once.
- Compress (headless): `tests/academy/test_ur_lessons_increment.py` asserted a
  37-wide permanent row, stale since T4 (`4c8324a59`) added the dealt-damage
  column on main; corrected to 38 here because it is one line and the suite
  is otherwise green. Not a rule of this Task.
- Realign (2026-11-02): origin/main is still `4c8324a59`; MTG-225 (T6),
  MTG-235 and MTG-236 are open on that base with no PR, each editing
  `managym/src/state/*` that this branch moved to `managym/state/src/`. The
  "stack on the effect-vocabulary Task" constraint could not be met without a
  published branch; the landing order is left to Jack Heart (plan, Status).
  WORLDS.md had claimed the authored-match and parity receipts were unchanged;
  corrected, since both were re-recorded for hash contract 12. MTG-236's
  hash.rs edit is not a version bump and T6's new hashed `copy` field is
  `skip_serializing_if`, so no second bump collides with 12.
- Concept review (2026-11-02, headless): the landing-order paragraph is
  treated as delivery sequencing, not a product choice; the Task directive
  ("start after T6 publishes, stack on MTG-235") already selects waiting, so
  no Blocked outcome was raised for it. The three remaining card-shaped
  types (`CreatureWithCounters`, T6's `CreatureWithNamesakes`,
  `BecomeArtifactCreature`) and the struct-of-bools keyword serialization
  were recorded as one content-digest-moving batch for the next implement,
  on the accepted rules (count conditions, batch digest moves, open
  vocabularies as rows); learning schema 9 → 10 is the one new consequence.
  `PyKeywords`' Python API change was left as a follow-on Task, since the
  Task outcome names the Rust types only.
- Implement, second pass (2026-11-02, headless): the direction said "bump
  learning schema 9 → 10"; the schema was already 10 (T4) and WORLDS.md's
  rule is that tokens derive from the definitions, so the keyword-list
  serialization changed every card's tokens without a schema change (the
  schema file is untouched). `CardPredicate.namesake` was not added: its
  only reader is Phoenix Fleet Airship, which arrives at the T6 sync, and a
  field nothing reads is a setup nothing uses. Deleting `CounterThreshold`
  converted Zhao's `StaticLandType.while_counters` to `condition:
  Condition::SourceCountersAtLeast`, the same "as long as" shape as the type
  grant. War Balloon's grant lists `[Creature]` only (a Vehicle is a printed
  artifact), as crew's `BecomeCreature` does. `permanent_is_animated` reads
  the permanent's own `This`-scoped Creature grant with a recorded-loss check
  only (the full loss check lives in the static index, which asks animation
  while it builds); predicate matching now asks animation only for
  type-reading predicates, so name and counter questions never recurse. The
  definitions guard gained `Place::SourceStatic` for a type grant's or
  land-type overwrite's own condition; a scoped static (`PtBuff`) still
  rejects source questions. The wave-memory line left by the previous step
  said the crew fold "waits for the next" digest move; corrected to say the
  batch moved it, since that line would otherwise ship false.
- Steer (Jack Heart, 2026-11-02, during the second implement): the batch is
  approved as one content-digest-moving change with one re-record, landing
  held until MTG-225 publishes. Its "learning schema 9 -> 10" repeats the
  concept review's number; the schema was already 10 before this branch and
  the schema file (kinds, roles, dtypes) is unchanged by the keyword list, so
  no bump was made. If a schema bump is wanted as a token-shape marker, it is
  a one-line change (`SCHEMA_VERSION`, `learning_schema.json`) at the sync.
- The decision capture's public definition (`ContentPack::public_definition`)
  keeps the observation's keyword object (`KeywordData`), as card and
  permanent rows do, rather than the definition's new keyword list: the
  capture contract (`DECISION_CAPTURE_VERSION` 4, the retained w15 capture
  fixture, `decision_capture.Definition`) is an observation surface and does
  not follow the content encoding.
- Implement, third pass (2026-11-02, headless): `CreatureWithNamesakes`,
  `Card::creature_with_namesakes` and the Airship definition are T5's
  (MTG-224, PR #405 onto main), not T6's; T6 (MTG-225) has merged T5 and
  still has no PR, so they arrive together at the T6 sync and the sync plan
  is unchanged. MTG-235 (#412) and MTG-236 (#411) published onto main the
  same day; both edit `managym/src/state/*`, so the sync onto T6 also takes
  whichever of them lands first across the crate move.
