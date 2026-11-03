# T5 questions and assumptions

Accepted decisions and pending checks live in
[jack-heart/a7234f-tla-t5-permanent.md](jack-heart/a7234f-tla-t5-permanent.md).
Historical parent notes (D1, T2–T4, archive/CI work and MTG-214's Wave rename)
are evidence, not directions for T5. Complete pre-compression notes:
`bbb5101eb:scratch/questions.md`; older stacked notes:
`1e9fd2d7c:scratch/questions.md` (lines 5–472).

## Current implementation assumptions

- Surveil reuses LookAndSelect: select moves the inspected card to the
  graveyard, decline keeps it. Kept cards retain order; exact for surveil 1,
  the only amount authored. Supporting larger amounts needs ordering choices.
- Copy exceptions share `Modification` with continuous records. The protocol
  calls the field `copy_of` because Pydantic's `BaseModel.copy` occupies `copy`.
- Airship uses self-relative `CreatureWithNamesakes` because
  `check_static_condition` has no source. It reads private layer stages,
  never recursively reads `characteristics()` during layer computation.
- T4 owns `PlayerTurnHistory`, sacrifice bookkeeping and its column after
  attacked. Text uses singular/plural correctly; no retained fixture holds
  the old “1 permanents” / “1 cards” wording.
- The engine measured 314 supported definitions (257 TLA eligible plus 57
  others; four TLA and one ths unsupported). Earlier 239/266 notes were
  pre-sync or based on stale main counts, not current measurements.
- The permanent row's name/creature-types fields are Aura-set markers; the
  card row states layered characteristics. Ward is printed unless abilities
  are lost; no authored effect grants ward. Text and mana cost stay printed
  until a modification changes them, then join `Characteristics`.

## Reversible choice awaiting a separate product decision

Since 1a/2a the projection's `base_power` / `base_toughness` contain layered
card-row P/T on the battlefield. No UI component reads them for printed P/T,
and no other Python-visible source exposes printed P/T. Keep and document
that value for T5; a printed-P/T accessor or field removal requires a follow-up
protocol decision by Jack Heart. This is not a T5 blocker.

The choice-navigation re-record (`889948346`) preserves recorded offer ids
when labels repeat, keeping original state digests and offer order. It is
published; Academy UI passed at `bbb5101eb` (CI 41539137476). Gate subsequently
confirmed all required CI at `1de85f7aa` (run 41541743395), plus local copy-fact
parity coverage and its CI selection. The latter remain uncommitted; see the
design's gate results for commands and evidence boundaries.
Shipping remains stopped by steer f39fef34; `/verify` is not a pre-merge check
because that skill finishes delivered Tasks.

## MTG-216 (decklist disclosure) — 2026-10-30, implementation (headless)

Jack Heart accepted one open/closed option, default open (supersedes the
kickoff's four-way assumption). Reversible choices made while implementing:

- No player-row "known" flag: an open main deck is never empty, so all-zero
  `decklists` rows mean closed. The player width is untouched (67 after the
  sync with main's T4 v5 widening) and the native width-binding rule is
  unchanged.
- No separate Python `DecklistHidden` class: refusals are `AgentError`
  ("decklist closed: …"). `Game::determinize` and
  `HiddenHandConstraints::for_viewer` assert; `Env` entry points check first.
- Search refusal at match build lists observation-only kinds
  (`checkpoint`, `heuristic`, `random`, `scripted_greedy`); every other kind is
  refused in closed matches. Academy refuses its search villain in closed games.
- Decklist pooling is semantic-pack only, like `known_hand`; generic-model
  policies ignore the `decklists` buffer. Adding `decklist_projection` changes
  the v5 semantic `ActorCritic` parameters, so v5 semantic checkpoints saved
  before this change fail strict load (open world; none frozen).
- Disclosure is not part of `world_binding` or deck identity: a checkpoint can
  play both modes, and in a closed game it sees zero opponent rows.
- `Deck.player()` (training deck pools) seats an open decklist; it takes no
  disclosure argument. Closed games come from `MatchConfig.decklists` or
  Academy only.
- `canonical_replay`/`study` privacy validators do not check decklist maps:
  frames carry no disclosure, and the engine projection is the authority.

Notes from the Tasks this branch builds on (TLA T2–T4 syncs, afbd6a, the
characteristics view, MTG-234) are main's and live at
`4c8324a59:scratch/questions.md`; earlier inheritance is pointed to from there.


## Sync onto main 230b2f563 — 2026-11-02

The design link above is absent in this checkout (removed with added scratch
files at 2c6cdae5f); the retained assumptions supply the resolution context.
Preserve T5's CopyView and layered battlefield P/T alongside main's decklist
disclosure fields. Navigation prefixes, offer IDs and state digests stay fixed;
recompute derived frame hashes for the combined projections. Reconcile the
current authored-match parity receipt's source manifest with the merged source,
then verify the unchanged command tape and behavioral evidence.

The first focused run passed all navigation cases but found stale parity viewer
hashes. The producer confirms that only `viewer_observation_sha256` and
`viewer_state_sha256` differ: combined layered P/T and decklist projections
require new hashes. State witnesses, command tape, transitions, presentation,
and stale-object proof remain identical. Refreshed those derived hashes from
the producer; no behavioral assertion was relaxed.

Focused verification: navigation plus authored-match parity initially reported
9 passed / 1 stale receipt; after correcting only the derived viewer hashes,
the failed parity test passed (1 passed). The advice schema equals its Pydantic
producer. Native extension rebuilt and replaced atomically. No push or landing.

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
- pr-publish (2026-11-02, headless): the graph stepped into pr-publish while
  MTG-225 (the directed base) still had no PR. Jack Heart's steer holds the
  branch until MTG-225 publishes and the Task says to stack, so publishing a
  ready PR against main was read as outside the boundary and skipped; a
  blocker was returned instead. If Jack Heart wants a PR against main before
  the T6 sync (review of the batch as it stands), `lf pr publish` from this
  checkout does it with no further change.

## MTG-238 sync onto main a8abf01b1 — 2026-11-02

Jack Heart's latest direction supersedes the earlier T6 hold above. The owned
sync combines the crate move and sparse Keywords with T5 copy effects,
Nonlegendary, layered card observations and main's decklist disclosure.
Airship now uses a conditional TypeGrant and a source-relative namesake
predicate; name matching reads layers 1–3 without the static index.

Current-world w17 evidence is regenerated from the combined tree, retaining
the recorded Commands. Historical worlds and frozen protocol evidence stay
unchanged. This sync step runs focused reconciliation proof only; the following
Gate owns the full debug/release, Python and arena-admission checks before
publication. The waiting sync caller owns the push.

Re-recording completed: 14 semantic cases / 3,396 Commands and 15 Phase rows;
authored parity covers 142 Commands / 858 viewer checks; public-commitment
parity covers 69 commitments with no gaps. Both parents' Command tapes and
game outcomes agree. Main's allocated object ids differ because earthbend
now creates effect records; the branch's terminal state digest is retained.
The rebuilt extension was installed atomically. Focused post-sync proof:
Airship/War Balloon, Ran and Shaw, copy facts and layered observations, plus
Academy retained navigation/parity and the four belief receipt bindings.
