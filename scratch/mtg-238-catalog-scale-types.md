# MTG-238: catalog-scale types

## Accepted outcome and direction — 2026-11-02

Sparse counters, enum-backed keywords, separate rules/card crates and generic
continuous effects and counted conditions replace catalog-sized fields and
card-shaped predicates. Jack Heart approved the conditional TypeGrant,
BecomeCreature fold and sparse keyword serialization as one content-digest
batch on 2026-11-02.

Jack Heart's later decisions that day (comments `f3e681f7` and `84c1ad84`)
supersede the T6 dependency hold and effect-vocabulary stacking prerequisite:
MTG-238 publishes against main first, as one PR for review, without auto-merge.
T6 integrates the crate move afterward. Main includes T5, so Airship's namesake
conversion belongs here. No product decision remains for this batch.

The sync onto main `a8abf01b1` is complete at merge `469b7eeba`, followed by
observation-test fixes in `3ed5b5baa`. T5's copy effects, Nonlegendary and
layered battlefield characteristics are preserved. The original plan and
superseded publication hold are retained at
`3ed5b5baa:scratch/mtg-238-catalog-scale-types.md`.

## Design (accepted shape, 2026-11-02; batch carried the same day)

1. **Counters** — `Counters(SmallVec<[(CounterKind, i32); 2]>)`, entries kept
   in `CounterKind` order with positive counts only. Same API (`get`, `change`,
   `iter`, `is_empty`) and the same serialization (map of present kinds in
   kind order), so neither content digests nor state hashes move. No longer
   `Copy`: the engine's by-value copies become clones.
2. **Keywords** — `enum Keyword` (14 variants, all evergreen today) and
   `Keywords` as a `u64` bitset: `has`, `insert`, `union`, `without`,
   `names`, `iter`, `Keywords::of(&[Keyword])` (no `remove`; `without`
   is the only subtraction, as the engine's layer-6 loss needs). Serialized
   as the sequence of keywords present in enum order (absent kinds omitted):
   a definition's tokens carry only its keywords and a new keyword changes no
   other card's encoding. Evergreen keywords keep their card/permanent tensor
   columns through `KeywordData`; the observability guard matches `Keyword`
   exhaustively, so the first non-evergreen keyword must name its fact row.
   Set files write `Keywords::of(&[Keyword::Flying])`.
3. **Animation** — `Permanent.animated` is gone. Earthbend records three
   `ContinuousEffect`s at one timestamp on the land: `BecomeCreature`
   (layer 4), `SetBasePt { 0, 0 }` (7b) and `AddKeywords(haste)` (6),
   `Indefinite`, so they end when the object leaves (CR 400.7) exactly as the
   flag did. Crew's "becomes an artifact creature until end of turn" is the
   same `BecomeCreature` record (a Vehicle is a printed artifact);
   `Game::animate_permanent` is the one earthbend writer. Removing a
   serialized `Permanent` field changes the canonical encoding, so
   `MATCH_STATE_HASH_VERSION` is 12.
4. **Counts as conditions** — `StaticTypeGrant` has an optional `condition`
   (`skip_serializing_if`); War Balloon is `TypeGrant { This, [Creature],
   SourceCountersAtLeast { Fire, 3 } }`. `StaticLandType.while_counters`
   is `condition: Condition` (Zhao: `SourceCountersAtLeast { Conqueror, 1 }`).
   `Condition::ControlsMatching` is `ControlsAtLeast { count, predicate }`.
   `Game::permanent_is_animated` = a `BecomeCreature` record or the
   permanent's own `This`-scoped Creature grant whose condition holds (no
   recorded ability loss); it reads only the card and records, never the
   static index, which asks it while being built. A type grant's condition is
   asked of its source (`source_condition_holds`) and predicate matching asks
   animation only for type-reading predicates, so a counter or name question
   never recurses. Phoenix Fleet Airship uses the same `TypeGrant` with
   `ControlsAtLeast { count: 8, predicate: namesake }`; names compare the
   source and candidate after layers 1–3 without asking the static index.
5. **Crates** — `managym/Cargo.toml` is a workspace root (package +
   members). `managym-state` (`managym/state/`) holds the former `src/state/`
   minus `hash.rs`, `player.rs` and `deck.rs`; the engine re-exports it as
   `managym::state` and keeps `state::hash`. `managym-cards` (`managym/cards/`) holds the set modules and
   `catalog()` / `incomplete()`; the engine reaches cards only through those.
   `tests/cards` lives in `managym/cards/tests/` (engine as dev-dependency;
   `tests/support` mounted by path so each primitive stays defined once).
   `cargo test` in `managym/` runs all three through `default-members`.

## Implemented behavior and evidence

All five design items are implemented, including Airship's namesake condition
and the removal of CreatureWithNamesakes and its dedicated readers. Incoming
T5 keyword literals and animation records use the catalog APIs. The engine's
Python-bound player/deck types remain in `src/state/`; shared scenario helpers
remain in `managym/tests/support/` and card tests mount them by path.

Counter serialization alone is byte-compatible. Removing `Permanent.animated`
changes canonical state encoding; effect-record allocation also changes main's
object ids. The accepted batch therefore uses hash contract 12 and regenerated
open-world w17 evidence. Frozen worlds, checkpoints and historical receipts
remain unchanged. Learning schema 10 already existed on main and remains 10:
keyword lists change definition-derived token values, not kinds, roles or dtypes.

The combined-tree extension was rebuilt and installed atomically. Current w17
semantic-kernel evidence (14 cases, 3,396 Commands, 15 Phase rows), authored
match/parity (142 Commands, 858 viewer checks), public-commitment parity
(69 commitments, no gaps), Academy fixtures and belief v3 pins were regenerated.
Both parents retain the same authored Command tape and game outcome; the
branch terminal digest is retained, while main's allocated object ids move.
Parity source identity now includes the state and card crates.

Post-sync focused validation passed 15 debug Rust card/observability checks and
11 Python navigation/parity/belief checks. The initial Rust compile found
23 merged assertions applying `Keywords::has` to observation `KeywordData`;
`3ed5b5baa` corrected them to the retained boolean observation fields, and the
rerun passed. No runtime code changed after the extension/evidence rebuild.
Earlier full debug/clippy and Python results (864 passed, 1 skipped) predate the
T5 integration and do not establish broad correctness of the combined tree.

## Concept review — combined T5/catalog tree, 2026-11-02

Outcome: nothing to change and no open product choice within MTG-238. Jack
Heart's accepted batch still fits the implementation after the main sync;
there is no new implement direction. The earlier review page's T6 hold is
superseded, not a remaining design dependency.

The concrete interaction remains one native decision surface: a player crews
Airship through its offered ability, or its eighth namesake makes it a creature
without an extra action. `TypeGrant.condition` derives the latter from current
state; crew records `BecomeCreature` until end of turn. Dropping below eight
removes the conditional grant, while a separately active crew effect survives.
Earthbend uses the same effect lifecycle with object-bound indefinite records;
leaving and returning creates a new incarnation, so old animation cannot leak
onto the returned land. Threshold loss, ability loss, expiry and object return
are the counterexamples that would invalidate a permanently stored animation
flag or a browser-derived creature state. Existing card scenarios cover the
threshold, ability-loss, crew-expiry and land-return paths; this review read
those scenarios and the implementation, and did not rerun them.

The consequential distinction is retained: sparse counter serialization alone
preserves bytes, but this combined batch changes definition tokens and state
identity. Recovery must bind current evidence to the new content/hash contract
while retaining frozen evidence; an unchanged Command tape is not an unchanged
digest. Usage guidance already describes the card/rules crate ownership and
native authority correctly. No further usage or skill changes are needed.
The next proof remains the combined-tree gate below, including T5 behavior and
the rebuilt-extension consumers; publication remains separate from merge.

## Publication validation — 2026-11-02

`lf sync --manual` confirmed the branch merges cleanly with current main.
The combined-tree publication checks are complete:

- Rust formatting and all-target/all-feature clippy passed. Full debug tests:
  876 passed, 2 ignored; full release: 878 passed, 2 ignored. The two added
  serialization regressions also passed in debug in the 10-test hash suite.
- The broad Python run covered env, Academy, belief, semantic, verification,
  model and package boundaries, plus the required arena-admission/monitoring
  and CI/release suites: 956 passed, 1 skipped, with one stale authored-parity
  source fingerprint. Regenerating that receipt changed only source hashes;
  its Commands, witnesses and viewer results stayed identical. The corrected
  parity and CI/release checks then passed (53 tests).
- Academy permanent-note tests: 7 passed. Svelte check: no errors or warnings.
  Frontend build passed. Ten isolated headless browser checks passed across
  choice navigation, Learn and trigger ordering; both servers stopped.
- Three formatting fixes and one merged test's `is_creature()` accessor were
  corrected. No runtime behavior changed during publication validation.
- Counter serialization now has a directed byte-level regression across
  inline-storage spill, removal, clone independence and saturation. Sparse
  keyword serialization has an ordered-present-kinds regression. CI selection
  and release staging explicitly cover both moved crates.

The rebuilt extension matched the installed wheel before broad validation;
the final formatted tree was also rebuilt and installed atomically. Its
authored parity, native/Python observation parity and static-token checks all
passed (28 tests).
Logs are local under `.runs/mtg-238-publication/`. Frozen evidence is unchanged.

Publication against main and Jack Heart's review remain outstanding. Publication
and merge are separate; no auto-merge is authorized. T6 is not a blocker.

Outside this Task: `PyKeywords` remains a 14-boolean observation surface;
the first non-evergreen keyword needs its fact representation and a Python API
migration. All current keywords are evergreen, and the exhaustive observability
guard requires an explicit decision for a new one. General layer dependency
ordering and the Oracle-text generator remain outside this Task.

Check result (realign, 2026-11-02): `git diff --check` — passed; existing post-sync focused proof retained, full combined-tree checks deferred to gate/CI.
