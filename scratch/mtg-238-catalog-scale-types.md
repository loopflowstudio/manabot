# MTG-238: catalog-scale types

## Interactive concept review — 2026-11-02

Jack Heart requested this review in the primary Task conversation. This section
is a proposal under discussion, not approval to expand implementation. The
earlier headless reviews' “no open choice” applies to the accepted batch; this
review asks whether to extend its completion boundary. Preserve the existing
repair work and the MTG-238-first, review-before-merge direction.

Accepted experience: author a card once in the shared rules types; the engine,
learning tokens and viewer-safe observations describe that same behavior.
Playing Airship offers ordinary crew; controlling eight namesakes grants its
creature type without a new user action. Losing the eighth removes that static
grant; a separate crew record remains until end of turn. Replaying the same
Commands in the bound world restores the result. An earthbent land that leaves
and returns is a new object and must lose its former animation. Forked caches
are derived machinery, never a second source of rules.

One mismatch to discuss: `Keywords(u64)` and `Keyword::bit` bind the rules
vocabulary to 64 parameterless keyword variants. Today's 14 fit; parameterized
abilities such as ward and firebending use separate rules types. This is a
storage limit, not an observed gameplay failure, and the Task's motivation
about all keyword abilities does not establish when 64 parameterless variants
will be needed. Sparse serialization already hides internal bit positions, so
removing this limit need not change today's definition bytes, tokens or digests.

Proposed extension (not implemented or approved): keyword-set capacity grows
with the declared vocabulary, while `of`, `has`, `insert`, `union`, `without`
and deterministic present-keyword serialization retain their meaning. Choose
storage in implementation; do not ask Jack Heart to select a container. No new
cards, mechanics, tensor columns, world bump or Python API migration is implied.
The existing requirement still makes the first real non-evergreen mechanic
reach fact rows, native/Python policy encoders, text and learning tokens together.

Draft authoring guidance if accepted: “Declare a parameterless keyword once in
the rules enum and use `Keywords::of`. Vocabulary growth must not alias or drop
an existing keyword. Keep present-keyword serialization order stable. A new
mechanic is complete only when its behavior and every shared observation path
are present; use fact rows for non-evergreen kinds.” Existing crate, native
authority and duration guidance stays unchanged. Draft agent instruction if
accepted: “When changing keyword storage, preserve existing wire bytes and test
membership, union and removal across a machine-word boundary. Never widen a
frozen observation version or regenerate evidence to hide a storage regression.”

Counterexamples and next proof: a keyword beyond bit 63 must remain independent
of bit 0 through insert/union/removal and clone/fork; existing present-keyword
bytes and learning tokens must remain identical. Test representation boundaries
with controlled fixtures rather than inventing supported mechanics. Preserve the
Airship threshold/crew-expiry and land-return cases. A changed existing digest
would refute the storage-only proposal, requiring investigation before publication.
An isolated `cargo check -p managym-cards` checks definitions without the engine;
behavioral tests and rebuilding the linked Python extension still use the engine.

Open question in this conversation: **1a** complete this storage boundary in
MTG-238 (recommended), or **1b** retain the current 64-variant capacity explicitly
and defer that work. No answer has been recorded. This review changes no runtime
code, runs no gameplay tests, and does not restart, publish or merge the Task.
The interactive review owns `scratch/concept-review.html`; keep its proposal
distinct from the Graph's accepted recovery requirements.

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

PR [#414](https://github.com/loopflowstudio/manabot/pull/414) was published
at `3235684b8`. Jack Heart's recovery direction (`d19b72d6`, 2026-11-02)
retains that PR and review boundary after main advanced. The subsequent sync
onto `a1a02a76f` is complete at `bb9853218`, incorporating `51fbacbbf`'s
derived-state caches and engine performance CI. Recovery checkpoints through
`0a9509a9b` are already on PR #414, which is mergeable against main
`a1a02a76f`. Final consumer validation and publication handoff remain; no
auto-merge or landing is authorized.

The earlier T5 sync at `469b7eeba` and observation-test fixes in `3ed5b5baa`
preserve copy effects, Nonlegendary and layered battlefield characteristics.
The original plan and
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

## Concept review — combined T5/catalog/cache tree, 2026-11-02

Outcome: nothing to change and no open product choice within MTG-238. Jack
Heart's accepted batch still fits the implementation after the cache sync at
`bb9853218` and the live audit-reference repair through `0a9509a9b`;
there is no new implement direction. The earlier review page's T6 hold is
superseded, not a remaining design dependency.

This review rechecked the current conditional-grant reader, animation writer,
sparse serializers, Airship threshold/ability-loss scenarios and fork identity
scenario against the accepted model. The recovery checkpoints change live
audit references and notes, not the engine model. No usage or skill revision
is needed. The next proof is the unfinished broad consumer suite and the
publication checks recorded below; completed recovery checks remain evidence
for this engine tree. This step ran no tests, made no implementation changes
and did not refresh remote CI status. The review page now reflects that
boundary rather than listing the completed Rust/extension checks as remaining.

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
That combined T5/catalog tree subsequently passed the publication checks below;
the newer cache merge has a separate recovery boundary.

The cache merge adds no player-facing concept or second authority. Search
forks inherit derived views, but changing an effect or an object's incarnation
must invalidate the fork's affected views without changing its parent. A
stale creature type after threshold loss or a returned land retaining haste
would violate the same accepted contract, regardless of cache reuse. The
effect writer, tracked accessors, fork-isolation scenarios and effect-lifecycle
scenarios were inspected in that concept review; tests were not rerun there.
The subsequent recovery checks below cover the merged engine, including debug
cache-recomputation assertions; broad consumer validation remains open.

One wording clarification: an empty affected set preserves derived-cache
revisions, not all state. `record_continuous` still allocates its timestamp
through `id_gen_mut` before `push_effects` returns. This matches main; treating
an empty call as a complete state/hash no-op would be a different contract.
The audit's relocated live anchors likewise do not change a rule disposition
or turn historical evidence into proof of the merged tree. No design change,
new implement direction or open product choice results from this review.

## First-publication validation — 2026-11-02, `3235684b8`

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

These results predate the cache merge and do not certify `bb9853218`.

## Recovery agreement and remaining work — 2026-11-02

The merged effect writer preserves the three earthbend records and uses
`TrackedState::id_gen_mut()` for timestamps. Empty effect batches return before
advancing derived-cache revisions; nonempty batches still invalidate characteristics,
triggers and mana. The sync record reports an atomic extension installation and
regenerated mutable parity receipts; the committed authored-parity diff changes
source fingerprints only. No runtime correction was needed during realign.

CI run `41551942865`, job `125641513310`, was read before repair: 77 tests
passed and the inventory test failed on the removed `src/state/mana.rs` path.
The live audit reviews and index now name the moved rules and card-test files;
engine-owned player references stay in place. A pre-existing trailing comma in
the index's CR 400.7j anchor was reconciled with its authored review. Validation
remains strict, all 3,322 rule dispositions and source spans/digests are retained,
and frozen audit receipts and source metadata are unchanged. The pinned raw TXT
is unavailable locally, so source-based regeneration was not claimed; offline
validation and exact agreement with authored overrides cover this repair.

Recovery evidence from the stopped publication attempt is retained under
`.runs/mtg-238-recovery/`. Both full Rust suites completed: 882 passed and
2 ignored in debug (development optimization level 1, assertions enabled)
and release. Workspace formatting and all-target/all-feature clippy passed.
The exact Python Unit Tests command passed all 103 tests after the final two
card-directory owner references were repaired in `0a9509a9b`; website checks
passed 21 tests. The fresh wheel was byte-identical to the installed extension
and was installed atomically. Authored parity, native/Python observation parity
and static-token checks then passed all 28 tests. No runtime source changed
between these checks and `0a9509a9b`.

The publication attempt ended on a provider-capacity error, not a recorded
validation failure. Its broad Python consumer log ends during progress with no
summary or successful exit receipt; it establishes no complete run. Remaining
local validation is the broad env/Academy/belief/semantic/verify/model and
package-boundary suite plus arena admission/monitoring, using `tests/verify`
(the earlier `tests/verification` invocation collected nothing). Its command is
`uv run --extra dev --extra play --extra notebook --extra artifacts --extra reports pytest tests/env tests/academy tests/belief tests/semantic tests/verify tests/model tests/test_package_boundaries.py tests/eval/test_evaluation_models.py tests/eval/test_mirror_admission.py tests/training/test_monitor_evaluation.py -q`.

At reconciliation, PR #414 points to `0a9509a9b`, is mergeable, and has no
auto-merge request. CI run `41566136106` has passed Python Unit Tests and
Semantic Kernel Conformance; the remaining checks, including Engine Budget,
are pending. Final publication preparation/handoff remains with the caller,
followed by Jack Heart's review. Completed recovery checks need not be replayed
without a relevant source change. Any required performance-baseline adoption
must retain the candidate artifact and reason under the existing policy; no
new performance result or baseline change is claimed here. T6 remains
downstream, not a blocker. No product decision is needed.

Outside this Task: `PyKeywords` remains a 14-boolean observation surface;
the first non-evergreen keyword needs its fact representation and a Python API
migration. All current keywords are evergreen, and the exhaustive observability
guard requires an explicit decision for a new one. General layer dependency
ordering and the Oracle-text generator remain outside this Task.

Check result (realign, 2026-11-02): `uv run --python 3.12 --extra dev pytest tests/infra/ tests/academy/test_clean_machine_verifier.py deploy/play/tests/` — 103 passed (reused completed recovery evidence); broad consumer completion and pending CI remain with publication.
