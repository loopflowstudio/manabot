# MTG-238 assumptions and evidence limits — 2026-11-02

Accepted design, current implementation and remaining checks live in
[the plan](mtg-238-catalog-scale-types.md). No product choice remains for this
batch. Jack Heart's MTG-238-first direction supersedes the earlier T6 hold;
the sync onto main `a8abf01b1` and Airship conversion are complete.

## Representation boundaries

- Counters preserve their serialized map and ordering. Removing the serialized
  animation flag does not preserve state digests: hash contract 12 and the
  open w17 re-record cover that change. Earthbend's records also allocate ids.
- Learning schema 10 was already on main. Keyword lists move content-derived
  token values without changing the schema's kinds, roles or dtypes; no further
  schema bump is needed for this batch.
- `Keyword` keeps its serde unit-variant representation; serializing through
  `name()` would change variant tokens into string tokens. The observability
  guard ties the two spellings together.
- `KeywordData` and `PyKeywords` keep the boolean observation contract, distinct
  from sparse definition serialization. `ContentPack::public_definition` uses
  `KeywordData`; decision-capture version 4 and the retained w15 fixture stay
  stable. All 14 current keywords are evergreen; non-evergreen fact rows and
  the corresponding Python API migration remain future work.
- Airship's namesake condition compares source-relative names at layers 1–3.
  Conditional animation must not call the static index while it is being built.
  Counter/name-only predicates do not ask animation. Source-dependent type and
  land-type conditions use the definitions guard's `Place::SourceStatic`;
  scoped P/T statics still reject source questions.
- `player.rs` and `deck.rs` stay in the engine: Deck's PyO3 methods must be in
  its defining crate. Cards depend on rules types and dev-depend on the engine;
  both test targets share the engine's scenario helpers.

## Inherited decisions still relevant

T5's copy exceptions share `Modification`; Python calls the field `copy_of`
because Pydantic already defines `copy`. Layered battlefield P/T remains in
`base_power`/`base_toughness`; a printed-P/T accessor or removal is an unresolved
follow-up protocol choice for Jack Heart, outside MTG-238. Surveil 1 uses
LookAndSelect; supporting larger authored amounts needs ordering choices.

Main's open/closed decklist authority remains native. All-zero opponent
`decklists` rows mean closed, search refuses closed opponents, and disclosure
does not enter deck/world identity. No observation-width or checkpoint-default
change is part of this batch.

## Evidence boundaries

The post-sync extension and mutable w17 corpora reflect the combined tree.
The authored tape and game outcome agree across both parents; allocated ids
from main move with earthbend records. Focused Rust and Python validation
passed after 23 test assertions were corrected to use the observation boolean
API. The combined-tree publication checks and the source-fingerprint-only receipt
refresh are recorded in the plan; broad pre-sync results were not used as proof. Frozen worlds and historical receipts were retained.

Choice-navigation re-records preserve the recorded offer id when labels repeat;
a label alone can select a different physical copy. The parity source manifest
covers the moved state/card crates as well as the engine.

Complete inherited T5/decklist notes, earlier holds, sync evidence and prior
validation are retained at `3ed5b5baa:scratch/questions.md`. Earlier source
references remain there (`bbb5101eb`, `1e9fd2d7c`, `4c8324a59`); the removed
T5 design link is historical, not a current checkout dependency.

## Post-publication sync — 2026-11-02

Merge `bb9853218` completed the sync onto main `a1a02a76f`, retaining MTG-238's
three-record earthbend animation and main's cache-preserving ID accessor.
Empty effect sets preserve derived-cache revisions, though `record_continuous`
still allocates a timestamp; they are not complete state/hash no-ops. Nonempty effect sets
still invalidate characteristics, triggers and mana through the shared writer.
Main's MTG-237 assumptions remain at `a1a02a76f:scratch/questions.md`;
its performance CI and baseline are retained without selecting new work here.
The two mutable w17 parity receipts retain this branch's vocabulary and hashes;
both were regenerated against the atomically installed combined extension.
Only source fingerprints changed; all non-source evidence matches the branch.
Realign repaired the live audit references and reconciled the index's existing
CR 400.7j anchor drift with its authored review. Offline inventory validation
and the exact Python Unit Tests command passed; the pinned raw TXT was not
available for source-based regeneration. Frozen audit evidence is unchanged.
The plan records the remaining merged-engine checks, republication and review.
