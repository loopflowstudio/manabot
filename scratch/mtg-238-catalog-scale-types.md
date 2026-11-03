# MTG-238: catalog-scale types

## Accepted direction and delivery

Jack Heart approved conditional TypeGrant, the BecomeArtifactCreature fold and
sparse keyword serialization together on 2026-11-02 (`4092cdb4`). Later that day,
`f3e681f7` and `84c1ad84` selected **MTG-238 first**, superseding the T6 hold and
stack prerequisite: publish against main for review; T6 integrates the crate
move afterward. Main's T5 supplied Airship, so its namesake conversion is here.
Recovery direction `d19b72d6` retains PR [#414](https://github.com/loopflowstudio/manabot/pull/414).
No auto-merge or landing is authorized; publication is separate from acceptance.

Jack Heart subsequently selected concept-review **1a** (`0544142c`, reinforced
by `8a0f7f33`): remove the single-word keyword capacity limit now. This resolves
the recorded scope blocker. Storage must grow with the declared parameterless
vocabulary while preserving the set API, Copy semantics, exact present-keyword
serialization, tokens, content identity and state digests. No new mechanics,
observation columns, Python migration or world bump accompany this extension.
The primary conversation owns the local `scratch/pr-review.html` walkthrough.
The extension is included in PR #414 at `bd4f909a9`; publication does not imply
completed validation or review acceptance.

The accepted experience is one card definition shared by engine, learning and
viewer-safe observations. Airship offers ordinary crew; eight namesakes grant
Creature automatically, losing the eighth removes that grant, and a separate
crew record survives until end of turn. An earthbent land that leaves and returns
is a new object with no former animation. Replaying bound Commands reproduces
these outcomes; fork caches never become another rules authority.

## Design and invariants

- **Counters:** `Counters(SmallVec<[(CounterKind, i32); 2]>)` stores positive
  entries in CounterKind order. `get`, `change`, `iter`, `is_empty` and map
  serialization retain their meaning and bytes. By-value consumers clone;
  adding a kind no longer enlarges every permanent.
- **Keywords:** one `keywords!` declaration generates Keyword, ALL and names.
  `Keywords` holds inline words sized by `ceil(ALL.len()/64)`; today's 14
  parameterless variants occupy one word. Private `KeywordBits<WORDS>` lets
  tests exercise real storage at ordinal boundaries without inventing mechanics.
  `of`, `has`, `insert`, `union`, `without`, `iter`, `names` and Copy remain.
  Serialization visits present enum variants in declaration order, never words
  or name strings: serde unit-variant identity also supplies learning tokens.
  Append variants to preserve order/indices. Parameterized ward/firebending
  remain separate rules types.
- **Observations:** all current keywords are evergreen. `KeywordData` and
  `PyKeywords` retain their 14-boolean surface and frozen columns.
  `ContentPack::public_definition` uses KeywordData to preserve capture v4 and
  historical w15 evidence. The exhaustive observability guard maps Keyword to
  columns; a first non-evergreen mechanic must add fact rows, native/Python
  encoders, text and tokens together, with its Python API migration.
- **Animation:** `Game::animate_permanent` writes three indefinite records at
  one timestamp: BecomeCreature (layer 4), SetBasePt 0/0 (7b), AddKeywords(haste)
  (6). They expire with the object (CR 400.7). Crew uses BecomeCreature on a
  printed artifact, with its own duration. The one effect writer invalidates
  caches for nonempty batches; it checks emptiness before mutably borrowing
  tracked state. `record_continuous` still allocates a timestamp for an empty
  batch through `id_gen_mut`, so empty batches are not state/hash no-ops.
- **Conditions:** War Balloon's own TypeGrant uses SourceCountersAtLeast(Fire,3);
  Zhao's StaticLandType condition uses SourceCountersAtLeast(Conqueror,1).
  Airship uses ControlsAtLeast(8, namesake). A grant's condition is asked of its
  source. Names compare layers 1–3 without the static index; counter/name-only
  predicates never ask animation while the index is being built. The definition
  guard allows source questions in type/land-type conditions (`SourceStatic`),
  while scoped P/T statics reject them. Threshold counting resolves the source
  once and stops at the requested count.
- **Crates:** `managym` → `managym-cards` → `managym-state`. Shared rules types
  are re-exported as `managym::state`; player/deck and canonical hash stay with
  the engine (Deck's PyO3 methods belong in its defining crate). Cards are one
  function each, collected through catalog()/incomplete(). Card tests live
  beside sets and mount the engine's `tests/support` by path. Default workspace
  members cover all three; isolated card checking does not build the engine.

## Delete — do not maintain

Completed cuts; no remaining predecessor path:

- Dense counter arrays; preserve their public operations and ordered map bytes.
- Boolean-field Keywords, Keyword::bit and Keywords(u64); separate ALL/name
  authorities. The enum declaration and vocabulary-sized inline storage survive.
- Permanent.animated and animated_creature; BecomeArtifactCreature and its reader;
  CreatureWithCounters, CreatureWithNamesakes, CounterThreshold and dedicated
  readers; ControlsMatching and StaticLandType.while_counters. Shared records,
  TypeGrant and Conditions own the behavior and retain its directed tests.
- Engine-owned `src/cardsets` and moved state modules; card tests in the old
  engine directory. Live audit, CI, packaging and source-manifest consumers use
  the new owners. Frozen source/evidence is retained.
- Observability's `set_keyword` string-to-enum round trip and indexed 14-row
  keyword table. Tests iterate Keyword::ALL directly through one exhaustive
  column mapping; they still check printed/granted tensor columns and tokens.
- The temporary Vec in `push_effects`; a peekable iterator checks emptiness
  before the shared mutation. No duplicate effect writer was introduced.

## Identity and evidence boundaries

The original batch moves content-derived token values, not learning schema 10
(which already existed on main). Counter serialization alone is compatible;
removing the serialized animation field changes match-state hash contract to
**12**, and earthbend records allocate ids. Mutable w17 corpora were rebound:
semantic kernel (14 cases, 3,396 Commands, 15 Phase rows), authored match/parity
(142 Commands, 858 viewer checks), public commitments (69, no gaps), Academy
fixtures and belief v3 pins. Both parents retain the same authored Command tape
and game outcome; allocated ids differ from main. Frozen worlds, checkpoints and
historical receipts are unchanged. Parity source identity includes both new crates.

Keyword capacity is storage-only. `.runs/mtg-238-keyword-capacity/` retains exact
pre/post snapshots: 308,291 definition JSON bytes, 440,288 serialized catalog
bytes (all tokens/offsets), and world/content identity
`w17 / b8f5d43374da2406f3e6cba475362b563e36f8ed651b69a4d30054aded33c7f2`.
They matched byte-for-byte. Authored parity matched 142 Commands, 143 checkpoints
per surface and 858 viewer checks; only source fingerprints changed. A moved
existing digest would refute the storage-only extension and must be investigated,
never hidden by regenerating game evidence.

## Completed validation and its limits

PR #414 was first published at `3235684b8`. Recovery merge `bb9853218` integrated
main `a1a02a76f`, including `51fbacbbf` cache preservation and Engine Budget CI.
Recovery/doc checkpoints through `cfa3aa182` were published and mergeable at the
last inspection (2026-11-02); runtime source was unchanged after `0a9509a9b`
until the keyword extension. These are dated observations, not current CI status.

Recovery logs in `.runs/mtg-238-recovery/` establish pre-extension debug/release
Rust suites (882 passed, 2 ignored each), workspace fmt/clippy, exact Python Unit
Tests (103 passed, including infra, clean-machine verifier and deploy/play),
website (21), and rebuilt-extension authored/observation parity plus tokens (28).
The audit failure in CI run `41551942865`, job `125641513310`, was read before
repair: stale `src/state/mana.rs` anchors followed the crate move, and CR 400.7j's
index comma drift was reconciled with its authored review. All 3,322 rule
dispositions and source spans/digests remain; offline validation passed. The
pinned raw TXT was unavailable, so source-based regeneration was not claimed.

Earlier first-publication checks (`.runs/mtg-238-publication/`) include Academy's
7 permanent-note tests, Svelte check/build and 10 isolated headless browser checks;
servers stopped. Broad Python had 956 passes, 1 skip and a stale source fingerprint;
refreshing that fingerprint changed no game evidence, and 53 focused checks passed.
These results predate the cache merge and do not certify later trees. Older T5
integration also corrected 23 tests to use observation KeywordData booleans;
pre-sync broad results are not combined-tree proof.

The capacity extension passed two debug state keyword tests, the engine's ordered
serialization test, workspace formatting, state-crate clippy and a release wheel
build/atomic installation. Controlled ordinals 0–191 cover membership, duplicates,
union, removal and copy independence; serialization pins all 14 current values.

Both recovery broad-consumer logs (`python-consumers.log` and
`python-consumers-complete.log`) stop without a completion result. Provider
interruptions are not validation failures, but neither filename proves a pass.
CI `41566136106` on `0a9509a9b` passed Python Unit, Rust, Academy UI, clean-machine
Play, protocol/semantic conformance and Academy Integration; general integration
and Engine Budget were cancelled when superseded, and aggregate Tests Result
failed. Its candidate artifact lacks a completed reference comparison.
No passing performance budget is claimed.

Reconciliation on 2026-11-02 confirmed main remains `a1a02a76f` and PR #414 is
open and mergeable at `bd4f909a9`, which includes the capacity extension and
compression changes. CI `41571072018` and release-packages `41571071905` are
running on that head; Detect Changes and package preflight passed. They supersede
`41567370391` and release-packages `41567370387` for current-head assessment,
not as completed proof.

## Concept review — 2026-11-02, `bd4f909a9`

Outcome: no design change and no open product choice for this Task. Jack Heart's
accepted 1a is implemented: declaring vocabulary determines storage capacity;
authors and consumers retain the same set operations and ordered enum values.
The earlier review page's unanswered 1a/1b choice is stale and is replaced by
the accepted result. Existing authoring guidance in AGENTS.md and the crate
ownership guidance in managym/README.md already express this contract.

The normal path, loss and recovery still have one authority: namesake conditions
derive creature status, crew records carry their own duration, and departing
objects cannot bring their former earthbend records back. A copied keyword set
must remain independent of its source across every word. Silent aliasing at
ordinal 64, a changed existing serialized value/token/digest, or recovery that
restores an expired effect would invalidate the design's implementation; the
retained boundary, serialization and directed gameplay evidence addresses these
contracts. The interactive Airship sequence on the page illustrates the accepted
contract; it is not a new combined gameplay proof.

No further implement direction follows from this review. The next proof remains
the gate's full affected suites and current-head CI/performance assessment below.
This step inspected source, the latest patch and retained evidence; it did not
rerun runtime tests, change code, publish, or select Graph navigation.

## Remaining owners

Compression evidence: 37 focused debug tests (state, continuous effects,
selected cards/observability and canonical hashes), release wheel build/atomic
install and exact definition/catalog/world comparisons passed. The retained
authored match reproduced all 142 Commands and 858 viewer checks; only three
source-file fingerprints and their aggregate changed. Full suites and CI remain
with gate/publication below.

Publication completed the affected full debug/release Rust, workspace clippy,
isolated `cargo check -p managym-cards`, and the broad consumer suite below.
The final results and the targeting repair are recorded in the publication section:

```sh
uv run --extra dev --extra play --extra notebook --extra artifacts --extra reports pytest tests/env tests/academy tests/belief tests/semantic tests/verify tests/model tests/test_package_boundaries.py tests/eval/test_evaluation_models.py tests/eval/test_mirror_admission.py tests/training/test_monitor_evaluation.py -q
```

The correct suite is `tests/verify`; the earlier `tests/verification` command
collected nothing. CI and packaging passed on `95c7eb7ed`, including Engine Budget;
no reference adoption is required. The final targeting repair is locally validated
and will start new CI on publication. Jack Heart reviews before merge; T6 remains
downstream. No product judgment remains for the accepted batch.

General layer dependency ordering, the Oracle-text generator, non-evergreen
mechanics and printed battlefield P/T protocol changes remain outside this Task.

Prior full plan/history: `cfa3aa182:scratch/mtg-238-catalog-scale-types.md`;
original design/hold: `3ed5b5baa:scratch/mtg-238-catalog-scale-types.md`.
Uncommitted pre-compression notes are preserved in
`.runs/mtg-238-compress/{plan,questions}-before.md`; pre-capacity concept notes in
`.runs/mtg-238-keyword-capacity/plan-before.md`. Engine memory now records the
accepted capacity extension and compatibility evidence; no child Wave directories
were present. Memory and scratch remain within their configured budgets.

Check (2026-11-02): `git diff --check` and `lf context` — passed; memory and scratch within budget. Retained runtime proof reused; full suites owned by gate.

## Publication validation — 2026-11-02

`lf sync` checkpointed the inherited notes at `95c7eb7ed` and merged main
cleanly (still `a1a02a76f`). It pushed that checkpoint, superseding CI
`41571072018` before its reference comparison completed. CI `41571694122`
and release-packages `41571694041` assess the checkpoint; the latter passed
Linux/macOS package and installation checks.

Publication review found a concrete regression in the new animation-read guard:
it checked only top-level predicate types, so Fire Nation Engineer's
"creature or Vehicle" target excluded an earthbent land. The expanded real-card
test failed on that omission, then passed with recursive
`CardPredicate::reads_types`, including resolution of the counter onto the land.
This preserves name/counter-only recursion avoidance and introduces no serialized
field or vocabulary. Full debug tests passed both before and after the repair
(885 passed, 2 ignored); the pre-repair full release suite also passed 885/2.
Workspace fmt/clippy and a fresh-target isolated card-crate check passed after
the repair. The broad Python consumer suite completed on the pre-repair extension
with 912 passed, 1 skipped.

Final repair `7910f4ed3` passed full debug and release Rust (885 passed, 2 ignored
in each), fmt, workspace clippy, isolated card checking and a release wheel build.
The wheel's CPython 3.12 extension was atomically installed. Definitions (308,291
bytes), serialized learning catalog (440,288 bytes) and world identity (107 bytes)
remain byte-identical to the pre-capacity snapshots. Authored parity reproduced
142 Commands, 143 checkpoints per surface and 858 viewer checks; its only changes
are two source hashes and their aggregate. The complete broad consumer command
above passed again against the repaired extension: 912 passed, 1 skipped in
282.55 seconds. Offline inventory validation retained all 3,322 entries.
Logs live in `.runs/mtg-238-publish-final/`.

Hosted CI `41571694122` completed successfully on pre-repair `95c7eb7ed`, as did
release packages `41571694041`. Engine Budget compared that candidate and reference
`4b5e9c0e4` on one runner: worst reply 11.211 vs 10.300 seconds, total search
342.434 vs 317.923 seconds, throughput 8,249.1 vs 9,037.6 steps/s. All search/step
digests and counts matched, and all timing checks passed the unchanged 10% budget.
No baseline adoption or speedup claim follows. This hosted result predates the
nested-predicate repair; publication starts new CI for it, which is not yet
completed proof. Main remains `a1a02a76f`; PR #414 remains for review, with no
auto-merge or Task completion.
