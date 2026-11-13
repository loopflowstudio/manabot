# MTG-320: runtime objects and checked card ids

## Outcome and accepted scope

Tokens and spell copies are objects, never cards. Runtime design and examples:
`docs/engine.md#objects-and-zones`. Preserve observation/Python/PyO3 names,
serialized numeric ids, zone order, seeded deals, and frozen evidence. Stop at
review; the task steer says "Stop at review. Do not enable auto-merge."

Steer dbf13009: "If the full split is too big for one PR, land the rename plus
the typed ids here, and say what is left in the PR body."
Steer f81d37a0: "Typed card ids. Add a `CardId` that can only name an object whose
kind is Card. Converting an ObjectId to a CardId is checked once; converting a
CardId to an ObjectId is free. Library, hand, decklists/sideboards and effects
that say \"card\" hold or take CardId"; "`card`/`card_id`/`cards` always holds a
CardId (or a collection of them), and `object`/`object_id`/`objects` always holds
an ObjectId"; "`game.state.objects[*card]` must not survive unless `card` is a
CardId". Definitions, published observation fields and card-name strings are
not runtime-id bindings.

## Zone decision

The preceding design permits this fallback: "If the visitor cannot be kept
byte-exact cheaply, take the second and change the two \"Zones\" rows and the
library/hand bullet." Use ordered `ObjectId` lists for hand/library as well as
graveyard/exile; narrow at card-only consumers with `GameState::zone_cards`.
A parallel card/visitor store would duplicate membership, insertion positions
and shuffle authority. Hidden sampling, observed library tops, zone size,
shuffles and draws must still see non-card occupants in their original places.
In particular, do not fix the existing tucked-token draw behavior here.

QA from the preceding rename saw Treasures in the graveyard at a mana-choice
decision after sacrificing them. That observation rules out removing tokens
inside the move instead of waiting for SBA.

## Delete — do not maintain

Completed in this branch: runtime `Card`, unchecked runtime `CardId`, `CardVec`,
`GameState::cards`, `card_to_permanent`, and the old public u32 `ObjectId`;
object-based deck/sideboard rosters and `ActivationSource::Card`; misleading
mixed-object helper names; the proposed second visitor store.

Compression also removes `ZoneManager::insert_to_zone` and
`remove_object_in_place`, the manual empty-zone constructor, the intermediate
object list in `resolution_cast_candidates`, and eager `foretold_cards` vectors.
`zone_objects_mut` owns zone dispatch; card-only consumers narrow before filtering.
There are no remaining deletion targets in this slice. The physical-card split
below is separate work, not a second implementation to maintain here.

## Implementation

The earlier object-kind/rename implementation is in df58f2bff and its history.
This slice adds checked store conversion, debug-asserted card indexing, typed
rosters, card-only candidates and card-specific play/cost/activation entry
points, including graveyard targets and card-type counts. Generic Action/selection
transport and captured ObjectRefs remain object-based: they also carry
permanents, spell copies and stale identities.
Command ingress narrows card-only operations before they run. Card choices
exclude transient tokens while object-inclusive observation readers keep them.

## Validation and remaining work

The following typed-id checks precede compression; compression results go below.

- `cargo check --manifest-path managym/Cargo.toml --workspace --tests -j 2`:
  passed before compression. Focused debug nextest: 3 regressions passed;
  the expanded card-choice/graveyard-target regression passed again after the
  typed-id narrowing (0.012 s). Commands used `CARGO_BUILD_JOBS=2`, `--profile ci`
  and `--test-threads 2`; the two id/zone tests are in `cr_400_identity.rs`.
- Root `uv run --extra play maturin build --release -i .venv/bin/python
  -m managym/Cargo.toml -o managym/target/wheels`: passed (5m 12s), cp313/arm64
  wheel. Replaced `_managym.cpython-313-darwin.so` atomically; fresh native import
  passed. Workspace rustfmt and `git diff --check` passed.
- Gate: affected suites and final-head semantic conformance, seeded before/after
  QA, and Engine Budget (unchanged digests; same-run numbers in PR body).
- Delivery: refresh the draft PR body and current CI status, preserve the Task
  context block, publish through the caller's delivery step. Do not auto-merge.

## Earlier evidence (not validation of this slice)

- CI 43325427802 on 2a3eae3b0: all 13 jobs passed, including debug Rust,
  conformance, Engine Budget, Clean-machine Play, Python and Academy.
- Same-run Engine Budget numbers and the runner baseline are in
  `scratch/pr-491-body.md`; they predate typed ids and compression.
- After 5dc528319, fmt/test build/clippy passed, focused debug engine tests 30
  passed and token/Azula tests 41 passed. After d608ff5ba, clippy and focused
  stack/copy/counter/cr_6 tests passed (160).
- Earlier QA: 60 random games, 150 Azula games and four Academy games finished.
  Permanent-spell-copy and spell-copy-to-hand paths were not reached in Academy.
  Semantic-kernel-w17-v2 reproduced after fixing ObjectLki counter serde drift;
  no evidence was regenerated for the rename.
- Historical validation details and older findings: git history of this plan
  at df58f2bff. The older Python failure attribution was inference, not a
  reproduced cause; subsequent CI passed.

## Compression rationale

Use the existing zone accessor for insertion/removal and derive empty defaults;
keep all ordered lists and their wire names. Build cast candidates as `CardId`s
and stop at a hand-only restriction before scanning other zones. Foretold scans
return an iterator, so sampling collects once and counts need no vector.
`flow/action.rs::chosen_card` owns the repeated action-boundary error. Observation
focus has one arm for actions that all identify an object; projection locals
name records as records, not ids. No action order, RNG call or serialized field
changes intentionally. The layer benchmark fixture now reads object kind instead
of the removed `is_token` field; all-target Clippy exposed that missed caller.

Compression validation (2026-11-12):

- `CARGO_BUILD_JOBS=2 cargo nextest run --manifest-path managym/Cargo.toml
  --lib --test rules_tests --test cards --test learn_setup_tests --profile ci
  --test-threads 2 -E 'test(cr_400_identity) | test(cr_704_sba) | test(foretold)
  | test(card_capture) | test(learn_tests) | test(learn_authored_setup)
  | test(partial_waterbend) | test(cascade) | test(drannith)
  | test(fire_lord_azula) | test(boiling_rock) | test(yue_casts)
  | test(solstice_revelations) | test(planetarium_casts)'`: 53 passed, no failures
  or errors (1.732 s; JUnit run 823626c6-c439-42d4-b354-a4b6b4911a47).
- Workspace `cargo fmt -- --check` and `git diff --check`: passed.
- `CARGO_BUILD_JOBS=2 PYO3_PYTHON="$PWD/.venv/bin/python" cargo clippy
  --manifest-path managym/Cargo.toml --workspace --all-targets --all-features
  -- -D warnings`: passed after fixing the benchmark caller (4.06 s).
- `CARGO_BUILD_JOBS=2 uv run --extra play maturin build --release
  -i .venv/bin/python -m managym/Cargo.toml -o managym/target/wheels`: passed
  (4m 00s). Atomically installed the matching cp313/arm64 extension; a fresh
  `uv run python` import of `managym._managym` passed.

## Later split

Not built: a tiny Copy physical Card store, a new object per zone change, the
inline CR 400.7 carry-over record, and measurements of layer-view/TriggerIndex
invalidation costs. CardId currently shares its object's row index. No new
emblem or copy-of-a-card kind until there is a producer.
