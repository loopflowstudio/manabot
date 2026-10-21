# Elemental implementation plan — MTG-161

Reconciled 2026-10-20. **Accepted outcome:** Jack Heart's 2026-10-20
MTG-157 direction covers all 42 owned cards through native and compiled rules,
complete legal choices, viewer privacy, deterministic replay and exact forks.
The ownership list and delivery constraints are in
[elemental-combat.md](elemental-combat.md); the accepted ten-deck selection is
in [elemental-decks.md](elemental-decks.md) and `elemental-decks.json`.
Native registration alone does not satisfy acceptance. This is an accepted
outcome with a partially implemented approach, not completed delivery.

## Decisions and evidence boundaries

Jack Heart accepted the selected content expansion and autonomous delivery on
2026-10-20. Cube caps apply per deck including its outside-game Lesson pool;
basics are unlimited and cross-deck reuse is allowed. Printed rules remain
mandatory, including on reused definitions. No reduced versions are accepted.
Final ten-deck admission belongs to the fourth Task; this Task still owns
complete native/compiled support and directed evidence for its 42 cards.

The earlier Rules chapter describes only Allies versus Lessons. This Task's
explicit expansion supersedes that content boundary for the selected pool,
without authorizing general expansion or Wave/chapter edits. Rules' GOAL.md
and the repository's retained `wave/rules/MEMORY.md` were read alongside
`.lf/directions/curated-rules-evidence.md`. The `.lf/wave/rules/` binding has
no MEMORY.md; neither Rules location has child Wave directories. Retained
memory preserves historical w4 proof and unresolved empirical acceptance;
it does not certify w7 or override this Task's accepted expansion. Wave
memory remains read-only under the accepted Task constraint.

The source snapshot in `elemental-oracle.json` contains all 42 owned names.
The previous implementation pass records its source as Scryfall's 2026-10-20
bulk export; reconciliation checked the retained contents, without fetching
new text. The cube CSV and deck JSON are also present. Registrations and
Scryfall shell conformance do not establish full behavioral fidelity.

HEAD and the locally available origin/main both identify `3d3737c6`.
Since the earlier `20fc96b7` comparison, `1c6ed80d` added Academy checkpoint
selection and `3d3737c6` aligned research and public naming. Neither changes
this Task's card ownership or admits historical checkpoints to w7. CI changes optimize
engine builds while retaining debug assertions and select checks by affected
areas. They do not waive debug validation, authored-world checks, pinned Linux
checks or configured release-browser coverage. No fresh remote comparison or
full-suite result is claimed by this reconciliation.

## Implemented native slice

`managym/src/cardsets/elemental.rs` registers all 42 owned real cards and Food and
Spirit tokens. All 42 have entries in the Scryfall fixture. Existing tests in
`managym/tests/rules/elemental_cards.rs` contain 42 directed cases spanning the
slice; this is not a claim of exhaustive rule coverage.

The shared `ManaCost` path represents two-color hybrid pips, preserves their
mana value and color identity, and backtracks payment assignments. Native and
Python observation metadata expose the pips without changing tensor widths.
Basic-land search derives private candidates from the current library for
legal actions, observations and validation. Candidate visibility does not pin
them as known top-of-library cards. Selection and failure to find converge on
one shuffle/continuation path; determinization refreshes opponent private
choices. Learn and revealing look/select use shared reveal-to-hand bookkeeping.

Other implemented primitives include optional discard/draw, counted forced
discard, noncreature-spell triggers, dynamic damage/pump values, conditional
combat restrictions, Spirit blocking restrictions, life-gain prevention,
triggering-Ally counters/haste, and activated tap costs. Independent activated
effects survive their source leaving the battlefield. Existing card definitions
were adapted to the new fields; preservation of their full behavior remains a
regression requirement.

The checkout identifies the unfinished world as w7, with match-state hash v4.
WORLDS.md leaves world freeze and release certification to gate.
Historical authored sources, IR and receipts retain their identities. The
CPython 3.12 extension was rebuilt and installed after this implementation;
the focused Python/native indestructible encoder parity check passes.

The supplied earlier feedback described 30 registrations, 23 tests,
resolution-time spell modes and first-target-only binding. Current inspection
supersedes those observations: spell modes precede targets/payment,
independent explicit clauses bind through casting, triggers and resolution,
and all twelve previously absent cards have registrations, Oracle fixtures
and directed play cases. Ability modes remain resolution-time choices.
Scry supports arbitrary top/bottom order. Graveyard targets use exact
ObjectRefs; mill/select, targeted tap activations, indestructible and
first-matching spell cost history are implemented. Counter mutation journals
its permanent at the event authority after a fork test exposed missing undo.

The first-target-only trigger helpers, duplicate action-target enum/conversions
and separate optional-target inference were removed. Shared
`effect_target_requirements` preserves implicit single-clause behavior;
`TargetClause` represents independent clauses. `state::target::Target`
re-exports the game-object type, preserving import paths and serialization.
Scry has one ordered continuation with private inspected-card pinning.

## Compiled admission and shared consumers

`content/semantic/v1/elemental.source.json` and its checked-in IR admit all 42
owned cards, five basics and four tokens as `elemental-combat-v1`. Shared typed
operations cover search, counted discard, quantities, modes, target clauses,
mill/select, returns and conditional statics. Opcode numbers 1–23 are unchanged.
All 51 definitions match the native rules structure, without target binding
normalization. A spell's inferred binding uses the selected mode and the shared
effect traversal; role declarations cannot override modal legality.

Static trigger guards and static programs share the typed condition parser and
lowerer. Explicit targets and clause roles share one requirement decoder, and
all token-creation operations use one checked token lookup. The offline compiler
compiles each nested instruction list through the same boundary; casting checks
clause-local distinctness directly against bindings without temporary lists.

Two 46-card validation rosters split the owned pool between seats, with five of
each basic per roster. Both orientations select the exact compiled catalog
entry and play to terminal. These are validation setups, not the final ten
product decks owned by the fourth Task. Historical two-deck and Jeong sources,
IR, learning schema, receipts and corpora retain their identities.

Learning schema 3 lives separately at `content/semantic/w7/learning_schema.json`.
It preserves typed hybrid pips and nested modes/clauses and exposes all new
quantities and statics without truncating the catalog. Runtime binding verifies
the exact source/IR hashes. The real policy consumer exposed a permanent-feature
slice that dropped indestructible; it now removes only the categorical join.
A compiled search Command passes through acting-view belief tracking and the
history-enabled policy with nonzero history gradients and determinization-safe
public history. These checks make no inference-quality or strength claim.

Control-change cases cover Boomerang's draw recipient, Zuko's Clue recipient
and the captured controller of an announced trigger. They inject the changed
controller through the scenario seam; they do not establish a general
control-change spell implementation. Structure parity compares full rules
definitions with ability ordering normalized, while replay executors share
engine primitives; neither is independent Oracle verification. Existing exact-fork,
stale-command, privacy and materialization cases remain on the native path;
compiled structure parity and seeded compiled replay/clone checks join them.

`conformance/semantic-kernel-w7-v1` contains eight terminal games and 1,107
commands, including four compiled Elemental games. CI/current conformance tests
select this root; bounded fuzzing also selects compiled orientations. The
Phase matrix is inherited evidence, not new external-engine card coverage.
The durable coverage note is `docs/rules/elemental-combat.md`.

## Remaining delivery

Implementation admission is complete for this Task's 42-card scope. Gate owns
affected full suites, bounded fuzzing, configured pinned Linux/browser/release
checks and final world-freeze judgment. Publication, waited landing and verified
Task completion remain the later authored Flow steps. Final ten-deck admission
remains the fourth Task's scope. No new product or human decision is identified.
Balance, mana-base quality and equal deck strength remain unmeasured targets.

Prior implementation evidence records 42 Elemental debug cases, 31 authored/
semantic debug checks and 38 distinct focused Python checks passing, plus the
extension rebuild/install and formatting checks. These were inspected rather
than rerun during reconciliation. The earlier request for source/IR, catalog/
corpus admission and control-change/belief/history evidence is now implemented;
the bounded proofs above retain their stated limits.

Check: `uv run pytest tests/semantic/test_elemental.py::test_elemental_artifacts_are_canonical_and_close_all_42_cards -q` — 1 passed; `git diff --check` — passed; full release checks remain with gate.

## Publication verification — 2026-10-20

Publication is blocked by
`tests/semantic/test_structural_katas.py::test_checked_artifacts_rebuild_byte_identically`:
the frozen structural-kata contract pins compiler SHA256
`f10f34d7b2e458a0ea01b124261a0c0fbbda4648e4452f277254aa6cc3250367`,
and rejects this Task's expanded compiler before rebuilding. Preserve the frozen
contract/artifacts; resolve historical reproduction compatibility before publishing.
No PR was published, auto-merge armed or Task completed by this step.

The broad semantic/projection/Academy/boundary Python run had 150 passes and two
failures. The stale Learn encoder assertion was corrected to address the fixed
outside-marker column and passes on rerun. The structural-kata failure reproduces.
Debug validation also exposed an 80-byte UndoEntry; permanent snapshots are now
boxed and included in owned-byte accounting, restoring the compactness invariant.
The release CPython 3.12 extension was rebuilt and installed after that fix.
Formatting, Ruff and diff whitespace checks pass.
The complete `cargo test --locked --manifest-path managym/Cargo.toml` debug run
passes after the journal fix, including the 257-case rules suite and all
search-state contract checks. Its output is `/tmp/elemental-publish-cargo.log`.


## Gate — 2026-10-20

Comparison: current tracked and untracked implementation versus HEAD and local
origin/main `3d3737c6`. No Task/Flow was launched, published or completed here.
Gate repaired the Disciple unblockable mutation's missing undo journal and
extended its source-removal test through nested branch-driver rollback. Casting
continuation state is grouped in `TargetSelection`; Clippy passes without a
new suppression. Modal diagnostics now pass priority before asserting effects.
Checkpoint rejection includes w6, and synthetic belief inputs include semantic
identity tensors. w7 belief hashes are pinned alongside retained older hashes.
The RUL-9 token census now accounts for outside candidates.

Frozen kata source/suite/contract files remain unchanged. Static rebuild checks
the exact original IR after removing only empty-sideboard setup additions and
checks the original schema after removing only schema 2's sideboard role. It
retains the historical producer identity; actual experiment runners still enforce
their preregistration. Regression tests reject altered IR and token schemas.

The w7 authority/parity receipts and Learn fixture are new files; historical
receipts and the w6 demo fixture remain byte-identical. Frozen consumers select
their historical authority/parity paths explicitly. New parity verifies current
source provenance strictly, without the older source/fingerprint normalization.
The Learn browser test waits for revision zero before replaying a new game.

Checks (logs under `/tmp/elemental-gate-*`; each command/result is one line):

- `cargo test --locked --manifest-path managym/Cargo.toml` — 446 passed in debug; after the casting signature refactor, `cargo test --locked --manifest-path managym/Cargo.toml --test rules_tests elemental_cards` — 42 passed.
- `cargo clippy --locked --manifest-path managym/Cargo.toml --all-targets --all-features -- -D warnings` — passed after grouping continuation state.
- `cargo fmt --manifest-path managym/Cargo.toml --check` — passed; changed Python files checked/formatted through `uv run ruff check --fix` and `uv run ruff format`; `git diff --check` — passed.
- `uv run --extra play maturin build --release -i .venv/bin/python -m managym/Cargo.toml -o managym/target/wheels` — passed, cp312 extension extracted into root package after final Rust edits.
- `uv run pytest tests/semantic/ -q` — 138 passed, including artifact rebuild and real Elemental policy/history consumer.
- `uv run scripts/generate_coverage_gaps.py --check` and `uv run scripts/compile_semantic_content.py --check` — passed; Elemental compiler `--check` with explicit source/fixtures/IR/coverage paths — passed.
- `cargo run --locked --manifest-path managym/Cargo.toml --bin semantic_conformance -- check --root conformance/semantic-kernel-w7-v1` — reproduced eight games, 1,107 Commands and 15 inherited Phase rows.
- `cargo run --locked --manifest-path managym/Cargo.toml --bin semantic_conformance -- fuzz --root conformance/semantic-kernel-w7-v1 --seed 24301 --cases 32 --max-commands 512 --failure-dir target/semantic-conformance/failures` — 32 games / 4,238 Commands passed.
- `uv run --extra dev pytest tests/env/ tests/model/ tests/verify/test_observation_parity.py tests/belief/test_state.py tests/belief/test_tracker.py tests/belief/test_runtime.py tests/test_package_boundaries.py tests/academy/test_ur_lessons_increment.py tests/academy/test_authored_match_authority.py tests/academy/test_authored_match_parity.py tests/academy/test_curated_pack.py tests/academy/test_play_modes.py tests/academy/test_release_prompt_matrix.py -q` — initially 334 passed, 1 skipped, 10 failures; all ten addressed by focused follow-ups.
- `uv run pytest tests/model/test_action_identity.py tests/model/test_attack_choice.py tests/belief/test_state.py tests/belief/test_runtime.py tests/academy/test_learn_setup.py tests/academy/test_learn_evidence.py -q` — 66 passed after fixes.
- `uv run pytest tests/academy/test_authored_match_authority.py tests/academy/test_authored_match_parity.py -q` — 5 passed on the w7 receipts; strict source-manifest regeneration follows final formatting.
- `npm --prefix frontend ci` — passed; `npm --prefix frontend run test:e2e:release -- --grep-invert 'release stack proves'` — navigation/combat and direct Learn demo passed; a reset race in the first Learn test was repaired; focused `--grep 'Learn modes|checked Learn demo'` — both passed.
- `npm --prefix frontend run test:e2e:release` (first invocation's grep matched no test title and therefore selected all 14) — first priority snapshot differed by 952 pixels on macOS; pinned Ubuntu visual acceptance is deferred to CI, with no snapshot rewrite.
- `uv run pytest tests/academy/test_public_commitment_parity.py tests/experiments/test_rul9_played_workloads.py -q` — 17 passed / 4 failed before the token-census repair; final targeted census recheck follows that repair. Two failures replay the historical public-commitment tape under w7 (expected hero revision 46, actual 41); one is the pre-existing RUL-9 parity hash mismatch.

Unresolved: world freeze is blocked by the historical public-commitment tape's
runtime divergence and RUL-9's frozen provenance mismatch. HEAD already pins
`af198bb3dcac542a6a34f1cbd8250938e4bfe0e79df7d284de4928873cc0f60b`
while HEAD's checked parity file hashes to
`2d63cc2433d5643f6d0621b34a1fdbe80a77e6535eb3cba947b451e673db3096`.
No frozen contract or receipt was overwritten, and no check was weakened or
skipped to hide these failures. Pinned Linux visual/reference and clean-machine
release checks remain with capable CI. No performance or strength improvement
is claimed; the meaningful before/after here is corrected rollback and complete
w7 live/headless/replay agreement. Publication/landing remain later Flow steps.

Final follow-up: `uv run --extra dev python -m academy.authored_match_parity write` followed by `uv run pytest tests/academy/test_authored_match_parity.py tests/experiments/test_rul9_played_workloads.py::test_real_semantic_census_is_ragged_and_fail_closed -q` — 4 passed after final formatting/source binding. The separately run authority suite passed 2 tests. The remaining known failures are the two historical public-commitment replay assertions and the pre-existing RUL-9 parity-file hash assertion. Ruff and diff whitespace checks pass; no historical fixture/receipt/contract file appears in the diff.
