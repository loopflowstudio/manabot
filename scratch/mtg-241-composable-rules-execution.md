# MTG-241 — composable rules execution

## Outcome and status

Accepted directive: Jack Heart, 2026-11-04. One PR using the code Graph, review
boundary only. Implementation began from `4463e57f5` (also the local main ref;
source findings at `5dd11cec4` still apply). The independent iteration and
contextual-read cut is implemented; the full Task is incomplete. The wider
replacement completion design requires review before its dependent cut.
No training, publication, landing or retained-evidence rewriting occurred.

Effect execution must retain nested operations, exact objects, scoped results
and root finalization across native choices. Replacement proposals must retain
transformed events, affected chooser, applied identities and logical batches,
re-evaluate applicability, and emit facts only at commit. Selection primitives
must preserve cardinality, visibility, order, shuffle, reveal and result binding.

## Owners and approach

`EffectFrame`/`run_frame` own execution and completion; `Decision` owns the current
question. Iteration retains its parent and remaining targets, creates a fresh
child scope, and returns without granting priority. Conditional branches use
the current scope. `proposed_event.rs` owns transformation and commit; callers
must retain their post-commit obligations rather than treating suspension as a
boolean result. `Game::select` and exact object references remain authoritative.
Observation continuation projection follows the same retained ownership chain.
`EffectFrame::new` owns fresh-scope initialization for roots and iteration
children. Callers bind only their source, targets and completion; result fields
start empty in one place. Frame fields and their serialized meanings stay intact.

## Delete — do not maintain

- Removed: the local ForEachTarget subframe loop and its no-decision assertion.
  Retained parent frames preserve target eligibility/order, isolated child
  results and root finalization. Continuation projection emits the parent chain.
- Removed: owner-based entry-condition reads and live-only SourcePower. The
  surviving reads use prospective entry control and exact-source current/LKI power.
- Removed: repeated fresh-frame literals in resolution, rule decisions and
  fixtures, and the one-use `with_parent` helper. New child scopes share the
  constructor; the interrupted parent retains its own results and cleanup.
- Specialized MillAndSelect, ReturnMilledThisWay, OptionalDiscardDraw, MayDiscard,
  SearchBasicLand and overlapping search/move handlers and their dedicated
  decisions, after migrating real card consumers to the shared selection owner.
  Preserve keyword-specific events, private viewing, decline and ordered remainders.
- Fixed-order/one-pass replacement application. Preserve no-replacement speed,
  exact applicability, pre-batch sources, transformed event and single commit.

## Remaining implementation and witnesses

1. Extend the reviewed completion protocol beyond effect scopes (see counterexample
   below); preserve the now-working nested iteration and contextual reads.
2. Shared selection migration and Surveil 2+ kept ordering; exact result use,
   decline/private inspection, simultaneous moves and replay/fork checks.
3. Resumable replacement batches: affected-player order, newly applicable
   replacements, once-per-event identity; Leaves from the Vine counters yield
   three or four with Caradora and Doubling Season in the selected order.
4. Native/Python policy, LLM, Academy, replay and reconnect projection; explicit
   version boundaries where meanings change. Preserve frozen inputs/evidence.
5. Gate owns affected suites and
   automated acceptance once, including same-runner Engine Budget and world
   admission checks from `docs/development.md`.

## Completion-boundary counterexample and revised design proposal

Discovered 2026-11-04 during implementation. An EffectFrame-only replacement
continuation cannot satisfy the accepted timing contract:

- `advance_mana_upfront` (`flow/mana.rs`) sacrifices a source, then replaces the
  retained ManaUpfront obligation with ManaChoice. `sacrifice_permanent` calls
  `move_card`; `commit_zone_move` calls `handle_leaves_battlefield`, which can
  return linked exiled cards. With competing entry replacements, that return
  must ask a question before the payment can finish. Installing a replacement
  Decision inside today's synchronous helpers would be overwritten by the
  payment's postamble, and falsely treating suspension as success credits output.
- `finalize_frame` similarly moves a permanent spell, then emits SpellResolved
  and restarts priority. A pending entry replacement is not a completed spell.
- Combat damage accumulates hits in a local vector; Saga entry recursively puts
  counters during zone commit. Neither has a retained completion boundary.
  Per-target iteration also does not, by itself, preserve one simultaneous event.

The initial local-frame approach is therefore insufficient for the replacement
cut. Proposed revision (not implemented): retain event transformation as a child
operation with explicit completion carrying event results to its caller. Root
completion distinguishes resolution, payment, turn action and SBA stabilization;
zone entry owns its remaining post-commit work, and a logical batch owns all
proposals plus pre-batch trigger sources until every replacement is settled.
The no-replacement case returns Completed inline. Pending is never represented
as a false/true commit result. The same protocol resumes the original payment
owner; a second replacement-only scheduler or replaying committed costs is not
an acceptable workaround.

MTG-242 owns the payment model; this exact call-path counterexample and the new
root EffectFrame fields were recorded in its Task comments. MTG-243 received
the retained-frame interface. MTG-244's characteristic algorithm is untouched.
The wider runtime cut needs an agreed completion interface across these owners;
selection migration depends on that move/batch owner and remains unstarted.
The Task is not ready for gate or publication merely because the independent
checks pass.

## Rules evidence

Wizards' Aetherdrift release notes (Caradora) state that the affected permanent's
controller orders counter replacements, including on entry; Foundations release
notes state Doubling Season also changes entry counters. Thus (1 + 1) × 2 = 4
and (1 × 2) + 1 = 3 are both meaningful choices.
- https://magic.wizards.com/en/news/feature/aetherdrift-release-notes
- https://magic.wizards.com/en/news/feature/foundations-release-notes

Validation (repository root; debug assertions enabled):

- `CARGO_PROFILE_DEV_OPT_LEVEL=1 cargo test --manifest-path managym/Cargo.toml -p managym --lib -j 2 -- resolution_tests flow::decision:: agent::continuation:: --test-threads=2` — 43 passed.
- `CARGO_PROFILE_DEV_OPT_LEVEL=1 cargo test --manifest-path managym/Cargo.toml -p managym-cards --test cards -j 2 -- combustion_man_uses_departed_source_power_across_a_choice_and_reentry cruel_administrator_entry_raid_reads_the_entering_controller --test-threads=2` — 2 passed.
- `CARGO_BUILD_JOBS=2 uv run --extra play maturin build --release -i .venv/bin/python -m managym/Cargo.toml -o managym/target/wheels` — built; cp312 extension atomically installed.
- `uv run --extra dev pytest tests/verify/test_continuation_input.py::test_native_fixture_relation_parity_and_other_viewer_safety -q` — 1 passed.
- `cargo fmt --manifest-path managym/Cargo.toml --all --check` and `git diff --check` — passed. Full suites/Clippy, wider observation/reconnect proofs, world receipts and Engine Budget remain with the completed cut and gate.

The iteration Command witness covers both seats, two nesting levels, divergent
forks, replay, stale rejection and single finalization. The separate timing
witness leaves lethally damaged creatures present through the sibling choice
and removes them only after root completion. The real-card LKI witness removes
and returns Combustion Man while its opponent's decision is pending: seven
last-known power applies, not the new object's four. Raid differs between owner
and entering controller. These tests do not prove the unimplemented event-batch
or shared-selection behavior.

Approved engine principles are in `managym/PRINCIPLES.md` with the README link.
Its final evidence paragraph follows Jack Heart's 2026-11-04 steer: public PRs
#42, #348, #353, #289 and #412 plus the existing speed investigation.
