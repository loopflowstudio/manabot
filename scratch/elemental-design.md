# Play Elemental sacrifice and landfall engines

Accepted scope, ownership and delivery constraints are in `elemental-sacrifice.md`.

## Accepted scope and current implementation — 2026-10-21

Jack Heart accepted the 17-card sacrifice/landfall scope and autonomous delivery
on 2026-10-20. MTG-161 delivered the shared Elemental primitives at
`1c92d12e4ab718539007422244e0c16bde265042`. The checkout now includes upstream
`60fc6abe` (#327), which adds the FiLM policy and `learnability-2` diagnostic.
This changes the learning validation baseline, not the accepted card rules.

All 17 owned cards execute through the native registry and the distinct
`elemental-sacrifice-v1` compiled pack. Its 27-definition closure includes five
basics and five tokens. Development world w8 uses match-state hash version 5
and learning schema 4; earlier source, schema, replay and receipt files remain
unchanged. Native choice descriptions carry sacrifice, search and mana meaning
through the ordinary Academy projection. Durable contracts and coverage are in
`docs/rules/elemental-sacrifice.md`.

### Reconciled implementation choices

- Activated stack objects retain the exact `ObjectRef`, replacing the duplicate
  `source_permanent_object_id` field and matching `StackObject` getter;
  observation projects the existing public permanent ID through the authority's
  current-object/last-known lookup. This preserves damage attribution and granted
  keywords after removal without following a later incarnation.
- Basic-land search retains one native `Effect::SearchBasicLand` and one private
  decision owner. Frozen source opcode 24 remains readable; opcode 51 lowers
  its richer search options into that same operation. No parallel runtime search
  representation or temporary registry-dump utility remains.
- `begin_payment` owns selected-cost setup for spells and activations;
  `commit_payment` produces and spends the whole announced mana cost once before
  sacrifices and stack insertion. This reserves colored base costs before
  spending generic alternatives and avoids redundant land taps. The former w8
  draft diverged at sacrifice-163 Command 77; it is preserved at
  `/tmp/manabot-sacrifice-before-combined-payment`. Only the unfrozen w8 corpus
  is regenerated; earlier world evidence remains unchanged.

### Evidence and limits

The implementation pass recorded 21 directed sacrifice tests, nine authored-match
tests and 13 focused semantic/Academy tests passing, plus formatting and lint
checks. The compiled/native test compares all 27 complete definitions; the
training-path test plays one compiled sacrifice game to termination without
observation truncation. These are bounded correctness checks, not balance or
full-game learning evidence.

The w8 corpus records ten terminal replays and 1,507 Commands. Its inherited
15-row Phase comparison retains three matches, five mismatches and seven
exclusions; a passing corpus check does not certify external-engine agreement.
The refreshed Academy receipt covers 156 Commands and 942 viewer checks for
UR Lessons/GW Allies, not the new sacrifice rosters. Its Rust-source fingerprint
must still match when gate runs. Earlier sources, schemas and receipts are
unchanged.

Rules' objective was read from `.lf/wave/rules/GOAL.md`; its referenced
`MEMORY.md` is absent and there are no immediate child scopes in this checkout.
No current Wave-memory lessons could be recovered. Accepted Task scope forbids
Wave/chapter edits; the broader expansion remains this Task's explicit exception
to the older Allies/Lessons chapter boundary.

### Remaining delivery boundary

Gate owns the full debug Rust/Clippy and affected Python suites, current-world
arena admission/monitoring, the full reward-driven `learnability-2` diagnostic
against the synced FiLM policy (including PPO; the `--pr` subset omits it),
release-browser checks and pinned Linux comparison. The inherited frozen
public-commitment tape divergence and RUL-9 provenance mismatch remain named
limits; no frozen input was rewritten. Publication, waited landing and Task
completion remain with the authored Flow. Final ten-deck admission and full
matrix integration belong to MTG-164.

No new product decision is unresolved. Release certification and world freeze
are not established by the focused implementation evidence.

Check: `uv run --extra dev --extra play pytest tests/academy/test_authored_match_parity.py -q` → 3 passed in 9.84s; current receipt/source fingerprint verified.

## Concept review — 2026-10-21

Outcome: no design change and no open product choice. Jack Heart's accepted
experience remains faithful play of the selected cards through the same choices
for people and agents. The implementation's payment continuations, search state
and source references support that experience without adding a second rules
authority or a separate user-managed transaction. Existing usage and execution
instructions remain applicable; no new skill or command is needed.

| Player interaction | Domain concept | Authority and state transition |
| --- | --- | --- |
| Cast Deadly Precision, select its target, then choose a sacrifice or pay {4} | One announced spell and its costs | `begin_payment` retains the announcement in `Payment`; `sacrifice_payment_action` selects the cost branch |
| Use Treasure and choose a color, then finish paying | Immediate mana production within the announcement | `activate_mana_ability` / `Decision::ManaChoice` resume `ManaPayment`; `commit_payment` spends the combined cost and completes the announcement before responses |
| Resolve Merchant's Food trigger; decline the search or search and optionally find a basic | Optional search, followed by its required consequences | `Decision::Modal` selects whether to search; the shared `Decision::SearchBasicLand` owns selection, exact-source return and shuffle |

Cost selection and resolution-time sacrifice must remain distinct internally:
Deadly Dispute sacrifices before responses, while Cycle of Renewal sacrifices
during resolution. Merging those timings would violate the accepted printed
rules. Conversely, richer basic-land searches correctly share one runtime owner;
their count, entry and source-return options do not require another lifecycle.
Native descriptions give Academy the choices' meaning without card-name rules.
Compiled content and learning consume the typed definitions; w8/schema 4/hash 5
provide new identities while frozen evidence retains its original meaning.

Failure and recovery preserve the same contract. A stale or repeated Command
must leave the current announcement and match unchanged; the caller uses the
current decision frame rather than repeating an old offer. Exact forks must
restore payment, mana, sacrifice events, per-incarnation limits and RNG along
with visible state. This is replay/branch recovery, not a proposed player undo
feature. Merchant is the counterexample to collapsing "decline search" and
"find nothing": declining leaves Merchant in play; searching without finding
still returns its exact source if present and shuffles. A source that leaves
and returns must never be followed into its later incarnation. Library choices
remain chooser-only, including across determinization.

Next proof remains the existing gate, with particular attention to payment
continuations and these recovery/privacy invariants through ordinary and
compiled consumers. This review inspected code, directed tests and the existing
contracts; it did not rerun tests or establish browser, learning or release
certification. It adds no implementation direction or delivery authorization
beyond the accepted scope above.

## Publication inspection — 2026-10-21

Inspected the full working diff against freshly fetched main at `60fc6abe`.
Publication fixes preserve compact undo entries by boxing last-known snapshots,
remove two Clippy violations, and update the stale-world rejection assertion to
w8. Refreshed only the development w8 Academy parity receipt after source edits;
it verifies 156 Commands and 942 viewer projections. The release extension was
rebuilt from the root. Selected publication paths exclude scratch and the local
Task Flow definitions.

Full checks and final publication result follow below. Browser release and pinned
Linux validation remain owned by the following gate step. Publication does not
arm merging or complete MTG-162.

Publication verification: 50 focused Python tests pass, including current-world
arena models, mirror admission and monitoring; full learnability-2 passes on the
final rebuilt native extension (three capacity seeds plus reward-driven PPO,
61.1 seconds). Clippy with all targets/features and denied warnings, Rust fmt,
Ruff lint/format and diff whitespace checks pass. Directed debug sacrifice checks
pass (22 selected, including 21 owned sacrifice cases); authored-match (9),
Oracle shell (2) and semantic conformance (6) checks pass. Full debug cargo test
is still running; do not describe it as passed before its final result.
