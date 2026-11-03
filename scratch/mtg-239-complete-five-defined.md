# MTG-239 — five defined cards

Implementation plan, reconciled 2026-11-03 against `6d30c9013` and the
subsequent admission cleanup. Jack Heart's
Task directive and November 3 playability clarification are accepted scope;
the admission cleanup and immediate ordering with clearer draw-order wording
are accepted by Jack Heart (November 3, comment 6b64d212). One PR,
with publication and broader acceptance owned by later code Graph steps.

## Outcome and design

Make Accumulate Wisdom, Cat-Gator, Lightning Strike, Suki, Kyoshi Warrior and
Water Tribe Rallier faithful through the existing rules and deck admission.
The official September 25, 2026 [Comprehensive Rules](https://media.wizards.com/2026/downloads/MagicCompRules%2020260925.txt)
confirm owner-chosen library order (401.4), four any-target categories (115.4),
CDA evaluation in all zones (604.3), and loyalty/defense removal from damage
(120.3c/h). No new planeswalker/battle cards, battle combat system, T6 content,
training run or corpus revision is included.

- Give LookAndSelect an explicit remainder-order rule. Random-order cards
  retain it; Accumulate Wisdom suspends on an uncapped native bottom-order
  choice. Each SelectCard commits the next card, nearest the top first. Use
  existing Commands, card focus, private inspection and forced-decision skip.
  Append a continuation code and text stage; keep frozen widths/codes stable.
- Replace the misleading CreatureOrPlayer specification with AnyTarget across
  its existing callers (all printed any-target cards). Legal and resolution
  checks share layered types and hexproof. Damage marks creatures and removes
  loyalty/defense counters as applicable, with one damage event and lifelink.
  Zero-counter unsupported-shell fixtures exercise ordinary graveyard SBAs;
  no new catalog card is implied.
- Add a contextual card-power reader: battlefield uses the layered view;
  stack uses controller; other zones/outside use owner. Reuse the existing CDA
  evaluator. Route zone power filters and visible card rows through it.
  Ability loss removes the battlefield CDA; leaving restores the off-field CDA.
- Support defined non-token cards through the existing Decklist/admit_decklists
  authority. Remove bug-based eligibility exclusions; preserve token, fixture
  and unknown-card rejection and whole-list main/sideboard coverage.

## Delete — do not maintain

All deletion targets are complete; none remain:

- `managym_cards::Incomplete`, `managym_cards::incomplete`, the empty
  `tla::INCOMPLETE` list and the pool's bug-based exclusion branch. Preserve
  token, rules-fixture and unknown-card rejection and whole-deck admission.
- Live documentation of the incomplete-card exception and stale TLA exclusion
  list. Preserve historical evidence and the accepted bug-tracking direction.


- TargetSpec::CreatureOrPlayer and its old text/fast-path assumptions: replace
  with AnyTarget, preserving ordering, exact target identity and ward handling.
- Unconditional random remainder in LookAndSelect and the Accumulate Wisdom
  test that accepts either order: preserve random semantics on other cards.
- Printed-power fallbacks in contextual zone predicates/observations: preserve
  printed-only structural matching where no game context is involved.
- `Game::bottom_in_random_order`: its sole caller now owns shuffle and
  placement together, sharing placement with a forced singleton. Both
  LookAndSelect exits reuse `continue_resolution` for suspension/resumption.

## Implementation outcome — 2026-11-03

The deletion cuts are implemented. Directed native and Python tests cover
the five cards, whole lists in main and sideboard, actual ordering Commands,
private inspection, stale rejection, replay/fork identity, native/Python/text
observations, definition tokens and loyalty/defense fact rows. The rebuilt
cp312 extension was installed atomically. Academy reconnect preserves the
ordering decision and duplicate submission commits it only once.

State-based deaths group non-destruction graveyard moves before checking lethal
damage, so candidate admission and the commit path share one precedence rule.
Counter descriptions reuse canonical names except the two display spellings.
The ordering encoder proof reads the native observation: captured viewer JSON
omits continuation frames, which the proof must exercise.

Jack Heart’s November 3 steer keeps this focused: known bugs in previously
playable cards belong in bug Tasks, not new play restrictions. AGENTS.md records
that direction. The empty incomplete-card API, TLA list and pool exclusion branch are removed,
and contributor/content/import/TLA docs now agree. Token, fixture and unknown
restrictions remain. No broader admission-policy redesign or new Task was made.

Current open-w17 evidence was regenerated through its producers. Rallier’s
correct CDA filter excludes Suki above power three at authored decision 90;
the resulting game has 135 Commands and 66 commitments instead of 142/69.
The old trace remains at `22e4aab71:conformance/authored-match-w17-v1/`. Three
navigation prefixes now explicitly order Accumulate Wisdom’s remainder; all
prior deliberate choices retain their IDs/labels and reach the intended
positions. Demo/trigger fixtures and current belief hashes bind the new content;
historical hashes, frozen receipts and MTG-213 corpus bytes are unchanged.

Recorded pre-sync proof includes 17 directed card/observability tests and 11
SBA tests in debug, admission and consumer checks, native/Python/text parity,
navigation/reconnect, trigger-order, demo and belief receipts, plus formatting
and lint. These results establish the pre-sync implementation; they do not
certify the merged tree. Their detailed commands remain at
`60b8071f8:scratch/mtg-239-complete-five-defined.md`.

## Integration and remaining checks

MTG-238 / PR #414 delivered at `44388ad42` and was merged here in `05eb48489`.
Definitions and card tests now live in `managym/cards`; shared rules types,
including sparse loyalty/defense counters, live in `managym/state`. Scalable
keywords and the upstream engine optimizations are preserved. No architecture
handoff remains. The original Task base `22e4aab71` is historical evidence.

After sync, the root release wheel was rebuilt and installed atomically. The
combined runtime regenerated the 14-case / 3,473-Command semantic kernel and
both parity receipts (135 Commands / 816 viewer checks; 66 commitments / zero
gaps). Three Lightning Strike debug tests and 14 parity/navigation/belief tests
passed; `6d30c9013` retains the completed receipt refresh. Reconciliation also
checked whole-deck admission and both text/native encoder cases on that runtime.

Implementation remaining: no known missing behavior within these five fixes.
Full acceptance remains open:

- Gate owns the merged-tree debug Cargo suite, clippy, affected Python/model
  suites, world-admission/monitor checks in AGENTS.md, and frontend
  generation/check/build plus focused browser interactions from
  `docs/development.md`. Pre-sync proof is not a substitute for these checks.
- Engine Budget CI compares against the delivered `1616b210f` reference from
  run 41699683808 with the unchanged 10% tolerance. Changed rules may move its
  digest; any reference adoption requires this PR's own artifact and a reason,
  not MTG-238's measurement or a local cross-machine timing comparison.
- Demo owns the final real-path walkthrough and human interaction judgment;
  the headless reconnect/shared-Command proof is available. Publication and
  delivery remain with the selected Graph. No product decision blocks these
  fixes; the no-new-battle-card boundary still excludes defeat-trigger/combat
  support for future battle definitions.

MTG-213's 134-to-306 eligible-deck projection is not a measured new corpus.
Its frozen cohort and running pilot, T6 content and T7 acceptance remain with
their existing owners. No corpus rewrite, training run, broader admission
redesign or additional Task is part of this change.

Check result (cleanup): `CARGO_PROFILE_DEV_OPT_LEVEL=1 cargo test -j 2 -p managym --lib cards::decklist::tests::` — 7 debug tests passed; `cargo fmt --check` passed; root release wheel rebuilt and installed atomically; `uv run --extra dev --extra play pytest` with the focused `test_five_completed_cards_admit_as_whole_decklists`, `test_decklist_errors_carry_each_failing_seat_with_line_issues`, `test_accumulate_wisdom_ordering_reaches_text_commands_and_both_encoders` and `test_accumulate_wisdom_order_survives_reconnect_and_commits_once` nodes — 5 passed across admission and wording/reconnect runs. Full merged-tree validation, Engine Budget CI and demo remain with their declared later steps.

Compression check: `uv run --extra dev --extra play pytest tests/semantic/test_text_observation.py::test_accumulate_wisdom_ordering_reaches_text_commands_and_both_encoders tests/semantic/test_text_observation.py::test_suki_hand_power_reaches_text_and_both_encoders -q` — 2 passed; Ruff passed. Runtime unchanged in this step; gate retains the merged-tree suites.

## Accepted interaction and review limits — 2026-11-03

Jack Heart accepted the exclusion cleanup and option 1a (comment 6b64d212):
immediate native SelectCard Commands with clearer draw-order wording. The
follow-up review corrected the earlier review's claim that admission guidance
already agreed: the empty API and pool branch still permitted bug-based
eligibility at `6d30c9013`. That mechanism and its documentation are now removed.
The earlier review is retained at `ae06995b1:scratch/mtg-239-complete-five-defined.md`;
no product decision remains open.

The shared instruction says: “Choose which remaining card to draw first. It
goes above the other remaining cards on the bottom. Each choice commits
immediately; the last card is placed automatically.” With Forest and Island
remaining, choosing Forest leaves Forest above Island at the bottom; choosing
Island reverses that order. Academy and LLM text consume this native instruction.
No draft, confirmation or undo lifecycle is approved. A future effect leaving
many cards could motivate revising a partial order, but that needs a shared
rules decision, never browser-only undo.

The native suspended resolution owns the remaining private cards. Ordinary
revision-bound Commands preserve placement, replay and fork identity. Reconnect
must return the actual pending choice; duplicate submission cannot place another
card and stale submission cannot revise a commitment. The reconnect proof covers
a retained server session, not process-crash durability. Neither navigation nor
a consumer may hide an order, choose it automatically, or expose private cards
to the opponent. The tests compare native observations directly because captured
viewer JSON omits continuation frames; both encoders explicitly use v5.

Other retained counterexamples: Rallier must not offer Suki above power three;
a future battle with a defeat trigger needs more than the current shell-only
zero-defense behavior. The review page is `scratch/concept-review.html`.
Merged-tree gate, Engine Budget CI, demo and publication keep their owners above;
this review establishes no new acceptance evidence or additional Task.
