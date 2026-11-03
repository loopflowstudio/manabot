# MTG-239 — five defined cards

Implementation plan, reconciled 2026-11-03 against `6d30c9013`. Jack Heart's
Task directive and November 3 playability clarification are accepted scope;
internal mechanisms are implementation choices awaiting review. One PR,
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
- Remove the five incomplete entries only with directed behavior and whole
  Decklist/admit_decklists coverage. Unknown and unsupported rows still fail.

## Delete — do not maintain

Remaining cleanup from concept review:

- `managym_cards::Incomplete`, `managym_cards::incomplete`, the empty
  `tla::INCOMPLETE` list and the pool's bug-based exclusion branch. Preserve
  token, rules-fixture and unknown-card rejection and whole-deck admission.
- Live documentation of the incomplete-card exception and stale TLA exclusion
  list. Preserve historical evidence and the accepted bug-tracking direction.

Completed cuts:

- TargetSpec::CreatureOrPlayer and its old text/fast-path assumptions: replace
  with AnyTarget, preserving ordering, exact target identity and ward handling.
- Unconditional random remainder in LookAndSelect and the Accumulate Wisdom
  test that accepts either order: preserve random semantics on other cards.
- Printed-power fallbacks in contextual zone predicates/observations: preserve
  printed-only structural matching where no game context is involved.
- Five TLA INCOMPLETE entries after their directed checks pass.
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
that direction. No broader admission-policy redesign or new Task was made.

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

Check result: `uv run --extra dev --extra play pytest tests/semantic/test_decklist_import.py::test_five_completed_cards_admit_as_whole_decklists tests/semantic/test_text_observation.py::test_accumulate_wisdom_ordering_reaches_text_commands_and_both_encoders tests/semantic/test_text_observation.py::test_suki_hand_power_reaches_text_and_both_encoders -q` — 4 passed on the merged runtime; broader validation remains with gate/CI.

## Concept review — 2026-11-03

Outcome: no design change and no open product choice within MTG-239. Reviewed
the accepted direction against `44388ad42..6d30c9013` and the reconciled plan;
this is a concept review, not a new validation pass or delivery approval.
Jack Heart's November 3 direction remains the boundary: fix these five cards,
preserve previously playable cards when bugs are discovered, and do not start
broader admission-policy work or additional Tasks here.

The consequential mismatch was a strategic ordering choice represented as
random engine work. The implementation now exposes that choice through the
same native decision and Commands used by humans and agents. No separate
browser ordering draft, confirmation lifecycle or recovery record is needed.
The suspended resolution owns the remaining cards; the ordinary match revision
and Command identity bind each placement. Academy labels and text observations
explain that decision rather than choosing on the player's behalf.

Concrete usage wording (implemented, not a new proposal): “Choose the card to
take with Accumulate Wisdom. Then choose the next bottom card, nearest the top
first. Each choice commits its placement. The last remaining card is placed
automatically.” With Forest and Island remaining, choosing Forest leaves
Forest above Island at the bottom; choosing Island gives the opposite order.
Only the chooser sees those inspected cards. On connection loss, resume the
session and act on its current pending decision. A duplicate Command does not
place another card; a stale choice cannot replace a committed placement.
The reconnect proof is for the retained server session, not process-crash
durability. Replay/forks retain the native state and explicit choices.

The other fixes remove distinctions consumers should not have to learn:
`AnyTarget` means the printed target categories, with legality and damage owned
by the engine; loyalty/defense use existing sparse counter facts. Contextual
power comes from `Game::card_power`, using the layered battlefield view or the
off-field CDA context. Rallier's filter and Suki's visible power therefore
agree. Decklist admission reads the same definitions for main and sideboard;
playability is not a claim of bug-free rules. Historical experiment populations
and receipts keep their identity; current open-world evidence binds the
corrected runtime. Existing README and AGENTS guidance already express these
contracts; no new skill or usage-document rewrite is needed.

Counterexamples retained: reconnect must not restart selection or randomize
the remainder; neither navigation nor a consumer may choose between two legal
orders; an opposing viewer must not receive the private cards; Rallier must
not offer Suki above power three; a future battle with a defeat trigger cannot
reuse the current shell-only zero-defense assumption without extending it.
That future card/combat lifecycle remains outside this Task. None requires a
new product decision for the current five cards.

Next proof remains the merged-tree gate and real-path demo listed above,
including debug invariants, consumer parity and browser interaction. No
implementation direction is added. The review page is
`scratch/concept-review.html`; loop-or-next owns the following Graph decision.

## Interactive concept review — 2026-11-03

Jack Heart invoked concept-review after the five fixes were implemented.
Review page: `scratch/concept-review.html`. This review edits design only;
the existing Graph remains responsible for execution. No new acceptance,
implementation restart, publication or merge is implied.

Accepted experience: a previously playable card remains playable when a rules
bug is found; record the defect in a Task. Finish the five fixes in this Task.
Human and policy consumers share the native legal decisions and observations;
reconnect restores the actual pending decision, stale commands fail, and
committed decisions cannot be undone without an explicit shared action.

Draft usage wording (proposed documentation):
“After choosing a card for your hand, choose which remaining card you want to
draw first. It goes above the other bottom card. Your click commits the order;
the final card is placed automatically. Reconnecting restores the pending
choice or the committed result.” This describes the implemented two-card case.

Draft contributor instruction (reflecting accepted policy; wording proposed):
“Production card definitions determine the playable card catalog. Track known
rules defects in Tasks; a bug report does not revoke a previously playable
card. Unknown cards, tokens and rules fixtures retain their existing deck-entry
restrictions. Playable is not a claim of bug-free rules.” Keep other guidance.

Consequential mismatch with accepted direction: the five incomplete entries
are gone, but `managym_cards::incomplete`, its empty TLA list, and the pool's
exclusion branch remain; AGENTS.md states the old exception immediately before
the new policy. This leaves a dormant way to recreate the rejected distinction.
Recommended consequence is to remove that empty bug-based eligibility path and
reconcile those sentences, while retaining token/fixture/unknown restrictions.
This is review direction, not an implementation made here; do not broaden into
a new qualification system or create additional Tasks from the discussion.

One unresolved experience choice: retain immediate native placement Commands
(recommended, 1a), or prefer a whole-order decision that can be rearranged before
one commit (1b). The second changes the shared decision contract for humans and
policies, not just browser navigation, and is not approved. For Accumulate
Wisdom the first has only one meaningful ordering click, because two cards
remain and the last is forced. Clearer draw-order wording is proposed.

Normal path: import whole deck → choose card to keep → choose earlier bottom
card → resume resolution. Types/APIs: CardDefinition → content_pool/world
descriptor → admit_decklists; LookAndSelect(random_order=false) →
OrderLibraryBottom → SelectCard via revision-bound Command → continue resolution.
Suki/Rallier share Game::card_power and card_matches_predicate; AnyTarget shares
native target legality and damage. No consumer-specific rules are added.

Failure/recovery: disconnect before submit restores both offers; disconnect
after acceptance restores the committed state; retrying the same Command must
not apply twice. A stale revision cannot change order. Only the acting viewer
sees the looked-at cards. Frozen worlds and retained training corpus stay fixed.
Counterexample to immediate-commit convenience: a future effect leaves many
cards and a user wants to revise a partially chosen order; solving that needs
a shared draft/commit contract, not a browser-only undo. No such redesign is
selected by this review. Next proof after a decision: actual browser walkthrough
with distinct remainder cards and disconnect/retry, alongside the existing
native/encoder/reconnect tests; full gate remains outstanding.
