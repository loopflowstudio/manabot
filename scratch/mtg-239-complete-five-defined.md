# MTG-239 — five defined cards

Accepted scope: Jack Heart's Task directive and November 3 clarification
(comment 1f3876a7), plus exclusion removal and immediate ordering with clearer
draw-order wording (comment 6b64d212). One PR, #415; the code Graph retains
gate, demo and delivery. CI recovery follows comment d35e6e94 and is
recorded in [ci-recovery.md](ci-recovery.md). No product decision is unresolved.

## Outcome and implementation

Complete Accumulate Wisdom, Cat-Gator, Lightning Strike, Suki, Kyoshi Warrior
and Water Tribe Rallier through shared rules and whole-deck admission.
The September 25, 2026 [Comprehensive Rules](https://media.wizards.com/2026/downloads/MagicCompRules%2020260925.txt)
are the design's rules reference: owner-chosen library order (401.4), four
any-target categories (115.4), CDAs in all zones (604.3), and loyalty/defense
removal from damage (120.3c/h).

- `LookAndSelect.random_order` preserves random placement for other cards.
  Accumulate Wisdom suspends on uncapped `OrderLibraryBottom`: ordinary
  SelectCard Commands place the next card nearest the top first. Each choice
  commits immediately; native forced-decision skipping places the last card.
  Continuation code 23, the text stage and definition tokens expose the rule
  without changing frozen widths or existing codes.
- `AnyTarget` replaces CreatureOrPlayer across printed any-target cards.
  Legality and resolution share layered types and hexproof. Damage marks
  creatures and removes loyalty/defense for every applicable type, with one
  damage event and lifelink. Non-destruction graveyard SBAs take precedence
  over lethal destruction; counter descriptions reuse canonical names.
- `Game::card_power` reads the layered battlefield view, the controller's CDA
  on the stack, and the owner's CDA elsewhere (including outside the game).
  Zone predicates and visible card rows share it. Ability loss removes the
  battlefield CDA; leaving restores the off-field CDA. Rallier excludes Suki
  when its library power exceeds three.
- Defined non-token cards are admitted through Decklist/admit_decklists;
  token, rules-fixture and unknown-card rejection remains. Known rules bugs
  belong in bug Tasks, not play restrictions; playable does not mean bug-free.
  Whole-list main/sideboard and training-source/checkpoint/replacement tests
  preserve positive admission and rejected-replacement root identity.

Inspection decisions share action construction only when the candidate list
and decline behavior agree. Their distinct decisions, commitment rules,
visibility and ordering remain owned by the existing resolution path. Scry
and LookAndSelect reuse `card_focus` instead of repeating object-ID projection.

## Delete — do not maintain

Completed; no remaining deletion targets:

- `managym_cards::{Incomplete, incomplete}`, `tla::INCOMPLETE`, the pool's
  bug-based exclusion branch, and live documentation of that exception.
- `TargetSpec::CreatureOrPlayer` and its old text/fast-path assumptions.
- Unconditional random LookAndSelect remainders and the Accumulate Wisdom
  test that accepted either order without offering a choice.
- Printed-power fallbacks in contextual predicates/observations; printed-only
  structural checks and private layer stages remain necessary.
- `Game::bottom_in_random_order`: placement and shuffle now belong to
  `library_bottom_decision`; both selection exits use `continue_resolution`.
- Negative admission fixtures using playable cards; Ally now supplies the
  rejection while repaired-card, unknown-card and rules-fixture coverage stays.

## Interaction contract and limits

Jack Heart selected immediate ordering (option 1a), without a draft,
confirmation or undo lifecycle. The shared instruction explains draw order
and automatic final placement: choosing Forest before Island leaves Forest
above Island at the bottom. Academy and LLM text consume that native instruction.
A future partial-order revision would need an explicit shared rules decision.

The suspended resolution owns the remaining private cards. Revision-bound
Commands preserve replay and fork identity; reconnect restores the actual
pending choice, duplicate submission cannot place another card, and stale
submission cannot revise a commitment. The reconnect proof covers a retained
server session, not process-crash durability. Neither consumer may hide a
choice, choose automatically, or reveal inspected cards to the opponent.
Encoder parity reads native observations directly because captured viewer JSON
omits continuation frames; both encoders explicitly use v5.

No new planeswalker/battle definitions or battle combat system are included.
Controlled shells cover damage and zero-counter SBAs; a future battle defeat
trigger needs additional handling. MTG-213's frozen cohort and running pilot,
T6 content and T7 acceptance retain their owners. The 134-to-306 eligible-deck
figure is a projection, not a new measurement. No training run, corpus rewrite,
broader policy redesign or additional Task is authorized here.

## Evidence and remaining acceptance

Concept review after CI recovery (2026-11-03, `e5627dd98`): nothing further
to change and no open product choice. Jack Heart's accepted admission and
immediate-ordering decisions still describe the implemented interaction.
The recovery changes rejection fixtures, not eligibility: an imported list
with the five cards is admitted, adding an Ally token fails, and a rejected
training-deck replacement leaves the existing roots intact. Removing that
token permits replacement for later roots without rewriting current games.
These failure/recovery cases preserve one admission authority across import,
training sources and checkpoints; they require no new user-facing concept.

Shared inspection-action construction retains separate pending decisions and
their commitment rules. Ordering still commits through native SelectCard
Commands, with retained-session reconnect and automatic final placement;
there is no draft to persist or undo. A changed rules benchmark digest calls
for an explained reference update under the existing budget procedure, not
restoration of random ordering or bug-based admission. The existing usage
guidance needs no further change. This is source/contract review, not a new
runtime proof: the next proof remains repaired-head gate and real-path demo,
with the counterexamples and durability limits above unchanged.

MTG-238 / PR #414 delivered at `44388ad42`, integrated in `05eb48489`.
Definitions/tests now live in `managym/cards`, shared types in `managym/state`;
sparse counters, scalable keywords and upstream optimizations are preserved.
The root release extension was rebuilt and installed atomically after sync
and after CI recovery.
On November 3, reconciliation confirmed remote main remains `44388ad42`;
no newer upstream integration is pending. Engine has no child memory scopes
in this checkout.

Open-w17 evidence was regenerated through its producers (`6d30c9013`):
14 semantic-kernel cases / 3,473 Commands; authored parity with 135 Commands /
816 viewer checks; 66 commitments / zero gaps. Rallier's corrected filter at
decision 90 changes the game from 142 Commands / 69 commitments. Three
navigation prefixes add explicit remainder ordering while retaining prior
chosen IDs/labels and intended positions. Demo/trigger fixtures and current
belief hashes bind the new content. Historical hashes, frozen receipts and
MTG-213 corpus bytes are unchanged; the prior authored tape remains at
`22e4aab71:conformance/authored-match-w17-v1/`.

Focused pre-sync and post-sync results are retained at
`60b8071f8:scratch/mtg-239-complete-five-defined.md` and
`72e29f5de:scratch/mtg-239-complete-five-defined.md`. The full prior plan/review
history is `8c3389884:scratch/mtg-239-complete-five-defined.md`; the earlier
claim that admission guidance agreed was incorrect at `6d30c9013` and was
resolved by the exclusion cleanup in `ae06995b1`. The accepted review page is
`scratch/concept-review.html`; causal policy evidence remains in
[defined-card-completeness.md](defined-card-completeness.md).

CI run 41711692234 exposed three Rust and four training-deck rejection
fixtures still naming playable cards. `8c3389884` repairs them and adopts
PR #415's own `ea316ff14` Engine Budget artifact. All timing checks passed;
the changed search/step digests and step count follow the corrected rules in
unchanged benchmark decks. The previous reference reproduced its digests.
The 10% tolerance stays unchanged; this is not equivalent-workload speedup
evidence. [CI recovery](ci-recovery.md) retains exact checks and timing deltas.

The simplification at `e5627dd98` shares identical action construction while
preserving separate decision semantics. Its focused debug checks passed 17
card cases and three Python consumer cases; the release extension was rebuilt.
The refreshed authored parity receipt retains 135 Commands / 816 viewer checks;
its diff from `8c3389884` changes source provenance only. Exact commands and
results remain at `e5627dd98:scratch/mtg-239-complete-five-defined.md`.

Publication verification on November 3: all 11 jobs in CI run `41734880466`
passed for runtime head `e5627dd98`. This includes the complete debug Cargo
suite, Python/model/world-admission/monitor checks, Academy build/check and
release browser interactions, both conformance jobs, clean-machine play,
and main integration's previously skipped evaluation/report/smoke steps.
The fresh Engine Budget comparison preserves all digests/counts and passes
the unchanged 10% tolerance. Linux/macOS package build and install workflow
`41734880090` also passed.

Supplemental local checks passed: 70 parity/navigation/Learn/content/learning/
belief tests, all-target/all-feature clippy, Cargo formatting, changed-Python
Ruff and `git diff --check`. [CI recovery](ci-recovery.md) records the exact
test paths, interpreter-path retry and same-run budget measurements.
`lf sync --manual` found no upstream changes beyond main `44388ad42`.
Publication changes only review notes; runtime code remains the tested head.

Demo owns the final real-path walkthrough and interaction judgment. Automated
validation does not establish demo acceptance, post-merge verification or new
shipping approval. Publication neither arms auto-merge nor completes the Task.
