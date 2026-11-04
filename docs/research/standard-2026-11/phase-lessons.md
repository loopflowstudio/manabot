# Phase: problems to anticipate, not an architecture to copy

Jack Heart's clarification, 2026-11-04: Phase is probably the most useful
comparison because it has encountered problems managym has not, while he judges
its code substantially less disciplined. The useful research question is which
failure classes its broader card coverage exposes, not whether to adopt its
implementation style.

This follow-up inspects regression-test source at Phase revision
`381dc910b946881420dc752120c02689d7d53b0f`. The tests and their explanations are
primary evidence of the failure modes they were written to prevent. They were
not executed here; historical failures are not claims that current Phase is
still broken. Their rules examples are often outside Standard, but the shared
mechanisms also occur in the reviewed Standard sets.

## The important architectural warning

**An operation that ordinarily needs no decision can still suspend when a
replacement applies.** That includes paying a cost, placing a counter, moving
an unselected library card, creating a token, or applying lifelink. A caller
cannot infer “runs to completion” from the ordinary effect's shape.

Our current non-suspending `ForEachTarget` and one-pass replacement processor
are therefore connected limitations. Completing replacement choices will expose
callers that never retained their continuation. A shared resumable execution
contract should be designed before separately extending every effect family.
This is an inference from their documented failures and our current source,
not proof that every analogous managym path is already defective.

## Specific failure classes and proposed witnesses

**1. A child decision can orphan its parent and break a later, unrelated spell.**
Phase's proliferate regression documents a parent frame left resident while
counter placement suspended for replacement ordering. The remaining proliferate
was lost; a later tutor exposed the stale frame. Its fixture checks both exact
counter totals and absence of stranded frames. For us: repeat an instruction,
interrupt its first iteration, complete all repetitions, then resolve a different
choice-producing spell. A final-board assertion alone misses a poisoned runner.
[Proliferate regression](https://github.com/phase-rs/phase/blob/381dc910b946881420dc752120c02689d7d53b0f/crates/engine/tests/integration/issue_7384_proliferate_counter_replacement_frame.rs).

**2. Cleanup is part of the continuation.** Surveil's rejected cards formerly
took a direct zone-move path that bypassed redirects. Once those moves could
pause, the retained-library ordering also needed a completion owner. For us:
surveil several cards, choose their order, redirect two discarded cards through
replacement choices, and assert every move and final reorder occurs once.
Clone and resume at the middle choice. Preserve private inspection throughout.
[Surveil continuation regression](https://github.com/phase-rs/phase/blob/381dc910b946881420dc752120c02689d7d53b0f/crates/engine/tests/integration/surveil_rest_pile_redirect_continuation.rs).

**3. Costs need retained obligations as well as a batch cursor.** Phase documents
a random-discard payment losing both its remaining moves and its final paid/not
paid settlement when a zone replacement paused it. The tests distinguish cost
discard from effect discard and include an unpayable control. For us: pending
payment owns chosen/random objects, remaining work, actual results and the
settlement branch across every nested choice. A redirected discard is not
automatically an unpaid discard. Test accepted and declined replacement paths.
[Discard-cost continuation regression](https://github.com/phase-rs/phase/blob/381dc910b946881420dc752120c02689d7d53b0f/crates/engine/tests/integration/random_discard_cost_replacement_resume.rs).

**4. Replacement products remain part of the modified event.** Phase's token
batch tests document lost appended tokens when replacements created nested
events rather than transforming the same event. New products can make another
replacement applicable; subsequent instructions can refer to all resulting
tokens. For us: combine a token modifier, a second modifier and an entry
replacement, choose different valid orders, and check counts, attached grants,
delayed cleanup and event identity. Do not reduce this to multiplying a count.
[Appended-token regressions](https://github.com/phase-rs/phase/blob/381dc910b946881420dc752120c02689d7d53b0f/crates/engine/tests/integration/appended_token_batch_replacement_order.rs).

**5. Simultaneity does not forbid internal rules choices.** Phase records a
combat-lifelink path that discarded gains when replacement ordering suspended,
based on an incorrect assumption that combat damage could not pause. Its tests
retain gains, controllers, batch events and completion, while delaying trigger
placement until the batch is complete. For us: separate an internal replacement
decision from a priority window. No player gets to cast in the middle, but the
affected player still makes the required choice. Test two lifelink sources and
both damage and life-gain observers through the pause.
[Combat/lifelink regressions](https://github.com/phase-rs/phase/blob/381dc910b946881420dc752120c02689d7d53b0f/crates/engine/tests/integration/combat_lifelink_replacement_ordering.rs).

**6. “Use LKI” is not one universal fallback.** Phase's self-copy and controller
tests distinguish live exact incarnations, departed-object copiable values,
current controller queries, event-time facts and specific zone-transition
exceptions. Our contextual-read audit should make those distinctions explicit.
For us: change control without changing incarnation; blink the object; copy it;
remove a grantor; compare each against an unmodified control. Never substitute
a later incarnation merely because its physical storage ID matches.
[Self-copy tests](https://github.com/phase-rs/phase/blob/381dc910b946881420dc752120c02689d7d53b0f/crates/engine/tests/integration/self_copy_last_known_values.rs),
[controller/LKI tests](https://github.com/phase-rs/phase/blob/381dc910b946881420dc752120c02689d7d53b0f/crates/engine/tests/integration/event_controller_last_known.rs).

## What the commit history adds

The proliferate tests arrived with
[August 15 fix #7403](https://github.com/phase-rs/phase/commit/89108c36792fe30de2c49b33e2ef0c02d2a508c0).
The commit explains a second failure beyond lost work: sharing counter-placement
machinery accidentally emitted a proliferate event for an operation that was
not proliferating. Caller-supplied completion data replaced the hard-coded
event. This supports a precise composition boundary: share mechanical work,
retain the enclosing instruction's domain event and emit it once.

[August 21 change #7581](https://github.com/phase-rs/phase/commit/d6160491ef18d6eaa76d09ea58bbd617fc587658)
shows that fixing lifelink suspension was not the end. Follow-ups addressed
triggers entering the stack during replacement choices, duplicate collection,
and delayed-trigger collection. Suppressing an entire event to prevent one
duplicate could suppress a different legitimate observer. For us, distinguish
committed event identity, observer collection identity, occurrence identity,
and the later placement batch. A global “event consumed” bit is too coarse.

Two recent fixes are especially relevant to our performance goal:

- [November 4, #10603](https://github.com/phase-rs/phase/commit/e524890ff0f7c9b0b8931976071dd240654c95f1)
  records an Anger grant staying inactive after milling. Layer invalidation
  considered other live readers of the changed zone but missed the moved
  source's own zone of function. The fix also covered a separate library
  placement path. Our cache dependency review must include **sources becoming
  active/inactive**, as well as values read by already-active sources, and every
  authoritative mutation path.
- [November 3, #10578](https://github.com/phase-rs/phase/commit/abb4f58d873f9ae7ffb2d2d33cca697d369157c4)
  describes a board wipe's cubic work: departure/observer pairs each rescanned
  the batch and cloned context. A collection-local index and early exclusion of
  observers without triggers removed repeated work. Borrow the indexing
  principle and bounded operation-count tests; their reported timings are not
  measurements of managym and do not justify local load generation.

[November 1, #10537](https://github.com/phase-rs/phase/commit/b46fc00de482a300393faff67610418a68eae687)
records compound ward paying only its first component. Independent affordability
checks also missed shared-resource conflicts, such as mana production spending
life needed by another cost. One payment authority now checks the supported
composite shape jointly. This echoes our Prism history: a typed list of costs
does not supply a valid joint payment witness by itself. Preserve supported-shape
limits explicitly rather than extrapolating completeness from this fix.

## What to take, and what to scrutinize

Take the counterexamples and the test discipline: prove the pause occurred,
name its chooser, assert exact outcomes and event multiplicity, inspect the
intermediate state, and perform a later action to expose stale continuation
state. Use a negative control that would fail if the feature never ran. Keep
tests bounded by expected transitions rather than arbitrary waiting.

Several inspected fixes introduce specialized pending records or completion
variants. They demonstrate the state that must survive; they do not establish
that one bespoke continuation per effect is the best architecture for us.
Prefer a small shared runtime context and explicit semantic results while
keeping native `Effect` definitions as the only executable card representation.
Choose allocation and sharing strategies using our own step/search measurements.

The random-discard fixture also documents excluding the resolving spell from a
redirect so a separate single-slot pending-replacement bug would not overwrite
the choice under test. That is useful scope honesty, but a passing narrow test
cannot establish the unrestricted workflow. Our review should record both the
isolated fix and the still-required composition witness, rather than forgetting
why a fixture was narrowed. This describes the test's retained limitation,
not a fresh reproduction of Phase's current behavior.

The recommendation is to maintain a small cross-mechanism regression matrix
derived from these failures. It should exercise the shared execution substrate
before hundreds of new definitions depend on it. It need not grow as every pair
of cards, and its planning coverage must not become a new runtime card allowlist.

Validation: primary test source and current managym audit reviewed; no external
code executed and no performance or current-conformance claims made.
