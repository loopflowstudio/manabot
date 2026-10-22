# Rules unification: draft review guide

This draft follows the accepted plan to find competing definitions of a rule,
adjudicate them against the pinned MTG rules, and share the faithful authority.
It preserves the original committed inventory and draft; it does not repeat
the whole-inventory audit or claim full MTG conformance.

## Concrete change

Stack-target offers independently enforced minimum mana value while acceptance
and resolution enforced it in `target_is_legal`. Both were faithful for the
inspected printed-cost subset. Offers now use that shared predicate; the
independent threshold comparison is removed. Explicit Bolt (1), Ogre (3) and
Mountain (0) witnesses prove choices at five thresholds, rejection of a
low-value target, successful countering, and failure of the entire ability
when its only target departs. Candidate order and stage-specific checks remain.

Review [the small native change](../../../managym/src/flow/triggers.rs) and
[independent witnesses](../../../managym/src/flow/target_authority_tests.rs).
[Authority adjudication](unification.md) names the competing definitions,
MTG verdict, chosen authority, removed logic and caller paths.

## Behavioral findings retained as separate proposals

- **Hexproof inconsistency:** creature/permanent target branches enforce
  CR 702.11b, while `SpellOrPermanent` bypasses it. A tiny explicitly failing
  probe establishes the discrepancy. The correction requires new world and
  admission evidence; it is not applied to the current world in this draft.
- **Trigger-order choice:** one ordering implementation exists, with fixed
  APNAP/enqueue order. At 2 life, simultaneous gain-2 and damage-2 triggers
  permit survival or loss depending on the controller's CR 603.3b choice.
  Both outcomes are verified; a separate failing probe shows that the shared
  action surface cannot offer the ordering choice. This remains a missing
  choice, not proved duplication. No browser-only workaround or new ABI is added.

The [detailed review](unification.md) gives exact remaining implementation and
validation obligations, along with inspected negative findings for target
identity, SBA/replacement/movement, combat and static queries. Existing mixed,
all-invalid and leave/reenter target coverage is retained. Distinct stages and
legitimate rechecks have not been collapsed into a universal rules engine.

## Ownership, checks and scope

The refreshed MTG-174 reference is stale: it currently identifies an Elemental
training Task. Task `4ba5dd` still owns composable optional payments; its recorded
landing failure and merged PR are documented without changing its lifecycle.
Payment repair remains outside this increment.

[Validation receipts](unification-validation.json) record passing debug checks,
two intentional failing proposal probes, the corrected initial fixture compile
failure, native rebuild/import, Python checks, cost and coverage limits.
[Line accounting](line-accounting.json) reconciles the original 621-line draft
with this bounded increment against the approximate combined 10k-line budget.
No frozen worlds, decks, experiment evidence, training budgets, CI selection or
required-check semantics change. No ready publication, merge, auto-merge or
Task completion is selected. The durable draft stops for review.

## Reference evidence

The [original evidence](evidence.md), [source pin](source.json),
[rule index](index.jsonl), [authored reviews](reviews.json),
[original validation](validation.json) and [CI analysis](ci.md) remain available.
The four individually reviewed target-related entries receive scoped partial
findings in this increment; unknown rules do not inherit those results.
The original report's counts describe its original snapshot.
