# Rules unification: draft review guide

This draft follows the accepted plan to find competing definitions of a rule,
adjudicate them against the pinned MTG rules, and share the faithful authority.
It preserves the original committed inventory and draft; it does not repeat
the whole-inventory audit or claim full MTG conformance.

Start with the [coverage map](coverage-map.md) for strengths and gaps by kind
of rule. The concept review below explains the desired experience and next proof.

## Concept review: one rule authority, checks at the right time

Jack Heart's accepted direction is to adjudicate competing implementations
against MTG and align around the faithful authority. The product consequence
is that human and bot choices come from the same native rules model; neither
consumer needs to learn which internal target branch or payment path was used.
The existing Academy usage contract already expresses this and needs no rewrite.

**Proposed usage for the hexproof correction:** choose a target from the legal
offers, then commit its bound Command. An opponent's hexproof permanent cannot
be offered merely because the instruction says “spell or permanent.” Your own
hexproof permanent remains targetable when otherwise legal. If the target gains
hexproof in response, resolution rechecks the new state. Reconnecting restores
the actual pending decision; it neither retargets automatically nor refunds
committed costs. These are intended outcomes, not verified repaired behavior.

| Your action | Shared concept | Authority/API responsibility |
|---|---|---|
| Inspect and choose a target | Legal offer in the current decision | Native `legal_targets_for_spec` enumerates through `target_is_legal`; Academy projects viewer-safe choices. |
| Commit the offer | Command bound to its frame | `hero_command` / `_apply_bound_command` validate binding and apply through native authority. |
| Respond, then resolve | Chosen target in the new game state | Resolution calls `target_is_legal` again; changed legality is an MTG recheck, not a conflicting definition. |
| Recover after disconnect | Same committed state and actual pending decision | Existing recovery/frame machinery retains identity and choices; the browser does not reconstruct rules. |

The most consequential mismatch is priority within this draft: the completed
consolidation removes equivalent logic, while the demonstrated hexproof
disagreement remains unrepaired. The next design direction is to centralize
permanent-wide targeting restrictions so target-spec branches cannot bypass
them. Keep spell/permanent distinctions and resolution-time checks; a single
boolean cached at offer time would be wrong after a response.

No new public targeting concept or competing UI workflow is needed. Preserve
exact object references, Command binding, persisted choices and historical
world identities. Behavioral repair must have a newly identified world and
fresh admission evidence rather than relabeling old receipts. The existing
draft-work instructions already require rule adjudication and recovery proofs;
no skill change is needed to express this direction.

**Next proof:** the same permanent under different target wordings, both
controllers, gaining hexproof in response, and leave/reenter identity; then
stale Command rejection, reconnect, replay/fork and required admission checks.
Trigger ordering follows as a separate missing shared choice. No unresolved
product choice is identified here: faithful choices and recovery were already
accepted. This design direction does not launch implementation.

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
