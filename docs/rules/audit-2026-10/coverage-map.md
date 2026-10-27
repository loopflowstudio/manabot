# Where the rules are strong, narrow or still open

The October 26 draft reviews **150 individual entries** in CR 115, 400.7,
603, 608 and 704 against checkout `324db65f2` and the pinned September 25,
2026 source. It adds two passing resolution witnesses and three explicitly failing,
ignored proposal probes. It changes no production rules or world identity.

The complete index has **3,322 entries**: 3 implemented, 92 partial, 57 missing,
330 not applicable and 2,840 unknown. Those are review dispositions, not a
percentage of MTG implemented. The 150 reviewed entries comprise 2 implemented,
74 partial, 56 missing, 8 not applicable and 10 unknown. Headings and other
families receive no inherited upgrade. Each entry in [reviews.json](reviews.json)
and the generated [index](index.jsonl) states its own scope, authority, comments,
evidence and remaining obligation.

## This pass: authority and limits

| Area | Established subset | Remaining obligation |
|---|---|---|
| Targets — CR 115 | Shared target predicate, explicit zones, clause-local distinctness, spell mode binding, optional zero targets and untargeted controller effects. | Any-target damage omits planeswalkers/battles. Auras and retargeting are absent. Target-inspection wording in 115.9 remains unknown. Activated source exclusion is broader than stack-self exclusion; no current-card failure established. |
| Zone identity — CR 400.7 | Permanent/ObjectRef incarnation invalidation, source LKI, `was_cast`, granted flashback retention. | **Earthbend return follows a land from graveyard into exile** in a directed probe; its destination incarnation is not retained. Stack-target CardId remains a separate representation concern without a supported-play failure. Aura, sticker and madness exceptions are absent. |
| Triggers — CR 603 | Event collection, ordinary ETB/death/sacrifice matching, source snapshots, target placement, APNAP, explicit intervening-if branches and resolution-time optional choices. | MTG-195 owns controller-selected ordering. True modal-trigger placement, reflexive/state triggers and most look-back exceptions are missing. `Effect::Modal` callers for color/token/may choices correctly choose at resolution; they are not evidence for 603.3c. |
| Resolution — CR 608 | LIFO, ordered resumable frames, target checks, current counts and selected source LKI, spell finalization through zone movement. A new witness survives a temporary zero-life state across a mid-resolution choice. | **Per-instruction legality rechecks skip damage after the same ability grants its target hexproof** in a synthetic probe. Activated abilities also lack a whole-object all-invalid gate. Abilities pop before effects finish. Full characteristic LKI and multi-player simultaneous instructions are unproved. Permanent entry uses owner rather than spell controller; spell-control changes are not admitted. |
| State-based actions — CR 704 | Life/draw losses, bounded toughness/damage matrix, token removal, legend choices and repeated stabilization. | **Deathtouch history survives an earlier SBA check** and causes a later erroneous destruction in a synthetic keyword-removal probe. Shared death-source snapshots do not prove complete pre-batch LKI. Loyalty, battle, attachment and other listed SBAs lack their required state/dispatch. Variant-only SBAs are outside admission. |

The three new failing probes use explicit controlled state changes. They establish
precise missing obligations, **not** that the selected decks can execute those
response sequences. Their assertions retain the CR-required outcomes, and their
ignore reasons keep the draft from turning known defects into passing expectations.
No new Task, repair, ABI migration or frozen-world change is selected here.

## Review priorities

1. **Existing shared-choice work:** MTG-195 (trigger order) and MTG-196
   (replacement/prevention order) retain their ownership and design boundaries.
2. **New exact obligations:** decide whether to sponsor destination-ObjectRef
   retention for earthbend (400.7e, 603.6, 603.7c) and consuming deathtouch history
   at the SBA boundary (704.5h), and retaining initial resolution legality
   across instructions (608.2b). Retain the positive first-departure, first-SBA
   and response-time target-recheck witnesses when repairing these gaps; shared-path parity cannot replace them.
3. **Before expanding authoring:** distinguish true modal triggers from
   resolution choices; add activated all-invalid suppression before compound
   independent effects; preserve ability stack lifetime; define effective/LKI
   characteristics for simultaneous events. These are inspected limits, not
   newly reproduced current-card failures.
4. **Remain unknown:** 115.9/a/b/c, 400.7h, 603.2h, 603.7h, 603.11, 608.2e and
   608.2m still need concrete consumers or independent witnesses. Their relevant
   representations were inspected, but the obligations were not established.

## Evidence retained outside this pass

Bounded mana arithmetic, selected combat keywords, land/priority timing and
selected replacement outcomes retain their earlier evidence; this pass does not
reaudit those families. The integrated Prism/Treasure regression is a historical
**pass** through explicit nested payment choices, not an open payment defect.
General continuous effects, long-tail mechanics and format setup remain narrow
or unreviewed. Unsupported mechanics are separated from admitted-game defects.

[Current receipts](obligation-validation.json) distinguish executed checks from
read-only evidence references. This implementation step ran only the new debug
tests/probes, source/index verification and inventory integrity tests. Gate owns
any broader affected checks; no training, native wheel rebuild, consumer matrix,
CI-cost study or remote compute was needed for this test/report-only change.
The [prior delivery review](README.md#prior-delivery-review) and
[authority adjudication](unification.md) remain historical evidence for PR354.
