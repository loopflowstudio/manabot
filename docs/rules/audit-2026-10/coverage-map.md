# Where the rules are strong, narrow or still open

The earlier October 26 audit reviewed **150 individual entries** in CR 115, 400.7,
603, 608 and 704 against checkout `324db65f2` and the pinned September 25,
2026 source. It added two passing resolution witnesses and three explicitly failing,
ignored proposal probes. That audit changed no production rules or world identity;
the subsequently selected resolution repair is described below.

The complete index has **3,322 entries**: 3 implemented, 92 partial, 57 missing,
330 not applicable and 2,840 unknown. Those are review dispositions, not a
percentage of MTG implemented. The 150 reviewed entries comprise 2 implemented,
74 partial, 56 missing, 8 not applicable and 10 unknown. Headings and other
families receive no inherited upgrade. Each entry in [reviews.json](reviews.json)
and the generated [index](index.jsonl) states its own scope, authority, comments,
evidence and remaining obligation.

## Resolution repair update — October 26

The selected repair replaces per-instruction legality rechecks with one entry-time
predicate and retained clause-specific eligibility. All three stack paths now
suppress all-invalid resolution; independent tests cover both controllers, mixed
slots, optional zero targets, type/hexproof changes, suspension, clone/undo and
object departure/reentry. The original hexproof proposal is enabled and passing.
CR 608.2b stays partial: this is bounded evidence, not exhaustive LKI/target support.
Final integrated world, receipts and consumer acceptance await MTG-195 handoff.
[Current repair evidence](resolution-validation.json) is separate from the original
failure receipts. Deathtouch history and earthbend destination identity remain open.

## Earlier audit: authority and limits

| Area | Established subset | Remaining obligation |
|---|---|---|
| Targets — CR 115 | Shared target predicate, explicit zones, clause-local distinctness, spell mode binding, optional zero targets and untargeted controller effects. | Any-target damage omits planeswalkers/battles. Auras and retargeting are absent. Target-inspection wording in 115.9 remains unknown. Activated source exclusion is broader than stack-self exclusion; no current-card failure established. |
| Zone identity — CR 400.7 | Permanent/ObjectRef incarnation invalidation, source LKI, `was_cast`, granted flashback retention. | **Earthbend return follows a land from graveyard into exile** in a directed probe; its destination incarnation is not retained. Stack-target CardId remains a separate representation concern without a supported-play failure. Aura, sticker and madness exceptions are absent. |
| Triggers — CR 603 | Event collection, ordinary ETB/death/sacrifice matching, source snapshots, target placement, APNAP, explicit intervening-if branches and resolution-time optional choices. | MTG-195 owns controller-selected ordering. True modal-trigger placement, reflexive/state triggers and most look-back exceptions are missing. `Effect::Modal` callers for color/token/may choices correctly choose at resolution; they are not evidence for 603.3c. |
| Resolution — CR 608 | LIFO, ordered resumable frames, target checks, current counts and selected source LKI, spell finalization through zone movement. A new witness survives a temporary zero-life state across a mid-resolution choice. | The per-instruction hexproof failure and missing activated all-invalid gate are repaired in the isolated draft above. Abilities pop before effects finish. Full characteristic LKI and multi-player simultaneous instructions are unproved. Permanent entry uses owner rather than spell controller; spell-control changes are not admitted. |
| State-based actions — CR 704 | Life/draw losses, bounded toughness/damage matrix, token removal, legend choices and repeated stabilization. | **Deathtouch history survives an earlier SBA check** and causes a later erroneous destruction in a synthetic keyword-removal probe. Shared death-source snapshots do not prove complete pre-batch LKI. Loyalty, battle, attachment and other listed SBAs lack their required state/dispatch. Variant-only SBAs are outside admission. |

The three original failing probes used explicit controlled state changes. They established
precise missing obligations, **not** that the selected decks can execute those
response sequences. Their assertions retain the CR-required outcomes, and their
ignore reasons keep the draft from turning known defects into passing expectations.
Those original results preceded the selected resolution repair above; no other repair or stack-target ABI migration is selected.

## Review priorities

1. **Existing shared-choice work:** MTG-195 (trigger order) and MTG-196
   (replacement/prevention order) retain their ownership and design boundaries.
2. **Remaining exact obligations:** decide whether to sponsor destination-ObjectRef
   retention for earthbend (400.7e, 603.6, 603.7c) and consuming deathtouch history
   at the SBA boundary (704.5h). The selected resolution repair (608.2b) awaits
   integrated acceptance. Retain the positive first-departure, first-SBA
   and response-time target-recheck witnesses when repairing these gaps; shared-path parity cannot replace them.
3. **Before expanding authoring:** distinguish true modal triggers from
   resolution choices; preserve ability stack lifetime; define effective/LKI
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

[Earlier audit receipts](obligation-validation.json) distinguish that audit's
executed checks from read-only evidence references. Its test/report-only step
ran debug probes and source/index checks, without rebuilding native code or
running consumers. The later resolution repair has separate receipts above;
broader integrated acceptance remains gate-owned.
The [prior delivery review](README.md#prior-delivery-review) and
[authority adjudication](unification.md) remain historical evidence for PR354.
