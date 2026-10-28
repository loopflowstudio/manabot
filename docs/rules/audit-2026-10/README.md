# Rules audit: integrated resolution repair

The selected CR 608.2b repair now retains clause-specific resolution eligibility
across spells, activated abilities, triggers and suspended choices. Effect handlers
use that result with separate object-presence guards. The retained hexproof probe
is enabled; earthbend destination identity and deathtouch history remain ignored,
unresolved proposals. See [resolution evidence](resolution-validation.json).

The repair is integrated with MTG-195’s merged PR386 in development world w15 /
hash 10. Policy-input and capture/text v4 expose retained eligibility while v2/v3
relation widths remain frozen. Fresh conformance/demo receipts bind the combined
source. Jack Heart authorized shipping this repair on 2026-10-28 after reviewing
the concept walkthrough. That authorization supersedes the earlier review-only
boundary for this increment; the whole-rules audit remains open. Historical
evidence stays unchanged. See [shipping acceptance](resolution-shipping.md) for
the integrated checks and remaining limits.

## Earlier obligation audit

The earlier audit reviewed **150 individual obligations** in CR 115, 400.7,
603, 608 and 704 on post-sync checkout `324db65f2`. The exact official TXT was
recovered and its pinned digest verified; [source.json](source.json) is unchanged.
That earlier audit changed no production behavior or world identity; its frozen
receipts remain unchanged.

Start with the [current coverage map](coverage-map.md) for prioritized findings.
The per-entry [reviews](reviews.json) and generated [index](index.jsonl) retain
specific authorities, evidence and limits. Overall dispositions are now
**3 implemented, 92 partial, 57 missing, 330 not applicable and 2,840 unknown**.
A missing unsupported mechanic is not a reproduced admitted-game defect.

Two ordinary debug tests passed: optional choices wait until resolution, and
SBA life loss waits until a suspended resolution finishes. Three ignored proposals
were explicitly run and failed at the required outcome: earthbend follows a land
through a second zone change; surviving deathtouch remains lethal at a later SBA
check; granting hexproof during resolution incorrectly suppresses a later damage
instruction of that same ability. These controlled fixtures do not establish
reachable authored-deck response sequences. Their exact commands, outputs and
source/test hashes are in [obligation-validation.json](obligation-validation.json).

Other findings include the absent true modal-trigger placement decision and
narrow LKI/ability-stack-lifetime semantics. Current resolution-time color/token
choices must not be moved to placement merely because their enum is called
`Modal`. Trigger ordering stays with MTG-195 and replacement ordering with
MTG-196. These earlier findings retain their dated scope; the shipping decision
above applies only to the resolution repair. The whole-rules audit remains open.

## Prior delivery review

PR354 merged as `4a2744705e0f493ce05c0643e5eafe59a5749c08`. It unified
stack-target mana-value checks and permanent-wide hexproof enforcement, with
independent target, identity and response-time witnesses. The combined targeting
and separately delivered payment repair shipped as w13; older worlds,
checkpoints and experiment evidence retain their identities. The Prism/Treasure
regression passes through explicit nested payment choices.

Jack Heart approved that bounded delivery on October 22. The whole-rules audit
remained open with 2,965 unknown entries. The separate October 28 authorization
selects this repair's delivery, not whole-audit completion. Earlier payment-owner identifiers are stale;
[the ownership review](unification.md#refreshed-ownership-and-delivery-boundary)
explains their historical context.

Historical evidence, including failures and limitations, remains in:

- [Authority adjudication](unification.md): competing paths, chosen authority,
  independent witnesses and remaining obligations.
- [Shipping integration](shipping-integration.json): combined-source checks,
  stale-world failures and repairs, and fresh w13 receipts.
- [Unification validation](unification-validation.json): earlier targeting,
  consumer and learnability results, including timeouts and corrections.
- [CI analysis](ci.md) and [line accounting](line-accounting.json): measured
  costs and the conservative authored-line budget.
- [Original audit](evidence.md) and [validation](validation.json): initial
  inventory and bounded coverage, not a whole-rules conformance claim.

The complete prior walkthrough is retained at
`dccf75632e18e88abc2fd92904ea4d81535cadfc:docs/rules/audit-2026-10/README.md`.
Its counts and acceptance results describe those checkpoints, not new checks
against this draft.

## Maintaining the inventory

Edit individual findings in [reviews.json](reviews.json). Regenerate the index
with the existing source-pinned tool; do not replay the retired one-shot review
author or infer child-rule status from family evidence:

```bash
uv run python scripts/audit_rules.py --source .cache/rules-audit/MagicCompRules-20260925.txt --write
```

Omit `--write` to verify exact agreement with the pinned text and authored
reviews. Without `--source`, the tool checks the committed inventory and anchors
offline. The raw Wizards text stays local; source metadata and digests remain in
[source.json](source.json).

## MTG-195 follow-up — 2026-10-27

World w14 replaces the fixed trigger-order owner identified by this audit with
controller-selected placement. The live anchors in `reviews.json` and its generated
index now point to that authority and its focused witnesses. CR 603.3b remains
partial: the two-pass scheduler has typed fixture coverage, while matching on
another ability triggering is still unsupported. The original source identity,
audited revision, draft and `unification.md` retain the dated audit findings.
See [the w14 contract](../../../WORLDS.md#w14-controller-selected-trigger-placement).
