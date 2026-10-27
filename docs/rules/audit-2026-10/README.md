# Rules unification: bounded delivery review

This increment follows the accepted plan to find competing definitions of a rule,
adjudicate them against the pinned MTG rules, and share the faithful authority.
It preserves the original committed inventory and draft; it does not repeat
the whole-inventory audit or claim full MTG conformance.

Start with the [coverage map](coverage-map.md) for strengths and gaps by kind
of rule. The review below explains the shared targeting contract.

## Concept review: one rule authority, checks at the right time

Jack Heart's accepted direction is to adjudicate competing implementations
against MTG and align around the faithful authority. The product consequence
is that human and bot choices come from the same native rules model; neither
consumer needs to learn which internal target branch or payment path was used.
The existing Academy usage contract already expresses this and needs no rewrite.

**Targeting contract:** choose a target from the legal
offers, then commit its bound Command. An opponent's hexproof permanent cannot
be offered merely because the instruction says “spell or permanent.” Your own
hexproof permanent remains targetable when otherwise legal. If the target gains
hexproof in response, resolution rechecks the new state. Reconnecting restores
the actual pending decision; it neither retargets automatically nor refunds
committed costs. The native repair has directed outcome witnesses; consumer checks exercise the existing shared contract.

| Your action | Shared concept | Authority/API responsibility |
|---|---|---|
| Inspect and choose a target | Legal offer in the current decision | Native `legal_targets_for_spec` enumerates through `target_is_legal`; Academy projects viewer-safe choices. |
| Commit the offer | Command bound to its frame | `hero_command` / `_apply_bound_command` validate binding and apply through native authority. |
| Respond, then resolve | Chosen target in the new game state | Resolution calls `target_is_legal` again; changed legality is an MTG recheck, not a conflicting definition. |
| Recover after disconnect | Same committed state and actual pending decision | Existing recovery/frame machinery retains identity and choices; the browser does not reconstruct rules. |

Permanent-wide restrictions now precede the target-spec match in
`target_is_legal`. Each branch retains its wording-specific restrictions, and
resolution rechecks current state. Grouping by target kind lets permanent branches
reuse one validated object and removes repeated presence checks. Both controller orientations, own hexproof,
response-time grants, mixed/all-invalid targets and leave/reenter identity have
bounded witnesses. The targeting repair originally advanced to w12. Shipping
integration with the separately landed shared-payment repair uses w13 and fresh
combined-runtime receipts. Tensor shapes and authored content are unchanged; earlier worlds,
checkpoints and `erd1` evidence retain their identities.

## Concrete change

Stack-target offers independently enforced minimum mana value while acceptance
and resolution enforced it in `target_is_legal`. Both were faithful for the
inspected printed-cost subset. Offers now use that shared predicate; the
independent threshold comparison is removed. Explicit Bolt (1), Ogre (3) and
Mountain (0) witnesses prove choices at five thresholds, rejection of a
low-value target, successful countering, and failure of the entire ability
when its only target departs. Candidate order and stage-specific checks remain.

Review [target legality](../../../managym/src/flow/resolution.rs),
[candidate enumeration](../../../managym/src/flow/triggers.rs), and
[independent witnesses](../../../managym/src/flow/target_authority_tests.rs).
[Authority adjudication](unification.md) names the competing definitions,
MTG verdict, chosen authority, removed logic and caller paths.

## Remaining separate finding

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

## Validation limits

Shipping integration uses **w13** after merging the independently owned payment
repair. The initial combined-source CI run exposed a stale w12 replay checkpoint
and a hard-coded w11 admission error assertion. The checkpoint matches the
payment repair's corpus exactly. Fresh w13 receipts preserve both older corpora
and all older demos; the error assertion now uses the runtime world version.

The combined source passes 6 conformance checks, 315 debug rules tests, 11
targeting witnesses, 51 world/admission/monitor tests, 31 Command/recovery/Study
runtime cases, all 7 Study branch cases and 43 inventory/selection tests. The
Prism/Treasure audit regression also passes after following the explicit nested
payment choices; it is no longer ignored. The separate trigger-order proposal
remains ignored. [Full w13 learnability](learnability-w13-integration.json) passes
in 39.4s, including 65,536 PPO transitions and checkpoint reload under unchanged
budgets. [Integration receipts](shipping-integration.json) retain failures,
source hashes and the repair's checks. New-head CI remains the landing gate.

The pre-sync enumeration cleanup has checks against its rebuilt, atomically
installed native extension: 11 debug targeting witnesses, 300 debug rules cases,
6 w12 conformance checks, 11 Command/recovery and decision-surface cases, and
21 dependency-blocked play/Study protocol/runtime cases. The known payment and
trigger-order probes remain ignored, with their failures retained separately.
Clippy, Rust formatting and changed Python lint pass. Clippy's initial finding
in the new mana matrix was fixed without changing its cases or expected outcomes;
that matrix was rerun. All 7 Study branch, 11 arena admission/replay and 20 monitor
cases also pass. The fresh [gate learnability diagnostic](learnability-w12-gate.json) passes in
43.0s with 144 capacity assertions and 65,536 reward-driven PPO transitions,
including checkpoint reload (after 1.0; no-update 0.640625). Its 480s total /
300s PPO watchdogs are unchanged. The selected checks total 317 passing debug
Rust tests and 116 Python tests, plus the targeted mana recheck after the lint
fix. See the gate receipts for source/native digests and timings.

At acceptance checkpoint `bda9b0aeb`, the w12 repair had 10 passing native
targeting witnesses and 6 debug conformance
checks over a fresh 14-case / 3,491-Command corpus. Dependency-blocked play/Study
protocol (17), Command/recovery (8), repaired decision surface (4), and tooling/
package-boundary (44) tests pass. All 11 arena admission cases passed across
bounded chunks. Frozen w11 and protocol fixture bytes are unchanged.

The full [integrated learnability diagnostic](learnability-w12-integrated.json)
passes in 43.5s after the upstream `07ddef0e` collector integration. It includes
all three capacity seeds and reward-driven PPO with checkpoint reload (65,536
transitions; after 1.0 versus no-update 0.640625). The positive control retains
its original one-seat recipe and 480s total / 300s PPO watchdogs. The earlier
[99.6s pass](learnability-w12-canonical.json) remains historical evidence.
Fresh dependency-blocked play/Study protocol checks pass all 17 tests against
the extension rebuilt for that checkpoint. Its native targeting, integrated
diagnostic and protocol results were fresh; conformance, arena, Command/recovery
and decision-surface counts reused receipts for then-unchanged production
sources. These are historical acceptance results, not full suites rerun after
the enumeration cleanup.

Study semantic retry, all seven Study branch cases and the four outstanding
monitor cases now pass. The branch fixture had been repeatedly persisting a
full growing replay while manually passing priority. Reusing returned frames
and enabling configured, recorded priority automation completes its seven
cases in 69.2s; all existing assertions and complete-match creation remain.
The four monitor cases pass in 26.1s; the other 16 also pass in a separate
14.8s bounded invocation after reviewing the upstream monitor rename. Earlier timeouts remain in the
[validation receipts](unification-validation.json), including a new bounded
failure with a persistence stack sample. No production persistence behavior,
training budgets, timeout limits or assertions changed.

Gate measured 331.942 command-seconds including native build, failed Clippy and
targeted rechecks, plus full learnability. This is local command duration, not
CI runner cost or session wall time. No required affected check is deferred;
the full platform/cross-language/fuzz matrix remains CI-owned. Recorded authored
churn is 4,274 lines against the approximate 10k budget.

## Ownership, checks and scope

The refreshed MTG-174 reference is stale: it currently identifies an Elemental
training Task. Task `4ba5dd` still owns composable optional payments; its recorded
landing failure and merged PR are documented without changing its lifecycle.
Payment repair remains outside this increment.

[Validation receipts](unification-validation.json) record passing debug checks,
historical failing proposal probes, the corrected initial fixture compile
failure, native rebuild/import, Python checks, cost and coverage limits.
[Line accounting](line-accounting.json) reconciles the original 621-line draft
with this bounded increment against the approximate combined 10k-line budget.
Earlier frozen worlds, decks, experiment evidence and training budgets remain unchanged.
CI conformance checks select the w13 root with unchanged required-check semantics.
On 2026-10-22, Jack Heart reviewed the walkthrough and approved shipping this
bounded increment, superseding older draft-only holds. Gate prepares this checkout;
the parent conversation owns publication and landing. MTG-177 remains open for the
unfinished substantive whole-rules audit.

## Reference evidence

The [original evidence](evidence.md), [source pin](source.json),
[rule index](index.jsonl), [authored reviews](reviews.json),
[original validation](validation.json) and [CI analysis](ci.md) remain available.
Five more target entries (115.1c/d, 115.2/3/6) now have individual scoped
findings, bringing the index to 30 partial, 3 implemented, 2 missing,
322 not applicable and 2,965 unknown entries. The inventory is complete;
the substantive whole-rules audit remains unfinished. Unknown rules do not
inherit these results. Stack CardId loses incarnation, but the new ordinary
recast/LIFO witness does not expose a supported-play stale-target effect.
The original report's counts describe its original snapshot.

## MTG-195 follow-up — 2026-10-27

World w14 replaces the fixed trigger-order owner identified by this audit with
controller-selected placement. The live anchors in `reviews.json` and its generated
index now point to that authority and its focused witnesses. CR 603.3b remains
partial: the two-pass scheduler has typed fixture coverage, while matching on
another ability triggering is still unsupported. The original source identity,
audited revision, draft and `unification.md` retain the dated audit findings.
See [the w14 contract](../../../WORLDS.md#w14-controller-selected-trigger-placement).
