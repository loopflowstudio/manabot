# MTG-177 — deliver the most valuable rules-audit work

## Outcome and delivery boundary

Jack Heart's original request was a whole-Comprehensive-Rules audit of
implementation, ownership, comments and behavioral tests, plus concrete bounded
improvements and a measured CI-cost assessment. Later direction prioritized
competing definitions: judge them against MTG, keep the faithful authority and
remove the contradictory implementation. On 2026-10-22, Jack Heart requested
this plan focus on landing the most important results against those goals.

Deliver a reviewable change that fixes a real supported-play defect, establishes
useful reviewed coverage, and makes the remaining rules investments clear.
The inventory is complete; the substantive whole-rules audit is not. Landing a
bounded increment must not be reported as completing that original obligation.
This plan orders the remaining work. The existing draft-review boundary still
applies: publication, merge and Task completion require their selected delivery
step; rewriting this plan starts none of them.

## 1. Finish acceptance of the targeting repair

The highest-value code change already exists: `target_is_legal` applies live
battlefield presence and opposing hexproof before target-wording restrictions.
Offers, acceptance and resolution share it. Own hexproof stays targetable;
responses can invalidate targets; mixed legal targets still resolve; departing
and returning objects do not inherit old targets. The independent stack-target
mana-value comparison is removed. Keep this repair as the core of delivery.

Close these concrete acceptance gaps on the final integrated source:

- Run the full learnability diagnostic after the upstream collector integration.
  The 99.6s canonical pass predates that integration and remains historical
  evidence. Keep the earlier collector timeout too; contention is plausible,
  not a proved cause. Preserve the existing 480s total / 300s PPO watchdogs.
- Finish Study semantic retry, all seven Study branch cases, and four monitor
  cases: random/explicit-matrix outcomes, admission-failure persistence and the
  saved dashboard. Inspect the retained timeout logs first and use focused,
  bounded invocations. Partial suites are not passes. If a check still cannot
  finish, identify the exact check and cause; “gate/CI owns it” is not closure.
- Verify the final w12 native build, conformance selection, arena admission and
  Command/recovery evidence against the delivered source. Reuse receipts only
  where their tested code and dependencies remain applicable. Do not rerun
  unrelated suites for prose changes.

**Done when:** affected behavior and required current-world checks have passing
receipts tied to the final source, or a specific unresolved acceptance failure
is plainly blocking delivery. Preserve all failures and frozen w11 evidence.
No assertion weakening, timeout inflation or speculative training repair.

## 2. Turn the inventory into a useful implementation audit

The main shortfall is evidence depth: 3,166 numbered rules/subrules plus 156
headings are indexed, but 2,970 entries remain unknown. Generic family triage
cannot satisfy the original per-rule request. Deepen the audit in this order:

1. **Targets and resolution (CR 115, 400.7, 608, 702.11).** Finish the reviewed
   obligations around this repair: each supported target form, restrictions,
   response-time changes, mixed/all-invalid targets and object identity. Inspect
   the unresolved `Target::StackSpell(CardId)` incarnation boundary explicitly.
2. **Triggers and state-based actions (CR 603, 704).** Establish which supported
   event/stack/ordering obligations are implemented and tested. Retain the
   controller-order counterexample: at 2 life, gain-2/damage-2 ordering changes
   survival. Fixed enqueue order does not provide the required player choice.
3. **Replacement interactions (CR 614–616), then current-card combat and layers.**
   Review real supported interactions and their owners, including affected-player
   replacement ordering. Distinguish legitimate stages from duplicate semantics.
   General timestamps/dependencies and unused mechanics need explicit scope.

For each reviewed rule/subrule, inspect the actual authority and callers,
explain the implemented subset, assess the nearby intent comment, and identify
an independent behavioral witness or precise gap. Compare competing paths
against the pinned rule before choosing one. A passing agreement test or matching
citation alone is insufficient. Update individual index rows and the coverage
map; keep unreviewed rows unknown and unsupported formats distinct from defects.

**Done for each landing increment:** its claimed rule set is explicitly listed,
individually adjudicated and traceable to code and evidence; the coverage map
explains strong subsets and consequential gaps. Keep the remaining whole-rules
review visible until actually completed. Do not claim a percentage implemented
from inventory statuses or reduce unknown counts through bulk relabeling.

## 3. Add only improvements that make those findings actionable

Retain the cheap mana and SBA matrices and the targeting regressions. Add missing
negative cases and bounded combinations discovered by the review. Put current
rule citations beside the implementation that owns the invariant, explaining
intent and limitations. Existing corrections cover targeting, setup, priority,
sacrifice and P/T layers; extend this within the rules actually inspected.

Repair small demonstrated supported-play contradictions through a single native
authority. Keep distinct phases, current-state rechecks and viewer-safe projections.
Trigger/replacement ordering are missing shared choices, not proven duplicate
engines: retain their witnesses and a concrete next implementation decision.
A large new decision model should not become an incidental addition to this
repair. Payment work remains with its owners; refresh ownership because the
historical MTG-174 payment reference is stale.

**Done when:** each added test, comment or consolidation closes a named finding;
remaining proposals state the user-visible consequence, native owner and next
proof. Avoid another round of equivalent-code cleanup ahead of known defects.

## 4. Finish the cost and review package

Keep the measured CI assessment and duration reporting. Existing samples show
35–46 runner-minutes for 9–11 minutes of wall span; sharding savings have not
been demonstrated. Make no topology change without evidence for both wall time
and total runner cost, duplicated setup, cancellation and required aggregation.
Do not weaken PR coverage to improve timing.

Refresh the concise review guide, coverage map and HTML walkthrough for the
final diff. Lead with the targeting behavior, reviewed rule families, corrected
citations and remaining gaps. Keep detailed inventory/receipts behind links.
Separate authored change cost from generated evidence: the last recorded charge
is 4,054 lines against the approximate 10k ceiling. Fresh conformance receipts
are evidence; runtime demo fixtures are charged product inputs. Preserve the
original pins and report observed incremental validation cost and checks not run.

**Ready for delivery review when:** the repair's acceptance is closed, the audit
claims match individually reviewed evidence, the line/cost accounting is current,
and the review surface explains what can land and what remains. Deliver the
bounded result through the selected PR operation. Keep the Task open if its
original substantive audit obligation remains unfinished.

## Evidence and constraints to carry forward

Current committed reference: `e6585adc2`. Code milestones: inventory/directed
checks `2b6e2b1b7`, shared stack-target predicate `c514d11b0`, permanent targeting
repair and w12 evidence `202416a8`. The latest upstream collector integration
is included; it is an independent contribution, not this audit's work.

Rules source: September 25, 2026 Wizards TXT, retained at
`scratch/MagicCompRules-20260925.txt`; SHA-256
`8d860e451f20f38865b725b42d82feb714c725373dd8f3b32b8652b3eeb070ca`.
Detailed evidence: `docs/rules/audit-2026-10/unification-validation.json`,
`learnability-w12-canonical.json`, `reviews.json`, `index.jsonl`, `ci.md` and
`line-accounting.json`. Historical plan detail remains in git.

Preserve complete human/bot legal choices, Command binding/stale rejection,
actual pending decisions on reconnect, costs and revealed information, viewer
privacy, replay/fork identity and identifiable priority-stop automation. Changed
rules meaning uses new world identity and fresh evidence; shapes alone do not
establish compatibility. No paid compute, stress loops, deck expansion, frozen
experiment edits or competing payment rewrite. All Python uses uv; Rust checks
run with debug assertions and two build jobs; replace rebuilt native extensions
atomically. Ordinary ceilings remain 900s cold Rust, 180s warm Rust and 120s
Python, with the existing full-learnability watchdog noted above.

Check: plan reconciled with current Task status and retained evidence; this edit
runs no engine tests and launches no Flow.
