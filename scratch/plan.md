# MTG-177 — continue the substantive rules audit

## Current implementation direction — 2026-10-26

Jack Heart requested `pursue` after the candidate review. Implement the first
priority: one rules-faithful resolution-legality authority across spells,
activated abilities and triggered abilities, including suspended resolution.
This supersedes the earlier audit-only/review-only restriction for that repair.
Deathtouch history and earthbend destination identity remain named follow-ups;
MTG-195/196 retain trigger/replacement ordering ownership. The whole-rules audit
also remains unfinished. Return the concrete implementation and evidence for
review; ready publication, merge and Task completion remain unselected.

The accepted behavioral contract is CR 608.2b: evaluate target eligibility when
resolution begins, per targeting clause, and retain that decision while effects
execute or suspend. Do not reinterpret gained hexproof or changed type as a new
target-legality decision after each instruction. Keep one shared predicate for
selection, acceptance and resolution-start evaluation. Separately retain the
object-incarnation/presence guards needed when an instruction moves an object;
a blanket removal of all effect checks is not correct. Resolve the precise
EffectFrame representation from existing state before editing, documenting any
consequential ambiguity rather than inventing a second authority.

Delete — do not maintain: migrate the repeated effect-handler target-legality
rechecks to the single resolution contract, and unify divergent entry handling
including the activated all-invalid gate. Remove superseded checks in the same
change. Preserve per-clause semantics, source exclusions, legitimate response-time
rechecks, untargeted effects, zero optional targets and current identity rules.

Proof: enable the retained hexproof-during-resolution regression; independently
assert both controllers, spell/activated/triggered entry paths, mixed/all-invalid
and multi-clause targets, suspended choices, and departure/reentry during an
effect. Assert expected outcomes, not just agreement between consumers. Preserve
the other two ignored proposals as unresolved and add relevant rule citations.
Use focused debug Rust checks with two jobs first. Required native rebuild,
world/admission, maintained model/choice-separability and consumer/replay checks
remain part of the affected acceptance, with gate owning broader checks once.
Do not resurrect the deliberately removed learnability verifier.

Coordination: fresh inspection found MTG-195 has uncommitted changes in trigger
placement, action/observation state, hashing, rollback, targeting tests and w14
fixtures in its own checkout. Do not adopt or overwrite that active writer's
files. Resolution.rs and the frame contract are this repair's center; reconcile
adjacent changes against actual supported handoff before integration. Do not
reuse w14 for different semantics or silently overwrite its evidence. Select the
final world identity and fresh affected receipts only against the integrated
source; surface an integration dependency if the other work is not yet available.

The old pursue completed audit implementation/compression and stopped during
sync. Its preserved draft is recovered and committed in `6b18ee3d9`; candidate
review is `f16ec7886`. No live audit Graph was observed. Continue from that work,
not a fresh audit. Keep the approximately 10k authored-line budget (7,493 already
charged), frozen evidence and existing resource bounds below. No paid compute,
training experiments, artificial load or unrelated card/format expansion.

## Accepted direction — 2026-10-26

Jack Heart accepted resuming the whole-rules audit and creating follow-up Tasks
with "sounds goo" after reviewing the proposed sequence. This pass produces a
readable audit draft for review. It does not authorize ready publication, merge,
or Task completion; the earlier "ship it" instruction delivered PR354.

Keep this Task as the audit owner. MTG-195 owns controller-selected trigger
ordering; its own conversation records design acceptance and implementation
authorization on October 26. MTG-196 owns affected-player replacement and
prevention ordering and remains queued. Do not implement either mechanism here.

## What has shipped

PR354 merged as `4a2744705e0f493ce05c0643e5eafe59a5749c08`: shared targeting
legality, permanent-wide hexproof enforcement, bounded independent tests,
rules citations, source-pinned inventory, coverage map and measured CI evidence.
Combined targeting/payment behavior was bound to w13 with fresh receipts;
earlier corpora, demos, checkpoints and experiments retain their identities.
The Prism/Treasure regression passes through explicit nested payment choices.
Do not repeat these repairs or treat historical payment failures as current.

The substantive audit remains unfinished: 2,965 of 3,322 indexed entries were
unknown after shipping. Unknown means unreviewed, not necessarily missing code.
The inventory contains 3,166 numbered rules/subrules and 156 headings.

## This pass

Audit individual obligations and subrules in CR 115, 400.7, 603, 608 and 704
against the updated checkout. Prioritize target/zone restrictions, resolution,
trigger collection/placement and SBA stabilization. Preserve the pinned official
September 25, 2026 source metadata in docs/rules/audit-2026-10/source.json; distinguish
any newer source rather than silently changing the pin. Recover the exact TXT
from that source if needed and verify its digest.

For each reviewed entry, inspect exact native authority and callers, relevant
comments/citations, independent behavioral evidence and exceptions. Decide which
implementation matches the rule when paths disagree. Distinguish legitimate
selection/resolution rechecks from competing semantics. Do not bulk-upgrade
subrules from family-level evidence or agreement between two shared callers.

Add only cheap directed tests and useful rule/invariant citations justified by
findings. Keep production behavior changes as explicit follow-up proposals for
review. In particular, do not implement trigger/replacement choice mechanisms,
start a stack-target ABI migration, expand cards/formats, rewrite payments, or
rerun CI-cost research without a new finding. Target/identity uncertainties need
a supported-play witness or precise missing obligation before a new Task.

Deliver an updated per-rule index and concise coverage map, prioritized findings,
bounded draft evidence, exact test receipts and line accounting. Keep working
direction here rather than growing an exhaustive narrative report. Leave other
families honestly unreviewed. This pass does not complete the whole-rules Task.

## Current source and constraints

The checkout was synced to main in `324db65f2` (upstream `558e0186b`), resolving
six conflicts and passing 39 focused CI-selection/world-admission checks.
Main moved arena models into shared evaluation and retired the old learnability
diagnostic. Use current supported test entry points; preserve historical receipts
without reviving removed production systems or relabeling old evidence.

Retain the original roughly 10k authored-line ceiling: the recorded charge after
shipping is 6,891. Reports/index/conformance evidence are excluded; runtime demos
and helper code count. Update conservative accounting for this increment.
Use debug Rust with two Cargo jobs, uv-managed Python, and bounded fixtures.
Limits remain 900s cold Rust, 180s warm Rust and 120s per Python check. No paid
compute, random-game searches, stress loops, assertion weakening or frozen
evidence edits. Do not run the full expensive matrix for this audit-only pass.

## Evidence and handoff

Read docs/rules/audit-2026-10/{README.md,coverage-map.md,unification.md,reviews.json,
index.jsonl,shipping-integration.json,line-accounting.json}; distinguish current
claims from pre-sync w12 history. Preserve all existing work. Keep agreed new
direction and a short per-command result here. Return the draft to the existing
MTG-177 primary conversation, with concrete choices and remaining findings.

## Delete — do not maintain

- Retired `scratch/update_review.py`, `scratch/finalize_audit.py` and
  `scratch/run_audit.py`, preserving their exact bytes in
  `.cache/rules-audit/retired-authoring-20261026.tar`. The one-shot author has
  drifted from the finished reviews and appends duplicate evidence on reruns;
  finalization depends on scratch receipts no longer present. Keep reviews.json
  as the authored source and scripts/audit_rules.py as the index generator.
  Historical helper lines remain charged; their removal does not refund effort.
- Replaced README.md's repeated PR354 walkthrough with the delivery result,
  receipt links and precise historical reference
  `dccf75632e18e88abc2fd92904ea4d81535cadfc:docs/rules/audit-2026-10/README.md`.
  Keep unification.md and all frozen receipts intact.
- Stale w12/current-payment wording and inherited MTG-184 questions are already
  removed. The latter remain at `324db65f2:scratch/questions.md`.
- No production path is replaced by this audit-only increment. Proposed
  behavior repairs remain review decisions, not deletion targets in this pass.

## Implementation — 2026-10-26

Completed individual inspection of all 150 obligations in the five selected
areas: 74 partial, 56 missing, 8 outside admission, 2 implemented and 10 still
unknown. Overall unknown count is now 2,840. The coverage map prioritizes the
findings; reviews/index retain exact authority and limits. No production changes.

Three deliberately ignored probes fail at their CR-required outcomes:
earthbend follows a second zone change; deathtouch survives an earlier SBA
check; per-instruction legality rechecks skip damage after the same resolving
ability grants hexproof. These are controlled obligation witnesses, not claims
of reachable authored-deck responses. Repair decisions remain with review.

Check: debug `cargo test ... cr_608_resolution` → 2 ordinary passes; explicit
ignored probes → 3 expected-outcome failures retained; inventory pytest → 6
passes; pinned-source generation/validation and final Rust format pass. Exact
commands, input hashes and output: `obligation-validation.json`; measured
command duration 33.542s. Broader affected checks belong to gate, without
repeating unchanged passing checks or the full expensive matrix.

Conservative total: 7,493 authored churn lines of roughly 10k, including local
helpers and inherited-question cleanup. Reports/index/evidence remain excluded.
Official input is retained at `.cache/rules-audit/MagicCompRules-20260925.txt`;
source.json is unchanged. Wave GOAL.md was read; its MEMORY.md is absent.

Remaining: review the three proposed repairs and ten unresolved obligations;
MTG-195/196 retain shared-choice ownership. This implementation pass is complete;
publication, merge and whole-Task completion remain outside its boundary.

Compression checks: `git diff --check` passes; `uv run python` verified all
three final artifact hashes in obligation-validation.json and README/coverage
links. Tests and authored reviews are byte-identical to their recorded receipts,
so those results are reused; broader checks remain gate-owned. `lf context`
reports scratch within both limits. No deletion targets remain in this increment.

## Operator recovery — 2026-10-26

The pursue Graph finished implement/compress, then stopped during sync without
an exit receipt. Fresh process inspection found no live Graph or sync worker.
Sync had merged main as `a52bfc5d3` and preserved the complete draft in stash
`2690e564e06d34988f8bb7e62a19c49ed8b7e270`. Applied that exact stash cleanly,
retaining the stash as a backup. No audit work was lost or rerun.

Verified all three final artifact hashes against obligation-validation.json,
verified the full pinned-source index, and passed `git diff --check`.
No managym or inventory-tool changes occurred between the tested base and the
sync merge, so the recorded focused tests remain applicable. The draft is
available for review; no new production repair, publication or merge is selected.

## Candidate review — 2026-10-26

Jack Heart accepted reviewing the three candidates with "sounds good". This
selects a review and priority recommendation, not implementation or shipping.
Fresh Task/Session/process reads confirm no live audit Graph; the interrupted
sync record remains historical. Rules backlog and open PRs were inspected.
No existing Rules Task owns these three repairs. MTG-195 still owns trigger
ordering and touches adjacent trigger/stabilization machinery; coordinate any
future repair against its actual diff. Open PR383 is adaptive game selection,
not an overlapping rules repair.

Recommended order, measured against the original authority-unification goal:

1. **Resolution legality first.** `target_is_legal` is already shared, but
   resolution entry and individual effect handlers call it at different times.
   CR 608.2b requires the initial resolution check; subsequent characteristic
   changes must not retroactively invalidate that decision. Establish one owner
   for clause-specific resolution eligibility across spells, activated abilities,
   triggered abilities and suspended frames. Preserve offer/acceptance checks,
   response-time rechecks, mixed/all-invalid behavior, and object-incarnation
   guards when an instruction itself moves an object. Merely deleting every
   effect-level check is not a sufficient design. Acceptance includes the retained
   hexproof probe, both controllers, multi-clause targets, interrupted resolution,
   and target departure/reentry. The probe is synthetic, not an admitted-deck
   exploit. Review the precise frame contract before implementation.
2. **Deathtouch history second.** CR 704.5h says since the previous SBA check;
   the current flag persists until cleanup. Consume the relevant damage history
   at the shared SBA boundary while preserving simultaneous checks, new damage,
   ordinary lethal damage, indestructibility and journal/undo behavior. Retain
   the positive first-check case and the negative later-check probe. This is a
   lifetime correction in one authority, not two competing engines. Coordinate
   changes to stabilization with MTG-195 rather than starting parallel edits.
3. **Earthbend destination identity third.** The watcher retains ObjectRef,
   but the return effect reduces identity to CardId plus graveyard/exile presence.
   CR 400.7e allows finding the first destination object, not following another
   zone change. Preserve that destination incarnation through trigger placement
   and resolution using existing identity machinery. Keep normal graveyard and
   exile returns working; reject a second move and leave/reenter, including a
   round trip to the same zone. No general stack-target ABI migration is needed
   merely because this narrower gap exists. Coordinate trigger context with
   MTG-195. Current probe directly moves zones, so real-card reachability remains
   an explicit evidence gap.

All three claims were checked against the pinned official text and native
authority. Existing test receipts still apply; this review changed no code and
ran no new behavior tests. Keep this as the reviewable repair plan. Shared-choice
work remains MTG-195/196; no duplicate Task or worker was launched by this review.
