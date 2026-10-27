# MTG-177 — continue the substantive rules audit

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
