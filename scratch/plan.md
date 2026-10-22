# MTG-177 — deliver the most valuable rules-audit work

Reconciled 2026-10-22. Status: targeting repair and enumeration cleanup approved
for this bounded delivery; gate passed. The substantive
whole-rules audit remains open. The parent conversation owns publication/landing.

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
On 2026-10-22, Jack Heart reviewed the updated walkthrough and instructed
"ok ship it". This authorizes gate, publication and landing of this bounded
increment, superseding the earlier draft-only delivery boundary. Complete the
cleanup's affected validation before landing. The substantive whole-rules audit
remains unfinished; shipping this increment does not complete that obligation.

## Concept review disposition — 2026-10-22

**No design change or open product choice for this targeting increment.**
Jack Heart's accepted shared-decision direction remains the contract. The
[review guide](../docs/rules/audit-2026-10/README.md#concept-review-one-rule-authority-checks-at-the-right-time)
already puts the interaction, concepts and APIs together; its usage wording and
the existing rules-unification skill need no revision.

Normal path: choose an offered target and submit its bound Command. Enumeration
derives ordered candidates from native state; `target_is_legal` owns the common
restrictions. `TargetSpec`, chosen `Target` and pending selection progress retain
their separate purposes. Clause distinctness, counts and source exclusions stay
with the relevant selection stage, including the documented activated-ability
limits. Resolution must still recheck current state; a cached offer cannot
replace that check. The cleanup removes enumeration machinery without adding
a second persisted legality model or changing the public action contract.

Failure and recovery: a stale Command is rejected with current recovery; a retry
of an already accepted command returns its receipt. Recovery supplies the actual
frame, pending decision and committed history. It must not reselect targets,
refund costs or erase revealed information. An opposing target gaining hexproof
after selection invalidates that target at resolution; own hexproof does not.
Mixed/all-invalid targets and leave/reenter identity remain counterexamples to
collapsing selection and resolution. Any changed legal offer, candidate order,
recorded outcome or replay identity would invalidate the cleanup's claimed
behavior preservation and require adjudication before reuse of w12 evidence.

The trigger/replacement decision ABIs and stack-incarnation migration remain
unselected follow-up designs, not prerequisites silently added to this increment.
The accepted faithful-play contract already rules out hiding required ordering
choices; this review does not approve a particular new interaction or ABI.
The next proof remains the affected consumer/conformance checks against the
rebuilt cleanup, under existing limits. This review inspected code and retained
receipts only; it adds no execution evidence or delivery authorization.

## Existing consolidation

- Already removed: independent stack-target mana-value comparison and
  permanent-wording-local hexproof guards. Candidate order, selection/resolution
  rechecks and frozen w11 evidence remain constraints.
- Already removed: the Study branch fixture's redundant `_experience_frame` calls and
  manual priority-pass setup. Use returned authoritative frames and configured,
  recorded priority automation; preserve complete-match creation, all seven
  branch assertions, Learn coverage, viewer isolation and exact return.
- Already removed: `stack_spell_targets` and the separate enumeration filters.
  Collect candidates in their existing order and apply `target_is_legal` once. Player,
  graveyard and mixed-target witnesses protect ordering and scope; existing
  stack-target tests stay on the surviving enumeration path.
- No replacement target representation or broad rules redesign is selected in
  this increment. Stack-spell incarnation reachability needs adjudication first.

## 1. Targeting repair acceptance and subsequent cleanup

The highest-value code change already exists: `target_is_legal` applies live
battlefield presence and opposing hexproof before target-wording restrictions.
Offers, acceptance and resolution share it. Own hexproof stays targetable;
responses can invalidate targets; mixed legal targets still resolve; departing
and returning permanents do not inherit old targets. Stack-card identity has
the separate evidence limit below. The independent stack-target
mana-value comparison is removed. Keep this repair as the core of delivery.

The integrated diagnostic passes in 43.5s with 144 capacity assertions and
reward-driven PPO/checkpoint reload (65,536 transitions, after 1.0 versus
no-update 0.640625), using unchanged 480s total / 300s PPO limits. Semantic
retry, all seven Study branch cases, all 20 monitor cases and 17
dependency-blocked play/protocol cases pass. At `bda9b0aeb`, debug targeting was
10 passed with the separate known trigger-order proposal ignored. That
acceptance follow-up changed tests and audit artifacts only; its conformance,
arena, recovery and decision-surface results reuse explicitly scoped receipts.

The later working cleanup changes production `legal_targets_for_spec`: it
collects candidates in existing order and filters once through `target_is_legal`.
Its focused debug run passes 11 tests with the same one ignored probe, including
new player/graveyard scope and ordering checks. The native wheel was rebuilt
and installed atomically. This is intended to preserve w12 meaning and shapes;
the earlier integrated diagnostic is evidence for the repair before this cleanup,
not a fresh full consumer run on the current source. The gate below supplies fresh broader checks; their assertions and budgets are
unchanged. Historical receipts remain scoped to their original sources.

The fresh branch timeout was in complete-match setup/persistence; using
returned frames and recorded priority automation closes it in 69.2s without
changing assertions. All failures and frozen w11 evidence remain. The earlier
collector timeout cause remains unproved; no training repair was made.

## 2. Turn the inventory into a useful implementation audit

The main shortfall is evidence depth: 3,166 numbered rules/subrules plus 156
headings are indexed, but 2,965 entries remain unknown. Generic family triage
cannot satisfy the original per-rule request. Deepen the audit in this order:

1. **Targets and resolution (CR 115, 400.7, 608, 702.11).** Finish the reviewed
   obligations around this repair: each supported target form, restrictions,
   response-time changes, mixed/all-invalid targets and object identity. Inspect
   the remaining exceptions. The follow-up individually reviews 115.1c/d,
   115.2/3/6 and refines 400.7/608.2b. Stack CardId loses incarnation, but the
   new ordinary recast/LIFO witness has no stale-target effect. General
   retargeting and activated multi-target/source-exclusion/whole-object
   invalidation remain outside the proven subset.
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
engines. Their native decision design remains unresolved: controller ordering
must preserve CR 603.3b's two-part process and targeted-trigger sequencing;
replacement ordering must expose the affected player's applicable choices.
Both need bound Commands, suspended-decision recovery, viewer safety and exact
replay/fork identity. Neither decision ABI is selected by this bounded repair.
The ownership refresh is complete: MTG-174 identifies training, while `4ba5dd`
owns composable payments. The immediate shared-payment owner remains unknown;
payment implementation remains outside this increment.

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
is 4,274 lines against the approximate 10k ceiling, including 136 lines for the
enumeration cleanup, its witness and ownership documentation, plus 6 gate lines to
remove stale payment-Task attribution and satisfy Clippy without changing coverage. Fresh conformance receipts
are evidence; runtime demo fixtures are charged product inputs. Preserve the
original pins and report observed incremental validation cost and checks not run.

**Ready for delivery review when:** the repair's acceptance is closed, the audit
claims match individually reviewed evidence, the line/cost accounting is current,
and the review surface explains what can land and what remains. Deliver the
bounded result through the selected PR operation. Keep the Task open if its
original substantive audit obligation remains unfinished.

## Evidence and constraints to carry forward

Integrated acceptance base: `dae5b8f9e`, retained in `bda9b0aeb`; current checkout
base for the cleanup: `6756acb0a`. Its upstream merge `0d9814995` changes release
documentation only and adds no runtime behavior. Code milestones: inventory/directed
checks `2b6e2b1b7`, shared stack-target predicate `c514d11b0`, permanent targeting
repair and w12 evidence `202416a8`. The latest upstream collector integration
is included; it is an independent contribution, not this audit's work.

Rules' objective still requires one faithful human/training authority and fresh
evidence for changed semantics. `.lf/wave/rules/MEMORY.md` is absent; ordinary
filesystem inspection found no child directories or child memories. No current
Rules memory could be recovered or curated. Proposed durable lessons remain
here: consumer agreement is not rules conformance; evidence reuse is scoped to
tested source; fixed ordering cannot replace a required player choice. The
available Intelligence memory contains only a heading and adds no guidance.

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

Check: `uv run python scripts/audit_rules.py --source scratch/MagicCompRules-20260925.txt` — passed exact-source/index agreement for 3,322 entries (2,965 unknown); existing debug evidence is retained in `unification-validation.json`.


## Gate handoff — 2026-10-22

**Passed; ready for the parent conversation's `lf land --wait-and-fix`.**
No publication, landing, auto-merge or Task completion was performed. Keep
MTG-177 open: 2,965 entries remain unknown and the substantive audit is unfinished.
Preserve the supplied untracked `.lf` Flows/skills and historical scratch inputs.

The cleanup preserves the existing w12 replay corpus, identity and demo bytes;
pre-w12 and protocol evidence also compare unchanged. Native installation was
atomic. Fresh checks cover 317 debug Rust tests and 116 Python tests, plus one
focused mana-matrix recheck after Clippy's equivalent-comparison fix. Two known
payment/trigger-order probes remain ignored, not counted as passes. Removed the
stale MTG-174 attribution from the payment probe without changing its behavior.

Full learnability passes in 42.990s: 144 capacity assertions, 65,536 PPO
transitions and checkpoint reload; after/no-update scores 1.0/0.640625. Original
480s total / 300s PPO limits, Python 120s bounds, debug assertions and two Cargo
jobs retained. No behavioral check failed. Clippy's original failure is retained
alongside the repair and focused recheck. Full platform/cross-language/fuzz CI
coverage remains CI-owned; no required affected check is deferred locally.

Durable receipts: `docs/rules/audit-2026-10/unification-validation.json`
(`enumeration_cleanup_gate`) and `learnability-w12-gate.json`. The runtime/test
source digest and native hash identify this cleanup, separately from historical
integrated acceptance. Gate cost: 331.942 measured command-seconds, including
the failed lint and targeted rechecks, not elapsed session time or CI cost.
Authored charge: 4,274 lines; report/index/diagnostic evidence excluded.

Commands and results (all limits retained):
- `uv run --no-sync maturin build --release -i .venv/bin/python -m managym/Cargo.toml -o managym/target/wheels` — cp312 release wheel built with two Cargo jobs. (25.910s).
- `uv run --no-sync python scratch/install_native.py` — Atomic native installation and canonical/compatibility import passed. (0.443s).
- `cargo test --locked --manifest-path managym/Cargo.toml --lib target_authority_tests -- --nocapture` — 11 passed; 1 known trigger-order probe ignored. (20.923s).
- `cargo test --locked --manifest-path managym/Cargo.toml --test rules_tests` — 300 passed; 1 known payment probe ignored. The later lint-only mana comparison was rechecked separately. (13.043s).
- `cargo test --locked --manifest-path managym/Cargo.toml --test semantic_conformance_tests -- --nocapture` — 6 passed; checked w12 replay corpus, metamorphic/property checks and phase evidence reproduced. (29.124s).
- `uv run --no-sync pytest tests/infra/test_rules_inventory.py tests/infra/test_ci_selection.py tests/test_package_boundaries.py -q` — 46 passed: inventory, CI selection and package boundaries. (4.594s).
- `uv run --no-sync pytest tests/academy/test_semantic_boundary.py tests/academy/test_optional_payment_recovery.py tests/academy/test_decision_surface.py tests/academy/test_server.py::test_protocol_v1_rejects_stale_and_dedupes_retry tests/academy/test_server.py::test_disconnect_after_acceptance_recovers_without_replaying_command -q --durations=5` — 11 passed: semantic boundary, payment recovery, decision surface, stale rejection and lost-response recovery. (8.146s).
- `uv run --no-sync pytest tests/academy/test_advice.py::test_default_play_import_does_not_require_training_dependencies tests/academy/test_study_protocol.py tests/academy/test_study_runtime.py -q --durations=5` — 21 passed: dependency-blocked play import, Study protocol and all 4 Study runtime cases. (43.644s).
- `uv run --no-sync pytest tests/academy/test_study_branch.py -q --durations=7 -o faulthandler_timeout=45` — 7 passed, all assertions and complete-match fixtures retained. (75.840s).
- `uv run --no-sync pytest tests/arena/test_models.py tests/arena/test_mirror_admission.py -q --durations=5 -o faulthandler_timeout=45` — 11 passed, current-world admission and mirror replay. (14.875s).
- `uv run --no-sync pytest tests/training/test_monitor_evaluation.py -q --durations=5 -o faulthandler_timeout=45` — 20 passed, including all 4 real arena outcome cases. (31.833s).
- `cargo fmt --manifest-path managym/Cargo.toml -- --check` — Passed before the gate test-only edits. (0.404s).
- `cargo fmt --manifest-path managym/Cargo.toml -- --check` — Passed after the gate test-only edits. (0.401s).
- `uv run --no-sync ruff check academy/play_demo.py manabot/arena/models.py scripts/audit_rules.py tests/academy/test_study_branch.py tests/academy/test_study_protocol.py tests/infra/test_ci_selection.py tests/infra/test_rules_inventory.py` — All changed Python files passed Ruff. (0.243s).
- `uv run --no-sync python scripts/audit_rules.py --source scratch/MagicCompRules-20260925.txt` — 3322 pinned entries reconstructed exactly; 2965 unknown. (0.078s).
- `cargo clippy --locked --manifest-path managym/Cargo.toml --all-targets --all-features -- -D warnings` — Failed on clippy::int_plus_one in the new mana matrix: count >= generic + 1. (13.373s).
- `cargo clippy --locked --manifest-path managym/Cargo.toml --all-targets --all-features -- -D warnings` — Passed all targets/all features with warnings denied after equivalent count > generic rewrite. (0.708s).
- `cargo test --locked --manifest-path managym/Cargo.toml --test rules_tests audit_generic_and_specific_mana_boundary_matrix` — 1 passed; unchanged 1152 finite oracle cases after lint fix. (5.323s).
- `git diff --exit-code 0d9814995 -- conformance protocol/fixtures academy/fixtures ':!conformance/semantic-kernel-w12-v1' ':!academy/fixtures/*w12.json'` — Pre-w12 conformance, protocol and Academy fixture bytes unchanged from upstream base. (0.019s).
- `git diff --exit-code 202416a8 -- WORLDS.md managym/src/lib.rs conformance/semantic-kernel-w12-v1 'academy/fixtures/*w12.json'` — w12 identity, conformance and runtime demos unchanged from the accepted repair. (0.021s).
- `bash -n .github/ci/select-tests.sh` — Bash syntax passed. (0.007s).
- `uv run --no-sync python -m manabot.verify.learnability --out .runs/learnability/rules-targeting-w12-gate` — passed full capacity and PPO/checkpoint diagnostic (42.990s).
