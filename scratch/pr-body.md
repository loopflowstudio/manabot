When several triggered abilities are waiting for one controller, fixed enqueue order removes a legal strategic choice. Controllers now choose the next ability to put on the stack through the same native authority used by Academy Commands, ordinary policies, replay and search.

<!-- loopflow:task-pr-context:start -->
> [!NOTE]
> **Task:** [Let controllers choose the order of simultaneous triggered abilities · MTG-195](https://linear.app/loopflow/issue/MTG-195/let-controllers-choose-the-order-of-simultaneous-triggered-abilities)
> **PR lifecycle:** Merge delivers the code. Record independent later work as Tasks and comments; complete the delivered outcome.
<!-- loopflow:task-pr-context:end -->

- One resumable placement round owns both APNAP passes, occurrence identity and target progress. Each choice commits immediately; later placements resolve first. Stale Commands reject before mutation, duplicate receipts do not repeat placement, and reconnect recovers the actual pending decision.
- Native labels describe the exact triggered ability or delayed payload and say “Put on stack next.” Public stack inspection, target-stage recovery and text capture reuse those descriptions. Academy retains all legal offers and full source inspection.
- World w14, semantic schema 7, capture version 3 and state hash 9 bind the new semantics. Policy input v3 joins each action to its trigger's complete contextual frame. Historical checkpoint configuration serialization is preserved, while old-world artifacts are rejected for current play. Oversized tensor inputs reject explicitly; native choices are not capped.
- Fresh w14 play fixtures and the 14-case, 3,420-Command conformance corpus preserve earlier frozen worlds and evidence. The live rules inventory now points to the controller-choice owner and retains partial CR 603.3b coverage; the dated audit remains intact. Synced main's deck-browser and forced-step automation changes; both browser suites remain selected.

The existing-card witness starts from an explicit retained scenario recipe, then casts Deadly Dispute and sacrifices Callous Inspector through ordinary Commands. With Bastion of Remembrance and the controller at 1 life, Inspector-first survives at 1; Bastion-first loses at 0 after normal priority passes. This is a constructed root, not a naturally reached seed-only prefix.

Validation:
- Final debug native ordering/target/rollback/capacity/description regressions: 7 passed. Clippy (all targets/features), Rust formatting and changed-file Ruff checks pass.
- Consumer/recovery/navigation/automation Python suite: 84 passed; final rebuilt-extension projection/fixture follow-up: 57 passed.
- Maintained model, both-preference fitting, PPO/checkpoint reload, arena and monitor checks: 234 passed, 1 inapplicable attention parameter skipped. Exact integration CI selection collects 669 tests.
- Frontend check/build pass. The first full release browser run passed 24 cases with 2 optional art/checkpoint cases skipped. The final run passed 23 cases (including both trigger-order proofs) and timed out on the checked Learn demo's unchanged five-second readiness assertion; its focused rerun also timed out while the page was still preparing the checked prefix. The checked prefix passes headlessly and previously opened in 1.2 seconds. Host load exceeded 240; the timing failure is retained, not counted as a pass.
- The complete debug Rust suite, semantic/protocol conformance, both platform package/install checks, and clean-machine play pass in CI. Local full-debug follow-ups were stopped after the final presentation refinement and during heavily contended compilation; they are not claimed as final local full-suite passes. The unchanged Learn readiness assertion remains enforced by Academy UI CI before merge.
- CI found a stale rules-inventory anchor and an observation-capacity test double missing the new continuation field. The live evidence references and fixture are updated; the exact Python unit job passes 73 tests locally, and the observation suite passes 54. The inventory regenerates from the pinned rules bytes with all 3,322 identifiers and coverage-status counts unchanged.

Limits: the two-pass scheduler is covered with typed fixtures; current content still cannot trigger on another ability triggering. General condition expansion and replacement ordering are outside MTG-195. Maintained two-preference fitting and ordinary PPO/checkpoint reload are integration/learnability evidence, not playing-strength or performance claims. No training experiment or paid compute was launched.
