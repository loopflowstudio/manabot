---
requires: a built card batch, its target interactions and focused check results
produces: unified concept, composability, simplification and behavior findings in `scratch/plan.md`
action_style: procedural
---
Review the whole card batch once across concepts, composability, simplification and rules behavior, without editing code.

1. **Outward concepts and contracts.** Judge the branch's new or changed rules
   types, wire/JSON fields, v11 observation columns, learning tokens, Commands,
   offer shapes and Academy prompts against the Comprehensive Rules and real
   consumers. Flag opaque wrapper names, derived consequence/interpretation
   flags where direct CR state belongs, arbitrary caps, broken replay/forks,
   missing stale rejection, viewer-privacy leaks, silent choice truncation or
   frozen input-version drift. When payment changed, verify competing legal
   spends (auto spends the most restricted applicable pooled mana first, manual
   keeps meaningful choices, receipt separates committed contributions from
   floating mana).
2. **Inward composability.** Follow at least two distinct representative cards
   through each new or changed mechanism. Flag card-specific engine dispatch,
   duplicated rules state, adapters, multi-axis special variants (fusing
   selectors, conditions, triggers, effects or transforms instead of composing
   them) and surviving obsolete predecessors. Every new variant should let the
   next 20 cards of its kind be card-file edits, not new engine code.
3. **Simplification and deletion (best of `compress`).** Inspect the branch
   diff (`main...HEAD` on first review, or since the last reviewed commit) for
   duplicated helpers, dead code, unnecessary indirection, leftover scaffolding
   and stale tests. Write a concrete **Simplify / delete** list—deepest
   obsolete owner first—so the fix pass deletes what doesn't earn its place
   alongside semantic fixes.
4. **Workspace behavior and cross-consumer checks.** Focused slice tests often
   miss stale consumers outside the edited set. Verify that the whole batch
   holds across the workspace: compile/run affected debug Rust tests (including
   `conformance_tests`, `rules_tests`, `scenario_tests` and touched card
   suites), check `tests/model/test_pinned_pilot.py`, LLM-text observation
   coverage, Academy fixture/schema parity and `tests/public_api.json` when
   those surfaces changed. Don't run full games or training checks.
5. **Write one batched fix pass in `scratch/plan.md`.** Replace earlier review
   findings rather than appending a second report. Group every defect,
   composability refactor and deletion into **one concrete fix pass** (with
   narrowest reproductions and exact file/symbol targets) so `implement`
   resolves them together in one ~30–50 minute session. Don't edit code here.

When deciding whether to continue:

- **Iterate** while any concept mismatch, multi-axis/card-specific engine hack,
  meaningful duplication/dead-code cleanup, failing consumer check or unbuilt
  slice remains.
- **Advance** when the batch plays by the rules, composes cleanly, leaves no
  worthwhile simplification behind and its workspace checks hold.
