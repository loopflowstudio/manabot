---
extends: concept-review
---
## Card coverage

Review the whole card batch across concepts, composability, simplifications and
workspace behavior, recording every finding (with narrowest reproductions and
file/symbol targets) on `scratch/concept-review.html` so `realign` can plan one
batched fix pass. Stay read-only on code and `scratch/plan.md`.

1. **Outward contracts and rules.** Check new or changed rules types, wire/JSON
   fields, v11 observation columns, learning tokens, Commands, offer shapes and
   Academy prompts against the Comprehensive Rules. Flag derived flags where
   direct CR state belongs, broken replay/forks, missing stale rejection,
   viewer-privacy leaks, silent choice truncation or frozen input-version
   drift. When payment changed, verify competing legal spends (auto spends the
   most restricted applicable pooled mana first, manual keeps meaningful
   choices, receipt separates committed contributions from floating mana).
2. **Inward composability.** Follow at least two distinct representative cards
   through each new or changed mechanism. Flag card-specific engine dispatch,
   duplicated rules state, adapters, multi-axis special variants (fusing
   selectors, conditions, triggers, effects or transforms instead of composing
   them) and surviving obsolete predecessors. Every new variant should let the
   next 20 cards of its kind be card-file edits, not new engine code.
3. **Simplification and deletion candidates.** Inspect the branch diff
   (`main...HEAD` on first review, or since the last reviewed commit) for
   duplicated helpers, dead code, unnecessary indirection, leftover scaffolding
   and stale tests, listing the deepest obsolete owner first.
4. **Workspace behavior and cross-consumer checks.** Focused slice tests often
   miss stale consumers outside the edited set. Verify the whole batch across
   the workspace: compile/run affected debug Rust tests (including
   `conformance_tests`, `rules_tests`, `scenario_tests` and touched card
   suites), check `tests/model/test_pinned_pilot.py`, LLM-text observation
   coverage, Academy fixture/schema parity and `tests/public_api.json` when
   those surfaces changed. Don't run full games or training checks.
