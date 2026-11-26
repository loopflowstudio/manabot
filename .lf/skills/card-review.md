---
extends: concept-review
---
## Card coverage

Record every finding with narrowest reproductions and file/symbol targets on
`scratch/concept-review.html` so `realign` can plan one batched fix pass.

1. **Outward contracts and rules.** Check new or changed rules types, wire/JSON
   fields, v11 observation columns, learning tokens, Commands, offer shapes and
   Academy prompts against the Comprehensive Rules. Flag broken replay/forks,
   missing stale rejection, viewer-privacy leaks, silent choice truncation or
   frozen input-version drift. When payment changed, verify competing legal
   spends (auto spends the most restricted applicable pooled mana first, manual
   keeps meaningful choices, receipt separates committed contributions from
   floating mana).
2. **Inward composability.** Trace at least two cards through each new or
   changed mechanism. Flag card-specific engine dispatch, duplicated rules
   state, adapters, fused multi-axis variants and surviving obsolete
   predecessors. Every new variant should let the next 20 cards of its kind be
   card-file edits, not new engine code.
3. **Prior art in other MTG engines.** When adding or reshaping a shared
   mechanism, check how Phase, XMage or Forge solve the same rules problem and
   surface adoptable techniques that fit `managym`'s single-authority,
   RL/search, sequential-offer and append-only v11 constraints (skip parser, IR
   or storage rewrites).
4. **Simplification and deletion.** Check the branch diff for duplicated
   helpers, dead code, unnecessary indirection, leftover scaffolding and stale
   tests, listing the deepest obsolete owner first.
5. **Workspace cross-consumer checks.** Compile and run affected debug Rust
   tests (`conformance_tests`, `rules_tests`, `scenario_tests` and touched card
   suites), plus `tests/model/test_pinned_pilot.py`, LLM-text observation
   coverage, Academy fixture/schema parity and `tests/public_api.json` when
   those surfaces changed. Skip full games and training checks.
