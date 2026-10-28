A target that gains hexproof during resolution no longer escapes later instructions of that same spell or ability. Activated abilities with no legal chosen targets now stop before any effect, and policy/text consumers can inspect the retained resolution-start result.

## What changes

- Share one clause-specific resolution-entry authority across spells, activated abilities and triggers. Preserve eligibility and slot order across suspension; check exact object presence separately. Payment ownership stays unchanged.
- Integrate [controller-selected trigger placement · PR386](https://github.com/loopflowstudio/manabot/pull/386) in **w15/hash 10**, with policy/capture/text **v4**. V2/v3 keep 32 relation columns; v4 adds eligibility at column 32 while validity remains at 31. Historical worlds, checkpoints and evidence are preserved.
- Retain the individual CR 115, 400.7, 603, 608 and 704 adjudications, fresh w15 conformance/demo evidence, and honest [coverage accounting](https://github.com/loopflowstudio/manabot/blob/jack/mtg-177-audit-all-mtg-rules-and-draft-low-cost-coverage-and/docs/rules/audit-2026-10/coverage-map.md). The 3,322-entry inventory still has **2,840 unknown entries**.

## Checks and limits

The [shipping gate](https://github.com/loopflowstudio/manabot/blob/jack/mtg-177-audit-all-mtg-rules-and-draft-low-cost-coverage-and/docs/rules/audit-2026-10/resolution-shipping.md) and [exact receipt](https://github.com/loopflowstudio/manabot/blob/jack/mtg-177-audit-all-mtg-rules-and-draft-low-cost-coverage-and/docs/rules/audit-2026-10/resolution-shipping.json) record 558 passing debug Rust tests, focused model/world/admission/monitor/active-training and recovery checks, 85 Command/replay/choice/Learn checks, and ten focused browser cases. Rust fmt/clippy, changed-file Ruff checks, frontend check/build and inventory validation pass. The receipt preserves the repaired fixture/lint failures, incorrect first browser harness, interrupted unoptimized native run, and investigated aggregate Python timeout.

Required Linux CI remains authoritative at the published head and is awaited before merge. The broader macOS nightly matrix and one CUDA-only shared-inference case are not certified by this local gate. Deathtouch-history and earthbend-destination proposals remain ignored, unresolved gaps; no strength baseline or release certification is claimed.

[Conservative authored churn](https://github.com/loopflowstudio/manabot/blob/jack/mtg-177-audit-all-mtg-rules-and-draft-low-cost-coverage-and/docs/rules/audit-2026-10/line-accounting.json) is 10,430 lines, 430 above the approximate 10k guideline, separately accounting for generated evidence and inherited changes. This is a new delivery after historical [targeting repair · PR354](https://github.com/loopflowstudio/manabot/pull/354). Merging this resolution repair **does not complete the whole-rules audit or MTG-177**.
