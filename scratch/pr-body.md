Academy made it difficult to connect a legal action to the exact card, target or payment step it affected. This PR makes cards inspectable without committing an action and explains pending decisions alongside explicit legal choices.

<!-- loopflow:task-pr-context:start -->
> [!NOTE]
> **Task:** Redesign Academy around the agent’s observations and actions · 295863
> **PR lifecycle:** Merge delivers the code. Record independent later work as Tasks and comments; complete the delivered outcome.
<!-- loopflow:task-pr-context:end -->

## What changes

- Share one decision renderer across Play, Watch, Replay and Study, showing who decides, what is happening and the available choices. Native context identifies selected targets, unpaid sacrifice reservations, committed costs and mana pools while preserving viewer privacy.
- Require explicit choices for sole legal mana colors and payment continuations in Academy. Priority-stop automation remains separately recorded; training defaults and frozen evidence contracts stay unchanged.
- Place full-width choices below the board, distinguish same-name objects, and keep the complete offer count visible while filtering. Battlefield lanes compact and page dense groups; focusing an offer reveals its subjects without submitting it.
- Preserve actual pending decisions on reconnect and let Study return to the original recorded decision after trying an alternative. Historical frames without context display “Details not recorded.”
- Add checked Learn and Deadly Precision demos, including Treasure-funded payment and the four-mana alternative. These use ordinary Commands and remain labeled `automated_validation`, including later human continuation; they are not human usability or curated-deck admission evidence.
- Cache Scryfall artwork for the curated roster and demos, retain procedural fallback, and provide artist/source credits.
- Move the Svelte frontend to `academy/ui` and update launch scripts, CI, packaging and deployment paths.

## Checks

Local: 106 UI unit tests, zero Svelte errors/warnings, production build, 22 focused Python surface/navigation/demo tests, and 19 Rust protocol/replay/Study tests pass after syncing the Elemental world. The earlier gate also passed the rules, arena/monitor, shared-table and Study checks. All 19 affected browser scenarios have passing focused evidence, including terminal focus, reduced motion, both Precision costs, navigation and all ten deck selectors.

Linux CI owns the new v8 screenshot corpus and comparison. The v7 images and matrix, prior world fixtures and frozen receipts are retained. New w10 demo/navigation fixtures bind the current explicit decision runtime. The live release matrix verifies 145 and 566 exact Commands in both seat directions; its browser completion proof and Linux references are in progress.

A historical public-commitment tape regeneration failure also reproduces with the unchanged base producer. Its strict assertions and frozen bytes remain intact; current-runtime checks do not certify that historical producer binding.

## Try it

Suggested walkthrough, not performed for this description: run `./scripts/play --demo precision`, click **New Game**, and inspect **Deadly Precision**. Inspection should leave the decision unchanged. Explicitly cast it, choose an Inspector target and reserve that Inspector as the sacrifice, then activate Treasure. The context should distinguish the unpaid reservation from the sacrificed Treasure; reloading should restore the pending **Add {B}** choice.
