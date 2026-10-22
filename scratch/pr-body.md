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

- Pinned Ubuntu 24.04 / Chromium release suite: **20 passed** while generating v8 references and **20 passed** on comparison; **8 Python matrix/navigation checks passed**. All 18 reference images are committed. Both seat directions reach terminal state through exact 145/566 Command sequences, with all ten prompt families, keyboard activation, reconnect, privacy, and accessibility assertions.
- After reconciling the Elemental world: **106 UI unit tests**, clean Svelte check, production build, **22 focused Python surface/demo/navigation tests**, and **19 Rust protocol/replay/Study tests** passed. The gate also passed rules, arena/monitor, shared-table, and Study checks.
- Prior screenshot corpora, world fixtures, manifests, and frozen receipt bytes are retained. New w10 demo/navigation fixtures bind the explicit decision runtime; v8 terminal reference names reflect their actual win/loss outcome.

A historical public-commitment tape regeneration failure also reproduces with the unchanged base producer. Its strict assertions and frozen bytes remain intact; current-runtime checks do not certify that historical producer binding.

## Try it

Suggested walkthrough, not performed for this description: run `./scripts/play --demo precision`, click **New Game**, and inspect **Deadly Precision**. Inspection should leave the decision unchanged. Explicitly cast it, choose an Inspector target and reserve that Inspector as the sacrifice, then activate Treasure. The context should distinguish the unpaid reservation from the sacrificed Treasure; reloading should restore the pending **Add {B}** choice.
