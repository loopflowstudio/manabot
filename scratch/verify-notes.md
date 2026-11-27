# Verify — MTG-487 (Pass 1)

- **Outcome:** **Fix forward** (`iterate`)
- **Merged PR:** [#627](https://github.com/loopflowstudio/manabot/pull/627) ("Offer one choice for game-identical copies instead of one per copy")
- **Exact merge SHA on `main`:** `65dcf2340a550b44ce75fadcabb0cf0d784de2f8` (`2026-11-27T07:48:29Z`)
- **Branch head before merge:** `ee8b62f528fcbd6c636c0452d903d15ff69feaa6`
- **Upstream diff between `ee8b62f528` and `65dcf2340a`:** `080ea64372` ("Steady Watch seats, hand ribbons, and open hands in Play") merged to `main` right before PR #627, changing 50 files in Academy/UI/docs (no changes under `managym/src`). Full verification against `main` was therefore required.

## Post-merge CI on `65dcf2340a550b44ce75fadcabb0cf0d784de2f8`

Run [45961017088](https://github.com/loopflowstudio/manabot/actions/runs/45961017088) (`conclusion: failure`):

- **Passing (14 jobs):** `Detect Changes`, `Semantic Kernel Conformance`, `Python Unit Tests`, `Rust Tests`, `Clean-machine Play`, `Academy Integration (contracts)`, `Academy Integration (play)`, `Engine Budget`, `Integration (experiment, model, training, sim, evaluation, evaluation-long)`.
- **Failing (2 jobs + `Tests Result`):**
  1. **`Protocol v1 Conformance`** (job `140378642243`, step `Certify TypeScript`):
     - `vitest run --run src/lib/protocol-conformance.test.ts` passed (5/5).
     - `npm run check` (`svelte-check --tsconfig ./tsconfig.json`) failed with 3 errors in `academy/ui/src/lib/board/view.test.ts`:
       - `view.test.ts:383:17`: `Error: Property 'hand' does not exist on type 'BoardView'.`
       - `view.test.ts:383:26`: `Error: Parameter 'item' implicitly has an 'any' type.`
       - `view.test.ts:383:65`: `Error: Parameter 'placed' implicitly has an 'any' type.`
  2. **`Academy UI`** (job `140378642259`, step `Focused UI checks and browser interactions`):
     - `vitest` failed: `FAIL src/lib/board/view.test.ts > where decisions land > places folded offer aliases on drawn copies on the board and in hand, keeping piles whole and trays deduplicated`
     - `TypeError: Cannot read properties of undefined (reading 'map')` at `src/lib/board/view.test.ts:383:22` (`1 failed | 49 passed (50)` files; `1 failed | 767 passed (768)` tests).
     - Playwright browser interaction checks did not run because `vitest` exited 1.

## Local checks run against `65dcf2340a`

1. `git diff --stat ee8b62f528 65dcf2340a -- managym/src` -> empty (no Rust engine drift between PR head and merge commit).
2. `uv run --extra dev --extra play python -m pytest tests/academy/test_interchangeable_offers.py tests/academy/test_decision_surface.py tests/academy/test_open_hands.py tests/academy/test_watch.py -q` -> `28 passed, 74 warnings in 14.84s`.
3. `grep -rn "offerPlacements\|view\.hand" academy/ui/src/` -> `academy/ui/src/lib/board/view.test.ts:383` is the sole remaining `view.hand` reference.
4. Inspected all 10 files touched by both `080ea64372` and `65dcf2340a`: `academy/ui/src/lib/board/view.ts` still attaches folded aliases to both `view.you.hand` and `view.opponent.hand` (`drawnObjects` includes both hands, `place` records object aliases in `drawn`, and `drawn.hand = ... offersAt(...)` populates both sides).
5. Note: `academy/ui/node_modules` is not installed in this worktree, so local `svelte-check` / `vitest` / Playwright were verified via the post-merge CI run logs on `65dcf2340a`.

## Gap and fix-forward plan

- **Root cause:** `080ea64372` moved `BoardView.hand` onto `SideView.hand` (`view.you.hand` and `view.opponent.hand`). PR #627 merged cleanly textually, leaving `academy/ui/src/lib/board/view.test.ts:383` referencing `view.hand` instead of `view.you.hand`.
- **Next pass (fix forward from `main`):**
  1. Sync `jack/mtg-487-canonicalize-indistinguishable-game-objects` onto `origin/main` (`65dcf2340a`).
  2. Update `academy/ui/src/lib/board/view.test.ts:383` from `view.hand` to `view.you.hand` (and confirm no other `BoardView` or hand-ribbon interaction breaks under `svelte-check`, `vitest`, and Academy UI Playwright specs such as `mana-payment.spec.ts` and `decision-surface.spec.ts`).
  3. Ship as a fix-forward PR and verify post-merge CI is green on `main` (including `Protocol v1 Conformance` and `Academy UI`).
