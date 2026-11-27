# MTG-487 Fix-Forward Plan

## Finished

- Original Task implementation merged to `main` in PR #627 (`65dcf2340a`). Post-merge CI run `45961017088` passed 14/16 jobs and failed `Protocol v1 Conformance` (`svelte-check`) and `Academy UI` (`vitest`) solely because `080ea64372` moved `BoardView.hand` to `SideView.hand` right before PR #627 merged (`scratch/verify-notes.md`).
- Synced branch onto `main` (`1bb77f9ff3`) and updated `academy/ui/src/lib/board/view.test.ts:383` from `view.hand` to `view.you.hand` (`ebb3032688`).

## Verification

- Local TypeScript/Svelte check: `cd academy/ui && npm run check` -> 0 errors, 0 warnings.
- Local UI unit suite: `cd academy/ui && npx vitest run` -> 50/50 files, 768/768 tests passed (including all 306 tests in `src/lib/board/view.test.ts`).
- Local Python Academy tests (against `65dcf2340a`, recorded in `scratch/verify-notes.md`): 28 passed (`test_interchangeable_offers.py`, `test_decision_surface.py`, `test_open_hands.py`, `test_watch.py`).
- Shape/overlap check against `080ea64372`: `academy/ui/src/lib/board/view.ts` still attaches folded offer aliases to both `view.you.hand` and `view.opponent.hand`; `Board.svelte` retains `.hand` on your hand ribbon and `Rail.svelte` keeps `choicesOpen = true` by default for `[data-testid="action-option"]`.

## Remaining

None to build. Ready to ship as a fix-forward PR.

## Deferred checks (CI / post-merge verify)

- Academy UI Playwright browser specs (`mana-payment.spec.ts`, `decision-surface.spec.ts`, full `Academy UI` CI job — skipped in post-merge run `45961017088` only because `vitest` failed first).
- Green `Protocol v1 Conformance` and `Academy UI` CI jobs on PR and post-merge `main`.
