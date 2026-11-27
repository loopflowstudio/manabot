# Verify — MTG-487 (Pass 2)

- **Outcome:** **Verified** (`advance`)
- **Merged PRs:**
  - [#627](https://github.com/loopflowstudio/manabot/pull/627) ("Offer one choice for game-identical copies instead of one per copy") — merged at `65dcf2340a550b44ce75fadcabb0cf0d784de2f8` (`2026-11-27T07:48:29Z`)
  - [#634](https://github.com/loopflowstudio/manabot/pull/634) ("Restore Academy UI checks after hand ribbons moved per side") — merged at `b611abb3d7ed28cc1f3b41b98d8a8e379fbed1d6` (`2026-11-27T08:46:28Z`)
- **Branch QA head:** `d6026151f7299ebe803dd7e5937f816afc7351a8` (`QA MTG-487 fix-forward at 67abc1a6e7`)

## Pass 1 → Pass 2 comparison

- **Pass 1 (`88c1ad6c1a`):** Checked PR #627 on `main` (`65dcf2340a`). Post-merge CI run [45961017088](https://github.com/loopflowstudio/manabot/actions/runs/45961017088) passed 14/16 jobs (`Rust Tests`, `Semantic Kernel Conformance`, `Python Unit Tests`, `Clean-machine Play`, `Academy Integration (contracts, play)`, `Engine Budget`, and all 6 `Integration` shards), but failed `Protocol v1 Conformance` (`Certify TypeScript` -> `npm run check`) and `Academy UI` (`vitest`) on `academy/ui/src/lib/board/view.test.ts:383` because PR #631 (`080ea64372`) moved `BoardView.hand` onto `SideView.hand` (`view.you.hand` / `view.opponent.hand`) just before #627 merged.
- **Pass 2 (this pass):** PR #634 updated `academy/ui/src/lib/board/view.test.ts:383` to `view.you.hand` and refreshed `.lf/qa/academy.md`. All three skip-rebuild conditions hold on `main` at `b611abb3d7ed28cc1f3b41b98d8a8e379fbed1d6`, and post-merge CI is green.

## Skip-rebuild check on `b611abb3d7ed28cc1f3b41b98d8a8e379fbed1d6`

1. **Tree equivalence with QA:**
   - Command: `git diff d6026151f7 b611abb3d7ed28cc1f3b41b98d8a8e379fbed1d6 -- . ':!scratch'`
   - Output: empty (exit 0). Every shipped file on `main` matches the exact tree exercised by QA (`d6026151f7`).
   - Command: `git diff HEAD origin/main -- . ':!scratch'` -> empty (exit 0).
2. **Post-merge CI for `b611abb3d7ed28cc1f3b41b98d8a8e379fbed1d6`:**
   - Push CI run [45968282282](https://github.com/loopflowstudio/manabot/actions/runs/45968282282) (`event: push`, `status: completed`, `conclusion: success`):
     - `Detect Changes`: success (7s)
     - `Academy UI` (job `140402228935`): success (8m16s):
       - Vitest (`npm test`, covering `view.test.ts` and `protocol-conformance.test.ts`): `Test Files 50 passed (50)`, `Tests 768 passed (768)`
       - TypeScript (`npm run check`): `svelte-check found 0 errors and 0 warnings`
       - Playwright `test:e2e:release`: `85 passed (2.9m)`
       - Playwright `test:e2e:launch`: `5 passed (19.2s)`
       - Playwright `test:e2e:policy`: `15 passed (40.8s)`
       - Playwright `test:e2e:search`: `5 passed (15.7s)`
     - `Tests Result`: success (3s)
     - Unchanged suites (`Rust Tests`, `Protocol v1 Conformance`, `Engine Budget`, `Integration`, `Clean-machine Play`, `Academy Integration`, `Semantic Kernel Conformance`, `Python Unit Tests`) were skipped by `Detect Changes` on `b611abb3d7` after already passing on `65dcf2340a` (run [45961017088](https://github.com/loopflowstudio/manabot/actions/runs/45961017088)), where `Protocol v1 Conformance` passed `Certify Rust and Python` and `Launcher contract tests` and failed only on `Certify TypeScript` (`npm run check`), now confirmed green in both `Academy UI` and locally.
   - Note on `Nightly baseline learning` run [45966127762](https://github.com/loopflowstudio/manabot/actions/runs/45966127762) on `65dcf2340a`: failed on `tests/verify/test_learning_regression.py::test_recipe_report_preserves_paired_deals_and_policy_inspection`, which is a pre-existing failure across all recent nightly runs since `2026-11-23` (`45139326522`, `45356665105`, `45576871262`, `45783721787`) and unrelated to MTG-487.
3. **No post-merge-only claims:** No Task outcome depends on a release install, deployment, or production state.

## Additional local check and claim review

- Local TypeScript conformance check on exact `b611abb3d7` tree:
  - Command: `cd academy/ui && npm test -- --run src/lib/protocol-conformance.test.ts src/lib/board/view.test.ts && npm run check`
  - Output: `Test Files 2 passed (2)` / `Tests 311 passed (311)` (`view.test.ts` 306/306, `protocol-conformance.test.ts` 5/5); `svelte-check found 0 errors and 0 warnings`.
- Verified Task outcomes against `main` (`65dcf2340a` + `b611abb3d7`) and cited QA (`d6026151f7`, `2949a63314`), ship-review (`1ea5705aa4`, `37358ee03f`), and CI records:
  - Manual payment stops fold indistinguishable sources (Kyoshi Village + 3 Plains -> 3 offers; 8 Forests + 1 Mountain -> 2 offers): verified in `Rust Tests` / `Semantic Kernel Conformance` on `65dcf2340a` and Play QA Scenario 1 (`d6026151f7`).
  - Fail-closed object-equivalence predicate across battlefield, hand, graveyard, combat, and targeting preserving all rule- and information-relevant distinctions: verified in `Rust Tests`, `Semantic Kernel Conformance`, `Academy Integration`, and QA Scenarios 1–2.
  - Academy shared decision surface (`aliases[]` on `Offer` and `InteractionOffer`, clickable/draggable board & hand copies sending canonical Command, Command binding, stale rejection, replay/fork identity, Play/Watch hand ribbons): verified in `Academy Integration`, `Academy UI` (768 Vitest + 110 Playwright specs on `b611abb3d7`), and QA Scenarios 1–3 (`d6026151f7`).
  - Docs & compatibility/budget: `managym/README.md` (Indistinguishable objects + updated manual mana sources replacing "eight Forests are eight actions") and `docs/academy.md` updated in `65dcf2340a`; `Integration` (including `test_pinned_pilot.py`) and `Engine Budget` passed on `65dcf2340a`.
- Out-of-scope note already dispositioned as Drop in `1ea5705aa4:scratch/ship-review.html`: `pileKey` keying on `offer.label` can visually stack Haste creatures from different turns that carry distinct `offer.id`s while keeping each copy individually clickable for its actual offer; no follow-up Task created.

## Decision

Every Task outcome holds on `main` (`b611abb3d7ed28cc1f3b41b98d8a8e379fbed1d6`), post-merge CI is green, and no repairable gap remains. Advance to complete MTG-487 (matching Jack Heart's approval in `c7dbc28e`: "proceed through deliver, auto-merge PR #627, verify, and mark MTG-487 complete once shipped").
