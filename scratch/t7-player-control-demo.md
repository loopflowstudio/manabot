# T7 player-control demo — 2026-11-04

Decision: **advance**. Reviewed implementation `63ac7b693bb9f5b396586f75b385f2e974508772`
on [PR #422](https://github.com/loopflowstudio/manabot/pull/422).
This is autonomous, delegated demo judgment, not new human feedback or merge
approval. Jack Heart's accepted option 1a remains the design: keep the viewer's
board stable, use native delegation instructions, and label inspected cards by
their actual owner and zone.

Design: [T7 plan](mtg-226-tla-t7-player.md). Dated human review:
[concept-review.html](concept-review.html), preserved unchanged. Scope and
remaining interaction limits: [TLA acceptance](../docs/rules/tla/acceptance.md).

## Intended progress and observed result

The preceding pass intended to implement option 1a, integrate merged T6, repair
pilot test isolation, and publish for review. Publication is now complete;
#422 replaces closed #418. Current-head CI reports **18 successful checks**,
including both integration suites, Academy UI, Clean-machine Play, Rust Tests
and Engine Budget. The release publish job is skipped. Evidence:
[CI run 41888896399](https://github.com/loopflowstudio/manabot/actions/runs/41888896399),
[release validation](https://github.com/loopflowstudio/manabot/actions/runs/41888895902),
and the local [status snapshot](../.cache/mtg226-demo/ci-status.json).

New execution evidence from this demo:

- **Eight runtime checks passed** through the native extension, policy/text
  projection, training Env, and real Academy TestClient WebSocket route.
  Both seats and durations preserve delegated Islandcycling ownership and
  intervening own-player priority. Exact search forks preserve the decision;
  selecting an Island removes library inspection. The submitted-deck cleanup
  session resumes the same frame, offers and inspected hand, commits once,
  acknowledges a duplicate and rejects a stale Command. Configured automation
  remains identified in the trace.
- **Two live Chromium cases passed**, at 390px and 1500px. The normal Academy
  backend and Vite UI ran on isolated ports 8126/5296. The new-game request used
  two valid 60-card submitted lists, seed 3 and a passive opponent. Ordinary
  Commands developed fourteen Islands and paid for whole-turn control.
  The page named “Seat 1 chooses for Seat 2” and labelled the eight inspected
  cards as the opponent's hand. Reload retained the exact frame hash, offers
  and private inspection. Keyboard selection committed the pending discard;
  control and private inspection disappeared. Both widths had no horizontal
  overflow, no page exceptions and no axe violations at the resumed decision.
- **Four native-frame Chromium cases passed**, covering both seats and both
  durations at each width. Captures came from the existing constructed native
  position and Academy's actual frame builder. The UI showed “Search cards,”
  native delegation instructions and “opponent's library”; it kept the viewer's
  mana below the opponent's through delegated, own-response, search and
  after-search frames. Own responses did not claim delegation. Library
  inspection disappeared after selection. No horizontal overflow or accidental
  Command was observed. This transport was a presentation fixture, not live
  gameplay or search reconnect.

The browser checks add evidence beyond the preceding source/SSR review. They
do not retroactively make that earlier review a browser proof.

## Evidence and reproduction

- Runtime log: `/tmp/mtg226-demo-runtime.log` — 8 passed.
- Live browser log: `/tmp/mtg226-demo-browser.log` — 2 live cases passed;
  its 4 presentation failures were harness label assumptions, corrected below.
- Corrected presentation log: `/tmp/mtg226-demo-search-browser.log` — 4 passed.
- [Browser walkthrough](t7-demo.spec.ts), [isolated configuration](t7-demo.config.ts),
  and [native capture script](t7-demo-capture.py).
- [Native frames](../.cache/mtg226-demo/search-frames.json).
- Live screenshots: `.cache/mtg226-demo/browser/t7-demo-live-delegated-cleanup-and-recovery-at-{390,1500}px/`,
  `controlled-before-reload.png` and `expired-after-keyboard-choice.png`.
- Search screenshots: `.cache/mtg226-demo/search-browser/t7-demo-native-search-presentation-seat-{1,2}-{combat,turn}/`,
  `search-390px.png` and `search-1500px.png`.

From the repository root, run:

```sh
uv run --extra dev --extra play pytest tests/semantic/test_player_control.py tests/academy/test_player_control_session.py -q
uv run --no-sync python scratch/t7-demo-capture.py
cd academy/ui
npx playwright test --config ../../scratch/t7-demo.config.ts
```

The harness initially used `import.meta` outside an ESM package and then looked
for a mana accessible label that is ordinary text. Its next assumption used
“Opponent” where live play names the selected bot (“Search 64”). Finally it
tried to replay older revisions in the same match when changing viewport;
the client correctly retained the newer revision. The corrected walkthrough
resizes each current frame instead. These were demo-script defects, not product
failures. Their logs remain at `/tmp/mtg226-demo-browser-harness-failure.log`,
`/tmp/mtg226-demo-browser-locator-failure.log`, the live browser log above, and
`/tmp/mtg226-demo-search-stale-fixture.log`. Product code was not changed.
All demo-owned servers and browser workers exited; neither port remains open.

## Judgment, limits and next action

No person observed this headless run, and no new product choice was accepted.
I exercised the supplied delegation by advancing because the accepted
presentation correction works, the resumed combined tree passes CI, and the
demonstration found no concrete product defect requiring another implementation
pass. No design change is proposed.

Live search WebSocket recovery and process-crash durability remain unproven;
exact search forks and live cleanup reload are separate witnesses. The
pairwise card combinations listed in the acceptance document, retained-checkpoint
compatibility and training strength remain outside these claims. Missing
coverage is not evidence of a failure and does not exclude defined cards.

Next action: advance to the caller's next review step with PR #422 and this
record. Jack Heart's review is still required before merge. This decision does
not authorize merge, auto-merge, Task completion or experiment execution.
