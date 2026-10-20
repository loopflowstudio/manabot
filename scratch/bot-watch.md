# Bot watch and decision inspection

2026-10-20: Task MTG-156; Jack Heart requested implementation and a stop at demo.

Implemented on GameSession's authoritative Commands, canonical replay rows and
retained Study roots. Watch advances one native decision per request, protected
by the table lock and an expected revision. Both seats select the configured
checkpoint or random/greedy/heuristic/search anchors, with decks and deal seed.
The existing replay route hosts the shared board, Advice and Study components.
Policy outputs persist at choice time. Playback never reruns inference.

Jack Heart's 2026-10-20 demo feedback: spectators should see both hands, and
this is a fine starting point; no further review is needed. Both hands now show
by default during bot watches, with a toggle for the acting bot's view. The
separate analysis snapshot supplies both hands because ordinary native
observations already redact the opponent hand. Active inspection still requires
the watch credential; ordinary trace APIs strip the analysis snapshot. Bot
inputs, decision evidence and Study retain their acting-view ExperienceFrame.

Arena/Tournament import verifies the original game trace and replay receipt,
replays Commands without policies, retains Study roots, and stores the source
digest under a new Academy identity. Multiple-game JSONL/gzip files have a game
selector. Missing old model outputs remain explicitly unavailable. Local-update
receipts expose only safe policy/search vectors and a belief summary. Compound
checkpoints expose their recorded sequence factors and root value; native
microchoice marginals are unavailable rather than reconstructed.

Delete — do not maintain: no watch predecessor. Watch's exact native microchoices
bypass the older structured grouping matcher for Study execution; the retained
structured Study path still serves its existing grouped-offer artifacts. Frozen
`erd1` digests and receipts are unchanged. `ed2` remains a pre-command address.

Proof: focused watch tests cover actor privacy, revision rejection, both bot
identities, both-seat isolated Study execution, actual search scores, scalar and
compound checkpoint outputs, verified arena import, tamper rejection and reveal.
The browser check steps, pauses, retries, returns, and reloads a saved watch.
The natural greedy-versus-heuristic game (seed 42) ended after 169 decisions.

Validation: `uv run pytest` focused watch/dependency checks: 7 passed; Study
protocol/runtime/branch/address and package-boundary checks: 42 passed.
`npm run check` has no diagnostics; production build and Playwright watch flow
pass. Local demo: http://localhost:5189/replay?watch=1 (backend 8030).
Screenshot: scratch/bot-watch-demo.png. Automated observation only; Jack Heart's
feedback accepts this as a starting point; no further review requested.

Both-hands adjustment verified by the focused watch test; frontend type check
has no diagnostics. Local demo backend refreshed; start a new watch for the
updated behavior. Separate follow-up created: MTG-160, "Redesign Academy's play
and viewing experience" (task_854ddae193334824aa605d956408c03b), unstarted,
without a checkout or execution.

Remaining: no further review requested. Stopped at the accepted demo boundary. No PR
publication, merge, Task completion or human acceptance is implied. Session roots
expire with ordinary tables; completed replay inspection remains durable.

## Gate — 2026-10-20

Polished formatted single-game JSON import (alongside JSONL and arrays), cleared
old import choices when opening another file, and refreshed replay history when
navigating from Watch to Replay within the same route. Watch controllers now
belong to GameSession and are released on closure. Addressed watch restoration
uses persisted watch metadata so completed seat-two inspection survives session
expiry; active restoration requires the watch credential as well as record access.

Checks on the resulting implementation:

- `uv run --extra dev ruff check` on all nine changed/new Python files: passed.
- `uv run --extra dev ruff format --check` on those files: passed.
- `git diff --check`: passed.
- `uv run --extra dev pytest tests/academy/test_watch.py tests/academy/test_advice.py::test_default_play_import_does_not_require_training_dependencies -q`: 7 passed, including both checkpoint modes and saved seat-two inspection after controller closure.
- `uv run --extra dev pytest tests/academy/test_watch.py tests/academy/test_server.py tests/academy/test_trace_api.py tests/academy/test_study_runtime.py tests/academy/test_study_branch.py tests/academy/test_study_protocol.py tests/academy/test_replay_index.py tests/academy/test_experience_protocol.py tests/test_package_boundaries.py tests/sim/test_structured_policy.py tests/sim/test_flat_mc.py -q`: 96 passed, 1 failed in 437s. The sole failure was the new test omitting the existing plural record-credential header; that test was corrected and passed in the focused 7-test run above. The broad process had already imported the earlier tree; it was not rerun after the focused fixes.
- `npm --prefix frontend run check`: 0 errors and 0 warnings.
- `npm --prefix frontend test`: 100 passed across 18 files.
- `npm --prefix frontend run build`: passed, including after the route-navigation fix.
- `npm --prefix frontend run test:e2e -- watch.spec.ts`: stepping, both-hand view, Study retry/return, pause/reload and formatted JSON import passed. The added navigation test initially had an ambiguous link locator; `npm --prefix frontend run test:e2e -- watch.spec.ts -g 'loads replay history'` passed after scoping it to the header.

Production performance signal: track step-to-inspected-board latency at p50/p95,
split by controller kind and legal-offer count. No before/after speed claim is
made; search time depends on the selected anchor and simulation budget. Full
cross-platform integration, clean-machine and visual-reference CI remain CI-owned;
this gate uses focused headless interaction coverage, with no manual walkthrough.
No publication, landing or Task-state change is part of this gate.
