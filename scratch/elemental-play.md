# Elemental local play

Accepted direction, Jack Heart, 2026-10-26: browse all ten Elemental decks,
choose a deck for each player, and play the latest MTG-183 model locally on CPU.
This expands MTG-79's human-play handoff. No training or hosted deployment.

Implementation: use the installed Elemental roster and existing checkpoint
catalog, GameSession, Command and attempt/replay store. Add a browsable roster
with full main lists and Lesson pools to game setup. Support choosing the next
matchup during a game without changing the running match until New Game.
Provide a local launcher that resolves the latest published MTG-183 checkpoint
by update, verifies bytes and CPU admission, then pins that file for this server.
Record actual model digest/update and origin; automated proof uses its own store.

Jack's follow-up direction: the deck browser and switcher must be the same
control, and deck lists need real card pictures. DeckSelector now owns browsing
and assignment for both Play and Watch; separate deck dropdowns are removed.
Legacy setups remain under Other decks within that browser. Card rows show
larger cached art and open full printed images. Normal Elemental launch fetches
the art cache; offline launch reuses it. Do not restore removed candidate-bundle
loading or duplicate checkpoint admission.

Jack also reported prompts with only one option and selected: "Auto-advance
forced steps and show them in the log." Keep skip_trivial=False; automate sole
legal actions through the existing bound Command path with auto=True. Log each
as Auto and preserve real choices, recovery and replay identity. The existing
automation checkbox can disable both forced advancement and priority passes.
Jack clarified that the human UI must expose the same choices/information as
the bot, and evaluation environments must use skip_trivial=True. Audited the
scalar evaluation path (sim.match → sim.replay → manabot.env.Env), scenario
evaluation, and vector rollout: all already enable native trivial skipping.
Academy now applies its recorded forced-action automation to both seats before
calling a policy or publishing a human prompt. Manual inspection remains
available with automation disabled. No training/evaluation world changed.

Checkout synchronized through `lf sync --manual` to main 558e0186b. Old HEAD
b51fc4e30 exactly matched landed bc454e671 outside experiments/data, whose local
files were retained while adopting main's untracked artifact policy. Merge
resolved to main's exact tracked tree; no preexisting working edits existed.

MTG-183 context: current training segment f3748bcabac8, seed 18301, Elemental
w13; earlier e7d47d8b220f ended at 3264 with monitoring export 3250 retained.
Read current published checkpoints, never mutate the training/evaluation work.

Implemented: roster browser in the existing Play setup, exact main/Lesson lists,
independent seat selection including mirrors, and next-match setup during a game.
`scripts/play-elemental` reads published Etude checkpoint records, verifies the
chosen bytes and CPU/world admission, then delegates to the existing launcher.
The printed URL configures the initial decks and exact checkpoint. `--offline`
reuses the pin without storage access. Instructions: docs/local-trained-challenger.md.

Latest published record inspected: segment f3748bcabac8, update 5950,
331180341 learner transitions; SHA-256
4f2ad656ba036a33d527212a7ac2badf2c5b9db136916f730846b07eef2fbea4.
Local policy in `.runs/elemental-play/checkpoints/`; its ten setups pass current
w13 admission and the loaded model is CPU. No training/evaluation service changed.

Initial validation: release browser suite 21 passed (including the new browser,
real checkpoint/reconnect, mirrors, mobile overflow, combat, Learn, Precision);
frontend type check/build pass; 35 targeted frontend unit checks, 5 checkpoint
catalog tests, 3 new launcher tests and 41 existing launcher/verifier tests pass.
Ruff and whitespace checks pass. Real CPU gameplay exercised 40 explicit legal
Commands each for UR/GW, BG/UG and UG/UG, with actual bot decisions retained in
`.runs/elemental-cpu-proof/result.json`. Automated records remain separate.
Desktop/mobile screenshots are in academy/ui/test-results/release/deck-browser-*.
Final narrow-screen correction constrains the long model selector to its parent;
7 affected browser checks pass after that change, including the real model at
390px, with a fresh type check/build. Live API confirmed exactly one compatible
checkpoint and zero human attempts before handoff; Jack now has one active
human_exploratory attempt. Preserve the running backend and that store.

Unified-browser validation: 22 release browser checks pass with cached art
required, including every Elemental list, assignment, mirrors, printed preview,
no runtime external requests, mobile width, and actual MTG-183 reconnect.
All 94 unique roster cards have crop and full images; 106 assets fetched with
none missing. Screenshot inspection confirms readable desktop art and printed
preview. Fresh Svelte check/build and all three launcher tests pass. Initial
long gameplay checks were disrupted by development reloads while art downloaded;
retest after cache completion. Cold-machine proof requires its dedicated clean
checkout harness and is not validated by an ordinary browser-test invocation.
Periodic Task automation is disabled; CI watcher is not running.

Forced-step checks: 10 focused backend tests pass, including explicit Commands,
automatic action receipts for both seats, policy-call exclusion, reconnect,
duplicate Commands, manual inspection and existing stop controls. Four slower
full-game stop tests were not part of that focused run. Thirty frontend store
and transport tests pass; Svelte check/build and Ruff pass. The release suite's
22 existing tests pass after human automation; the added log test needed to
open the collapsed Decision log, then all four deck-browser tests passed.
An ordinary UR/GW browser game reached terminal. Game-history's initial run
hit a pre-hydration native form submission; its check now waits for the loaded
profile before submitting search. Its full-game proof now uses retained seed
1255757736 instead of an unbounded random deal and passes, including public
history, private feedback and replay. The 12 server regression tests also pass;
their log-buffer fixtures now use the renamed action log. After applying forced
automation to the bot as well, all four deck-browser checks pass again against
the real CPU checkpoint. Cold-machine proof still requires the dedicated
empty-artifact harness.

On final checkpoint refresh the latest published MTG-183 export remains update
5950, digest 4f2ad656ba03. The earlier human server was no longer running after
the pause; the retained human attempt remains in SQLite and must not be deleted.

Local human server originally started via `./scripts/play-elemental --offline`.
After the pause it was restarted from the verified installed runtime with the
same pin and human record directory, without dependency installation during
browser proof. Current tool PTYs: backend 55027, frontend 70491; API 18479,
browser 18480. API reports the checkpoint available and the prior human attempt
interrupted; frontend returns HTTP 200. Future restart: `./scripts/play-elemental --offline`.
Human records: `.runs/elemental-play/human-play/play.sqlite`, human_exploratory.
Hand-off URL:
http://localhost:18480/?hero=elemental_ur&villain=elemental_gw&opponent=4f2ad656ba036a33d527212a7ac2badf2c5b9db136916f730846b07eef2fbea4

Jack reviewed the local changes on 2026-10-26: "ok seems ok", then explicitly
requested "let's ship this". Proceed through the ship Graph: gate, then land
with wait-and-fix. This authorizes publication and merge of the implementation.
Fresh human records show one interrupted attempt and one active BG/UW attempt;
neither is terminal. Preserve the live server and SQLite store. Shipping the
implementation does not satisfy MTG-79's separate terminal-human-play and
return-feedback completion criteria.

Gate, 2026-10-26: comparison is the working tree against `558e0186b`;
the synchronized HEAD has that exact tracked tree. Reviewed the full diff and
the accepted local-play scope. Gate fixes: prepare the native engine before
the Elemental wrapper imports checkpoint admission, explain a missing offline
pin, label forced actions as automation in replay logs and transition copy,
and update Academy's stale explicit-forced-prompt documentation. Added Watch
seat-assignment coverage and launcher bootstrap/failure checks. The stops
browser comparison now supplies its claimed fixed seed instead of relying on
the server's random-deal default; forced skipping can still change the random
opponent's sampling sequence, so this is a click-reduction check, not matched
trajectory timing.

Gate commands/results (installed environment preserved with `--no-sync`):
- `uv run --no-sync pytest --collect-only tests/academy tests/infra deploy/play/tests -q` — 322 tests collected together; no module-name collisions.
- `uv run --no-sync pytest tests/academy/test_forced_step_automation.py tests/academy/test_stops.py tests/academy/test_server.py tests/academy/test_elemental_play_launcher.py tests/academy/test_play_launcher.py tests/academy/test_clean_machine_verifier.py tests/academy/test_checkpoint_catalog.py tests/academy/test_study_runtime.py tests/academy/test_presentation.py tests/academy/test_experience_protocol.py tests/academy/test_recorded_attempts.py -q --durations=10` — 98 passed in 444.51s, including all full-game stop tests; no timeout. The manual-vs-automatic complete-game stop comparison accounted for 206.22s.
- `uv run --no-sync pytest tests/academy/test_elemental_play_launcher.py -q` — 6 passed after the gate bootstrap/missing-pin fixes.
- `npm --prefix academy/ui test` — 106 passed; `npm --prefix academy/ui test -- src/lib/replay.svelte.test.ts` — 3 passed after the replay-label fix (one new test).
- `npm --prefix academy/ui run check` — zero errors/warnings, including after the replay fix; `ACADEMY_STATIC_BUILD=1 ACADEMY_FRONTEND_BUILD="$PWD/.runs/elemental-gate/spa" npm --prefix academy/ui run build` — final SPA build passed.
- `npm --prefix academy/ui run test:e2e:release` with static build, ports 18579/18580, the pinned update-5950 checkpoint, required cached art and `automated_validation` origin — 24 passed in 2.1m. Includes Watch, mobile layout, every deck/lesson list, printed previews, no external runtime requests, real CPU checkpoint/reconnect, command choices, combat and Learn. Screenshots remain in `academy/ui/test-results/release/`.
- `uv run --no-sync ruff check` and `ruff format --check` on changed Python files, plus `git diff --check` — passed.
- `uv run --no-sync pytest tests/infra/test_ci_selection.py -q` — 40 passed; new launcher paths select Academy and clean-machine checks, and Academy CI explicitly runs the new launcher/forced-action contracts.

The additional built-SPA browser harness initially used `academy.server:app`,
which serves only the API: `/games` returned 404 and no game started. That run
was interrupted, its subprocesses reaped, and the temporary harness corrected
to the documented `deploy.play.asgi:app` with `/games` as its readiness probe.
Its logs are retained under `.runs/elemental-gate/`; this was harness setup,
not a gameplay timeout or a passing result.

The corrected built-SPA run passed game history/private feedback/shared replay
(14.3s), keyboard F6, and UR/GW play to terminal (44.0s). Its full-game stops
comparison exposed a test race: after Play Again it accepted the prior game's
still-mounted board, saw the old Game Over overlay, and returned zero clicks.
The retained new attempt had four opening events, confirming the replacement
arrived after that assertion. The test now waits for the new authoritative
action controls; the failing case was rerun alone. It also now pins seed 0
explicitly. Logs: `.runs/elemental-gate/browser-extra-final.log` and
`.runs/elemental-gate/stops-final.log`.

The corrected pointer-driven stops comparison then hit its unchanged 300s
overall test budget, during automated input 63 after 289 manual inputs. This
was not the individual 30s response deadline: the overall budget expired
inside that assertion. Retained attempts show the first game completed after
644 native events and the second progressing through 695 events to turn 30;
the page remained connected with a pending response and no error alert.
The two complete-game assertions, seed, action budgets, and 300s timeout remain.
The driver now focuses each legal control and presses Enter (the same supported
input path used by the passing history proof), avoiding repeated pointer
scroll/layout stability waits. Release interactions retain pointer coverage.
Its final run is recorded in `.runs/elemental-gate/stops-keyboard.log`.

Final result: `npm --prefix academy/ui exec -- playwright test --config
"$PWD/.runs/elemental-gate/playwright.config.ts" stops.spec.ts --grep 'stops cut'`
passed (3.8m test, 4.4m including startup/cleanup), with 289 deliberate inputs
without automation and 73 with it. Both games reached authoritative completion
(644 and 807 native events respectively), and narration assertions passed.
This closes the required timeout investigation without raising the timeout or
removing the full-game assertions. Together with the three passing built-SPA
cases and 24 release cases, 28 distinct browser checks pass. All gate subprocesses
were reaped; ports 18579/18580/18581 are free. Original human listeners remain on
18479/18480 (PIDs 14802/15219), and their record directory was not accessed or
modified by this gate. No hosted/training change, publication, merge or Task
completion was performed. Ready for the primary conversation's
`lf land --wait-and-fix`; MTG-79's human-game/return-feedback criteria remain open.

Cold-machine verification is deferred to the selected Ubuntu CI job:
`./scripts/verify-clean-machine` requires removing this checkout's installed
environment, native extension and frontend artifacts, which would violate the
active human-server preservation boundary. Launcher/verifier contract tests
passed locally. Neither cold-start budget nor reference-machine performance is
certified by this warm local gate. For a future hosted rollout, track time to
first authoritative choice, action-to-authority response latency, and missing
cached previews; no production deployment or Wave-memory update is part of this
gate.
