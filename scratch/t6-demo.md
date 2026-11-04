# T6 headless behavior demonstration — 2026-11-03

## Current result — decision pass 2

**Advance.** The concrete failures identified by pass 1 are repaired. Fresh
demonstration on published head `d17e8bfd779739a20f1ae0344fcad542e3e694e0`
passes **67 Python consumer checks and seven Svelte rendering checks**.
No new repairable behavior gap or unresolved product question was observed.
This advances the demo step, not merge approval: [PR #417](https://github.com/loopflowstudio/manabot/pull/417)
remains open for Jack Heart's review, with final-head CI still running.

Current design: [T6 working design](mtg-225-tla-t6.md),
[accepted direction](t6-direction.md), and
[concept review](concept-review.html). The pass-1 account below is retained
as failure evidence; its iteration recommendation and conflict status are
superseded by this pass.

### Intended progress versus observed progress

Pass 1 required isolating frozen pilot scheduling/analysis fixtures from live
runtime admission, explicitly pinning historical input capacities, and syncing
the conflicting branch. The implementation now uses an opt-in compatible-world
fixture for synthetic outcomes, including the actual fresh-process analysis CLI.
The full 33-case pilot module passes, including independent negative checks
that frozen loading, run declaration and evaluation reject the real changed
world. The declared input remains v6/256 facts/64 actions. Frozen protocol and
regime files have no diff from the preceding demo head `d21fae20d`.

Main `fb5ad2c13` is integrated. GitHub reports this published head as
`MERGEABLE`; the previous conflict is resolved. Publication also found and
repaired a further concrete integration gap: typed encoders rejected T6's
relation kinds 7–9 and schedule frames 11–12. The new consumer run exercises
both typed encoders through actual one-update training, checkpoint export and
reload, as well as directed context/gradient checks for inheritance, captured
subjects, frequency and schedule order. These are bounded correctness smokes,
not strength or throughput measurements. This is additional observed progress,
not merely a repeated implementation claim.

### Demonstrated experience and evidence

The real Academy `/ws/play` endpoint ran in process through ASGI TestClient,
using native game state, ordinary Commands and default trivial-decision skipping.
The three gameplay prefixes below use no injected game state.

| Action | Observed result on this head |
| --- | --- |
| Play Bumi, resolve its combat trigger, enter the extra combat, reconnect and retry an old Command | Pending restriction becomes current even with zero pending combats. Both closed-deck viewers see the public schedule without opponent hand disclosure. Reconnect preserves the frame; stale rejection preserves digest/events. |
| Cast Fatal Fissure, reconnect while its watcher waits, then advance through cleanup and reconnect | Exact captured subject and once/this-turn lifetime survive retained-session recovery and clone/capture checks. The unmatched watcher expires at cleanup and remains absent. |
| Reach Sozin chapter II, select one card, reconnect, retry and stop searching | Remaining private choices and budget survive; duplicate/stale Commands cannot repeat the exile. Stopping removes inspection and retains the committed exile. |
| Encode delayed lifetimes, transformations and restricted combat | Native/Python paths agree across the selected version cases; older v4 exclusions remain intact. |
| Consume T6 records with both typed policies | Context responds to capture, frequency and schedule differences; storage-row permutation preserves meaning. Both actual training/export/reload smokes pass. |
| Render ordered turns, current/pending combat restrictions and historical absence | Seven component checks pass, including a current restriction with no pending combat and “Schedule not recorded.” |

Fresh commands, both exited successfully:

```sh
uv run --extra dev --extra play --extra artifacts --extra evaluation pytest tests/academy/test_schedule_recovery.py tests/academy/test_named_search_recovery.py tests/verify/test_observation_parity.py::test_waiting_delayed_trigger_rows_agree_and_stay_out_of_v4 tests/verify/test_observation_parity.py::test_t6_transform_parity_and_schedule_relation_versions tests/verify/test_observation_parity.py::test_bumi_scheduled_combat_restriction_has_native_python_fact_parity tests/eval/test_tla_pilot.py tests/model/test_state_records.py -q
npm --prefix academy/ui test -- src/lib/components/GameBoard.svelte.test.ts
```

Logs: [67 consumer passes](../.runs/t6-demo/pass2-consumers.log),
[seven rendering passes](../.runs/t6-demo/pass2-schedule-render.log),
[published-head state](../.runs/t6-demo/pass2-pr-state.json), and
[CI snapshot](../.runs/t6-demo/pass2-pr-checks.json).

The preceding publication's complete debug run remains applicable to unchanged
Rust: **952 passed, two ignored**, including Sozin payment-before-targets,
optional search, Visions timing, independent delayed lifetime combinations,
countered occurrences, cleanup survival and exact departure retirement.
Its log is [combined-rust.log](../.runs/t6-publication/combined-rust.log).
The repaired typed consumer suite and actual mutable parity builders are also
recorded in the working design; the parity regression log has **20 passes**.
No Rust or UI source changed after that gate. These broader results were
inspected and reused, not represented as fresh runs in this demo.

### Feedback, limits and next action

There was no new human observation or acceptance. Jack Heart's existing 1a
decisions for schedule completion and independent lifetime composition remain
the design authority. Acting under the explicit headless instruction to stand
in at demo, the agent chooses advance because the required behavior holds in
the configured consumer paths and the specific preceding failures now have
passing proofs. No agreed design change is needed, and no new design is proposed.

Retained-session recovery is not process-crash durability. Component rendering
is not browser geometry: this surface has no rendering environment, and no new
screenshots, interactive page or human browser walkthrough are claimed. Prior
Playwright evidence retains its documented limits. No retained-checkpoint
binary compatibility, historical-world admission or training strength is claimed.

Recommended next action: continue from demo to the existing review boundary,
retaining final-head CI as pending validation. At inspection six checks had
succeeded and ten were in progress, with none failed; the PR merge-state label
was `BLOCKED`, distinct from merge conflicts or a missing product decision.
Final CI success is not claimed. No merge, auto-merge, Task completion or new
Graph was requested here. All locally started commands have finished; no
server or watcher remains running.

## Historical result — decision pass 1 (superseded)

Decision pass 1: **iterate to implement**. The accepted Academy behavior holds
in the focused demonstration, but final-head CI exposes a reproducible
experiment-fixture integration failure. This is concrete repair work, not an
unanswered product choice or lack of progress.

Reviewed head: `d21fae20d3ef0e65fa7814124785f08034d6741d`, published on
[PR #417](https://github.com/loopflowstudio/manabot/pull/417).
Design: [T6 working design](mtg-225-tla-t6.md),
[accepted direction](t6-direction.md), and
[preserved concept review](concept-review.html).

## Intended progress and observed results

The preceding pass intended to finish independent delayed event/frequency/expiry
composition, preserve exact selector capture, complete public scheduling, refresh
mutable w17 receipts, and publish the cohesive T6 change for review. Its recorded
debug, consumer and browser evidence is in
`../.runs/t6-publication/final-*.log`. Publication reached the reviewed head.
Those results are progress; this pass does not discard them because broader CI
found another gap.

Autonomous demonstration used the real Academy `/ws/play` ASGI WebSocket endpoint
and native engine, with normal Commands and trivial-decision skipping. TestClient
hosts the app in process; this proves retained-session reconnect, not process-crash
durability or a deployed network service. No state injection drives these three
gameplay prefixes.

| Action | Expected and observed |
| --- | --- |
| Play Bumi, resolve its combat trigger, advance into extra combat, reconnect, then submit an old Command | Pending land-creature-only combat becomes current with an empty pending queue; restriction remains. Both closed-deck viewers receive the public schedule without private hand disclosure. Reconnect returns the same frame; stale rejection changes neither state nor events. Passed. |
| Cast Fatal Fissure, reconnect while its delayed watcher waits, then advance through cleanup and reconnect again | Once/this-turn lifetime and one exact captured object persist; capture and clone digests agree. Stale Command is rejected. The unmatched watcher expires at cleanup and stays absent after recovery. Passed. |
| Reach Sozin chapter II, choose a name, exile one card, reconnect, retry the Command, and stop searching | Remaining choices and budget survive with chooser-only inspection. Retry is duplicate; fresh-id stale retry is rejected. Stopping ends inspection and preserves the committed exile. Passed. |
| Encode real waiting delayed abilities through native/Python v4/v5/v6 paths | Once and repeating cases agree; v4 excludes newer rows. Both passed. |
| Render the board with ordered schedules, current restriction, pending combats and historical absence | Seven Svelte rendering checks passed, including distinct turn orders, viewer labels, current restriction with zero pending combats, and “Schedule not recorded.” |

Fresh evidence:

- `../.runs/t6-demo/academy-recovery-parity.log`: five passed in 1.26s.
- `../.runs/t6-demo/schedule-render.log`: seven passed.
- Prior complete debug log explicitly passes independent lifetime combinations,
  no-match expiry, countered occurrence without rearming, queued/stacked survival,
  exact departure retirement, Sozin payment-before-targets and Visions timing.
  Those suites were inspected, not rerun here.

No browser screenshots or layout proof were produced on this headless surface.
The previous 25-pass/two-skip Playwright proof remains prior evidence, not a new
Bumi browser walkthrough. No interactive demo page or new human response is
claimed.

## Concrete failure and next implementation direction

[Final-head CI integration job](https://github.com/loopflowstudio/manabot/actions/runs/41774414354/job/126391517284)
reports **27 failed, 278 passed, one skipped**. All 27 failures in
`tests/eval/test_tla_pilot.py` stop at `experiments/tla_pilot_v3.py:91`:
`pilot runtime world differs from the prospective protocol`.
Full failure evidence: `../.runs/t6-demo/ci-integration-failure.log`.

Local reproduction:

```sh
uv run --extra dev --extra play --extra artifacts --extra evaluation pytest tests/eval/test_tla_pilot.py::test_pilot_pins_independent_population_and_elemental_learning_rule -q
```

It fails at the identical world guard; evidence:
`../.runs/t6-demo/pilot-world-reproduction.log`. The frozen v3 experiment correctly
rejects a different world. Its tests incorrectly require that historical protocol
to match the evolving current runtime before they can exercise population,
evaluation scheduling and controlled analysis outcomes.

Return to implement with this bounded direction: separate frozen-protocol test
fixtures from live-runtime admission. Exercise those historical scheduling and
analysis contracts under explicitly controlled compatible identity/configuration;
retain a negative test proving the real changed runtime is rejected. Inspect
the next configuration guard too: the frozen input contract must not inherit
fresh observation defaults. Preserve frozen protocol/regime bytes, evidence and
production rejection semantics; do not refresh historical experiments, weaken
admission, skip the suite wholesale or launch training. Rerun the full pilot
module with artifacts/evaluation extras and the affected discovery/analysis paths,
then publish the repaired PR head and check CI.

GitHub also reports `CONFLICTING`/`DIRTY`; snapshots are
`../.runs/t6-demo/pr-state.json` and `../.runs/t6-demo/pr-checks.json`.
Reinspect authoritative base/mergeability and resolve any real integration
conflicts through `lf sync` before republishing. This pass did not merge or
diagnose a conflict locally. Rust, Python unit, Academy integration/UI, protocol,
kernel, clean-machine play and Engine Budget checks succeeded at the reviewed
head; those successes do not cancel the failing integration job.

## Feedback and decision authority

Jack Heart previously accepted schedule completion and independent delayed
lifetime composition (both 1a). There is no new human observation or approval in
this headless pass. Acting under the caller's instruction to decide autonomously,
the agent selects iteration because the requested cohesive, validated PR still
has a reproducible integration gap. No accepted product design changes.
No unresolved product question prevents this repair. Keep PR #417 and its
stop-for-review boundary; this decision authorizes neither merge nor auto-merge.

All test commands finished; no server or watcher was left running.
