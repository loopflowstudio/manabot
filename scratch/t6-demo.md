# T6 headless behavior demonstration — 2026-11-03

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
