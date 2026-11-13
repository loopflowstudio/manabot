Contributors waited over 24 minutes for the slowest Integration job. This PR reduces that to 5 minutes 49 seconds in the PR-triggered Ubuntu run, including installation and compilation, while preserving test coverage and native debug assertions.

<!-- loopflow:task-pr-context:start -->
> [!NOTE]
> **Task:** Get CI under 8 minutes · 38ef25
> **PR lifecycle:** The pull request is published for review; no Task settlement is requested.
<!-- loopflow:task-pr-context:end -->

## What changes

- Balance Integration across six groups using `pytest -n logical`. Integration and Academy Integration install dependencies once and build managym once, removing the duplicate compilation.
- Keep the five-minute test watchdog in each worker. Replace admission-test polling with injected time, and use two complete-game streams for the real-worker size smoke while retaining its update assertions.
- Document the eight-minute job budget, parallel-test requirements and how to rebalance groups without dropping tests.

The budget measures individual jobs, not total workflow duration. Engine Budget retains its same-runner reference comparison and is exempt; the macOS nightly is outside this timing target. The PR run leaves 131 seconds of margin; the preceding full dispatch left only 32 seconds, so runner and download variability can still push a job over budget. Eight minutes is a documented target, not an enforced timeout.

This also delivers MTG-263, “Shard the Integration core pytest group so it stops setting CI’s critical path”: measured balancing and process-safe parallel tests, with no coverage dropped. MTG-263 can be closed when this change lands.

## Checks

- [Baseline CI](https://github.com/loopflowstudio/manabot/actions/runs/43312163427): slowest Integration job **24m03s**; Academy Integration **3m41s**.
- [Full Ubuntu CI](https://github.com/loopflowstudio/manabot/actions/runs/43334548912) at `30cfcb2e6`: all jobs passed; every non-exempt job finished within **7m28s**, and Academy Integration took **2m54s**. Engine Budget passed in **14m56s**. The subsequent commit only records this evidence.
- [PR Ubuntu CI](https://github.com/loopflowstudio/manabot/actions/runs/43337105047) at `1fc4e1b24`: all jobs passed; every non-exempt job finished within **5m49s**, and Academy Integration took **3m06s**. Engine Budget passed in **10m13s**. Training passed all 346 tests, including the changed admission and real-worker size checks.
- Recorded local validation passed: 53 CI-selection tests, five active-training tests, Ruff check/format, `uv lock --check` and `git diff --check`. Controlled watchdog checks confirmed healthy parallel runs continue and stuck tests print stacks and fail.

After merge, inspect ordinary Ubuntu job durations for timing regressions, especially evaluation and dependency installation. The timing evidence includes both full manual dispatches and the PR-triggered run on the current code.
