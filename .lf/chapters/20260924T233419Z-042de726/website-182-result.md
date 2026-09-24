# PR #182 repair evidence — 2026-09-24

Finish: rebased, locally verified website behavior; published exact head with auto-merge enabled; shared notes restored untracked. No deployment or new watcher.

Watched head: 05a1f35c2b1330998e1a116268307b136966d570.
Base after lf rebase --manual: 196edd5861acd312db7c6e984e2ec76030885990.
Loopflow skipped merged parent commits; website content unchanged by rebase.
Original GitHub check-runs: 0; commit statuses: 0 (pending aggregate). Unknown, not failed.
Original website suite: 10 passed.
Stronger XML regression: 1 failed / 13 passed; missing concatenation operator made XML preamble the join separator and omitted opening urlset.
Fix: restore + before join; assert parsed blog-only sitemap and plain /healthz behavior plus Fly probe configuration.
Post-fix: uv run --python 3.12 --extra test pytest tests/ -q from website/: 14 passed.
Focused: uv run --python 3.12 --extra test pytest tests/test_site.py -k healthz -q: 4 passed, 10 deselected.
git diff --check: passed.
Article identical to original watched head. No Rust changes; Rust checks not needed for website scope.
Gate guidance updated to require website suite with parsed XML and health probe coverage.

## Published handoff

Observed UTC: 2026-09-24T20:32:39.859806+00:00

PR: https://github.com/loopflowstudio/etude/pull/182

Published head and local HEAD: `2a58adf728209ba86815e03e9454e63dfcea0b10`.

`lf pr arm` returned successfully. State: OPEN; merge state: BLOCKED; auto-merge enabled: True; merged at: None.

Exact-head check-runs at handoff:

- Visual Reference Gate: in_progress, conclusion pending. https://github.com/loopflowstudio/etude/actions/runs/36055453082/job/107821111435
- Rust Tests: in_progress, conclusion pending. https://github.com/loopflowstudio/etude/actions/runs/36055453082/job/107821111377
- Python Unit Tests: completed, conclusion success. https://github.com/loopflowstudio/etude/actions/runs/36055453082/job/107821111360
- Protocol v1 Conformance: in_progress, conclusion pending. https://github.com/loopflowstudio/etude/actions/runs/36055453082/job/107821111336
- Clean-machine Play: in_progress, conclusion pending. https://github.com/loopflowstudio/etude/actions/runs/36055453082/job/107821111069

Commit status aggregate: pending; individual statuses: 0.

CI is still pending, not a local pass or merge claim. The existing supervisor owns subsequent CI observation and merge; no new watcher was started.

## Preservation and scope

Article bytes match the watched original head; SHA-256 `4f57957e3a5118f788ba1d1df3a73c8abb8218d73236ff802282fb90941ebaa5`. Root redirect, blog-only sitemap, plain 200 `ok` health response and Fly `/healthz` probe retained.

Protected open-work.md, open-work-inventory.json, review-history and all ship-logs were moved outside the checkout before commit/clear-scratch and restored as untracked/ignored local notes. Non-log hashes verified; original log inodes restored for ongoing writers. No protected path is tracked.

Preservation manifest and intermediate evidence: `/Users/jack/src/etude-182-preserved-85vcl11q`.

No deployment, #179 mutation, #183 reopen, PR retirement, or branch cleanup performed. No Co-Authored-By trailer added; published collapsed commit has none.

## Continued Clean-machine Play repair — 2026-09-24

Finish line: reproduce the failed boundary, preserve the 60,000 ms budget and
website behavior, verify the narrow fix, publish and arm the exact repaired
head, then restore protected local notes and log inodes. A warm launcher pass
alone is not clean-machine performance evidence. No merge wait or deployment.

Watched failed head: `2a58adf728209ba86815e03e9454e63dfcea0b10`.
`lf rebase --manual` completed successfully before CI investigation.
Exact job 107821111069 in run 36055453082 failed at
`frontend/e2e/clean-machine.spec.ts:244`: elapsed 61,590 ms, required <60,000 ms.
Launcher internal readiness was 52,634.1 ms; this excludes uv setup and browser
startup. Every other substantive check passed; Tests Result only propagated
Clean-machine Play failure. The website was not the failing surface.

Current hypothesis: serial proof-runner/browser startup after launcher readiness
adds avoidable measurement overhead. Check a cold-artifact baseline locally
before changing this scheduling. Local macOS results will not be represented as
a fresh Ubuntu reference-host certification.

Protected paths and complete exact-job log are preserved at
`/Users/jack/src/etude-182-ci-preserved-5r38ruhp`. Existing build/dependency
artifacts are also preserved there for restoration after the local cold run.

### Diagnosis and verified repair

The verifier serialized npm/Playwright/Chromium startup after
`ETUDE_PLAY_READY`, adding proof setup to an already marginal cold launch.
It now starts the browser runner when the launcher's locked npm installation
marker appears, overlapping the remaining native build. The browser waits for
the actual ready record before navigation. The original external clock and
strict 60,000 ms assertion remain unchanged, as do all offline/recovery checks.
The shell reaps the browser runner on launch failure or readiness timeout.
No launcher, engine, dependency lock, website, or budget change was needed.

Local evidence:

- Original cold-checkout-artifact browser proof: pass, 26,534 ms externally,
  24,179 ms launcher-internal. This did NOT reproduce the Ubuntu timeout.
- New scheduling/cleanup regressions against the old verifier: 3 failed.
  Same regressions plus launcher contracts against the repair: 31 passed.
- Repaired cold-checkout-artifact browser proof: pass, 24,080 ms externally,
  23,444 ms launcher-internal. Offline reload preserved session, visible state,
  pack identity, and accepted another action; zero public requests.
- Both browser runs used local macOS/arm64, Node 24.12.0, CPython 3.12.12,
  uv 0.9.22, and Chromium 149.0.7827.55. Project artifacts were absent before
  each run; shared host download caches were available. These results verify
  behavior and scheduling, not Ubuntu reference-host performance. TypeScript
  checking overlapped the end of the repaired run; no controlled speedup is
  claimed from these two wall-clock samples.
- `npm --prefix frontend run check`: zero errors and warnings.
- Website suite: 14 passed. Website diff against the failed head is empty,
  preserving article, root redirect, repaired XML sitemap, and `/healthz`.
- `sh -n scripts/verify-clean-machine` and `git diff --check`: passed.
- No Rust source or build configuration changed; no new Rust test run needed.

The Python Unit Tests CI job now includes the new verifier regressions.
`.lf/steps/gate.md` records the cold-artifact proof, timing boundary, port
isolation, cleanup checks, and limits of local host evidence.
Original local dependencies/build outputs were restored after both runs.
Receipts and logs are under the preservation directory's `baseline-artifacts`
and `repaired-artifacts` trees; `verifier-regression-before.log` captures the
three expected failures against the old implementation.

### Repaired publication handoff

Observed UTC: 2026-09-24T20:44:09.843915+00:00

Updated Python Unit Tests command: 30 passed.

`lf pr arm` succeeded. PR: https://github.com/loopflowstudio/etude/pull/182

Published head and local HEAD: `7c9dda0826688318db79f59879c0e854cd9f0163`.

State: OPEN; merge state: BLOCKED; auto-merge enabled: True; merged at: None.

- Python Unit Tests: in_progress, conclusion pending. https://github.com/loopflowstudio/etude/actions/runs/36056779748/job/107825683654
- Visual Reference Gate: in_progress, conclusion pending. https://github.com/loopflowstudio/etude/actions/runs/36056779748/job/107825683576
- Rust Tests: in_progress, conclusion pending. https://github.com/loopflowstudio/etude/actions/runs/36056779748/job/107825683507
- Protocol v1 Conformance: in_progress, conclusion pending. https://github.com/loopflowstudio/etude/actions/runs/36056779748/job/107825683445
- Clean-machine Play: in_progress, conclusion pending. https://github.com/loopflowstudio/etude/actions/runs/36056779748/job/107825683339

CI on this exact head remains pending; no merge is claimed. The existing landing supervisor resumes observation. No watcher was started and no merge wait was performed. No deployment, #179/#183 change, PR retirement, or provider-materializer interference occurred.

All protected notes and the complete ship-logs directory were restored locally after arm. Original inodes verified for every protected file, including live logs; unchanged review-note hashes verified. Updated result notes remain untracked. Published commit contains no Co-Authored-By trailer.
