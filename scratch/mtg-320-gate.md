# MTG-320 gate — 2026-11-13, 33968d5c10

Ready for review within the checked-card-id scope. No code changes or new
blockers found. The accepted design and complete earlier gate receipts are
recoverable at `fe30e33d62:scratch/mtg-320-plan.md`; the runtime design is in
`docs/engine.md#objects-and-zones`. Preserve the recorded direction:
"Stop at review. Do not enable auto-merge."

Reviewed kind construction/transitions, checked CardId construction/indexing,
ordered library/hand entries and their wire representation, mixed-zone
narrowing, selection of created tokens, stale identity rejection, SBA cleanup,
canonical hash compatibility, and the related tests/docs. Graveyard and exile
keep mixed ObjectId lists; physical-card storage and new rows per zone change
remain outside this PR. Existing documentation accurately states these limits.

## Check plan and result reuse

Runtime/zone changes reach native identity, SBA, selection/cost and card tests;
projection/hash changes reach Python, conformance and browser checks. Reuse the
previous gate and CI for identical sources. Only scratch was changed after the
tested revision, so no compiler, suite, formatter or native rebuild was rerun.

- `git diff --exit-code 349d643e52 HEAD -- . ':!scratch'`: passed; all non-scratch inputs identical.
- `git diff --no-renames --name-only -z origin/main...HEAD | bash .github/ci/select-tests.sh`: selects all nine CI areas; each has passing coverage in run 43493623873.
- `gh run view 43493623873 --json headSha,status,conclusion,jobs`: completed successfully at 349d643e52; all 16 jobs passed, including debug Rust, conformance, Python, Academy, six integration groups and Engine Budget.
- `gh run view 43493623564 --json headSha,status,conclusion`: completed successfully at 349d643e52; package/install/artifact matrix covers CPython 3.12/3.13 on Linux x86_64 and macOS arm64; release publication intentionally skipped.
- `git diff --check`: passed.
- `gh pr view 491 --json state,headRefOid,autoMergeRequest,title,body`: open at 349d643e52, auto-merge null; published body still described CI as in progress.

Reused local receipts from the preceding gate: 22 focused debug tests passed
(run 77e314ef-542b-4697-9c97-2c9585c8455f), workspace all-target/all-feature
Clippy with warnings denied passed, and workspace rustfmt passed without edits.
Exact commands are in the earlier plan. No deferred check is claimed as a new
local pass. Broad suites remain covered by the passing CI runs above.

Same-runner Engine Budget artifact reported search 90.484 → 98.364 seconds
(+8.71%), worst reply +8.48%, and throughput -5.36%; all digests/counts matched.
This passes the existing 10% threshold and does not establish a speedup.
Final recorded QA at 6b28b642d6 used the same engine tree: 568 seeded games,
137,518 decisions with matching observations/state hashes, and three Academy
games reaching Game Over without page/console errors. No new human demo.

The historical backend capture failure hit its 120-second training budget
(141 seconds overall), not pytest's 300-second limit. Prior investigation
located the stage/regime deadline checks; the cause of exhaustion is unknown.
The steer says "Let CI judge it on this head; this machine is heavily loaded."
That exact test passed in CI in 28.83 seconds; no deadline was relaxed. The
historical Academy stall remains unexplained; the shared tap-cost failure
occurred on main too. Preserve these limitations without assigning a cause.

Restored the prepared PR title/body from fe30e33d62, which already contains the
completed CI results, conversion audit, measured budget and limits. Pinned
`.pr-copy-ref` to local HEAD. These are local delivery inputs; gate did not
publish, merge, enable auto-merge or change Task state. No background process
was started.
