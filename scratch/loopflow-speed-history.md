# Loopflow speed history, Aug–Nov 2026

Source: `git -C /Users/jack/src/loopflow log --since=2026-08-01` (586 commits), PR bodies
(commit messages carry the full PR description), and `gh run list -R loopflowstudio/loopflow
--workflow ci.yml` (successful runs, createdAt→updatedAt; includes queueing).

## Measured CI trend (merge_group, successful, daily median wall time)

| Period | Median | Note |
|---|---|---|
| Aug (W33–W36) | 5.4–7.0 min | small matrix |
| 09-25..09-27 | ~10 min | before the Sep 28–29 caching batch |
| 09-30..10-04 | 6.0–7.3 min | after #1320–#1346 batch (confounded: #1361 deferral same week) |
| 10-08..10-12 | 10.8 → 19.6 min | test growth; baseline doc measured rust-test 20:09 median |
| 10-13..10-18 | 5.8–7.2 min | after #1543 (build-once + partitions) |
| 10-20..10-21 | 13.7–15.3 (max 70) | lf-world tests landed (#1617/#1619) |
| 10-22..11-05 | 9.6–11.9 min | creep back despite #1647 budgets; macOS moved out of ci.yml by #1749 |

PR-event medians of 0.2 min from W41 on are deferred checkpoint runs (#1361), not real proofs.

## Significant changes

| Commit / PR | Date | Category | Mechanism | Measured effect (source) |
|---|---|---|---|---|
| af1805ba2c #1543 | 10-13 | CI structure / test runner | Compile & archive Rust once (nextest archive), download into 5 lib + 3 integration partitions on 6 slots; consolidate 41 integration test binaries into one `integration` binary (+14 dedicated); `check_rust_test_inventory.py` rejects unregistered test files; omit CI debug symbols (keep debug-assertions); prepared fixtures per cell | Required CI ~20 min → 4:25–4:55 (PR body, runs 38431676003/38427516700); fixture cells 105s→10s; integration archive 380→196 MB; merge-queue median 19.6→6.4 min (gh run data). Biggest single win. |
| 845e8b7c8c #1647 | 10-22 | Test tiers / budgets | nextest `pr` profile: `slow-timeout = {60s, terminate-after=2}` on PRs only, per-test overrides for known slow tests (~1.5x worst of 5 queue runs); Swift per-PR 5 min budget + slow-test report | Prevents creep (5-whys: 3 PRs took Swift step 2–5→13–14 min in a day); no before/after speed number. `.config/nextest.toml` |
| 24b5a338c7 #1745 | 10-27 | Graph/step design (verification scope) | gate/implement skills plan from diff (`scripts/test.py --list`), run only reachable suites once, `--rust-filter` nextest expr, `--reuse-passing` for identical tree, one cargo build at a time (tests then clippy on same build), never whole-workspace in implement; 900-line TESTING.md split to per-module READMEs + short `scripts/testing.md` | Baseline only: gate 52.8 h over 204 steps; implement 391.6 h over 25 days. After: unmeasured. |
| 192124063f #1320 | 09-29 | Caching (local evidence) | Reuse identical passing evidence; empty plan returns before resource scans; cancel superseded PR runs | Evidence reuse 8.02s→0.12s; empty plan 16s→0; landing regression 73.7s→13.9s (local, PR body) |
| 8c5eeb0cf0 #1547 | 10-14 | Worktree/build sharing | Prepare fixture CLI once per verification; prune completed worktrees' build output (24h, 64 GiB) | Small edit/test median 106.3s→70.0s (-34%); warm fixture -29.5%; allocation 4.47→3.39 GiB (local, PR body) |
| scripts/testing.md (via #1745, #1820) | 10-27 | Worktree/build sharing | sccache as `rustc-wrapper` in user `~/.cargo/config.toml`, shared across worktrees; repo sets none (CI has no sccache) | unmeasured |
| fd2b2c18b5 #1749 | 10-28 | CI structure (path gating) | macOS jobs split to reusable workflow, path-classified per job; not run in merge queue; post-merge `macos-main.yml` tests since `last-green`, bisects and opens revert PR; land merges main only when GitHub requires; split conflict hot-spot files | Removes 10–49 min macOS wait for Rust-only PRs (PR body, baseline claim); after unmeasured |
| 732f2f493c #1646 | 10-22 | CI structure | Path filter skipping macOS for docs/Python; per-test 300s kill naming the test; cancel superseded merge-queue runs | unmeasured (density world build 219s→~50s locally) |
| 5402d93974 #1336 | 09-29 | CI structure | Main push reuses the merge-queue run's exact-SHA proof; only warms caches | Avoids ~24.4 runner-min per main push (baseline #1334); push medians fell to 1–2 min (gh data) |
| 4a696c0740 #1361 | 09-30 | CI structure | Defer full matrix on checkpoint PRs containing scratch; `--no-fail-fast` | runner savings unmeasured |
| 4820e3670c #1332 | 09-29 | Caching | Include workspace `[profile]` fingerprint in rust-cache key; only main saves caches | ~50s SQLite rebuild on exact hits (log observation); gain unmeasured |
| 2283951248 #1335 | 09-29 | Caching | build.rs stopped registering missing `.git/packed-refs` as input (was rebuilding every invocation) | 40.8s rebuild → 0.09s reuse (E2E job/local) |
| 97e0926da0 #1328 | 09-29 | Other (profile) | `opt-level=2` for bundled SQLite in dev/test profiles only | migration tests 12.99→5.97s; first build +12s once |
| c3e8372a3b #1321, bc51da30a4 #1329 | 09-29 | CI structure | Reuse binaries built by earlier test step instead of rebuilding | ~40–50s and 54s removed (log observations) |
| d4b283a873 #1343, 7d1158dac2 #1346, 37ca5fca81 #1546 | 09-29..10-14 | Caching | Restore mtimes for unchanged Swift/Xcode sources so cache hits don't recompile | 23.5s→0.3s, 19.5s→~2s (local) |
| 21c020035e #1340 | 09-29 | Other (observability) | `time`-style CPU/RSS reports on expensive CI commands | instrumentation only |
| a1d2f8a591 #1431 | 10-04 | Graph/step design | `pursue` reduced to one implement loop; dropped demo + loop-decide steps from pursue | unmeasured |
| e88c34a674 #1367, 7944efd647 #1748, 0b17cf980f #1672, 7cc5f59f35 #1714 | 09-30..10-27 | Graph/step design (sync) | Merge main instead of rebase (no replayed conflicts); sync commits uncommitted edits before merging so conflicts reach resolver; plan once; shorter sync lock so CI repair runs beside live work | Correctness/lost-work fixes; LOO-659 lost hours of implement edits; speed unmeasured |
| 0b7bda54a8 #1770, 234be45e3b #1597 | 10-18..10-29 | Task stacking | Heal stacked child base after parent squash-merges; stop Auto-Ship after identical repeated failures | Stopped a 38x demo→gate→land loop (LOO-669) |
| 521d9e119c #1645 | 10-22 | Test tiers | Deleted slow world tests until fast | Swift step 13–14 min back toward ~3 min (per #1647 5-whys) |

## What transfers to manabot

manabot today: `ci.yml` runs plain `cargo test` (one job) with `CARGO_PROFILE_DEV_OPT_LEVEL=1`,
rust-cache, ~20 separate integration binaries in `managym/tests/`, `debug = 1`, no nextest.

1. **Build once, partition with nextest archives (#1543).** Highest measured payoff (20→~5 min).
   manabot: `cargo nextest archive` in one job, `--partition hash:k/N` in N jobs; give
   `semantic_conformance_tests` its own partition or its own job. Keep `debug-assertions` on
   (CI must stay debug per CLAUDE.md); dropping debuginfo (`CARGO_PROFILE_DEV_DEBUG=0` in CI)
   is safe for debug_assert! semantics. Caveat: more runners, artifact upload/download ~1 GB for
   loopflow; partition skew (one slow partition set the 5:15 floor).
2. **Consolidate integration binaries (#1543).** Each `tests/*.rs` links the whole engine;
   merging into one `tests/integration.rs` with modules cuts link time and archive size.
   manabot already shares `tests/support/`; caveat: per-binary `cargo test --test X` commands in
   docs/CI (e.g. `experience_protocol_tests`) and the cards crate's single `tests/cards.rs`
   already follow this pattern.
3. **Per-PR test budgets with nextest `pr` profile (#1647).** Cheap, prevents the creep loopflow
   saw (W43–W45 drifted 6→11 min even after budgets — budgets name tests but don't remove cost).
4. **Diff-scoped verification in implement/gate steps (#1745).** Directly applicable to manabot's
   multi-pass Graph: `--list` plan from changed paths, named-module filters, reuse-passing for an
   identical tree, one cargo build at a time per target dir. Caveat: effect never measured; a
   filter that misses a dependent test is caught only by CI.
5. **sccache shared across worktrees (user-level config).** Fits agents in separate worktrees;
   unmeasured in loopflow; note manabot sets `incremental = false` already (sccache-compatible).
   Caveat: maturin/PyO3 builds and different RUSTFLAGS/opt-level envs fragment the cache.
6. **Main reuses merge-queue proof (#1336), cancel superseded runs, path gating (#1646/#1749).**
   manabot's ci.yml already path-selects jobs; Engine Budget could skip on main when the queue
   ran it on the same SHA. Post-merge bisect/revert is heavier machinery.
7. **Cache-key and rebuild-trigger hygiene (#1332, #1335, #1328).** Audit `build.rs`
   `rerun-if-changed` paths and profile changes in cache keys; opt-level overrides for hot
   dependencies in test profile.
8. **Sync: merge not rebase; commit edits before merging (#1367, #1748).** Reduces repeated
   conflict resolution across implement/review/sync passes.

Not transferable / caveats: Swift/Xcode mtime caching; loopflow's CI median crept back to
~11 min by November, so gains need budgets to hold; many PR bodies report local, not hosted,
numbers.
