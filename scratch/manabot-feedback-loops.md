# manabot feedback loops — measured from existing records (2026-11-05)

No suites were run for this study; numbers come from GitHub Actions logs and
local tool state.

## CI critical path (green runs)

Per-job wall time, main pushes 41909331179 / 41935280348 and PR 42048447244:

| Job | Minutes |
|---|---|
| Integration (main shard) | 27.1 / 21.9 / 24.8 — **critical path** |
| Integration (long) | 8.0 / 8.5 / 8.5 |
| Engine Budget | 5.1 / 6.5 / 8.5 |
| Rust Tests (debug, opt-level 1) | 3.8 / 4.3 / 4.4 (2 min compile, ~2 min tests) |
| Academy UI | 3.3 / 3.2 / 12.9 |
| Others | ≤ 5.6 |

Integration main (job in run 41935280348): 1.3 min install, 1.0 min extension
build, **10.0 min** first pytest step (1052 tests, single process, 593 s),
**8.5 min** second pytest step (two-seat/Etude/eval), 0.6 Perf proof. No
pytest-xdist; `fail-fast: false` matrix of only `main`/`long`.

Any change under `managym/src`, `state/src`, `cards/src` or `content/` selects
every job (`.github/ci/select-tests.sh`), so every engine PR waits ~25 min
minimum for green.

## The debug-oracle blowup on the MTG-244 branch

Rust Tests per-target times, main (run 41935280348) vs MTG-244 PR #427
(run 41958999646, head before 2026-11-05 06:22 PT):

| Target | main | MTG-244 | ratio |
|---|---|---|---|
| lib unit tests | 8.65 s | 41.83 s | 4.8× |
| catalog pair/mirror | 12.2 s | 85.1 s | 7× |
| random smoke (13 tests) | 40.5 s | 227.9 s | 5.6× |
| semantic_conformance_tests (6) | 42.2 s | **5771.9 s** | **137×** |
| Rust Tests job | ~4 min | 105.5 min (failed) | |

Earlier #427 runs were cancelled at 60.3 and 80.5 min in Rust Tests.
Five `managym-cards` failures (`tla::honest_work_*`, `phoenix_fleet_airship`,
`energybending`, Toph observability) ran in **18.7 s** but were reported only
after the 96-minute conformance target, because `cargo test` runs test
binaries serially in name order.

Mechanism (diff `main...origin/jack/mtg-244-complete-composable-layer-evaluation-with-concrete`,
`managym/src/flow/statics.rs`): in debug builds the layer code skips its fast
paths (`if independent && !cfg!(debug_assertions)`, `!cfg!(debug_assertions) && !plausible(a, b)`)
and runs the full pairwise clone-and-apply dependency oracle for every layer
computation; `layer_board` also `debug_assert_eq!`s the cached board against a
full `compute_board(None)` on every cache hit. Release (Engine Budget) never
executes these, so no budget sees their cost. Long conformance tapes
(2000 commands) multiply it.

## Local loops

- Host: 16 cores, 128 GB. `~/.cargo/config.toml` sets `rustc-wrapper = sccache`
  and `jobs = 8` for every worktree.
- sccache: 73.6 % Rust hit rate, but 3,386 non-cacheable calls (bin 963,
  cdylib 55, proc-macro 176, other crate-type 1,834): test binaries, the
  cdylib and linking always rebuild. Workspace crates keyed on worktree path
  miss across worktrees.
- `managym/Cargo.toml`: `incremental = false`, `debug = 1` for dev/test
  ("agent worktrees build this crate once or twice"). Every pass after a
  change recompiles managym from scratch (CI: ~2 min compile on 4 vCPU).
- `managym/tests/` has 20 separate integration-test binaries (each linked
  against the full engine); `managym-cards` already uses one `tests/cards.rs`.
- docs/development.md asks for `CARGO_PROFILE_DEV_OPT_LEVEL=1` locally (CI's
  comment: ~3× faster tests) and `-j 2` on shared hosts; nothing enforces
  either (no repo `.cargo/config.toml`, no wrapper script).
- `cargo-nextest` is installed but unused by the repo and CI.
- 36 git worktrees exist; 8 have a `managym/target`.

## MTG-241 elapsed time by Graph step

From `lf graph show <id> --processes --json` step start/completion (UTC,
2026-11-04/05). Step records start 20:14Z, two hours before the brief's 22:16Z
claim time; the source of that gap is unexplained.

| Graph | Span | implement | compress | sync | realign | loop-or-next | End |
|---|---|---|---|---|---|---|---|
| 22ee206e (5 passes) | 249 min | 140 (15→13→22→46→44) | 82 (last pass 56) | 14 (pass 5 failed after 11.6) | 10 | 3 | sync failed |
| b421d936 (3 passes) | 300 min | 137 (16→26→95) | 24 | 23 (pass 2: 21.7) | 115 (pass 3 failed after 89.5) | 2 | realign failed (lf store mismatch) |

- ~9.2 h wall, no PR. Implement passes grew 3–6× from pass 1 to the last pass
  in each Graph; that growth coincides with the integrated MTG-244 sibling and
  full-suite verification under three-agent host contention (brief), not
  measured here per pass.
- Tooling/sync failures directly consumed ≥101 min (failed sync 11.6, failed
  realign 89.5) and ended both Graphs; nothing from them reached review.
- Not yet split inside implement passes: agent work vs test/build wait vs
  rework. That needs the per-pass conversation logs.
