# MTG-234: characteristics lookups fast again

Status: implemented and gated 2026-10-30; PR #410.

## Design (as built)

- `Game::state` is `TrackedState` (`flow/game.rs`): `Deref` to `GameState`,
  `DerefMut` advances a revision. `priority_mut()` / `rng_mut()` write fields
  no characteristic reads without advancing it.
- `StaticsCache` (`flow/statics.rs`, `Game::statics_cache`, a `RefCell`, empty on
  clone) holds, per revision, the `StaticIndex` (attached permanents with their
  exact attached object; static sources that have not lost abilities) and each
  permanent's `Arc<Characteristics>`.
- `Characteristics` owns its strings (no lifetime); `Game::characteristics`
  returns `Arc<Characteristics>`.
- Debug builds recompute on every cache hit and `debug_assert_eq!` (view and
  index), so debug `cargo test` checks the cache everywhere.
- `TrackedState` / `StaticsCache` Debug print state only / opaque.

## Measurement (`managym/examples/bench_search.rs`, release, seeds 1–3 × 40
decisions, Search-64 = 16 worlds × 4 rollouts, both seats, curated UR Lessons v
GW Allies mains; same score digests in all three builds)

| build | worst reply | total |
| --- | --- | --- |
| pre-T3 `b73a6b5db` | 4.34 s | 128.7 s |
| T3 base `f122d515b` | 9.82 s | 305.5 s |
| this branch | 4.97 s | 151.8 s |

Profile after: characteristics ~32% of playout time (was >50%); remaining
characteristic-side cost is owned-string allocation (~7%) and cache churn from
legitimate mutations (taps). Not pursued: `Card` string interning.

## Checks

- Debug `cargo test --no-fail-fast` (all targets, incl. semantic conformance,
  with the cache debug asserts) passed; clippy `--all-targets --all-features`
  and fmt clean; extension rebuilt.
- Python: elemental world, skip-forced, public-commitment and authored-match
  parity pass. The authored-match w17 receipt differed only in
  `identity.relevant_source` hashes (behavior fields equal) and was
  regenerated.
- Compress: `Game::restart_priority_round` replaces the repeated
  active-player priority writes; inline shuffles use `shuffle_library`.
  Debug `cargo test --no-fail-fast`, clippy, fmt, authored-match /
  public-commitment / skip-forced pytest pass; receipt hashes regenerated.

## Departure from the Task text (draft, for review)

- The Task asked for an index maintained at enter/leave/attach/detach. It is
  instead rebuilt lazily once per state revision (see `questions.md`,
  MTG-234): every write passes `TrackedState::deref_mut`, so it cannot drift.
  Reviewer may still ask for incremental maintenance.
- Target was "near pre-T3": worst reply 4.97 s vs 4.34 s (+15%), total
  151.8 s vs 128.7 s (+18%).

## Remaining

- Gate: full affected Python suites; PR body with the table above and the
  departure. Only two receipts bind source hashes (authored-match, regenerated;
  public-commitment, unchanged and passing).

Check (2026-10-30 realign): no new commits on `origin/main`; code matches the
design above; no checks rerun.

Concept review (2026-10-30, headless): the model is one reader
(`Game::characteristics`), one revision (`TrackedState`) and one derived cache
keyed on it; no second index to keep consistent, no new public concept, and
the CLAUDE.md rule already describes it. Lazy per-revision rebuild instead of
incremental enter/leave/attach maintenance is an implementation choice that
preserves the Task's outcome (identical behavior, speed near pre-T3), recorded
in `questions.md`; not a product choice. Nothing to change; no open choice.

Gate (2026-10-30, headless): reviewed the full diff against the Task and
AGENTS.md; no changes needed. Code is unchanged since the recorded checks
(only scratch moved after `65f87c164`), so they stand: debug `cargo test
--no-fail-fast`, clippy `-D warnings`, the four Python suites above. Reran
`cargo fmt --check`: clean. The world/observation-shape suite does not apply
(no world or observation change). Before/after: worst reply 9.82 s → 4.97 s,
total 305.5 s → 151.8 s (pre-T3 4.34 s / 128.7 s). CI owns the full matrix.
