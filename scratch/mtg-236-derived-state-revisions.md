# MTG-236: derived state revisions, indexed effect records, cache-carrying clones

Task brief: `.lf/prompts/20261102-120246-*-summary.md` (Jack Heart's concept
review of 2026-11-02, Performance section). Base `4c8324a59` (T4 landed);
`origin/main` is still that commit. Branch head `86af3387f`, two commits.

## Status (reconciled 2026-11-02)

Items 1-4 exist and are checked (below). Item 5 is not implemented and is
proposed for removal (decision pending). Remaining: the quiet-machine
acceptance measurement, publication with the measured delta in the PR body,
and ordering against T5/T6, which touch the same files.

## Outcome (brief's payoff order)

1. Done. `Revisions { characteristics, mana_pools, triggers }` on
   `TrackedState` (`flow/game.rs`); `DerefMut` advances all three, typed
   accessors advance the subset their field feeds. One `DerivedCache`
   (`flow/statics.rs`) holds the static index and views (keyed on
   `characteristics`, views also on `mana_pools` while a static condition
   reads a pool) and the trigger index (keyed on `triggers`); debug builds
   recompute every cached value on read and assert equality.
   Departures from the brief, recorded in `questions.md`: `mana_pools`
   replaces the brief's `offers` counter (no derivation caches offers), and
   only the write sites the revision probe showed hot moved to typed
   accessors; the rest keep advancing every counter through `DerefMut`.
2. Done. `EffectRecords` (`state/effect.rs`): records in timestamp order
   plus by-object / by-player index; `effects_on` reads one object's records.
3. Done. `Game::clone` carries the derived cache (immutable `Arc`s) and
   revisions.
4. Done. `TriggerIndex` (sources by event kind) rebuilt on `triggers`,
   sources in object order so simultaneous triggers are numbered as `main`
   numbers them; `ObjectLkiLog` with latest-by-entity lookup.
5. Not done: interned subtypes in `Characteristics`. After 1-4 a 25 s
   `sample` of the release bench puts all of `Game::characteristics` at ~2%
   inclusive and its `String`/`Vec<String>` building under 1% (~32% before
   this branch), so interning cannot move the acceptance numbers. The
   remaining profile is mana payment search (`Mana::payment_remainders`,
   `PaymentModel::witness`, the `PaymentSearchState` hash map) and allocator
   time.
   **Decision needed (Jack Heart):** drop item 5 from this Task on that
   evidence, or implement it anyway for the brief's stated reason (allocation
   share). Proposal: drop it and open a separate Task for mana payment
   search if the quiet-machine bench misses acceptance. Not accepted yet.

Acceptance (unchanged): `bench_search` worst reply <= 4.34 s and total
<= 128.7 s, seeds 1-3, 40 decisions, same score digests as `main`; debug and
release `cargo test` green; no rules change (`WORLDS.md` untouched).

## Remaining work

- **Quiet-machine `bench_search`** (`cargo run --release --example
  bench_search --manifest-path managym/Cargo.toml`), base and branch. Every
  run here was made with load averages of 12-70 from other sessions, so the
  absolute thresholds are unverified; the recorded CPU-second delta is the
  evidence so far. Never induce load; a CI runner or an idle laptop is fine.
- **Publish** with the delta in the PR body. The brief asked for a delta per
  item; items 1-4 landed in one commit (`884090f1b`) plus a fold (`86af3387f`),
  so only the aggregate delta exists. Per-item deltas would need four
  separate builds; proposed: report the aggregate unless Jack Heart asks.
- **Ordering against T5 and T6.** The brief said to start after MTG-225 (T6)
  published; T6 is still an unpublished worktree
  (`manabot.mtg-225-tla-t6-sagas`, head `d344572db`, same base) and T5 is
  open PR #405. Both overlap this branch in `flow/statics.rs`,
  `flow/zones.rs`, `state/effect.rs`, `state/hash.rs`, `flow/decision.rs`,
  `flow/resolution.rs`; T6 also in `flow/triggers.rs` (+128 lines),
  `flow/effects.rs`, `flow/tick.rs`, `flow/play.rs`. Whichever lands second
  rebases. Their new writes through `DerefMut` stay correct by construction;
  T6's new trigger-source kinds must join `TriggerIndex` by event kind (or
  be matched outside it, as graveyard triggers are) and new effect-record
  kinds go through `EffectRecords::push`/expiry. Debug recompute-and-assert
  fails `cargo test` on any miss.

## Delete — do not maintain

All removed on this branch: `TrackedState::revision()` single counter,
`GameState::effects` as a bare `Vec` (the canonical hash still serializes
the ordered records), `object_lki` linear scans in `flow/triggers.rs` and
`flow/identity.rs`, `Game::clone` starting with an empty cache, and the
separate `TriggerIndexCache`. Nothing remains to delete.

## Measurements

`bench_search`, release, seeds 1-3, 40 decisions, this machine on
2026-11-02 while other sessions kept the load average at 25-70 (the brief's
4.97 s / 151.8 s were measured on a quiet machine, so only the delta and the
digests are comparable).

| build | worst reply | median (s1/s2/s3) | total (s1+s2+s3) | digests |
| --- | --- | --- | --- | --- |
| base `4c8324a59` (load ~40) | 10.97 s | 2.58 / 1.61 / 1.46 | 125.95 + 78.95 + 75.57 = 280.5 s | 3929171216812047577 / 4710253461277237463 / 8287291947930727881 |
| items 1-4 (load ~50-70) | 8.12 s | 1.41 / 0.98 / 1.13 | 82.13 + 52.89 + 53.09 = 188.1 s | identical |
| base, paired run (both binaries at once, load ~33→19) | 11.65 s | 3.12 / 1.31 / 1.46 | 150.51 + 69.65 + 74.55 = 294.7 s | same |
| items 1-4, paired run | 11.06 s | 2.45 / 1.20 / 1.19 | 118.40 + 58.27 + 56.66 = 233.3 s | identical |

Wall-clock worst replies are set by the other sessions' load (seed 1's worst
decision is 4-5x its median in every run, base and branch alike).

CPU seconds (`/usr/bin/time -p`, each binary alone, load ~22-30):

| build | user | sys | real | wall totals (s1+s2+s3) |
| --- | --- | --- | --- | --- |
| base `4c8324a59` | 212.32 | 3.95 | 222.83 | 102.49 + 57.04 + 63.28 = 222.8 s |
| items 1-4 | 147.34 | 2.88 | 180.20 | 70.12 + 69.96 + 40.10 = 180.2 s |

CPU work drops 31% (digests identical). Applied to the brief's quiet-machine
numbers for the cache branch (4.97 s / 151.8 s) that projects to ~3.4 s /
~105 s, inside the acceptance bound; a projection, not a measurement.

Revision-site probe on the base (sampled `deref_mut` backtraces, seed 1,
8 decisions): the cache was flushed every step by `clear_mana_pools` on empty
pools, `clear_damage`, the step flags, `emit`'s three pushes, taps and
`invalidate_mana_cache`, none of which a layer reads.

## Checks

- Debug `cargo test`, all 19 managym test targets plus lib: green, with every
  cached view, static index and trigger index recomputed and asserted on
  read (lib, cards, rules, search, conformance, engine,
  search_state_contract, match_state_hash, scenario, canonical_replay,
  authored_match, elemental_world in the implement pass; the remaining
  targets on 2026-11-02 realign: green, exit 0).
- Release `cargo test --release --manifest-path managym/Cargo.toml`
  (2026-11-02 realign): green, all targets, exit 0 with `--no-fail-fast`.
- clippy `--all-targets --all-features -D warnings` and `cargo fmt --check`
  clean.
- Python (extension rebuilt from this branch): authored-match parity,
  authored-match authority, public-commitment parity and skip-forced suites
  pass; both w17 receipts were regenerated and only their source hashes
  changed.

## Proposed Engine memory (not written; wave memory is read-only here)

- Derived state is cached per revision subset, not maintained incrementally:
  `TrackedState` revisions name what each cache reads, typed accessors move
  only the counters their field feeds, and debug builds recompute-and-assert
  so a misclassified accessor fails `cargo test` (MTG-234/236, 2026-11-02).
- Bench evidence under load: wall-clock worst reply is unusable when other
  sessions load the machine; record CPU seconds and score digests, and
  verify thresholds only on a quiet machine or CI runner.
