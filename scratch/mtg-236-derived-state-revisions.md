# MTG-236: derived state revisions, indexed effect records, cache-carrying clones

Task brief: `.lf/prompts/20261102-120246-*-summary.md` (Jack Heart's concept
review of 2026-11-02, Performance section). Base `4c8324a59` (T4 landed).

## Outcome (payoff order; each measured with `bench_search`, release, seeds 1-3, 40 decisions)

1. `Revisions { characteristics, triggers }` on `TrackedState`; `DerefMut`
   advances both, typed accessors advance the subset their field feeds.
   `StaticsCache` keys on `characteristics`; debug recompute-and-assert stays.
2. `EffectRecords` (`state/effect.rs`): records in timestamp order plus
   by-object / by-player index; `effects_on` reads one object's records.
3. `Game::clone` carries the statics cache (immutable `Arc`s) and revisions.
4. `TriggerIndex` (sources by event kind) rebuilt on `triggers`;
   `ObjectLkiLog` with latest-by-entity lookup.
5. Interned subtypes in `Characteristics`: not done. After 1-4 a 25 s
   `sample` of the release bench puts all of `Game::characteristics` at
   ~2% inclusive and its `String`/`Vec<String>` building under 1% (it was
   ~32% before this branch), so interning cannot move the acceptance
   numbers. The remaining profile is mana payment search (`Mana::
   payment_remainders`, `PaymentModel::witness`, the `PaymentSearchState`
   hash map) and allocator time; that is a separate Task.

Acceptance: worst reply <= 4.34 s and total <= 128.7 s with the same score
digests as `main`; debug and release `cargo test` green; no rules change.

## Delete — do not maintain

- `TrackedState::revision()` single counter (replaced by `Revisions`).
- `GameState::effects: Vec<ContinuousEffect>` as a bare list (becomes
  `EffectRecords`; the canonical hash still serializes the ordered records).
- `GameState::object_lki: BTreeMap` linear scans in `flow/triggers.rs` and
  `flow/identity.rs` (`ObjectLkiLog` owns the lookups).
- `Game::clone` starting with an empty `StaticsCache`.

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

Wall-clock worst replies on this machine are set by the other sessions'
load (seed 1's worst decision is 4-5x its median in every run, base and
branch alike), so the acceptance thresholds (4.34 s / 128.7 s, measured
quiet) cannot be verified here; CPU seconds for both binaries are recorded
below as the load-insensitive comparison. Re-run `bench_search` on a quiet
machine or CI runner before landing.

CPU seconds (`/usr/bin/time -p`, each binary alone, load ~22-30):

| build | user | sys | real | wall totals (s1+s2+s3) |
| --- | --- | --- | --- | --- |
| base `4c8324a59` | 212.32 | 3.95 | 222.83 | 102.49 + 57.04 + 63.28 = 222.8 s |
| items 1-4 | 147.34 | 2.88 | 180.20 | 70.12 + 69.96 + 40.10 = 180.2 s |

CPU work drops 31% (digests identical). Applied to the brief's quiet-machine
numbers for the cache branch (4.97 s / 151.8 s) that projects to ~3.4 s /
~105 s, inside the acceptance bound, but that is a projection, not a
measurement.

Revision-site probe on the base (sampled `deref_mut` backtraces, seed 1,
8 decisions): the cache was flushed every step by `clear_mana_pools` on empty
pools, `clear_damage`, the step flags, `emit`'s three pushes, taps and
`invalidate_mana_cache`, none of which a layer reads.

Checks: debug `cargo test` for lib, cards, rules, search, conformance,
engine, search_state_contract, match_state_hash, scenario, canonical_replay,
authored_match and elemental_world targets: green, with every cached view,
static index and trigger index recomputed and asserted on read. clippy
`--all-targets --all-features -D warnings` and `cargo fmt --check` clean.
Python (extension rebuilt from this branch): authored-match parity, authored-
match authority, public-commitment parity and skip-forced suites pass; both
w17 receipts were regenerated and only their source hashes changed.
