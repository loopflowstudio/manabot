Since the characteristics cache landed in [Halve Search-64 reply times by caching permanent characteristics per state revision · PR #410](https://github.com/loopflowstudio/manabot/pull/410), every mutable touch of game state flushed it: a tap, a damage mark or an emptied mana pool recomputed every permanent's layered characteristics on the next read, and each cloned search world started cold. The engine now recomputes a derived view only when a field that view reads has changed, and cloned worlds keep what the parent already computed, so a Search-64 reply stays fast for the whole playout with identical play.

<!-- loopflow:task-pr-context:start -->
> [!NOTE]
> **Task:** [Derived state: revisions by what the layers read, indexed effect records, cache-carrying clones · MTG-236](https://linear.app/loopflow/issue/MTG-236/derived-state-revisions-by-what-the-layers-read-indexed-effect-records)
> **PR lifecycle:** The pull request is published for review; no Task settlement is requested.
<!-- loopflow:task-pr-context:end -->

## What changes

- `TrackedState` carries `Revisions { characteristics, mana_pools, triggers }`, one counter per derivation, named for what reads it. `DerefMut` still advances every counter; typed accessors (`tap`, `mark_damage`, `events_mut`, `move_card`, `mana_pool_mut`, step flags, stack and trigger queues, incarnations) advance only the counters their field feeds. Only the write sites a revision probe showed hot in playouts moved; every other write keeps the conservative path.
- One `DerivedCache` in `flow/statics.rs` holds the static index and views (keyed on `characteristics`, and on `mana_pools` only while a present static reads a pool, discovered from the sources through `Condition::reads_mana_pools`) and the trigger index (keyed on `triggers`). Debug builds recompute every cached value on read and assert equality, so a misclassified accessor fails `cargo test` in debug.
- `EffectRecords` (`state/effect.rs`) keeps continuous-effect records in timestamp order with a by-object and by-player index; `effects_on` reads one object's records instead of filtering all of them per attachment.
- `TriggerIndex` (`flow/triggers.rs`) looks up trigger sources per event kind, in object order so simultaneous triggers are numbered as before; `ObjectLkiLog` answers departed-object facts with a latest-by-entity lookup instead of a history walk.
- `Game::clone` carries the derived cache and revisions as immutable `Arc`s, so search worlds inherit the parent's views.
- AGENTS.md records the accessor contract. The current w17 receipts were refreshed after integrating T5: the authored-match receipt changes only relevant-source hashes; the public-commitment receipt also updates its six content-bound tracker digests (repeated in the headless summary). Command tapes, state witnesses, commitments, viewer-isolation and atomicity proofs, and frozen RUL-9/INT-12 references remain unchanged.

No rules change: `WORLDS.md` is untouched, the canonical hash still serializes the ordered effect records and the LKI log, and the pre-T5 benchmark score digests match base `4c8324a59`. The synced implementation also passes live/headless/replay parity.

## Why it matters

Historical measurement before the T5 merge: `bench_search`, release, seeds 1-3, 40 decisions, each binary alone under `/usr/bin/time -p`, score digests identical to base `4c8324a59`. These numbers are not a new measurement of the synced head `d2136d4ad`:

| build | user CPU s | wall total (s1+s2+s3) |
| --- | --- | --- |
| base `4c8324a59` | 212.3 | 222.8 s |
| items 1–4, before T5 integration | 147.3 | 180.2 s |

CPU work drops 31% with identical digests. Every run was made while other sessions kept this machine's load average at 20-70, so wall-clock worst replies (8-11 s on both builds) were set by that load and are not comparable to the brief's quiet-machine numbers.

**Accepted check still to run on a quiet machine or CI runner:** `cargo run --release --example bench_search --manifest-path managym/Cargo.toml` on seeds 1-3, 40 decisions, must show a worst reply at or below 4.34 s and a total at or below 128.7 s with the same digests. Applying the 31% CPU drop to the brief's post-#410 quiet numbers (4.97 s / 151.8 s) projects about 3.4 s / 105 s; that is a projection, not a measurement.

**Deferred on evidence:** item 5 of the brief, interned subtypes in `Characteristics`, is not built. After the four items above, a release profile puts all of `Game::characteristics` at about 2% inclusive and its string building under 1%, so interning cannot move the acceptance numbers. The remaining profile is mana payment search (`Mana::payment_remainders`, `PaymentModel::witness`) and allocator time. Suggestion: a separate Task covers mana payment search if the quiet-machine bench misses acceptance. The producible-mana cache is the one remaining hand-invalidated derivation and is a possible follow-on outside this scope.

**Integration:** T5 [PR #405](https://github.com/loopflowstudio/manabot/pull/405) is integrated at checkout base `a8abf01b1`; the merge retains its `CreateTokenCopy` behavior. MTG-225 saga work overlaps the same engine modules and will need integration against the resulting main. New state writes through `DerefMut` invalidate conservatively; new trigger-source and effect-record kinds must join the indexes and have directed debug coverage.

## Checks

- The earlier implementation passed the full debug and release Rust suites. At synced head `d2136d4ad`, CI's full debug Rust suite passes with cache recomputation assertions enabled; main integration is still running. Academy UI/integration, protocol, semantic conformance, clean-machine Play, Python unit, long integration, and Linux/macOS package/install checks pass ([CI](https://github.com/loopflowstudio/manabot/actions/runs/41549044249), [packages](https://github.com/loopflowstudio/manabot/actions/runs/41549044194)).
- Fresh focused debug and release checks pass and cover fork identity, shared cached views and fork-local invalidation, continuous-effect expiry, LKI, trigger ordering, Ozai's mana-conditioned characteristics, and Joo Dee's copy decisions.
- The preceding headless demo passed all 7 authored-match/public-commitment Python tests on unchanged native sources and receipts: 142 commands, 143 checkpoints, 858 viewer checks, stale-command/object rejection, and hypothesis isolation.
- `cargo fmt --check` and clippy `--all-targets --all-features -- -D warnings` pass. The full release matrix predates the T5 merge; the current gate adds focused release coverage of the merged paths. The strengthened fork test is locally verified and awaits publication/CI.

The quiet-machine worst-reply ≤4.34 s and total ≤128.7 s acceptance remains unverified. Shipping approval does not certify those targets; the Task must retain that measurement gap.
