# MTG-236: derived state revisions, indexed effect records, cache-carrying clones

Task brief: [.lf current directive](../.lf/prompts/20261102-160437-d0a2fbad-54dc-4d18-a85e-9117cbafece0-0I3Lcf-summary.md)
(Jack Heart's concept review of 2026-11-02, Performance section).
Original measurement base: `4c8324a59` (T4 landed). Current checkout base:
`a8abf01b1` (T5 integrated); reviewed head: `d2136d4ad`.

## Status (reconciled 2026-11-02)

Items 1-4 exist and are checked (below); subtype interning remains deferred.
PR #411 is published, synced with T5, and approved for shipping by Jack Heart
(see Shipping approval). The headless demo below advances to the remaining
delivery steps. Quiet-machine latency acceptance remains unmeasured; this
decision does not certify those thresholds or complete the Task. Earlier
publication and T5-ordering entries below describe their dated state and are
superseded by the sync and shipping sections.

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
   Settled (concept review, 2026-11-02): dropped from this Task. The brief
   lands each item with its measured `bench_search` delta; an item whose
   entire upper bound is under 1% of playout time has no delta to land
   with, and its premise (7% allocation share) was measured before items
   1-4 and no longer describes `Game::characteristics`. The PR body keeps
   this as deferred evidence. Jack Heart can reopen it; mana payment search
   is a separate Task if the quiet-machine bench misses acceptance.

Acceptance (unchanged): `bench_search` worst reply <= 4.34 s and total
<= 128.7 s, seeds 1-3, 40 decisions, same score digests as `main`; debug and
release `cargo test` green; no rules change (`WORLDS.md` untouched).

## Concept review (2026-11-02, headless, against the brief and `d0b86ce63`)

Accepted experience, unchanged: a Search-64 reply as fast as before T3 with
the same play (identical score digests). Nothing a player, policy or viewer
sees changes; the concepts this Task adds are for the engine's readers.

Model as implemented, and what each name must mean:

- `Revisions { characteristics, mana_pools, triggers }`: one counter per
  derivation, named for what reads it, not for what writes it. `DerefMut`
  advances all three; a typed accessor advances the subset its field feeds.
  The names match the brief's "revisions by what the layers read";
  `mana_pools` for the brief's `offers` is recorded in `questions.md`.
- `DerivedCache`: everything computed from state and keyed on `Revisions`
  (static index, views, trigger index), cloned with the game as immutable
  `Arc`s. One cache, one keying rule.
- `EffectRecords`, `TriggerIndex`, `ObjectLkiLog`: indexes over records the
  state already owned; they add lookups, not new facts. The canonical hash
  still serializes the ordered records and the LKI log in object order.

What settles the contract "a write a layer reads advances its counter":

- The permanent fields behind typed accessors (tapped, damage, deathtouch
  mark, summoning sickness, exhaust and once-per-turn limits, dealt-damage
  flag) are read only by rules questions (`permanent_can_tap`,
  `has_lethal_damage`), offers, and source/frame conditions
  (`SourceTapped`, `TargetDealtDamageThisTurn`). `check_static_condition`
  debug-asserts that no static carries a source or resolution-only
  condition, so no static can read them. Verified by reading
  `flow/statics.rs` at this head.
- Pool reads by statics are discovered from the sources present
  (`StaticIndex::reads_mana_pools`, from `Condition::reads_mana_pools`),
  not declared by hand. A new condition that reads another untracked input
  must extend the same predicate; it is a one-input form of "which extra
  inputs do the present sources read", to generalize only when a second
  input appears.
- Debug builds recompute every cached value on read and assert equality,
  so a misclassified accessor fails `cargo test` in debug (CI).

Settled implications (no code change in this Task):

- Item 5 is dropped (above). It is an implementation means whose expected
  delta the brief's own standard cannot land; not a product choice.
- No rename or merge is needed among the added types. The three-counter
  struct is the smallest form that keeps tapping and mana payment off the
  layer cache; folding `mana_pools` into `characteristics` would flush the
  views on every mana tap, and a single counter was the problem.

Follow-on direction, outside this Task (not in the brief, not in the
profile; record for a separate Task if wanted): the producible-mana cache
(`mana_cache`, `cached_producible_mana`, `invalidate_mana_cache`) is the one
remaining hand-invalidated derivation, with about 25 invalidation sites
across `zones.rs`, `decision.rs`, `effects.rs`, `resolution.rs`, `sba.rs`,
`search.rs`, `proposed_event.rs` and `scenario.rs`. It reads availability
(`permanent_can_tap`: tapped, summoning sickness) and `abilities_lost`, so
it cannot join `DerivedCache` without an availability counter. Adding one
would let it key on `Revisions`, delete the manual sites and gain the same
debug recompute-and-assert. Weigh against the bench before doing it; it is
not needed for acceptance.

Unresolved product choices: none. What remains is delivery (below).

## Remaining work

- **Quiet-machine `bench_search`** (`cargo run --release --example
  bench_search --manifest-path managym/Cargo.toml`), base and branch. Every
  run here was made with load averages of 12-70 from other sessions, so the
  absolute thresholds are unverified; the recorded CPU-second delta is the
  evidence so far. Never induce load; a CI runner or an idle laptop is fine.
- Published 2026-11-02 as PR #411
  (https://github.com/loopflowstudio/manabot/pull/411), ready for review,
  with the aggregate CPU-second delta and the unverified quiet-machine
  threshold in the body. Items 1-4 landed in one commit (`884090f1b`) plus a
  fold (`86af3387f`), so only the aggregate delta exists; per-item deltas
  would need four separate builds. Load was 21-30 at publication, so the
  quiet-machine bench is still outstanding. Not merged; merge and Task
  completion are separate steps.
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

## Local PR walkthrough (2026-11-02, continued primary conversation)

- `scratch/pr-review.html` reviews published base `4c8324a59516b9738fe55f925ea35ecbb28e369c`
  to head `06472d4c89ad6317d0bc86f54d53f63669958a47`, with pinned source
  excerpts for revision invalidation, payment/error continuation, cache-carrying
  clones, effect indexes and trigger-source filtering.
- Rendered in headless Chromium at 1440×1000 and 390×844; inspected the opening
  and code sections, tested a disclosure, no page overflow or script errors.
- Fresh GitHub read: all 17 executed checks succeeded, package publication
  skipped; PR #411 open and mergeable, no auto-merge. Task is at interactive
  `demo`; its completed code Graph has no live process in `lf ps`.
- Performance figures remain prior-run evidence, not fresh acceptance. The
  quiet-machine 4.34 s / 128.7 s thresholds are unverified. No benchmark run,
  scope approval, merge authorization or Task completion occurred in this pass.
- Earlier headless notes call subtype deferral settled; the conversation
  contains no subsequent acceptance from Jack Heart. The walkthrough labels
  the deferral as the implementation decision and preserves this distinction.

## Sync after status return (2026-11-02)

Fresh PR status became conflicting after main reached `a8abf01b1` (T5 token
copies, following decklist disclosure). `lf sync --manual` and `--continue`
created merge `e088c99b6`. Kept main's `CreateTokenCopy` resolution instead of
restoring the obsolete `CreateNonlegendaryCopy` implementation; retained the
branch's typed accessors. Kept main's compressed questions plus MTG-236 notes.

Focused debug proof (`CARGO_PROFILE_DEV_OPT_LEVEL=1`, two Cargo jobs): four
copy/observability tests and four Joo Dee tests passed. Rebuilt the release
Python extension and replaced it atomically. Authored-match parity passed all
three tests after refreshing only its relevant-source manifest.

The initial combined parity run had five passes and two public-commitment
receipt failures. Inspection found only source-manifest differences and the
six prior/posterior/history digests (duplicated in the headless summary).
These bind the content manifest, which T5 changed; main also updated the native
belief-binding expectations for that reason. No command, state witness,
commitment, support size, viewer-isolation assertion, atomicity proof or frozen
evidence differed. Refreshed the current open-w17 receipt from the producer,
asserting equality after removing only those source/digest fields; retained
frozen RUL-9 and INT-12 references unchanged. Focused rerun follows.

The rerun passed: four public-commitment tests plus main's native belief
projection receipt test (5 passed in 33.28 s). Together with the three authored
parity tests and eight directed debug Rust tests, the focused sync proof is
green. Full post-sync CI and quiet-machine latency acceptance remain separate.

## Shipping approval (2026-11-02)

Jack Heart reviewed the current result and said "lg2m ship it" in the primary
Task conversation. Ship PR #411 through the remaining code Graph. The current
synced head is `d2136d4ad`; focused sync proof is recorded above and CI is
running. Preserve the quiet-machine 4.34 s / 128.7 s acceptance gap explicitly
in the durable PR description: shipping approval is not a latency measurement.
Subtype interning remains deferred in the reviewed implementation.

Before landing, refresh the PR copy's stale statements: T5 PR #405 is integrated
at main `a8abf01b1`; the public-commitment receipt changed content-bound tracker
digests as well as source hashes after that sync. All state/command/privacy
proofs and frozen evidence references remained identical. The original 31%
CPU measurement predates this sync and is not a new measurement of this head.

## Demo decision pass 1 — 2026-11-02 (headless)

Experience reviewed: Academy's UR Lessons versus GW Allies match through
the real `GameSession` server, headless engine, persisted replay and public
commitment/hypothesis paths, with the derived-state implementation at
`d2136d4ada56ffb06e98d0d52146b2ccc7433e98`.
The design is the Outcome and Concept review sections above; implementation
is [PR #411](https://github.com/loopflowstudio/manabot/pull/411).

**Decision: advance.** The preceding implementation intended to avoid
unnecessary recomputation without changing play. Its retained evidence shows
31% less user CPU time and matching benchmark digests before the T5 sync.
New evidence in this pass establishes the synced implementation still
reproduces the configured match and viewer contract. No concrete repairable
implementation defect surfaced that warrants returning to implement.

Fresh demonstration command (completed, exit 0):

```sh
uv run --extra dev --extra play pytest tests/academy/test_authored_match_parity.py tests/academy/test_public_commitment_parity.py -q
```

Observed: **7 passed in 31.07 s**. These tests execute the receipt producers,
including FastAPI `TestClient` against the real Academy app; they do not merely
read retained JSON. No receipt was rewritten in this demo.

- Live, headless and replay agree on 142 commands, 143 checkpoints and 142
  ordered transition groups per surface, with 858 viewer projection checks
  and no first divergence. Both player replay projections are admitted;
  spectator projection is refused.
- Object `96@3` is rejected as stale, and a retained command is rejected for
  stale revision, preserving state witnesses and semantic event cursors.
- Public commitment proof reproduces 69 commitments, 55 consumed commitments,
  and zero provider gaps, identity-stream mismatches or negative-proof
  mutations. Materializing a hypothesis with support size 484 preserves the
  source match, event cursor and viewer observation. Frozen RUL-9 and INT-12
  references remain unchanged.
- Supporting evidence:
  [authored-match receipt](../conformance/authored-match-parity-w17-v1/release-live-headless-replay-seed-0.json),
  [public-commitment receipt](../conformance/public-commitment-parity-w17-v1/release-live-replay-hypothesis-seed-0.json),
  [authored producer](../academy/authored_match_parity.py), and
  [commitment producer](../academy/public_commitment_parity.py).
  The latter normalizes the relevant-source identity only for comparison;
  behavioral fields remain strict. The authored receipt comparison is exact.

Fresh remote evidence at this head: [CI run 41549044249](https://github.com/loopflowstudio/manabot/actions/runs/41549044249)
has successful debug Rust tests (debug assertions retained), Academy UI,
Academy integration, protocol and semantic conformance, clean-machine Play,
Python unit tests and long integration. Main integration was still running
at the last check. Package builds/install checks also succeeded in
[run 41549044194](https://github.com/loopflowstudio/manabot/actions/runs/41549044194);
publication was skipped. PR #411 was open and mergeable. `WORLDS.md` has no
diff against checkout base `a8abf01b1`.

Human feedback: the existing Shipping approval section records Jack Heart's
exact statement, "lg2m ship it". No person participated in this headless
demo, and there is no new human observation or approval. Acting under the
requested stand-in authority, the reviewer interprets that explicit approval,
the preserved contract and the demonstrated CPU reduction as grounds to
advance delivery with the already disclosed measurement gap. No design
change was made or newly attributed to Jack Heart.

Limits and next action: host load averages were 28.00 / 27.59 / 21.94 before
the demonstration. The brief requires a quiet machine or CI runner for
`bench_search`, so it was not rerun here. The 4.34 s worst-reply / 128.7 s total
bound remains unverified, and the older CPU delta is not a measurement of
this synced head. Complete the remaining delivery checks and correct the PR
body's stale T5 overlap, source-only receipt-change and measurement-head
statements as already directed above. Retain the latency gap in that body;
do not claim the Task's performance acceptance is met. No further product
judgment is needed for this demo decision. No servers or other long-lived
processes were started; the test process exited.

Review record: `scratch/mtg-236-derived-state-revisions.md` (this section).
No `scratch/demo.html` was generated: this invocation is headless with no
rendering surface; the interactive runbook requirement does not apply.

## Gate plan — 2026-11-02

Compare against checkout base `a8abf01b1`, including the existing uncommitted
headless-demo notes. Changed paths are the 17 `flow` modules listed by the
Task snapshot, `search_state.rs`, four `state` modules, AGENTS.md and the two
current w17 parity receipts. The implementation changes cache invalidation,
effect/LKI indexes and fork reuse; it changes no UI, world, schema or Python API.

`.github/ci/select-tests.sh` selects all eight CI areas for the native sources
and receipts. Use those existing exact-head CI results for the full matrix;
there is no content-keyed local runner. Locally run focused debug and release
search-state contracts, continuous-effect/identity/trigger rule tests and the
Ozai mana-condition and Joo Dee copy cases. Add cache-sharing and invalidation
assertions to the existing fork identity regression: existing semantic tests
alone would also pass if clones silently started cold again. Then run required
rustfmt and clippy (`--all-targets --all-features`, pinned PyO3 interpreter),
serially with two Cargo jobs and CI's optimized debug assertions.

Retain the immediately preceding 7-test Python demo result: its native sources,
receipts and test inputs have not changed. CI owns the broad native suite and
browser/integration matrix; there is no new UI interaction to capture locally.
Prior full release results predate T5; the focused release run checks the merged
paths without repeating the entire package. Quiet-machine latency measurement
is deferred: current load averages are 19.21 / 24.21 / 21.43. No benchmark is
started on this busy host. The 4.34 s / 128.7 s bounds remain unverified.

## Gate results — 2026-11-02

Review found no implementation defect requiring a native-source change.
Extended `managym/tests/search_state_contract.rs` to assert that warmed views
survive cloning and event draining, a departed object's view is invalidated,
and fork invalidation preserves the parent's cached view and witness. These
assertions test the promised reuse without timing thresholds. Native sources,
world identity and receipt inputs are unchanged; no extension rebuild or
receipt regeneration is necessary for this test-only addition.

Commands below ran serially; all completed without failure or timeout:

- `CARGO_PROFILE_DEV_OPT_LEVEL=1 cargo test -j 2 --manifest-path managym/Cargo.toml --test search_state_contract` — 4 passed, including the strengthened fork regression.
- `CARGO_PROFILE_DEV_OPT_LEVEL=1 cargo test -j 2 --manifest-path managym/Cargo.toml --test rules_tests cr_613` — 6 passed (layers, expiry, hashing and fork restoration).
- `CARGO_PROFILE_DEV_OPT_LEVEL=1 cargo test -j 2 --manifest-path managym/Cargo.toml --test rules_tests cr_400` — 6 passed (identity/LKI and related stale-reference cases).
- `CARGO_PROFILE_DEV_OPT_LEVEL=1 cargo test -j 2 --manifest-path managym/Cargo.toml --test rules_tests cr_603` — 7 passed (ordering, matching and resolution).
- `CARGO_PROFILE_DEV_OPT_LEVEL=1 cargo test -j 2 --manifest-path managym/Cargo.toml --test cards ozai_keeps_unspent` — 1 passed (mana-conditioned characteristics).
- `CARGO_PROFILE_DEV_OPT_LEVEL=1 cargo test -j 2 --manifest-path managym/Cargo.toml --test cards joo_dee` — 4 passed (merged copy/decision behavior).
- `cargo test --release -j 2 --manifest-path managym/Cargo.toml --test search_state_contract` — 4 passed.
- `cargo test --release -j 2 --manifest-path managym/Cargo.toml --test rules_tests <filter>` — `rules::cr_613_continuous_effects`: 6 passed; `rules::cr_400_identity`: 3 passed; `rules::cr_603_triggers`: 6 passed (three sequential invocations).
- `cargo test --release -j 2 --manifest-path managym/Cargo.toml --test cards <filter>` — `ozai_keeps_unspent`: 1 passed; `joo_dee`: 4 passed (two sequential invocations).
- `cargo fmt --check --manifest-path managym/Cargo.toml` — passed.
- `CARGO_PROFILE_DEV_OPT_LEVEL=1 PYO3_PYTHON="$PWD/.venv/bin/python" cargo clippy -j 2 --manifest-path managym/Cargo.toml --all-targets --all-features -- -D warnings` — passed.
- `git diff --check` — passed.
- `gh pr view 411 --json headRefOid,statusCheckRollup` — at published `d2136d4ad`, 15 checks passed, package publication skipped, main integration still running; the aggregate has not reported yet. The unpublished test additions have local debug/release proof, not a new CI result.

The immediately preceding Python demo's 7 passes remain applicable to unchanged
native/Python/receipt inputs. CI owns the full debug and browser matrix. The
prior full release matrix predates the T5 merge; this pass adds focused release
coverage of the merged paths, not a new whole-package release certification.
The busy host precludes the brief's quiet-machine benchmark; no latency target
is marked passed and no new performance number is claimed.

Prepared corrected delivery copy in `scratch/pr-title.txt`,
`scratch/pr-body.md` and `scratch/.pr-copy-ref`. It distinguishes the historical
31% CPU improvement from the synced head, records T5 integration and the
content-bound receipt digest refresh, and retains the outstanding latency
acceptance. Publication should consume that copy and retain this evidence in
the PR before scratch cleanup. The remote PR body has not been edited here.

Gate can advance to the delivery step with these explicit limits. Landing
still requires successful CI, including any checks selected by publishing the
test addition. Shipping approval is already recorded above; Task performance
acceptance remains open. No background process remains from this gate.
