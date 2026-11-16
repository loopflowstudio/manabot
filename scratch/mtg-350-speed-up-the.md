# MTG-350 — bounded mana payment work

## Current outcome (2026-11-15)

The reservation/life repair is preserved and integrated onto main
`2176882620`. A fresh sequential release comparison gives 12.80x counted-tap,
2.60x diverse-mana, 2.01x mixed and 3.24x restricted-filter, with matching
witnesses, outputs and closing digests. Mixed remains marginal; this is local
measurement, not Engine Budget CI. Earlier pre-integration comparisons passed
all four 2x targets in both orders. The retained experimental game advances
under the authorized isolated reservation/life overlay: its candidate-owned
Command returns in 0.81 ms. Implementation checks pass; gate/CI and delivery remain.

Jack Heart, 2026-11-15: "launch an auto task stacked on this to fix the yue bug?"
The Task comment separately requires "Keep legal actions, first witnesses and
the Engine Budget digests unchanged". No Wave is selected. History in git:
prior design, exact-replay contradiction, diagnosis and concept review at
`592f499546:scratch/mtg-350-speed-up-the.md`; every intermediate measurement,
allocation table, rejected-pair timing and concept review of this pass at
`02ccb36f7f:scratch/mtg-350-speed-up-the.md`.

## Contract and implementation

An offer must preserve completion of the announced payment and every saved
caller. `Payment` is the announced work; `PendingChoice::Waterbend` owns the
remaining cost and committed tap count. `Decision` / `SpendCompletion` own
nested callers. Reservations and ordered obligations are derived planning
inputs, never a parallel serialized payment lifecycle.

`payment_model` now reads the announced payment once for ManaPayment,
SacrificePayment and Waterbend. A shared internal helper reserves its source
and unpaid life fee; the same helper follows an announced payment held in
`SpendCompletion::Announced`, including nested Upfront/ManaChoice callers.
ManaUpfront life fees remain Output obligations, so they are not debited a
second time. Actual cost commits and caller restoration use the existing path.
No browser operation, automatic source preference, cancellation, fallback
action, policy layout, world identity or replay-schema change is introduced.
The README describes the search bounds, exact cache and fork ownership;
development docs describe the separate allocation diagnostic.

Directed proof: Yue owes her own tap and waterbend {5}. With one floating mana
and Prism plus three Ogres, tap two Ogres, leaving {3}. Activating Prism spends
one and returns one, taps Prism and leaves only one Ogre. The remaining pool
plus that Ogre pays two, not three: Yue's promised tap cannot be spent twice.
The repaired authority removes this infeasible offer. With two floating mana,
the same Prism activation remains legal and completes. Both continuations
retain every previous tap, consume the complete pool, leave all permanents in
play and reach a legitimate decision with native trivial skipping on. Commands
at intermediate payments verify exact forks, source immutability and stale
rejection. Empty-hand Yue resolves through the forced decisions.

A synthetic payment-model fixture isolates a two-life announced fee with
Sunbaked Canyon's one-life mana fee. At two life the nested activation is
infeasible; at four life it is payable. The announced fee survives Waterbend,
spending and a saved upfront caller. Execution and a fork pay Canyon once,
then the announcement once, leave one life and no pooled mana, and complete
without an empty pending decision. Existing nested Prism/Treasure and
sacrifice/suspended-resolution regressions also pass.

## Frozen evidence and experimental integration dependency

Original engine source: `fd7a7ed36c21553d419504daafeb36f70b3ab4ed`.
Study source: `3b519a3e94edd5570d240e606d636fb9d358fd96`.
Required flags before import: `MANAGYM_EXPERIMENT_COMBAT_PICK=0`
`MANAGYM_EXPERIMENT_SOURCE_TAPS=1`.

Retained stream 24 / seed 37037 / input 196 (Add {G}, actor 0, turn 25 precombat main)
previously reproduced on a CPython 3.13 rebuild of the unchanged failing source.
It matched the complete original frame, fingerprint
`9b85ba019f7270c05cea9aa92779b40915bfc3a4a6febdc113aa305933880b91`, in
0.045749 s after reset, then exceeded the approximately 58-second subprocess
bound and was killed/reaped. This was **changed-runtime reproduction**, not
the original private binary. Rebuilt extension SHA256:
`9458a2ae4652d8098d9439740aafc88713a80dec29f161bc9b2356f955ad14ec`.

The diagnostic found an empty Waterbend space, not a stuck witness: remaining
{3}, tapped 2, announced Yue permanent 21, pool G, no suspended decision. Native
trivial skipping repeatedly executed no action. A source-reservation-only
snapshot diagnostic first changed offer 193 from one mana activation plus two
waterbend taps to two waterbend taps. Its continued positional inputs after
that divergence were **not** a corrected continuation and prove no fix of the
original game. The snapshot is diagnostic only, never shipping candidate code.

The Task base lacks source taps and automatic fixed-source preference. Those
features remain outside the shipping diff. `scratch/landing-direction.md`
resolves the earlier prerequisite question: use the complete exact failing
snapshot for isolated diagnostics and overlay only this repair.

2026-11-15 retained-game proof: unchanged source rebuild matched all 197 frames
including the complete retained input-196 binding. The repaired snapshot
matched every frame through input 192. At input 193, actor 0's removed offer
is Rumble Arena (object entity 181/incarnation 2), ability 0, its {1}, {T}
any-color filter. Both Knowledge Seeker tap choices (155/3 and 163/3) survive.
No recorded numeric action is consumed after this divergence.

Resources at divergence: empty pool; untapped Forest 122, Mountain 133,
Rumble Arena 181 and Knowledge Seekers 155/163. Yue 153 owes her own tap;
all other permanents are tapped. These are exactly five units for waterbend
{5}. Arena's filter spends one, produces one and consumes Arena's fixed-mana
tap: at most four units remain for Yue. Her reserved tap cannot supply the
missing fifth. This proves the removed activation infeasible rather than a
preference change. Selecting the candidate's first Seeker through a fresh
revision-194 semantic Command completes the forced remaining payment and
reaches revision 195, Yue's legitimate free-cast choice, in 0.0008095 s.

Evidence: `scratch/yue-repair-proof.json` retains original/candidate frames,
object/ability identities, resources, candidate Command and receipt. It is
explicitly changed-runtime, preserving the original receipt/context bytes.
Original rebuild extension SHA256 `9458a2ae…ad14ec`; repair-only snapshot
extension `af23cebf5ddcff2a93b65def0c4b7dca0d459532dc84a3d0f05176bb11a98dc0`.
The exact overlay patch and full per-frame observations remain ignored under
`.lf/tmp/mtg350/`. This is an isolated proof of the shared repair, not shipping
experimental source-preference code. A production Rumble Arena regression
also verifies all five resources commit, no pool remains and native forced
skipping reaches a legitimate priority decision.

Original receipt SHA256 remains
`2a575ed1eba230fda450c2be7534182f6ba663922a35ecbeda9a963bf8d9c007`;
context remains `120d22c30608762ac8f01e11d44fc906dc0ba5a58179c011cb3b1075458bc387`.
No paid training, rental, active-job control, population change, deck exclusion
or merge of PRs #523/#524/#538/#540 belongs to this repair.

## Performance implementation

`PaymentCache` belongs to Game: 64 FIFO searched queries, copied by value on
clone, sharing only immutable Arc entries (negatives and complete first
witnesses). Equality covers ordered sources (identities, abilities,
restrictions, outputs, untapped state), payable life, pool classes, ordered
obligations, SourcePolicy and selected counted-tap identity. Debug hits
recompute the uncached query; arithmetic/direct plans bypass the cache. Source
lists, output alternatives and tap-candidate lists are shared immutable
payloads; reservations detach with Arc::make_mut only on actual change. The
cache is outside persistence, hashes, worlds, offers and policy inputs.

Search bounds: an all-class conservation bound (restricted seeds included,
gross output minus upfront fee per possible activation, optimistic) is kept
separate from the purpose-aware gross bound; the restricted-seed filter is the
counterexample to merging them. Per-cost color feasibility projects allowed
classes after per-class modal maxima; concrete pools and exact keys keep every
class. Automatic-prefix budgets start at the optimistic minimum; forced
zero-cost Pay/Tap/single-Output transitions advance before full-board bounds.
Ability-use flags are sparse (source, ability) pairs with the former canonical
order. Branch state is an immutable Arc reused as its memo key. Spending
branches are skipped when the pool cannot cover the whole current cost.
`production_alternatives` consumes owned modes (in-place dedup, first
occurrence kept; deterministic bonus mutates in place). Zero-use abilities
contribute no optimistic output.

Lessons kept: a deep-copy cache moved the zero-payment mixed block 18.066 →
46.156 ms; optimistic reachability closure barely pruned (34,306 → 33,825
restricted calls) and slowed diverse; inline resource vectors regressed counted
274.781 → 292.034 ms. All three were removed. The zero-payment control drifts
up to ~2x between runs with no established host cause, so decisions use
sequential matched pairs in both orders, deterministic witnesses and separate
`bench_states --allocations` traffic counts (requests, not live heap).

## Latest measurement (mixed production pass)

Release `benchmark` feature, `bench_states counted-tap diverse-mana mixed
--sizes 16 --layers search` then `restricted-filter --sizes 4 --layers search`,
sequential on one machine, no compiler/test runner; median of three workloads.
Order: before-pass, baseline, final.

| Required cell | Baseline ms | Before pass ms | Final ms | Baseline / final |
|---|---:|---:|---:|---:|
| counted-tap 16 cast-choice | 1015.588 | 275.857 | 289.905 | 3.50x |
| diverse-mana 16 cast-choice | 382.944 | 147.047 | 140.889 | 2.72x |
| mixed 16 cast-choice | 278.208 | 138.678 | 137.064 | 2.03x |
| restricted-filter 4 priority | 106.920 | 40.326 | 36.685 | 2.91x |

Zero-payment mixed block 15.968 / 14.035 / 16.026 ms. The opposite order gave
2.37x/2.23x/2.28x/3.35x with control drift 17.004/21.856/18.389 ms. Within-pass
timing gains are small (counted 5% slower); retention also rests on mixed
allocation traffic 3,590,140 → 2,436,070 requests. All 14 checkpoint first
witnesses, plan counts, non-payment counters and simulation outputs match the
baseline; closing stress digests remain size-16
`ee476afb928628d61f01469a7de6fe2eefa95121bdd1006ca1a270c3faa1abb3` and
restricted-4 `adecfb769e039154e870dc7d36956035cb8c772ff25058ea5718ef70f6d43ecb`.
Raw rows are ignored `.lf/tmp/mtg350/production-*.jsonl`.

Checks on final code: debug `flow::mana` lib tests — 36 passed; cards
`yue_reserves_her_tap` — 1 passed; cards `heartbeat` — 7 passed;
`tests/model/test_record_coverage.py` — 4 passed. Root cp313 release wheel
rebuilt, extension replaced atomically, native import passed (SHA256
`7afe6f46a8efbd9ee0d032042461ad747f418a0a052da37406a64dd0b8933480`). Workspace
fmt check and `git diff --check` passed. Before repair both new directed tests
failed on old production code. `tests/training/test_game_errors.py`'s three
`test_resource_limit_receipt_replays_its_retained_encoding_boundary` variants
fail on unchanged base too (`binding_mismatch`); main's `452d7b5357` resolves
it on rebase. No binding check was changed.

## Current-main integration measurement

Main baseline `2176882620` exported into an isolated source directory; candidate
includes only this Task's net repair/performance diff. Release `benchmark`
`bench_states`, same four cells/arguments as above, sequential baseline then
candidate, no compiler or test runner. Complete first witnesses, plan counts,
non-payment counters and simulation outputs match at every checkpoint.

| Required cell | Main baseline ms | Candidate ms | Baseline / candidate |
|---|---:|---:|---:|
| counted-tap 16 cast-choice | 3540.790 | 276.582 | 12.80x |
| diverse-mana 16 cast-choice | 346.694 | 133.286 | 2.60x |
| mixed 16 cast-choice | 258.851 | 128.626 | 2.01x |
| restricted-filter 4 priority | 108.592 | 33.500 | 3.24x |

Closing digests match between this main and candidate: size-16
`ee476afb928628d61f01469a7de6fe2eefa95121bdd1006ca1a270c3faa1abb3`;
restricted-4 `ce8fcd164f2c4e5e918af37e56b6d79753267a3c024f4d5739d87611e798739b`.
The latter differs from the older baseline above across main's intervening
changes; it is equal in this same-revision comparison. Engine Budget's reference
has not moved. Raw rows/comparison are ignored `.lf/tmp/mtg350/main-*`.

## Remaining work

### Concept review — 2026-11-15

Outcome: nothing to change and no open product choice. Advance to the next
Graph step; no additional implement direction. Reviewed the current patch
against the retained-game proof and the existing README/QA instructions;
this review ran no new games or timing measurements.

The experience remains one announcement with nested payment choices. The
largest mismatch was an offered filter that spent a resource already owed
to that announcement, leaving no completion. Reservation at the shared native
authority fixes that mismatch without teaching people a reservation operation.
Payment and saved callers own the lifecycle; reservations, search obligations
and cached witnesses remain derived. The existing usage docs already express
this model, so no further documentation or skill redesign is needed.

Normal path: Yue's nearby funded Prism choice remains available and completes.
Failure path: the underfunded filter is absent before resources are committed
to it. Recovery retains committed taps and the actual pending decision;
stale Commands are rejected and exact forks preserve that decision while
cache mutations stay independent. This is prevention, not recovery of an
already stranded historical frame: the retained changed-runtime proof starts
at the first changed offer and uses a fresh candidate-owned Command. Original
evidence is never rebound or rewritten.

Counterexamples remain decisive: one more floating mana makes Prism legal;
restricted mana can seed a filter even when the parent cannot spend it;
selected tap identity can change feasibility; an upfront life fee must not
be charged again as an announced fee. None permits merging resource classes,
dropping caller obligations or turning the cache into persisted authority.
No API, persistence, browser or policy-input migration follows from this
review. Next show gate/Engine Budget results and the focused delivery copy,
including changed-runtime provenance and the marginal local mixed speedup.
The one-screen review is `scratch/concept-review.html`.

Implementation and focused checks are complete on main `2176882620`. The
retained-game proof and four-cell witness parity pass. Current-main timing is
one sequential matched pair; earlier pre-integration timing passed in both
orders. Mixed is close to the local 2x target, so retain that limitation in
review. Exact keys, the 64-entry FIFO, independent branch mutations and complete
first witnesses remain required; no rejected optimization should be revived.

Gate owns CI, broad debug nextest/clippy and Engine Budget, then delivery can
publish and land this independent repair. PR #523 is no prerequisite. Never
move the Engine Budget reference to disguise a changed digest. Retain the
repair-only snapshot provenance and candidate-owned continuation in PR copy
before scratch is cleared; no experimental source-preference feature ships.

## Delete — do not maintain

Main removed the parent report's opt-in `mana/outcomes.rs`,
`PaymentShapeCapture`, capture hooks and `is_surplus`. Keep those deletions
while integrating the performance work; remove the exclusive `into_uncached`
helper as well. This Task ships no reporting API or source-preference feature.
Synthetic source ids use main's `payment_object` allocator; live fixture writes
use `state[object]` so appearance identity is validated.

Removed the one-off `payment_cache_profile` test and its exclusive
`CachedPayment::clone`: allocation diagnostics belong to `bench_states
--allocations`, while cache retention, exact equality and fork independence
remain covered by directed tests. Removed the temporary reservation list in
`Game::payment_model`; each caller applies its reservation directly to the one
model (reservations only remove capabilities, so their order cannot change the
result). The current-cost optimistic bound lives in `cost_coverable`, separate
from branch traversal. `prefix_lower_bound` returns as soon as the shortfall is
covered, including a funded pool before allocating/sorting output counts.

Compress validation (2026-11-15): `CARGO_BUILD_JOBS=2 cargo nextest run
--manifest-path managym/Cargo.toml --profile ci -p managym -E
'test(flow::mana::)' --test-threads 2` — 36 passed. The same command with
`-p managym-cards -E 'test(yue_reserves_her_tap) | test(heartbeat)'` — 8 passed.
Release `bench_states` compared sequentially against the pre-compress binary:
all 14 search checkpoints have identical complete work counters, first
witnesses and outputs; both closing digests above match. Before/after ms:
counted cast 271.281/275.449, diverse cast 124.021/122.840, mixed cast
118.072/121.313, restricted priority 30.183/33.038 and restricted cast
15.403/16.829. These are a parity check with local timing variation, not a new
speedup claim. Raw rows: `.lf/tmp/mtg350/compress-{before,after}-{16,4}.jsonl`.
Restricted priority's slower sample prompted an opposite-order check:
before/after 32.183/30.481 ms, with identical work and digest (raw rows
`compress-reverse-{before,after}-4.jsonl`). No consistent regression was found.
Root cp313 wheel rebuilt and atomically installed; native import passed,
SHA256 `c9e3d2ae2d942ab257b7e2deefbb6e6d9018c835275b3d7fce062b386acae0ba`.
`uv run --extra dev pytest tests/model/test_record_coverage.py -q` — 4 passed.
Workspace fmt and `git diff --check` passed. After curation, `lf context`
reports scratch below both configured limits; no limit was raised.

No dormant deep-copy cache, reachability closure, inline resource vectors,
dense per-source use rows, duplicated announced-payment readers, late forced
branches, independent memo copies, doomed partial-spending copies, or owned
mode copies for absent/deterministic bonuses. Preserve one payer/model, shared
announced caller ownership, exact resource/fee semantics, forced-decision
skipping, receipts and replay binding authority. No parallel serialized
lifecycle, global cache, cutoff, fallback action, source-tap prerequisite
import, schema/data migration or unrelated source preference belongs here.

QA (2026-11-15, `scratch/qa.html`): 1,188 seeded random games (authored,
TLA pool, Yue pools) have identical per-decision digests on base `ddc337f764`
and this checkout; all 17 stress states keep closing digest `84379ad0…21b2` and
every witness. The `actions` layer had become a warm payment-cache hit from
preparation; the harness now clears payment answers there (benchmark feature
only) and its rows search again with base's call counts.

Implementation proof (2026-11-15): repair-only snapshot root cp313 release build
and retained-game comparison passed; debug nextest filter
`test(yue_rejects_rumble_arena_filter)` — 1 passed (0.016 s).

Current-main checks (2026-11-15): debug nextest `test(flow::mana::)` —
30 passed; cards `test(yue_reserves_her_tap) |
test(yue_rejects_rumble_arena_filter) | test(heartbeat)` — 9 passed. The
removed six outcome-capture tests belong to main's deleted parent-report path.

Final integration checks (2026-11-15): root cp313 release wheel rebuilt and
extension replaced atomically; import passed, SHA256
`1d95d76a879af46309635f0fe7b7a1f9fa86f541bd5f354189cfd6ede5f0d8dd`.
After syncing main's locked Etude migration dependency, pytest
`tests/training/test_game_errors.py -k
test_resource_limit_replay_rejects_old_schema_and_reproduces_current_boundary`
— 3 passed; `tests/model/test_record_coverage.py` — 4 passed. Main renamed the
receipt regression; the old filter selected no tests and is superseded by this
result. Workspace fmt and diff whitespace checks pass. Original receipt and
context hashes still match the retained values. Scratch remains below both
configured limits.
