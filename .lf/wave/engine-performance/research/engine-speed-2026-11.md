# Engine speed investigation — 2026-11-03

Jack Heart requested a measure → plan → optimize loop for MTG-238 / PR #414,
then asked how to reach gains of 50% or more. Acceptance is either a shared
`Env::step` improvement, or a search improvement that is almost certainly
neutral for training/evaluation engine time. Jack accepted regular `Env::step`
as the training proxy; these measurements are not end-to-end training results.

## Workloads and evidence

The starting engine is `cf069a3bab91fb34d78350821151a3c587aca930`, including
main's MTG-235 vocabulary merge. The committed CI reference is a separate
identity from the local starting engine. Adoption below uses the improved
candidate's own hosted artifact and preserves the existing 10% tolerance.

Measurements use the unchanged release `bench_steps` and `bench_search`:

- Steps: 512 seeded complete games, 99,923 decisions, no capped games,
  digest `12247096877573042301`, native trivial-decision skipping enabled.
- Search: seeds 1–3, 40 replies each, 16 worlds × 4 rollouts per action.
  Ordered full-score digests are `5192957023818819312`,
  `14020561159006433540`, and `12455483057923843581`.

Local experiments save each release binary and compare steps serially in
ABBA order. No build or test runs alongside our timed workload. The shared
M4 Max host sometimes had unrelated activity; runs with unstable controls are
retained as negative measurement evidence, not attributed to code. Hosted
same-runner comparisons own delivery acceptance. Raw local evidence is under
`.runs/mtg-238-optimize/` in the Task checkout.

The final local search candidate took 96.252s total versus 148.454s for the
starting engine, with worst reply 2.923s versus 4.800s. All score digests match.
This serial local comparison is supporting evidence, not a hosted result.

Hosted [CI run 41699683808](https://github.com/loopflowstudio/manabot/actions/runs/41699683808)
measured optimized commit `1616b210f7f4765aa8032debad50b7f18450e8b1` and
reference `4b5e9c0e4` on one AMD EPYC 7763 runner:

| Metric | Reference | Optimized | Less time |
|---|---:|---:|---:|
| 99,923 regular Env steps | 11.166 s | 7.768 s | 30.43% |
| Total Search-64 replies | 317.027 s | 177.771 s | 43.93% |
| Worst search reply | 10.526 s | 5.272 s | 49.91% |

That is 43.74% more regular steps/s and 78.33% more search work/s, with all
counts and digests exact. The repository adopts this candidate's own artifact
as the new performance reference so future regressions cannot consume the gain.
The 10% tolerance is unchanged. The hosted reference differs from the local
starting engine; do not combine ratios across those comparisons.

Repeated combined step comparisons measured 35–38% less time than the starting
engine. The small remaining differences were below reliable attribution on this
shared host, so the local loop stopped at that measurement/design boundary.
This does not mean that no engine optimization remains.

## Correctness validation

The final source passes debug and release Rust suites (893 tests, 2 ignored in
each), formatting and clippy. Debug retains both cache recomputation and the
full-fork waterbend oracle. The release CPython 3.12 extension was rebuilt and
atomically installed. Definition JSON (307,652 bytes), learning catalog
(449,297 bytes) and world identity (107 bytes) are byte-identical to the starting
extension. Authored parity preserves 142 Commands, 143 checkpoints and 858
viewer checks; only source fingerprints change. Public parity preserves 69
commitments with no gaps. Frozen game evidence, schemas and tolerances remain
unchanged. The broad Python consumer suite passes 945 tests with one skipped;
the performance-budget tests pass 17 tests after reference adoption.

## Retained changes

The useful changes remove repeated work in the shared engine:

1. Cache structural land-type setter candidates per characteristics revision.
   Read their conditions and ability loss live; keep this cache independent
   of the static index to avoid recursive construction. Debug recomputation
   and a Zhao entry/counter/ability-loss/fork/departure scenario check it.
2. Project card-type observation flags in one traversal. All 512 type subsets
   are checked against the rules query API.
3. Use a BTreeMap for the sparse private effect-record index. The ordered
   record vector remains the serialization and state-hash authority.
4. Borrow battlefield traversal in read-only queries instead of allocating
   intermediate vectors. Mutating callers retain owned snapshots.
5. Compute priority actions once, including the sole-Pass test used by native
   trivial-decision skipping.
6. Reserve hypothetical waterbend taps in the payment model instead of cloning
   the entire game and its event history. Every affected debug query compares
   its answer with the original clone → tap → payment calculation.
7. Reuse a completed stabilization result across automatic priority passes
   within one `tick_priority` invocation. Stack resolution invalidates it;
   every new invocation starts without a proof. No cross-action cache is added.
8. Check terminal payment success and the admissible resource bound before
   allocating/hashing a visited-state key. Explored states retain cycle
   detection and the same transition order.
9. Reuse the already-established absence of legend conflicts for the same SBA
   batch. Explicit legend choices still compute their current loser list.

Two proposals were discarded: omitting the disabled profiler's clock read did
not give a convincing repeatable gain; SmallVec's union feature reduced
Counters from 32 to 24 bytes but its apparent speedup reversed on repetition.
Neither change remains in production.

## The next architectural boundary

A bounded search sample after the first four changes attributed approximately
30% to state-based checks, 24% to payment work and 31% to static layers. These
regions overlap and cannot be added. Allocation/freeing appeared throughout
about a third of sampled execution. Later changes remove some of that work;
these percentages are not final-engine attribution or promised savings.

Search already uses `Game::step` directly and avoids observation construction
during random rollouts. About 99% of that sample was inside random playout;
outer branch setup was not the dominant cost. Internal legality probes were
a distinct copying cost and have been addressed above.

Further large gains need explicit dependency/ownership designs:

- Reuse a payment source inventory across questions at one decision, with
  separate reservations and obligations. Its inputs include tapping, activation
  usage, life, mana, turn, characteristics and pending/suspended payments.
  `PendingChoice` is outside `TrackedState`; the characteristics revision alone
  is not a valid cache key.
- Preserve stabilization proofs across committed actions only with a revision
  covering life, empty-library draws, damage, zones, attachments, counters,
  legendary characteristics and trigger placement. The existing characteristics
  revision intentionally excludes several of these inputs.
- Attribute characteristic-cache misses and observation allocations before
  changing invalidation granularity or buffer ownership. Preserve one layered
  authority and its debug recomputation oracle.

Independent rollout parallelism could reduce reply latency by spending more
cores, but is not reduced per-game CPU cost or a training-throughput claim.
Prior clone/undo/COW experiments did not justify replacing compact full clone;
a new proposal needs new evidence. None of these broader designs is implemented
or accepted by this investigation.

Use explicit units when discussing a large target: 50% higher throughput means
33% less time; twice the throughput means 50% less time. A credible 2× design
must eliminate enough non-overlapping work to support that target. Passing the
budget, or summing overlapping profile percentages, does not establish it.
