# Engine speed: measure → plan → optimize

Jack Heart requested this loop on 2026-11-03, continuing MTG-238 / PR #414 until
we hit a wall. Run here without another Graph or worktree. Publish verified
improvements for review; no landing approval is implied.

Start: cf069a3ba, integrated main 22e4aab71. Hosted CI 41683698086 and packaging
41683698087 passed, including Engine Budget. This is the loop's local reference,
not a replacement for the committed CI reference 4b5e9c0e4.

Each iteration:
1. Measure the unchanged release bench_steps (512 seeded games) and inspect a
   sampling profile of that actual workload. Save binaries and source patch.
2. Write a specific hotspot, hypothesis and smallest semantics-preserving change.
3. Build once; compare saved binaries serially in ABBA order, requiring identical
   counts and digests. No build/test runs alongside timed workloads. Discard
   noisy/non-improving candidates, retaining their evidence.
4. Confirm useful candidates with the full three-seed Search-64 workload and
   ordered-score digests. Add directed semantic tests where the change affects
   cache ownership/invalidation; retain debug recomputation guards.
5. Re-profile the improved engine and repeat. Finish with full affected checks,
   rebuilt extension, source-sensitive replay proof and published current-head CI.

A wall means the remaining profiled costs have no repeatable useful reduction
from tested, scoped changes, or require an architectural/semantic decision beyond
this optimization. Passing the 10% budget alone is not a wall. Report negative
results and the remaining costs; never label local noise a speedup or weaken a
correctness check/budget. No stress load or artificial pressure.

Evidence: `.runs/mtg-238-optimize/`; serial comparisons via
`uv run --no-sync python -I .runs/mtg-238-optimize/compare.py <before> <after>`.

## Iterations

- 01 plan: sample-00 places the land_type_overwrite battlefield scan among the
  largest application frames. Cache only structural setter locations per
  characteristics revision, independently of StaticIndex to avoid recursive
  index construction. Conditions, timestamps and ability loss remain live reads.

- 01 step result: 5.194s → 4.896s mean (-5.74%); both candidate runs faster
  than both controls, all counts/digests exact. Full search comparison running.
  Added a Zhao scenario covering cached absence, entry, counter threshold,
  ability loss in a clone, original independence, counter removal and departure.

- 02 plan: CardTypeData::from_types is another sampled hot frame. Traverse the
  type set once instead of repeated membership searches; verify all 512 subsets
  against the existing rules query API. This changes only projection computation.

- 01 search timing limit: while search-01 ran, a separate Loopflow checkout's
  test process used ~758% CPU and rustc ~520% on this shared host. Candidate
  seed 1 took 67.885s versus control 63.792s, inconsistent with step pairs and
  not attributable to this patch. Retain the run for exact score correctness;
  do not count its timings as a speed result. Dedicated hosted CI remains
  authoritative; do not stop or change other work to manufacture quiet time.

- 03 plan: ObjectRef hashing is prominent in effect lookups. Compare a BTreeMap
  for the private object-to-record-index map, retaining the ordered record vector
  as serialization/hash authority and exact incarnation keys. This targets the
  sparse active-effect workload without changing the public effect API.

- 02 result: 4.8325s → 4.7705s (-1.28%), each candidate faster than each
  control; exact decisions. Exhaustive 512-subset type projection test passed.
- 04 plan: payment-model construction repeatedly allocates an intermediate
  battlefield-id vector. Add a borrowed traversal sharing the same primitive,
  use it for the three read-only mana queries, retain snapshots for mutation.

- 03 result: 4.747s → 4.4945s (-5.32%); separated ABBA ranges and exact
  decisions. The index remains private, record ordering/serialization unchanged.
- 05 plan: disabled Profiler::track still calls Instant::now per environment
  step. Test making its start optional; expect a small gain at most. Keep only
  a repeatable improvement, not a speculative cleanup.

- 04 result: 4.6075s → 4.5555s (-1.13%), separated ABBA ranges and exact
  decisions. Retain the three read-only mana traversals; mutating callers stay
  on the same owned snapshot API.
- 06 plan: the installed smallvec 1.16.4 docs describe its stable union feature
  removing one word of tag overhead. Counters currently measures 32 bytes and
  Permanent 144. Test the existing sparse-counter implementation with that
  feature, retaining API/serialization and requiring measured workload benefit.

- 05 first comparison is invalid for timing: control ranged 7.153s–5.102s;
  another checkout's test used ~893% CPU alongside two rustc processes. The
  apparent 21% gain cannot come from removing one clock read per Env::step.
  Keep all raw results, restore the profiler implementation pending a quiet
  comparison, and do not count this as an optimization.

Jack Heart asked to think beyond 1% changes toward 50%+ gains. Extend attribution
from individual functions to work boundaries: search rollout versus observation,
legal/payment queries, characteristic invalidation, and branch copying. A target
is a hypothesis, not a forecast; identify enough measured cost before redesign.
The existing Search-64 path already calls Game::step/random_playout directly and
does not build Observation each rollout step, so removing observations cannot be
claimed as a new search optimization. Training Env::step is a separate consumer.

- 05 repeat: 4.6495s → 4.6135s (-0.77%), but one control is within 0.2% of
  candidates. Not a convincing useful gain; discard the profiler change.
- 06 repeat: 4.6045s → 4.6495s (+0.98%). The earlier -1.55% did not repeat;
  discard union feature despite the real 32→24-byte Counters layout reduction.
- Combined 00→04 ABBA: 5.1925s → 4.7025s (-9.44%), all counts/digests exact.
  Do not add the individual iteration percentages. Full search proof follows.

- 04 full Search-64: 135.075s summed, 4.320s worst, all three ordered-score
  digests exact. Earlier 00 control was 150.644s; that is an unpaired local
  comparison, so hosted comparison still owns acceptance.
- Profile after 04: a bounded 10s seed-1 Search-64 sample contains ~29.6% SBA,
  24.1% payment, 30.7% static layers, 33.6% allocator ancestry. These overlap,
  are approximate sampled attribution and cannot be added. Observation is 0%.
  ~99.3% of samples are inside random_playout; outer search cloning is not the
  dominant cost. Game clones also occur *inside* legality/payment probes.
- 07 plan: priority first calls can_player_act, then compute_player_actions,
  repeating successful legality queries. Compute the complete action list once
  and use its sole Pass action to recognize trivial priority. Same order, native
  skipping and decisions; no cache/invalidation expansion required.
- 07 result: 4.5705s → 4.3705s (-4.38%); both candidates faster than both
  controls, all decisions/counts exact. Retain single priority action calculation.
- 08 plan: sampled Game::clone occurs inside waterbend legality probes and
  copies accumulated event history. Represent hypothetical non-mana taps as
  PaymentModel source reservations instead. Keep a debug differential oracle
  against clone→tap→payment for every affected query, so future dependencies
  cannot silently invalidate the optimization. Cover activation, pending cast
  and suspended resolution payment paths; preserve sacrifice-only abilities.

## Larger design direction (investigation, not an accepted rewrite)

Use 2× throughput / 50% less time as the ambitious target, distinguishing it
from 50% higher throughput (33% less time). Neither is a forecast.

1. Incremental stabilization: record a proof that SBAs reached a fixed point and
   invalidate it only on relevant committed changes. Currently stabilization
   resets sba_done at entry and performs full scans. A revision must include
   life, empty-library draws, damage, zones, attachments, counters, legendary
   characteristics and trigger placement; the existing characteristics revision
   deliberately excludes several of these and is not a sufficient key. Begin
   with counters attributing unchanged scans, then a reference/differential path.
2. Reuse payment facts across legality questions at one decision. Separate source
   inventory from reservations and search obligations. Its identity includes
   tapped state, activation usage, life, pools, current turn, characteristic
   dependencies and pending/suspended payments. PendingChoice is outside
   TrackedState; an unqualified state-revision cache would be incorrect.
3. Narrow characteristic invalidation and reuse scratch storage only after
   measuring miss causes and allocation sites. Preserve one layered authority;
   use current debug recomputation guards as the correctness oracle. Copy effects,
   global buffs, controller changes and static-source changes need broader
   invalidation than an individual permanent.
4. For reply latency, independent rollout parallelism is a separate lever: it
   spends more cores, does not reduce per-game CPU work, and must preserve RNG
   streams and deterministic score reduction. It is not a single-core engine
   speedup or evidence of training throughput gains under saturation.

Reject speculative GPU/whole-engine rewrite and clone→undo as the first move:
this profile puts ~99% inside actual playout, not outer branch setup. Prior
compact-clone experiments also failed to justify undo/COW; new adoption needs
new workload evidence. Internal legality probes are a distinct clone cost.

The current sampled percentages overlap. Even eliminating a 24% payment region
cannot alone halve total runtime. A credible 2× proposal must identify a broad
non-overlapping critical-path share and remove most of it, or combine independent
reductions. Measure every architectural prototype against unchanged full-score
and replay oracles, on the same machine, before accepting it.

- 08 focused debug proof: 23 waterbend tests passed with the full-fork oracle
  active (1 engine unit, 2 rules, 20 cards). Release ABBA 07→08 measures
  4.3195s → 3.835s (-11.22%), all decisions exact.
- Combined 00→08 ABBA: 5.0855s → 3.8325s, 24.64% less time per step,
  approximately 32.7% higher throughput. This is a direct comparison, not
  accumulated percentages. Full Search-64 before/after is running serially.
- 09 plan: a bounded version of incremental stabilization needs no new state
  revision. Within one tick_priority invocation, an automatic pass mutates only
  PriorityState holder/pass count. Reuse that invocation's completed fixed-point
  proof across such passes; reset after any stack resolution, and start unknown
  on every new invocation. Do not cache across external actions or other writes.
  This targets repeated unchanged-board scans while preserving the existing full
  authority whenever a rules transition occurs. Measure after 08's search ends.

Jack Heart's acceptance constraint (2026-11-03): do not make training or regular
evaluation slower to make search faster. Report native steps, full search,
training and ordinary evaluation separately. Native Env::step throughput is
necessary but is not end-to-end training/evaluation proof. Retain the baseline
extension and compare the same bounded consumer configurations before/after;
use fresh subprocesses and atomic extension replacement, never overwrite a mapped
binary. Revise/drop a candidate that regresses those consumers.

Jack Heart clarified that regular Env::step is an acceptable training proxy.
Use the unchanged native step workload for that acceptance; no end-to-end PPO
benchmark is required. Keep search and regular evaluation measurements separate.

Final acceptance clarification from Jack Heart: retain either (1) a search win
that is almost certainly neutral on training/evaluation engine time, or (2) a
pure Env::step win. No separate evaluation benchmark is needed for clearly shared
Env::step improvements. Investigate a separate consumer only for a plausible
tradeoff; do not expand validation into an unrelated training benchmark project.
- 08 full search: 148.454s → 102.349s total (-31.06%); worst reply
  4.800s → 2.975s (-38.02%). All full-score digests/counts match. Shared host
  still had unrelated compiler activity; dedicated CI remains timing authority.
- 09 result: 3.925s → 3.586s (-8.64%), separated ABBA ranges and exact steps.
- 10 plan: PaymentModel::witness clones/hashes a visited key before terminal
  success or the admissible resource bound. Evaluate those read-only exits first;
  memoize only states that need exploration. Preserve transition ordering and
  cycle detection for every explored state. This targets sampled payment
  allocation/hash traffic, not a different payment strategy.
- 10 result: 3.638s → 3.5375s (-2.76%), separated ABBA ranges, exact steps.
- 11 plan: the ordinary SBA path discovers legend groups, finds none, then
  rediscovers the same groups before committing the batch. Pass the known empty
  loser list directly to the shared batch implementation. Explicit legend choices
  still rediscover their current losers. Borrow battlefield traversal while
  grouping legends. Terminal precedence and simultaneous commit order stay intact.
- 11 result: 3.509s → 3.4245s (-2.41%), separated ABBA ranges, exact steps.
- 12 plan: the new profile still shows battlefield-vector collection among the
  largest allocation sites. Extend the borrowed traversal to read-only static
  predicates, mana bonuses, candidate filters and priority ability enumeration.
  Mutation loops keep owned snapshots. Measure this wider allocation removal;
  drop it if the remaining benefit does not repeat.
- 12 initial ABBA invalid for timing: 11 varied 5.646→4.846s, 12 varied
  8.098→6.714s, then the immediately following 12 runs took 3.322/3.280s.
  This cannot identify a code effect. Keep raw noisy data and repeat serially
  after full search; all counts/digests remain exact.

Jack Heart authorized shipping on 2026-11-03: "go for it when done." Finish
validation and hosted performance acceptance, then land and verify the merged
outcome. This supersedes the earlier review-only/no-merge boundary for this batch.
