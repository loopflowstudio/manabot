# W2-179: ContentPack local performance diagnostic

Measurement date: 2026-07-15.

## Scope

This is a local before/after diagnostic for the behavior-preserving
definition/state separation in W2-179. It is **not a comparable clone or
rollout baseline** and must not be used to select a branching representation.
W2-182 owns the canonical whole-rollout evidence contract, the later
clone/undo/page-COW comparison, and any selection decision.

The measured implementation keeps the fixed action system and current flat-MC
algorithm unchanged. It gives card definitions deterministic `CardDefId`
indices in a process-wide, schema-versioned `ContentPack`; match-local `Card`
records retain their object ID, owner, definition ID, legacy registry-key value,
and a shared definition handle.

## Environment

- Base revision for the before sample: `82bae3a15e6875dcb92f7037f01fbc66cce4d1bb`
- Host: Apple M4 Max, 128 GiB RAM, arm64
- OS: Darwin 25.0.0
- Rust: `rustc 1.96.1 (31fca3adb 2026-06-26)`
- Build: Cargo `--release`, single process/thread
- Command: `cd managym && cargo run --release --example measure_content_pack`
- Workload seed: 377 (`0x179`)
- Deck: 60-card interactive mirror per player (lands, creatures, Bolt,
  Counterspell, Recall, and Pyroclasm)

The executable uses a counting wrapper around Rust's system allocator. Byte
counts are requested allocation sizes, not live or RSS bytes. Timings below are
medians of three warm executable runs; deterministic allocation counts are from
the same runs.

## Non-comparability with W2-182

W2-182's canonical contract was reviewed after this diagnostic was captured.
That contract, rather than this file, owns comparable evidence. In particular:

- Fixtures/decks: this diagnostic uses one synthetic interactive mirror;
  W2-182 freezes `ur-gw-opening`, `ur-gw-interactive`, `ur-gw-heavy`, and
  `ur-gw-suspended` recipes and action tapes.
- Seeds: this diagnostic uses a small local seed set; W2-182 versions seed
  derivation across fixture, repeat, worker, actor, decision, world, root
  action, rollout, and ply.
- Warmup/counts: this diagnostic uses its own 200 clone warmups and three
  process runs; W2-182 requires one untimed warmup and at least five measured
  repeats whose root-decision work lasts at least two seconds.
- Loads: this diagnostic is single-process; W2-182 defines sequential and
  retained worker x actor x rollout cells from solo through CPU/teacher and
  retained stress loads.
- Reset/termination: this diagnostic resets ended step-throughput games and
  applies a 2,000-step rollout cap locally; W2-182 makes fixture construction,
  root decisions, cap hits, empty-action errors, timeouts, and partial failure
  behavior manifest/schema data.
- Timing: this diagnostic times its three local loops directly; W2-182 starts
  native timing only after fixture construction and warmup and separates fork,
  determinization, apply, projection, policy-stub, tail, process startup, and
  reporting boundaries.
- Memory: this diagnostic counts requested allocator calls/bytes and has no
  peak RSS result; W2-182 samples aggregate live worker RSS every 5 ms and
  requires absolute peak, post-warmup baseline, peak delta, and parent RSS.
- Provenance/raw data: this Markdown records basic host metadata and manual
  samples; W2-182 requires full hardware/tool/source metadata and keeps every
  repeat in an atomically written, verified raw JSON schema from which its
  Markdown throughput/RSS summaries are derived.

After W2-179 lands, W2-182 can run that unchanged canonical contract against
the shared-pack layout. `ContentPack::definition_entries` supplies stable
`(CardDefId, CardDefinition)` input for its content digest without coupling the
benchmark to pack storage, while `GameState::content` and each card's
`definition_id` expose the layout seam. W2-179 does not add the canonical
driver, scenarios, RSS sampler, artifact schema, or summary renderer.

## Workloads

- Step throughput: 20,000 `Env::step` calls, including observation creation and
  reset when a game ends.
- Clone cost: 20,000 direct clones of the same 80-action midgame `GameState`
  after 200 warmup clones. The fixture contains 120 cards, 28 allocated
  permanent slots, and 498 events.
- Representative rollout throughput: eight repeats of current
  `Env::flat_mc_scores(4, 4, seed, 2000)` at a deterministic 48-action midgame
  with six root actions: 768 total simulations per run.
- Object size: Rust `size_of` evidence. These are shallow inline sizes; owned
  allocations are reported separately by the allocator measurements.

## Results

| Metric | Before | After | Change |
|---|---:|---:|---:|
| Environment steps/s | 155,763 | 158,885 | +2.0% |
| Step allocations/step | 127.416 | 119.603 | -6.1% |
| Step allocated bytes/step | 20,287.647 | 18,767.559 | -7.5% |
| `GameState` clone latency | 29,638.535 ns | 1,772.771 ns | -94.0% (16.7x faster) |
| `GameState` clones/s | 33,740 | 564,089 | +1,572% |
| Clone allocations/clone | 1,313 | 25 | -98.1% |
| Clone allocated bytes/clone | 208,622 | 29,787 | -85.7% |
| Flat-MC simulations/s | 2,563 | 2,721 | +6.2% |
| Rollout allocations/simulation | 9,037.147 | 7,695.480 | -14.8% |
| Rollout allocated bytes/simulation | 991,411.062 | 805,124.604 | -18.8% |
| Shallow `Card` size | 656 B | 32 B | -95.1% |
| Shallow `GameState` size | 1,696 B | 1,672 B | -1.4% |
| Shallow `Game` size | 1,936 B | 1,912 B | -1.2% |
| Definition-store header | 32 B per state | 56 B once per shared pack | moved out of state |

For the 120-card clone fixture, the inline card vector payload falls from
78,720 B to 3,840 B before considering the removed deep-owned strings, effect
trees, abilities, targeting clauses, costs, and subtype vectors. The 648 B
shallow `CardDefinition` plus its owned data now exists once per definition in
the 58-definition shared pack, rather than once per physical card and again in
every cloned branch.

### Warm timing samples

| Sample | Steps/s before | Steps/s after | Clone ns before | Clone ns after | Rollouts/s before | Rollouts/s after |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 155,763.240 | 153,289.693 | 29,691.719 | 1,795.552 | 2,563.182 | 2,721.393 |
| 2 | 152,760.818 | 158,885.103 | 29,638.535 | 1,772.771 | 2,553.259 | 2,755.306 |
| 3 | 156,379.043 | 162,565.390 | 29,368.877 | 1,738.317 | 2,575.704 | 2,638.969 |

## Behavior checks

- Before and after used six root actions and completed 768 simulations with no
  cap hits.
- The deterministic rollout score checksum was exactly `27.250000000` both
  before and after.
- `content_pack_tests` checks stable IDs across independently built packs,
  legacy registry-key equality, shared pack/definition identity across matches
  and search clones, copy-on-write isolation for the existing scenario mutation
  seam, and equality of two seeded fixed-action traces through all sampled
  rules state.
- The Python extension and ABI regression suites are rebuilt and run separately
  from this microbenchmark.

## Conclusion

Within this local diagnostic, definition sharing removes the dominant measured
clone allocation without changing the search algorithm or representation of
mutable rules facts. These numbers are directional evidence for the seam only.
W2-182 must establish the comparable full-clone throughput/RSS result under its
canonical contract after integrating this layout.
