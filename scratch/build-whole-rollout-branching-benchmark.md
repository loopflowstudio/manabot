# W2-182: whole-rollout branching benchmark harness

Status: implemented and verified against directive v5.

The canonical measurement authority is
`docs/benchmarks/search-branching-contract-v1.md`. This scratch note records
the Task boundary and receipts without redefining the contract.

## User-visible outcome

An engine/search developer can run:

```bash
uv run scripts/bench_branching.py run
uv run scripts/bench_branching.py verify
```

The first command produces an exact current-driver full-clone baseline at the
current single-worker and saturated worker x actor x rollout shapes. The second
recomputes all contract, source, artifact, repeat-checksum, cell, and summary
checks. The derived report leads with whole-rollout simulations/s,
transitions/s, root latency, and peak RSS; clone latency is secondary.

## End-to-end proof

The release binary reconstructs the exact 48-action midgame and 80-action
heavy fixtures, validates their drift fields, and passes all four equivalence
seeds on both fixtures. The `uv` runner launches fresh native worker groups,
waits for warmup/ready, launches a fresh process group per cell/repeat, records
exact worker argv, samples aggregate worker RSS every 5 ms after a start
barrier, preserves the complete timestamped sample series and every raw worker
result, derives summaries, writes
atomically, and verifies the artifact. Every primary cell repeats its first
root in a fresh group and must reproduce the ordered result checksum.

The checked-in proof is:

- `experiments/data/w2-182-search-branching-v1.json`
- `experiments/w2-182-search-branching-v1.md`
- `uv run scripts/bench_branching.py verify`

## Source of truth

- `Game` plus its current `ActionSpace` is the rules/legality authority.
- `BenchmarkManifest` in `managym/src/benchmark.rs` is the executable workload
  authority and matches the canonical Markdown contract.
- The checked-in raw JSON is the measurement authority; the Markdown report
  is generated only from it.
- Semantic hashes use ordered logical JSON plus BLAKE3. Physical cards carry
  object ID, stable registry key, and owner; definitions occur once via a
  sorted content digest. The snapshot contains no randomized map iteration,
  pointers, allocator state, timing, or RSS.

## Affected surfaces and consumers

- Native rules/search state and release benchmark binary
- Python process/RSS orchestration (always invoked through `uv`)
- Raw JSON schema and derived experiment report
- Future clone-plus-undo and page-COW-plus-undo drivers through the same
  `BranchDriver` hooks

The Python extension ABI, policy observation ABI, play UI, and current search
entry points remain unchanged.

## The branching question this harness is built to answer

The state choices are not mutually exclusive. Immutable definitions now live
in `Arc<ContentPack>` while mutable match facts remain in dense
vectors and structs. The open question is how those dense facts should cross a
search branch boundary. The likely shape is exact isolated snapshots at the
outside of search and a transaction-like mark/rollback loop inside each owned
branch, but contract v1 is designed to let measurements overturn that prior.

Use the following notation when discussing scale:

- `W`: worker processes;
- `A`: actors per worker;
- `L`: legal root actions in the fixture;
- `R`: worlds times rollouts per world, per root action;
- `S`: measured root seeds.

A flat comparison therefore executes `S * W * A * L * R` complete
simulations. The retained comparison keeps `W * A * L * R` logical branches
live at once for each seed. This is the realistic Monte Carlo surface. Clone
latency by itself measures none of rules execution, rollback, dirty-page
amplification, retained pools, or worker saturation.

### Candidate 1: compact exact full clone

This is the implemented reference driver. `Game::clone` copies mutable match
facts and cheaply clones the immutable content-pack `Arc`. Each world is an
exact fork of the root; each simulation is an exact fork of its world. The
reference `mark` is another full clone and rollback is state replacement.

Why keep it:

- Rust ownership makes root/sibling isolation obvious;
- the implementation is the simplest semantic oracle for optimized drivers;
- dense vectors and structs remain cache-friendly while a rollout executes;
- after definition separation, a clone no longer duplicates card programs and
  printed characteristics.

What may stop scaling:

- every sibling pays mutable-state copy and allocator costs;
- flat search multiplies those costs by `W * A * L * R`;
- retained search materializes every live state even when most facts are
  unchanged;
- clone-and-drop microbenchmarks can understate allocator pressure and memory
  bandwidth when rules transitions are interleaved.

### Candidate 2: compact outer clone plus worker-local undo

The ownership boundary remains an exact compact clone: a worker, actor, or
determinized world never mutates another branch. Within that owned branch, one
dense scratch state can serve sequential siblings:

```text
world = fork_exact(root)
scratch = fork_exact(world)
for action in legal_actions:
  for rollout in rollouts:
    mark = scratch.mark()
    scratch.reseed(rollout_seed)
    scratch.apply(action)
    scratch.play_to_terminal_or_cap()
    record_outcome(scratch)
    scratch.rollback(mark)
```

The mark should eventually be a journal cursor plus the minimum scalar
checkpoint, not a cloned `Game`. Every mutation path must be journaled:
overwritten scalar/slot values, vector pushes and truncations, zone order and
reverse indices, RNG, event ledgers, stack/choice/trigger queues, allocation
watermarks and free lists, caches, and the surfaced action space. Tests must
include allocations and zone changes so a rollback that only handles common
combat fields cannot pass.

This candidate is strongest in flat sequential search because it removes the
inner simulation clone while retaining dense execution. It cannot pretend that
retained alternatives are sequential. The retained cells still require an
independent exact fork for every simultaneously usable slot; undo can help only
with a nested probe owned by one slot. That retained-cell limitation is useful
evidence, not a reason to change the workload.

### Candidate 3: dense page-COW outer fork plus undo

Mutable state is divided into fixed-size, refcounted pages. `fork_exact`
shares clean pages; the first branch-local write copies its page. The branch
then uses the same journal/mark discipline for repeated inner work. Immutable
definitions remain outside this system in `ContentPack`.

This is the fullest version of "safe snapshot forks outside, dense
transactional execution inside":

- root, actor, world, and retained-slot forks are isolated even while sharing
  physical pages;
- hot rules code mutates typed dense pages rather than a persistent HAMT;
- rollback reuses branch-private storage rather than building a fresh state;
- retained pools pay roughly for dirty pages instead of whole logical states
  when mutations are sparse.

The costs must be measured rather than wished away: page metadata and
refcounts on reads/forks, first-write copies, false sharing when unrelated hot
facts occupy one page, and duplicated bookkeeping when both COW and an undo
journal record the same mutation. Page size and field grouping are benchmark
parameters only in a future contract version; v1 drivers may not tune them by
changing fixtures or the `W * A * R` workload.

### Exact boundary contract

| Boundary | Required lifecycle | Why |
|---|---|---|
| Definitions | Shared immutable `Arc<ContentPack>` | Programs and printed facts are content, not match state |
| Match/root to worker or actor | `fork_exact` | Failure, cancellation, and concurrent mutation cannot leak |
| Root to determinized world | `fork_exact`, then deterministic mutation | Worlds are common across root actions and must remain independently reproducible |
| World to sequential sibling | `mark`/`rollback` is allowed | Only one sibling is live, so transaction reuse preserves the workload |
| World to retained slot | `fork_exact` is required | All alternatives are simultaneously live and projected for policy work |
| Rules execution inside an owned branch | Dense mutable application | This is the locality advantage manabot wants to preserve |

`fork_exact` is a semantic promise, not a requirement to eagerly copy bytes.
Likewise, `mark` is currently a reference clone but is intended to become a
cheap transactional checkpoint. Both must restore the canonical snapshot,
legal actions, visible observation, event boundaries, allocation watermark,
and RNG continuation.

### Apples-to-apples decision evidence

Every driver runs the same fixture tapes, content digest, legal action order,
seed derivation, determinization, step cap, policy stub, process topology,
timing boundary, and 5 ms RSS sampler. The artifact's driver ID and supported
counters are the intended differences. The deterministic equivalence gate and
fresh-process repeat checksum must pass before timing is considered.

Primary evidence is:

- total complete simulations/s and transitions/s;
- p50/p95/p99 whole-root latency;
- absolute aggregate peak RSS and peak RSS delta;
- maximum simultaneously live logical states;
- peak journal bytes, peak branch-private COW bytes, checkpoint copies,
  eager forks, and allocator counts/bytes when supported.

Read flat and retained, single and saturated cells together. Full clone may
win on simplicity and transition locality, clone-plus-undo may dominate flat
rollouts, page-COW-plus-undo may dominate retained pools, or a fourth design
may beat all three. A mixed production endpoint is valid: safe COW snapshots
for the outer pool and a dense undo journal inside each worker is one coherent
result, not a compromise the benchmark needs to hide.

### Future implementation slices (not W2-182 scope)

1. Implement a journaled dense driver and make every mutator go through a
   rollback-audited write seam. Start with the sequential cells; retained
   slots remain exact forks.
2. Implement typed page storage behind the same logical accessors, then expose
   dirty-page and peak-private-byte counters.
3. Run all three drivers through the unchanged equivalence gate and raw result
   schema, on the same host/build in an interleaved order.
4. Pre-register acceptable correctness, throughput, latency, and memory
   thresholds before examining optimized-driver results.

The concrete executable seam already exists as `BranchDriver`; the native
manifest owns exact workload dimensions and the artifact schema already has
driver identity plus journal/COW/checkpoint counters. Future work changes the
driver implementation, not the measurement contract.

## Absent and error states

Fixture drift, empty nonterminal action spaces, terminal reuse, step-cap
overflow, failed equivalence, mismatched repeated checksums, worker failure,
missed ready/RSS sampling, malformed JSON, missing required metadata, or stale
summary/report data invalidates the run. Journal, page-COW, and allocator
counters are `null` with an explicit unsupported reason for the full-clone
driver; they are never reported as zero.

## Operational boundary

Native release workers contain all timed rules work. Fixture construction,
warmup, semantic correctness hashing, process startup, RSS polling, and JSON
serialization are excluded from native root timers. Aggregate process RSS is
sampled at 5 ms and may double-count shared pages. The canonical saturated
cell requires at least eight physical cores unless explicitly labeled
oversubscribed.

## Exclusions

This Task does not implement or select clone-plus-undo or dense page-COW. It
does not add neural inference, change search policy, establish a regression
threshold, or claim that clone latency alone justifies a storage design.

## Verification receipts

- Integrated onto `origin/main` at `2863105`, including W2-179's shared
  `ContentPack` and stable `CardDefId`. Its explicitly non-comparable local
  diagnostic remains intact; this generated report describes only the current
  driver at the recorded source state.
- Rust: 16 unit tests, 2 conformance, 3 ContentPack, 13 engine, 162 rules, 6
  scenario, and 10 search tests pass; clippy is clean with warnings denied.
- Python: 15 focused contract tests pass; Ruff check/format pass. An optional
  repository-wide run reached 263 passing tests with no failures before being
  stopped after seven minutes in a long verification matchup.
- CPython 3.12 release wheel rebuilt and the worktree native extension imports.
- Full canonical benchmark generation and artifact/report verification pass;
  artifact SHA-256:
  `20c403beed10408a84b80428986e7cb6acbcc4f09782260bfededba34c897bf1`.
