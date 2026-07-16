# Search branching benchmark contract v1

Contract ID: `manabot.search-branching.v1`

Owner: W2-182 (harness and current full-clone baseline).

This file is the canonical measurement contract for the current driver and
future branching drivers. Until W2-182 first publishes it, validation and
evidence fields may tighten as part of the same pre-release review. After that
publication, a result that changes a fixture, seed, workload dimension, reset
rule, timing boundary, RSS method, or required schema field is a different
contract and must use a new contract ID. Raw results carry the SHA-256 of this
file.

The contract measures search-state implementations. It does not select one.
Clone latency is diagnostic; total whole-rollout throughput and peak RSS are
the primary decision evidence.

## Evidence boundary

- W2-182 runs every cell in this contract for
  `full_clone/current_game_v1`, records deterministic equivalence evidence,
  and owns the raw/summary artifact format.
- W2-182 measures the current post-ContentPack driver only. Its numbers are not
  a W2-179 before/after comparison, and no non-contract W2-179 number may be
  combined with them.
- Future clone-plus-undo and dense-page-COW-plus-undo implementations use this
  unchanged contract through the driver hooks below. They are not implemented
  or selected by W2-182.

## Engine and build boundary

- Native Rust, Cargo `--release`, system allocator, one Rust thread per worker
  process.
- Current fixed-action ABI and current rules engine. No neural inference is in
  a timed region.
- Python is orchestration only and every Python command uses `uv run`.
- `skip_trivial = true` everywhere.
- Step cap: 2,000 surfaced `Game::step` calls per simulation, including the
  root action.
- A nonterminal state with no legal action is an invalid run, not a draw.
- Reaching the step cap is a cap result scored as 0.5 and counted separately.
- A terminal game is never stepped again. A new measured root is reconstructed
  from the fixture recipe; mutable roots are never recycled across samples.

## Exact deck

`interactive-mirror-60-v1` is used by both players. Counts are exact and total
60 cards per player:

| Card | Count |
|---|---:|
| Island | 12 |
| Mountain | 12 |
| Gray Ogre | 6 |
| Wind Drake | 6 |
| Man-o'-War | 4 |
| Raging Goblin | 4 |
| Lightning Bolt | 6 |
| Counterspell | 4 |
| Ancestral Recall | 3 |
| Pyroclasm | 3 |

Player 0 is named `hero`; player 1 is named `villain`. Deck iteration is the
engine's `BTreeMap` order.

This is the v1 deck because it is the current flat-MC/search benchmark deck.
UR Lessons versus GW Allies may become a later fixture version; it must not be
mixed into v1 comparisons.

## Exact fixtures

### `interactive-midgame-48-v1`

Primary step and whole-rollout fixture.

1. Construct `Game::new`/`Env::reset` with `interactive-mirror-60-v1`, setup
   seed `377` (`0x179`), and `skip_trivial = true`.
2. Do not reseed after setup. The setup shuffle and subsequent fixture action
   selection therefore consume one continuous `ChaCha8Rng` stream.
3. Advance exactly 48 surfaced decisions. At each decision call the current
   engine's `Env::random_action_index`, record the returned fixed-ABI action
   index, then call `Env::step` once.
4. Do not reset during the 48-action prefix. Do not drain or normalize engine
   history beyond what `Env::step` already does.
5. The fixture is valid only if it is nonterminal and has exactly six legal
   root actions after decision 48. The harness stores the realized 48-index
   tape, canonical root hash, legal-action hash, action kind, and shape counts.
   Any mismatch aborts the run as fixture drift.

### `interactive-heavy-80-v1`

Clone and retained-state fixture.

1. Construct `Game::new` directly with the same deck/configuration and setup
   seed `377`.
2. Immediately call `Game::reseed(0xc10e)`.
3. Advance exactly 80 surfaced decisions. At each decision choose with
   `game.state.rng.gen_range(0..legal_action_count)` and call `Game::step`.
4. Do not call `take_observation_events` during the prefix.
5. The fixture is valid only if it is nonterminal. For the v1 engine it has
   120 physical cards, 28 allocated permanent slots, and 498 committed events;
   those values and the realized 80-index tape are drift checks, not values
   silently updated by the harness.

Fixture construction, fixture validation, and canonical hashing are never in a
timed region.

## Deterministic seeds

All seeds are unsigned 64-bit integers. `mix_seed` is the existing
SplitMix64-style function in `managym/src/flow/search.rs`.

- Fixture/setup seed: `377`.
- Heavy-fixture action reseed: `0xc10e` (`49422`).
- Equivalence trace seeds:
  `[0x5eed, 0x5eee, 0x5eef, 0x5ef0]`
  (`[24301, 24302, 24303, 24304]`).
- Whole-rollout warmup seed: `0xbeee` (`48878`).
- Whole-rollout measured root seeds:
  `[0xbeef, 0xbef0, 0xbef1, 0xbef2, 0xbef3, 0xbef4, 0xbef5, 0xbef6]`
  (`[48879, 48880, 48881, 48882, 48883, 48884, 48885, 48886]`).

For a measured root seed `s`, derivation is:

```text
worker_seed = mix_seed(s, worker_index)
actor_seed = mix_seed(worker_seed, actor_index)
world_seed = mix_seed(actor_seed, world_index)
rollout_seed = mix_seed(
    world_seed,
    root_action_index * rollouts_per_world + rollout_index + 1,
)
policy_stub_seed = mix_seed(rollout_seed, ply_index)
```

Determinization uses `world_seed`. Rules RNG uses `rollout_seed`. Retained
policy-stub choices use `policy_stub_seed % legal_action_count`; the stub is
external to rules RNG. Worlds are common across root actions, matching current
flat MC.

## Warmup and measured counts

Warmup never contributes timing, allocation, checksum, or RSS summary samples.

### `step-v1` (diagnostic and regression evidence)

- Fixture family: `interactive-midgame-48-v1` deck/configuration.
- Warmup: 2,000 successful `Env::step` calls.
- Measured: 20,000 successful `Env::step` calls.
- Action selection calls and fixture construction are outside step timing.
- If a game terminates, reconstruct using setup seed `377 + completed_games`.
  Reset time is accumulated separately as `reset_seconds` and excluded from
  `step_seconds`.
- Each `Env::step` timing includes rules application, trivial-action collapse,
  observation-event consumption, observation construction, reward, and info.
- Summary: steps/s, p50/p95/p99 step latency, resets, and reset seconds.

### `clone-v1` (secondary diagnostic)

- Fixture: `interactive-heavy-80-v1`.
- Warmup: 200 `Game::clone` plus immediate drop operations.
- Measured: 20,000 `Game::clone` plus immediate drop operations.
- Timer starts immediately before `Game::clone` and ends after the clone is
  dropped. `black_box` prevents elision.
- Fixture construction, semantic hashing, and shape inspection are excluded.
- Summary: p50/p95/p99 clone-and-drop latency, clones/s, allocation counts and
  requested bytes when a counting allocator is active.

### Whole-rollout cells (primary evidence)

Every worker performs one untimed full-shape root evaluation using warmup seed
`0xbeee`, then one measured root evaluation for each of the eight measured
seeds. A root evaluation covers every legal root action.

`A = 6` for `interactive-midgame-48-v1`.

| Cell ID | Fixture | Shape | Workers | Actors/worker | Worlds | Rollouts/world | N/action | K policy plies | Exact measured simulations |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `flat-single-64-v1` | midgame | sequential | 1 | 1 | 16 | 4 | 64 | 0 | 3,072 |
| `flat-saturated-64-v1` | midgame | sequential | 8 | 1 | 16 | 4 | 64 | 0 | 24,576 |
| `retained-single-8-v1` | heavy | retained | 1 | 1 | 8 | 1 | 8 | 8 | `8 * A_heavy * 8` |
| `retained-saturated-16-v1` | heavy | retained | 1 | 8 | 16 | 1 | 16 | 8 | `8 * 8 * A_heavy * 16` |

`A_heavy` is a fixture drift field because the heavy root's primary purpose is
retained memory pressure. Its realized value is stored once in the fixture
record and must be identical across all drivers in a comparison. The raw
artifact stores both the formula and realized simulation count.

Eight workers is the current seat-balanced/search experiment saturation point.
One worker with eight retained actors is the current cross-game batching shape
used to avoid MPS process contention. A host with fewer than eight physical
cores must reject the saturated flat cell unless explicitly run as
oversubscribed; oversubscribed results are labeled and cannot be the canonical
v1 baseline.

## State lifecycle and whole-rollout boundary

### Sequential flat cells

For each actor and measured root seed:

1. Reconstruct the exact root fixture outside timing.
2. Start the root-decision timer.
3. For each world, exact-fork the root and determinize it with `world_seed`.
4. For each legal root action and rollout, exact-fork the world, reseed with
   `rollout_seed`, apply the root action, then choose uniformly from current
   legal action indices using the branch rules RNG until terminal or cap.
5. Score hero win `1.0`, loss `0.0`, draw/cap `0.5`; retain outcome, step count,
   terminal semantic hash, event boundary, and RNG probe in the checksum.
6. End the timer after the last outcome is accumulated.

The primary timer includes root/world clones, determinization, rollout clones,
reseed, root apply, all rules transitions, and outcome aggregation. It excludes
fixture construction, correctness hashing time, process startup, RSS polling,
and JSON serialization. Those excluded durations are recorded separately.

### Retained cells

For each actor and measured root seed, retain all
`worlds * legal_actions * rollouts_per_world` branches simultaneously. Timing
starts before pool construction. It includes world/simulation forks,
determinization, root action application, observation projection for active
slots, eight deterministic policy-stub plies, random legal tails, and outcome
aggregation. It excludes neural inference. Timing ends after every slot is
terminal or capped.

The pool is destroyed after the measured root. No slot or root is reset in
place between measured seeds.

## Deterministic equivalence gate

No timing or baseline claim is valid until the driver passes all four
equivalence seeds on both fixtures:

1. Root and `fork_exact` have identical authority, legal-surface, fixed-viewer,
   and diagnostic witness components.
2. Mutating a fork cannot change its root or sibling.
3. Two exact forks driven by the same external action sequence compare equal
   after every step through terminal/cap.
4. Full-clone `mark`/`rollback` is implemented as snapshot replacement and
   restores exact equality. This is a reference checkpoint, not an undo
   journal.
5. Repeating a measured root seed produces the same ordered result checksum;
   timing and RSS may vary.

Search-witness schema 2 is representation-neutral. Its authority fingerprint
serializes logical rules facts, not Rust layout or `Debug` output:

- content schema/digest;
- physical cards as object ID, stable definition ID, and owner (definitions
  themselves are represented once by the content digest);
- permanents, players, exact zone order/reverse membership, turn, priority,
  stack, combat, mana, pending/suspended choices, triggers, delayed links,
  event ledgers, allocation watermark, legal action space, and skip state;
- eight `next_u64` values from a cloned RNG.

Serialize as canonical JSON with fixed struct field order and ordered maps,
then hash with BLAKE3. Pointer addresses, allocator state, copied definition
layout, and randomized hash maps are prohibited. This keeps hashes comparable
across branching representations despite definition sharing. Separate typed
components fingerprint the legal surface and each fixed viewer projection;
diagnostics expose event boundaries, terminal state, and the RNG continuation
probe. The aggregate is deliberately a witness, not restorable state.

## Branch driver hooks

Every implementation supplies the same logical operations:

```rust
trait BranchDriver {
    type State;
    type Mark;

    fn fork_exact(&self, source: &Self::State) -> Self::State;
    fn determinize(&self, state: &mut Self::State, viewer: PlayerId, seed: u64);
    fn reseed_rollout(&self, state: &mut Self::State, seed: u64);
    fn mark(&self, state: &mut Self::State) -> Self::Mark;
    fn apply(&self, state: &mut Self::State, action: usize) -> ApplyResult;
    fn rollback(&self, state: &mut Self::State, mark: Self::Mark);
    fn witness(&self, state: &Self::State) -> SearchStateWitness;
    fn counters(&self) -> DriverCounters;
}
```

`full_clone/current_game_v1` uses `Game::clone`; its mark is a full snapshot
and rollback replaces state. Journal and COW counters are `null` with an
unsupported reason, never misleading zeroes. Later clone-plus-undo and dense
page-COW-plus-undo drivers implement the hooks but are outside W2-182.

### Three-strategy comparison endpoint

Contract v1 deliberately keeps the logical state lifecycle separate from its
physical representation. The planned comparison has three named strategies;
they are hypotheses, not an exhaustive list or a preselected winner.

| Strategy | Safe outer fork | Repeated inner work | Expected strength | Risk the whole-rollout cells expose |
|---|---|---|---|---|
| `full_clone/current_game_v1` | Clone the compact `Game`; immutable definitions remain shared through `Arc<ContentPack>` | Clone again per world and simulation; `mark` is a reference full snapshot | Simple ownership, exact isolation, contiguous mutable facts | Copy bandwidth and allocator traffic as `W * A * L * R` rises |
| `compact_clone_undo/*` | Clone once at each independently owned worker/actor/world boundary | Reuse one dense scratch state with a journal cursor or checkpoint marker, then roll back after each sequential simulation | Dense access while deleting most inner full-state copies | Journal volume, rollback cost, and writes that escape the journal |
| `dense_page_cow_undo/*` | Fork refcounted fixed-size mutable pages; copy a page on its first branch-local write | Journal mutations within the private branch and roll back to a cursor | Cheap safe forks plus dense transactional execution, especially for retained pools | Dirty-page amplification, refcount/indirection cost, and paying for both COW and undo |

Here `W` is worker processes, `A` is actors per worker, `L` is legal root
actions, and `R = worlds * rollouts_per_world` is simulations per action. For
`S` measured seeds, a flat cell performs exactly `S * W * A * L * R`
simulations. That formula, rather than an isolated `clone()` loop, is the
scaling surface that the three drivers must share.

The likely hybrid endpoint is **safe exact forks outside and dense
transactional execution inside**:

1. `ContentPack` is immutable and shared by every match and branch.
2. Match roots, worker-owned roots, actors, worlds, and every simultaneously
   retained slot are isolated by `fork_exact`. The physical fork may be an
   eager clone or page COW, but no mutation can cross that boundary.
3. A sequential probe or rollout takes `mark`, applies and advances dense
   mutable facts, records its outcome, then `rollback`s before the next sibling.
4. A retained cell cannot replace its simultaneous slots with one rollback
   buffer without changing the workload. Each live slot still requires a safe
   exact fork; undo is useful only for nested probes within that slot.

This distinction prevents a misleading benchmark in which the undo driver
wins by serializing a workload that the clone and COW drivers execute with all
branches resident.

### Driver-specific lifecycle and counters

For sequential cells, an optimized driver may replace the inner
`fork_exact(world)` with `mark(scratch)` / run / `rollback(scratch, mark)` only
if `scratch` is restored to the same canonical world snapshot before every
action/rollout pair. The journal must restore RNG state, zone order and reverse
membership, events and observation events, choices, stack and triggers,
allocation watermarks, arena lengths/free lists, caches, and the legal action
space. A root action that allocates and later rolls back is part of the gate,
not an exceptional path.

For retained cells, pool construction and destruction remain inside the root
timer for every driver. A page-COW implementation may share clean pages among
slots, but every first write must become branch-private. A compact-clone-plus-
undo implementation still materializes one independent slot per live branch;
it may not alias slots or reconstruct them outside timing.

Existing result fields have fixed meanings for later drivers:

- `eager_forks`: full logical-state copies performed by the driver;
- `checkpoint_copies`: full snapshots taken to implement a mark or reset;
- `journal_bytes`: peak live undo storage, not cumulative bytes appended;
- `cow_bytes`: peak branch-private copied page bytes, excluding shared clean
  pages and the immutable content pack;
- `allocation_count` / `allocation_bytes`: timed-region allocator totals when
  supported;
- `max_live_states`: logical states simultaneously usable, independent of
  physical sharing.

Unsupported counters remain `null` with a reason. A zero is valid only when
the driver measures that counter and observed no corresponding work.

An optimized driver is executable under contract v1 when it exposes a stable
driver ID, implements the same `BranchDriver` operations, passes the unchanged
equivalence and repeat-checksum gates, and emits the same raw worker/result
shape. Driver ID and driver counters may differ; fixture hashes, action order,
seed paths, workload dimensions, cap, timer boundaries, and RSS protocol may
not. A new representation such as arena deltas or OS-assisted snapshots can
join the comparison without becoming one of the three named strategies if it
honors those constraints.

### What decides, and what does not

The comparison is made on matched whole-rollout cells on the same host and
build profile. It reports simulations/s, transitions/s, root p50/p95/p99,
absolute peak RSS, peak RSS delta, maximum live states, and the driver counters
above. Flat cells expose sequential transaction overhead; retained cells expose
the memory cost of many live alternatives. Results should be read at both the
single and saturated `W * A * R` shapes so an optimization that only moves a
bottleneck between cores or actors is visible.

Clone-and-drop latency remains a diagnostic for explaining a result. It cannot
select a strategy: a faster clone may lose on mutation locality, an undo journal
may grow with rules activity, and page COW may save retained memory while
reducing transition throughput. Correctness equivalence is a prerequisite, not
a weighted metric. No action cap, shallower rollout, altered determinization,
or different scheduler may be used to improve a driver's score.

## Peak RSS protocol

Each workload cell and repeat uses a fresh process group.

1. Before starting any worker, the parent records the repeat's UTC start time.
   The first worker becomes the leader of a new process group and every other
   worker in that repeat joins it. No process group may be reused by another
   cell, repeat, or deterministic replay.
2. The parent starts the exact worker count and retains the exact argv passed
   to each process. Each worker echoes its actual process argv in its ready
   receipt; a mismatch invalidates the repeat.
3. Every worker constructs its fixture, completes warmup, emits a `ready`
   record, and blocks on stdin.
4. After all workers are ready, the parent samples aggregate worker RSS once as
   `rss_baseline_bytes`, begins 5 ms polling, and releases one byte to every
   worker as a start barrier.
5. Every polling attempt is retained in order with a contiguous sequence,
   UTC capture timestamp, monotonic offset, aggregate RSS, and a per-worker
   PID/state/RSS receipt. An exited worker contributes zero but remains present
   in every later sample. The final sample records every worker exited.
6. `rss_peak_bytes` is the maximum sampled sum of worker RSS from barrier
   release until all workers exit. `rss_peak_delta_bytes` is peak minus
   baseline. Parent RSS is recorded separately.
7. The report states that summed RSS double-counts pages shared across worker
   processes and that 5 ms sampling can miss shorter spikes. The method is
   identical for every driver.

Canonical results require the complete aggregate-worker RSS sample series and
a completed sampler receipt. A sampler exception or lost live PID, ready or
start-barrier timeout/failure, worker signal/panic/OOM/nonzero exit, malformed
child record, incomplete final sample, or worker timeout aborts the run before
an artifact is written. Do not substitute `ru_maxrss` or a peak retained from
another cell.

## Hardware and execution metadata

Raw results require:

- a UTC `started_at` captured before build, fixture, equivalence, or worker
  execution begins, a UTC completion time, and the local timezone;
- OS name/version, kernel, architecture;
- CPU model, physical cores, logical cores;
- total physical memory;
- power mode and thermal state when the OS exposes them, otherwise `null` with
  reason;
- Rust, Cargo, uv, and Python versions;
- Cargo profile and allocator;
- process topology, thread count, oversubscription flag;
- RSS method and polling interval;
- contract ID and SHA-256, result schema, source-tree SHA-256, engine/content
  schema/digest, the orchestrator's exact process argv, each worker's exact
  invoked/acknowledged argv, and working-tree/PR provenance supplied by
  Loopflow receipts.

Missing optional thermal/power data is allowed. Missing contract/source
digests, build profile, seed, workload dimensions, timing, or RSS invalidates a
canonical result.

## Raw result schema v1

Authoritative file:
`experiments/data/w2-182-search-branching-v1.json`.

```json
{
  "schema": "manabot.search-branching.result.v1",
  "contract": {
    "id": "manabot.search-branching.v1",
    "sha256": "..."
  },
  "run": {
    "started_at": "...",
    "completed_at": "...",
    "argv": [],
    "source_sha256": "...",
    "driver": "full_clone/current_game_v1",
    "status": "complete"
  },
  "hardware": {},
  "build": {},
  "fixtures": [
    {
      "id": "interactive-midgame-48-v1",
      "deck": "interactive-mirror-60-v1",
      "setup_seed": 377,
      "action_tape": [],
      "semantic_hash": "...",
      "action_hash": "...",
      "shape": {}
    }
  ],
  "equivalence": {
    "seeds": [24301, 24302, 24303, 24304],
    "checks": [],
    "passed": true
  },
  "cells": [
    {
      "id": "flat-single-64-v1",
      "dimensions": {},
      "warmup": {},
      "repeats": [
        {
          "status": "complete",
          "started_at": "...",
          "completed_at": "...",
          "root_seed": 48879,
          "process_group_id": 1234,
          "commands": [],
          "ready": [],
          "workers": [],
          "barrier_released_at": "...",
          "barrier_wall_seconds": 0.0,
          "sampler": {
            "status": "complete",
            "interval_seconds": 0.005,
            "sample_count": 2
          },
          "rss_baseline_bytes": 0,
          "rss_peak_bytes": 0,
          "rss_peak_delta_bytes": 0,
          "rss_samples": [
            {"sequence": 0, "phase": "baseline", "workers": []},
            {"sequence": 1, "phase": "complete", "workers": []}
          ],
          "result_checksum": "..."
        }
      ],
      "summary": {}
    }
  ],
  "artifact_sha256": "..."
}
```

Each repeat also retains parent RSS and every timestamped aggregate-worker RSS
sample. Each worker result contains actor/world/rollout dimensions, derived seed roots,
root action count, simulations, transitions, outcomes, caps, fork/apply/tail/
hash durations, max simultaneously live states, terminal checksum, and driver
counters. Raw repeat records are retained; summary-only JSON is invalid.

`artifact_sha256` hashes canonical JSON with that field omitted. Writes use a
temporary file and atomic rename. Verification recomputes the artifact hash,
contract/source digests, required cell set, summaries, and generated Markdown.
It also rejects a missing/failed status, pre-execution timestamp, argv receipt,
fresh process-group receipt, barrier receipt, sampler receipt, raw RSS series,
worker coverage, or nonzero worker result even if aggregate summaries remain.

## Summary metrics

The generated report must lead with, per whole-rollout cell:

- total completed simulations / barrier wall second;
- total rules transitions / barrier wall second;
- p50/p95/p99 root-decision latency;
- absolute aggregate peak RSS and peak RSS delta;
- max live states, cap rate, and deterministic checksum/equivalence status.

It then reports step throughput/latency and clone-and-drop latency/allocation
diagnostics. Clone results cannot justify a storage choice without the
whole-rollout throughput and RSS cells.

No regression gate or implementation selection is part of v1 baseline
capture. Later comparisons must preserve the raw v1 results and pre-register
their throughput/RSS decision thresholds before running optimized drivers.
