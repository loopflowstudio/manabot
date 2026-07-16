# Whole-rollout branching evidence: `compact_clone_undo/current_game_v1`

Contract: `manabot.search-branching.v1` (`9c8dfc3cae6bb85642caf61c074bb3ba2c0845f555c3024238f3d2d6fa85708f`)
Driver: `compact_clone_undo/current_game_v1`
Run: `2026-07-16T13:53:40.049280Z`; canonical: `true`
Evidence scope: current driver at this source state only; this is not a W2-179 before/after comparison.

## Primary whole-rollout evidence

| Cell | simulations/s | transitions/s | root p50 / p95 / p99 | peak RSS | peak delta | max live | cap rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| `flat-single-64-v1` | 288.9 | 37557.6 | 0.299s / 0.315s / 0.315s | 24.7 MiB | 6.3 MiB | 3 | 0.000% |
| `flat-saturated-64-v1` | 2097.2 | 276050.9 | 0.332s / 0.348s / 0.348s | 193.2 MiB | 20.5 MiB | 3 | 0.000% |
| `retained-single-8-v1` | 332.8 | 32361.7 | 0.008s / 0.010s / 0.010s | 15.1 MiB | 4.0 MiB | 17 | 0.000% |
| `retained-saturated-16-v1` | 361.7 | 33815.9 | 0.122s / 0.133s / 0.133s | 71.5 MiB | 22.6 MiB | 264 | 0.000% |

Peak RSS is the 5 ms sampled sum across worker processes and can double-count shared pages. Clone latency is diagnostic, not a storage decision.

## Branch lifecycle and journal counters

| Cell | eager forks | checkpoints | fork time | mark time | rollback time | journal peak | journal entries | marks / commits / rollbacks |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `flat-single-64-v1` | 128 | 3072 | 0.001s | 0.000s | 0.415s | 4.64 MiB | 34385 | 6144 / 3072 / 3072 |
| `flat-saturated-64-v1` | 1024 | 24576 | 0.005s | 0.001s | 3.558s | 4.65 MiB | 35727 | 49152 / 24576 / 24576 |
| `retained-single-8-v1` | 192 | 0 | 0.001s | 0.000s | 0.000s | 0.11 MiB | 86 | 128 / 128 / 0 |
| `retained-saturated-16-v1` | 3072 | 0 | 0.009s | 0.000s | 0.000s | 1.66 MiB | 86 | 2048 / 2048 / 0 |

`null` counters are unsupported by this driver, not observed zeros: system allocator has no counting hook; clone-plus-undo does not implement page COW.

## Step and clone diagnostics

| Cell | samples/s | latency p50 / p95 / p99 | resets | reset time |
|---|---:|---:|---:|---:|
| `step-v1` | 125077.7 | 6.9µs / 17.1µs / 23.0µs | 116 | 0.002s |
| `clone-v1` | 216322.1 | 4.6µs / 4.8µs / 4.9µs | 0 | 0.000s |

## Reproduction and evidence

```bash
uv run scripts/bench_branching.py run --driver compact_clone_undo/current_game_v1
uv run scripts/bench_branching.py verify --driver compact_clone_undo/current_game_v1
```

Equivalence: `true` across 8 fixture/seed checks, each replayed twice.
Each primary cell also repeated its first measured root in a fresh worker group and matched the ordered deterministic result checksum.
Artifact SHA-256: `d6dcfc3ae14a9957afa0d71624d1380728ca250b20bdafec53bf50eba843e359`.
Source SHA-256: `8c0ad849119ca8fd81a87fac68b83a65867d5142e3ec71746d96c3b2832ce992`.
Source method: `git-ls-tree-sha256-v1` over 124 tracked paths recorded in the raw receipt.

`source_sha256` hashes the canonical NUL-delimited Git tree entries for the receipt's explicit source closure. Run and verification fail closed when an admitted input is dirty, missing, or gains an undeclared production compile-time dependency. Checkout paths, `.git` representation, `.claude` worktrees, generated evidence, and unrelated tracked files are excluded by construction; changing an admitted engine, embedded-content, harness, dependency, test, or contract input requires regenerating every candidate receipt.

The raw artifact contains the pre-execution UTC start, exact orchestrator and worker argv, a fresh process group per cell/repeat, barrier and sampler receipts, the complete timestamped 5 ms aggregate-worker RSS series, hardware, versions, fixture tapes and hashes, all worker results, seeds, timings, outcomes, and deterministic checksums.

This artifact measures one candidate. Comparing candidates means comparing two artifacts from the same host and source state, matched by their equal per-cell result checksums. No page-COW representation is implemented, measured, or selected here.
