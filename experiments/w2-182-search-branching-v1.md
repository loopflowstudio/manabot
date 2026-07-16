# Whole-rollout branching evidence: `full_clone/current_game_v1`

Contract: `manabot.search-branching.v1` (`9c8dfc3cae6bb85642caf61c074bb3ba2c0845f555c3024238f3d2d6fa85708f`)
Driver: `full_clone/current_game_v1`
Run: `2026-07-16T13:51:48.883110Z`; canonical: `true`
Evidence scope: current driver at this source state only; this is not a W2-179 before/after comparison.

## Primary whole-rollout evidence

| Cell | simulations/s | transitions/s | root p50 / p95 / p99 | peak RSS | peak delta | max live | cap rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| `flat-single-64-v1` | 297.6 | 38691.7 | 0.236s / 0.254s / 0.254s | 16.0 MiB | 0.7 MiB | 3 | 0.000% |
| `flat-saturated-64-v1` | 2194.6 | 288878.3 | 0.263s / 0.283s / 0.283s | 128.4 MiB | 3.2 MiB | 3 | 0.000% |
| `retained-single-8-v1` | 326.6 | 31753.7 | 0.008s / 0.010s / 0.010s | 14.4 MiB | 3.5 MiB | 17 | 0.000% |
| `retained-saturated-16-v1` | 362.6 | 33900.7 | 0.117s / 0.120s / 0.120s | 65.9 MiB | 17.7 MiB | 264 | 0.000% |

Peak RSS is the 5 ms sampled sum across worker processes and can double-count shared pages. Clone latency is diagnostic, not a storage decision.

## Branch lifecycle and journal counters

| Cell | eager forks | checkpoints | fork time | mark time | rollback time | journal peak | journal entries | marks / commits / rollbacks |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `flat-single-64-v1` | 128 | 3072 | 0.000s | 0.007s | 0.022s | null | null | null / null / null |
| `flat-saturated-64-v1` | 1024 | 24576 | 0.004s | 0.057s | 0.185s | null | null | null / null / null |
| `retained-single-8-v1` | 192 | 0 | 0.001s | 0.000s | 0.000s | null | null | null / null / null |
| `retained-saturated-16-v1` | 3072 | 0 | 0.009s | 0.000s | 0.000s | null | null | null / null / null |

`null` counters are unsupported by this driver, not observed zeros: system allocator has no counting hook; full-clone baseline has no undo journal or page-COW counters.

## Step and clone diagnostics

| Cell | samples/s | latency p50 / p95 / p99 | resets | reset time |
|---|---:|---:|---:|---:|
| `step-v1` | 128781.8 | 6.7µs / 16.6µs / 22.4µs | 116 | 0.001s |
| `clone-v1` | 220947.3 | 4.5µs / 4.6µs / 5.5µs | 0 | 0.000s |

## Reproduction and evidence

```bash
uv run scripts/bench_branching.py run --driver full_clone/current_game_v1
uv run scripts/bench_branching.py verify --driver full_clone/current_game_v1
```

Equivalence: `true` across 8 fixture/seed checks, each replayed twice.
Each primary cell also repeated its first measured root in a fresh worker group and matched the ordered deterministic result checksum.
Artifact SHA-256: `990e52911961bed8629b0011101cdb3156fddb24fd391cea8dfe02c6d1d7c12d`.
Source SHA-256: `8c0ad849119ca8fd81a87fac68b83a65867d5142e3ec71746d96c3b2832ce992`.
Source method: `git-ls-tree-sha256-v1` over 124 tracked paths recorded in the raw receipt.

`source_sha256` hashes the canonical NUL-delimited Git tree entries for the receipt's explicit source closure. Run and verification fail closed when an admitted input is dirty, missing, or gains an undeclared production compile-time dependency. Checkout paths, `.git` representation, `.claude` worktrees, generated evidence, and unrelated tracked files are excluded by construction; changing an admitted engine, embedded-content, harness, dependency, test, or contract input requires regenerating every candidate receipt.

The raw artifact contains the pre-execution UTC start, exact orchestrator and worker argv, a fresh process group per cell/repeat, barrier and sampler receipts, the complete timestamped 5 ms aggregate-worker RSS series, hardware, versions, fixture tapes and hashes, all worker results, seeds, timings, outcomes, and deterministic checksums.

This artifact measures one candidate. Comparing candidates means comparing two artifacts from the same host and source state, matched by their equal per-cell result checksums. No page-COW representation is implemented, measured, or selected here.
