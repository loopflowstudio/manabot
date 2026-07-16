# Whole-rollout branching evidence: `compact_clone_undo/current_game_v1`

Contract: `manabot.search-branching.v1` (`9c8dfc3cae6bb85642caf61c074bb3ba2c0845f555c3024238f3d2d6fa85708f`)
Driver: `compact_clone_undo/current_game_v1`
Run: `2026-07-16T19:46:40.592943Z`; canonical: `true`
Evidence scope: current driver at this source state only; this is not a W2-179 before/after comparison.

## Primary whole-rollout evidence

| Cell | simulations/s | transitions/s | root p50 / p95 / p99 | peak RSS | peak delta | max live | cap rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| `flat-single-64-v1` | 278.7 | 36232.6 | 0.312s / 0.335s / 0.335s | 24.6 MiB | 4.9 MiB | 3 | 0.000% |
| `flat-saturated-64-v1` | 1823.4 | 240013.8 | 0.376s / 0.487s / 0.487s | 193.2 MiB | 22.1 MiB | 3 | 0.000% |
| `retained-single-8-v1` | 318.3 | 30951.4 | 0.009s / 0.010s / 0.010s | 16.6 MiB | 4.7 MiB | 17 | 0.000% |
| `retained-saturated-16-v1` | 350.3 | 32756.1 | 0.127s / 0.135s / 0.135s | 84.2 MiB | 30.7 MiB | 264 | 0.000% |

Peak RSS is the 5 ms sampled sum across worker processes and can double-count shared pages. Clone latency is diagnostic, not a storage decision.

## Branch lifecycle and journal counters

| Cell | eager forks | checkpoints | fork time | mark time | rollback time | journal peak | COW peak | journal entries | marks / commits / rollbacks |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `flat-single-64-v1` | 128 | 0 | 0.001s | 0.000s | 0.439s | 4.64 MiB | null | 34385 | 6144 / 3072 / 3072 |
| `flat-saturated-64-v1` | 1024 | 0 | 0.007s | 0.001s | 4.124s | 30.23 MiB | null | 263176 | 49152 / 24576 / 24576 |
| `retained-single-8-v1` | 192 | 0 | 0.001s | 0.000s | 0.000s | 0.11 MiB | null | 86 | 128 / 128 / 0 |
| `retained-saturated-16-v1` | 3072 | 0 | 0.015s | 0.000s | 0.000s | 1.66 MiB | null | 86 | 2048 / 2048 / 0 |

`null` counters are unsupported by this driver, not observed zeros: system allocator has no counting hook; clone-plus-undo does not implement page COW.

## Step and clone diagnostics

| Cell | samples/s | latency p50 / p95 / p99 | resets | reset time |
|---|---:|---:|---:|---:|
| `step-v1` | 126836.2 | 6.8µs / 16.9µs / 22.8µs | 116 | 0.002s |
| `clone-v1` | 159974.2 | 6.2µs / 6.5µs / 7.9µs | 0 | 0.000s |

## Reproduction and evidence

```bash
uv run scripts/bench_branching.py run-matrix
uv run scripts/bench_branching.py verify-matrix
```

Equivalence: `true` across 8 fixture/seed checks, each replayed twice.
Each primary cell also repeated its first measured root in a fresh worker group and matched the ordered deterministic result checksum.
Artifact SHA-256: `389bd14afcb925489b5a29f7c0dacb34baa388886e8f3eb0cad13aa9fa1bb395`.
Source SHA-256: `4354b41a80706409cdc23b26979ca2679f88b62a8d48516e18fc7b9cd3ab8beb`.
Measurement revision: `2d804a766757cace5c3e0d41c3a95f36b7e31cc6`.
Release binary SHA-256: `af197d6d7f8297cdf2cdf72a2b417af328310cac33ef5f78bffaa6e2a5e80a83`.
Source method: `git-ls-tree-sha256-v1` over 126 tracked paths recorded in the raw receipt.

`source_sha256` hashes the canonical NUL-delimited Git tree entries for the receipt's explicit source closure. Run and verification fail closed when an admitted input is dirty, missing, or gains an undeclared production compile-time dependency. Checkout paths, `.git` representation, `.claude` worktrees, generated evidence, and unrelated tracked files are excluded by construction; changing an admitted engine, embedded-content, harness, dependency, test, or contract input requires regenerating every candidate receipt.

The raw artifact contains the pre-execution UTC start, exact orchestrator and worker argv, a fresh process group per cell/repeat, barrier and sampler receipts, the complete timestamped 5 ms aggregate-worker RSS series, hardware, versions, fixture tapes and hashes, all worker results, seeds, timings, outcomes, and deterministic checksums.

This artifact measures one candidate. The matrix verifier compares all three candidates from the same host, source revision, and release binary and requires equal logical work before the decision record is accepted.
