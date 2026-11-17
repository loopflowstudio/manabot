# Whole-rollout branching evidence: `dense_page_cow_undo/event_pages_4k_v1`

Contract: `manabot.search-branching.v1` (`9c8dfc3cae6bb85642caf61c074bb3ba2c0845f555c3024238f3d2d6fa85708f`)
Driver: `dense_page_cow_undo/event_pages_4k_v1`
Run: `2026-07-16T19:46:40.592943Z`; canonical: `true`
Evidence scope: current driver at this source state only; this is not a W2-179 before/after comparison.

## Primary whole-rollout evidence

| Cell | simulations/s | transitions/s | root p50 / p95 / p99 | peak RSS | peak delta | max live | cap rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| `flat-single-64-v1` | 272.7 | 35447.9 | 0.325s / 0.357s / 0.357s | 24.2 MiB | 5.0 MiB | 3 | 0.000% |
| `flat-saturated-64-v1` | 1686.8 | 222036.7 | 0.410s / 0.535s / 0.535s | 193.2 MiB | 22.0 MiB | 3 | 0.000% |
| `retained-single-8-v1` | 311.5 | 30291.9 | 0.009s / 0.011s / 0.011s | 13.9 MiB | 3.2 MiB | 17 | 0.000% |
| `retained-saturated-16-v1` | 331.7 | 31011.0 | 0.126s / 0.203s / 0.203s | 37.7 MiB | 2.1 MiB | 264 | 0.000% |

Peak RSS is the 5 ms sampled sum across worker processes and can double-count shared pages. Clone latency is diagnostic, not a storage decision.

## Branch lifecycle and journal counters

| Cell | eager forks | checkpoints | fork time | mark time | rollback time | journal peak | COW peak | journal entries | marks / commits / rollbacks |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `flat-single-64-v1` | 0 | 0 | 0.000s | 0.000s | 0.475s | 4.64 MiB | 0.00 MiB | 34385 | 6144 / 3072 / 3072 |
| `flat-saturated-64-v1` | 0 | 0 | 0.006s | 0.001s | 4.727s | 30.23 MiB | 0.03 MiB | 263176 | 49152 / 24576 / 24576 |
| `retained-single-8-v1` | 0 | 0 | 0.000s | 0.000s | 0.000s | 0.11 MiB | 0.07 MiB | 86 | 128 / 128 / 0 |
| `retained-saturated-16-v1` | 0 | 0 | 0.003s | 0.000s | 0.000s | 1.66 MiB | 1.01 MiB | 86 | 2048 / 2048 / 0 |

`null` counters are unsupported by this driver, not observed zeros: system allocator has no counting hook; cow_bytes measures branch-private copied 4096-byte event pages.

## Step and clone diagnostics

| Cell | samples/s | latency p50 / p95 / p99 | resets | reset time |
|---|---:|---:|---:|---:|
| `step-v1` | 125677.7 | 6.9µs / 17.0µs / 23.1µs | 116 | 0.002s |
| `clone-v1` | 818472.9 | 1.2µs / 1.3µs / 1.5µs | 0 | 0.000s |

## Reproduction and evidence

```bash
uv run scripts/bench_branching.py run-matrix
uv run scripts/bench_branching.py verify-matrix
```

Equivalence: `true` across 8 fixture/seed checks, each replayed twice.
Each primary cell also repeated its first measured root in a fresh worker group and matched the ordered deterministic result checksum.
Artifact SHA-256: `707d346625a198af57b0b24af9237969b7fc5ea9bb84979ed8e80cc3db33dcd3`.
Source SHA-256: `4354b41a80706409cdc23b26979ca2679f88b62a8d48516e18fc7b9cd3ab8beb`.
Measurement revision: `2d804a766757cace5c3e0d41c3a95f36b7e31cc6`.
Release binary SHA-256: `af197d6d7f8297cdf2cdf72a2b417af328310cac33ef5f78bffaa6e2a5e80a83`.
Source method: `git-ls-tree-sha256-v1` over 126 tracked paths recorded in the raw receipt.

`source_sha256` hashes the canonical NUL-delimited Git tree entries for the receipt's explicit source closure. Run and verification fail closed when an admitted input is dirty, missing, or gains an undeclared production compile-time dependency. Checkout paths, `.git` representation, `.claude` worktrees, generated evidence, and unrelated tracked files are excluded by construction; changing an admitted engine, embedded-content, harness, dependency, test, or contract input requires regenerating every candidate receipt.

The raw artifact contains the pre-execution UTC start, exact orchestrator and worker argv, a fresh process group per cell/repeat, barrier and sampler receipts, the complete timestamped 5 ms aggregate-worker RSS series, hardware, versions, fixture tapes and hashes, all worker results, seeds, timings, outcomes, and deterministic checksums.

This artifact measures one candidate. The matrix verifier compares all three candidates from the same host, source revision, and release binary and requires equal logical work before the decision record is accepted.
