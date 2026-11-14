# Whole-rollout branching evidence: `full_clone/current_game_v1`

Contract: `manabot.search-branching.v1` (`9c8dfc3cae6bb85642caf61c074bb3ba2c0845f555c3024238f3d2d6fa85708f`)
Driver: `full_clone/current_game_v1`
Run: `2026-07-16T19:46:40.592943Z`; canonical: `true`
Evidence scope: current driver at this source state only; this is not a W2-179 before/after comparison.

## Primary whole-rollout evidence

| Cell | simulations/s | transitions/s | root p50 / p95 / p99 | peak RSS | peak delta | max live | cap rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| `flat-single-64-v1` | 294.5 | 38285.0 | 0.240s / 0.258s / 0.258s | 15.9 MiB | 0.9 MiB | 3 | 0.000% |
| `flat-saturated-64-v1` | 2001.7 | 263485.9 | 0.280s / 0.379s / 0.379s | 128.7 MiB | 2.7 MiB | 3 | 0.000% |
| `retained-single-8-v1` | 321.7 | 31283.1 | 0.008s / 0.010s / 0.010s | 14.9 MiB | 3.5 MiB | 17 | 0.000% |
| `retained-saturated-16-v1` | 350.6 | 32782.0 | 0.121s / 0.125s / 0.125s | 81.7 MiB | 29.9 MiB | 264 | 0.000% |

Peak RSS is the 5 ms sampled sum across worker processes and can double-count shared pages. Clone latency is diagnostic, not a storage decision.

## Branch lifecycle and journal counters

| Cell | eager forks | checkpoints | fork time | mark time | rollback time | journal peak | COW peak | journal entries | marks / commits / rollbacks |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `flat-single-64-v1` | 128 | 3072 | 0.001s | 0.009s | 0.023s | null | null | null | null / null / null |
| `flat-saturated-64-v1` | 1024 | 24576 | 0.005s | 0.108s | 0.230s | null | null | null | null / null / null |
| `retained-single-8-v1` | 192 | 0 | 0.001s | 0.000s | 0.000s | null | null | null | null / null / null |
| `retained-saturated-16-v1` | 3072 | 0 | 0.014s | 0.000s | 0.000s | null | null | null | null / null / null |

`null` counters are unsupported by this driver, not observed zeros: system allocator has no counting hook; full-clone baseline has no undo journal or page-COW counters.

## Step and clone diagnostics

| Cell | samples/s | latency p50 / p95 / p99 | resets | reset time |
|---|---:|---:|---:|---:|
| `step-v1` | 127368.0 | 6.8µs / 16.9µs / 22.6µs | 116 | 0.002s |
| `clone-v1` | 161759.2 | 6.1µs / 6.3µs / 7.7µs | 0 | 0.000s |

## Reproduction and evidence

```bash
uv run scripts/bench_branching.py run-matrix
uv run scripts/bench_branching.py verify-matrix
```

Equivalence: `true` across 8 fixture/seed checks, each replayed twice.
Each primary cell also repeated its first measured root in a fresh worker group and matched the ordered deterministic result checksum.
Artifact SHA-256: `0d623a67b186e9c138cbee99ab14cc0df948d30e03ec999c52b7394818af90f0`.
Source SHA-256: `4354b41a80706409cdc23b26979ca2679f88b62a8d48516e18fc7b9cd3ab8beb`.
Measurement revision: `2d804a766757cace5c3e0d41c3a95f36b7e31cc6`.
Release binary SHA-256: `af197d6d7f8297cdf2cdf72a2b417af328310cac33ef5f78bffaa6e2a5e80a83`.
Source method: `git-ls-tree-sha256-v1` over 126 tracked paths recorded in the raw receipt.

`source_sha256` hashes the canonical NUL-delimited Git tree entries for the receipt's explicit source closure. Run and verification fail closed when an admitted input is dirty, missing, or gains an undeclared production compile-time dependency. Checkout paths, `.git` representation, `.claude` worktrees, generated evidence, and unrelated tracked files are excluded by construction; changing an admitted engine, embedded-content, harness, dependency, test, or contract input requires regenerating every candidate receipt.

The raw artifact contains the pre-execution UTC start, exact orchestrator and worker argv, a fresh process group per cell/repeat, barrier and sampler receipts, the complete timestamped 5 ms aggregate-worker RSS series, hardware, versions, fixture tapes and hashes, all worker results, seeds, timings, outcomes, and deterministic checksums.

This artifact measures one candidate. The matrix verifier compares all three candidates from the same host, source revision, and release binary and requires equal logical work before the decision record is accepted.
