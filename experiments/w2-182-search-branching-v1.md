# Whole-rollout branching evidence: `full_clone/current_game_v1`

Contract: `manabot.search-branching.v1` (`9c8dfc3cae6bb85642caf61c074bb3ba2c0845f555c3024238f3d2d6fa85708f`)
Driver: `full_clone/current_game_v1`
Run: `2026-07-16T12:19:20.001033Z`; canonical: `true`
Evidence scope: current driver at this source state only; this is not a W2-179 before/after comparison.

## Primary whole-rollout evidence

| Cell | simulations/s | transitions/s | root p50 / p95 / p99 | peak RSS | peak delta | max live | cap rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| `flat-single-64-v1` | 288.6 | 37518.8 | 0.240s / 0.249s / 0.249s | 16.0 MiB | 0.8 MiB | 3 | 0.000% |
| `flat-saturated-64-v1` | 2022.9 | 266270.6 | 0.284s / 0.303s / 0.303s | 134.6 MiB | 12.9 MiB | 3 | 0.000% |
| `retained-single-8-v1` | 325.4 | 31638.0 | 0.008s / 0.009s / 0.009s | 14.5 MiB | 3.5 MiB | 17 | 0.000% |
| `retained-saturated-16-v1` | 352.0 | 32909.8 | 0.119s / 0.123s / 0.123s | 70.7 MiB | 25.1 MiB | 264 | 0.000% |

Peak RSS is the 5 ms sampled sum across worker processes and can double-count shared pages. Clone latency is diagnostic, not a storage decision.

## Branch lifecycle and journal counters

| Cell | eager forks | checkpoints | fork time | mark time | rollback time | journal peak | journal entries | marks / commits / rollbacks |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `flat-single-64-v1` | 128 | 3072 | 0.000s | 0.005s | 0.020s | null | null | null / null / null |
| `flat-saturated-64-v1` | 1024 | 24576 | 0.004s | 0.053s | 0.182s | null | null | null / null / null |
| `retained-single-8-v1` | 192 | 0 | 0.001s | 0.000s | 0.000s | null | null | null / null / null |
| `retained-saturated-16-v1` | 3072 | 0 | 0.009s | 0.000s | 0.000s | null | null | null / null / null |

`null` counters are unsupported by this driver, not observed zeros: system allocator has no counting hook; full-clone baseline has no undo journal or page-COW counters.

## Step and clone diagnostics

| Cell | samples/s | latency p50 / p95 / p99 | resets | reset time |
|---|---:|---:|---:|---:|
| `step-v1` | 126676.5 | 6.8µs / 16.9µs / 22.6µs | 116 | 0.002s |
| `clone-v1` | 223252.3 | 4.4µs / 4.6µs / 5.6µs | 0 | 0.000s |

## Reproduction and evidence

```bash
uv run scripts/bench_branching.py run --driver full_clone/current_game_v1
uv run scripts/bench_branching.py verify --driver full_clone/current_game_v1
```

Equivalence: `true` across 8 fixture/seed checks, each replayed twice.
Each primary cell also repeated its first measured root in a fresh worker group and matched the ordered deterministic result checksum.
Artifact SHA-256: `aedbeec0fcfb566c3208413d3c8bd529063c576f93e01feaa7ab29277215bb7a`.
Source SHA-256: `ba778d315bf1aab095c8eb5adeea7a0161184f1fc8d7a3c1820cc2c78e46c634`.

`source_sha256` hashes the whole tree except generated evidence, so this receipt is bound to one exact source state: it must be generated at the final landing tree and landed before `origin/main` moves under it. Any later rebase or source edit, even one touching no benchmark file, invalidates it and requires re-running `run`; re-check `verify` immediately before submitting. Regenerating this receipt is also what repairs the W2-182 baseline, which had been left failing `verify` with a contract-digest mismatch after the witness refactor edited the contract and benchmark without regenerating the artifact.

The raw artifact contains the pre-execution UTC start, exact orchestrator and worker argv, a fresh process group per cell/repeat, barrier and sampler receipts, the complete timestamped 5 ms aggregate-worker RSS series, hardware, versions, fixture tapes and hashes, all worker results, seeds, timings, outcomes, and deterministic checksums.

This artifact measures one candidate. Comparing candidates means comparing two artifacts from the same host and source state, matched by their equal per-cell result checksums. No page-COW representation is implemented, measured, or selected here.
