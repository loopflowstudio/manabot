# W2-182: whole-rollout branching baseline

Contract: `manabot.search-branching.v1` (`a3a6c721bd4a349787d32491e04103d81ce0ee3d71318bb83046414e3ace6089`)
Driver: `full_clone/current_game_v1`
Run: `2026-07-15T23:04:44.325776Z`; canonical: `true`
Evidence scope: current driver at this source state only; this is not a W2-179 before/after comparison.

## Primary whole-rollout evidence

| Cell | simulations/s | transitions/s | root p50 / p95 / p99 | peak RSS | peak delta | max live | cap rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| `flat-single-64-v1` | 622.3 | 80894.3 | 0.158s / 0.276s / 0.276s | 8.6 MiB | 0.7 MiB | 3 | 0.000% |
| `flat-saturated-64-v1` | 2617.8 | 344575.5 | 0.309s / 0.642s / 0.642s | 68.7 MiB | 2.6 MiB | 3 | 0.000% |
| `retained-single-8-v1` | 487.5 | 47401.1 | 0.008s / 0.009s / 0.009s | 8.6 MiB | 1.7 MiB | 17 | 0.000% |
| `retained-saturated-16-v1` | 396.3 | 37052.7 | 0.175s / 0.329s / 0.329s | 45.5 MiB | 10.1 MiB | 264 | 0.000% |

Peak RSS is the 5 ms sampled sum across worker processes and can double-count shared pages. Clone latency is diagnostic, not a storage decision.

## Step and clone diagnostics

| Cell | samples/s | latency p50 / p95 / p99 | resets | reset time |
|---|---:|---:|---:|---:|
| `step-v1` | 148622.0 | 6.0µs / 13.2µs / 17.5µs | 116 | 0.002s |
| `clone-v1` | 313375.2 | 3.1µs / 3.7µs / 4.0µs | 0 | 0.000s |

## Reproduction and evidence

```bash
uv run scripts/bench_branching.py run
uv run scripts/bench_branching.py verify
```

Equivalence: `true` across 8 fixture/seed checks, each replayed twice.
Each primary cell also repeated its first measured root in a fresh worker group and matched the ordered deterministic result checksum.
Artifact SHA-256: `d8c1e84a16d63ddc804d87b194213b779faa1a8d00f31dc1d4780d77be3a40b4`.
Source SHA-256: `ace6bb6f61f659d896bc584d290ac888010a108b964c894d70118f9dd9485ee3`.

The raw artifact contains the pre-execution UTC start, exact orchestrator and worker argv, a fresh process group per cell/repeat, barrier and sampler receipts, the complete timestamped 5 ms aggregate-worker RSS series, hardware, versions, fixture tapes and hashes, all worker results, seeds, timings, outcomes, and deterministic checksums. No undo or page-COW implementation was selected or measured.
