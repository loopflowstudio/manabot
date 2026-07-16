# Structured policy decoder benchmark

Status: **PASS**. Workload `structured-policy-v1` (`sha256:9168f071c207a4daec4b396b9c829fdf0936fe37c786bd6d1c3d93c37cb15a78`).

Run with:

```sh
uv run scripts/bench_structured_policy.py \
  --workload experiments/workloads/structured-policy-v1.json \
  --out experiments/data/structured-policy-v1.json \
  --report experiments/structured-policy-decoder.md
```

## Correctness

The fixed frontier reached 35 explicit target candidates and 64 represented attacker declarations. It recorded 0 overflows, 0 illegal decoder outputs, and 0 trace mismatches.

Shared-state action agreement was 6435/6435 overall and 106/106 on structured attacker decisions.

| Adapter | UR win rate | GW win rate | draws | cap hits |
|---|---:|---:|---:|---:|
| Structured hybrid | 12.5% | 87.5% | 0 | 0 |
| Legacy adapter | 12.5% | 87.5% | 0 | 0 |

| Deck/seat | Structured | Legacy |
|---|---:|---:|
| UR on play | 25.0% | 25.0% |
| UR on draw | 0.0% | 0.0% |
| GW on play | 100.0% | 100.0% |
| GW on draw | 75.0% | 75.0% |

The game list is seat-balanced by alternating which deck is seat 0/on the play. Win rate is migration evidence for the fixed synthetic scorer, not a policy-strength claim.

## Performance

| Adapter | focused p50 | focused p95 | games/s | legacy-equivalent actions/s | peak RSS |
|---|---:|---:|---:|---:|---:|
| Structured hybrid | 116.4 µs | 191.8 µs | 180.107 | 147732.5 | 25.7 MiB |
| Legacy adapter | 90.2 µs | 206.0 µs | 186.994 | 153381.7 | 25.7 MiB |

Latency includes offer projection, ragged flattening, deterministic scoring, decoding, and application on alternating Bolt/attacker fixtures. Peak RSS comes from fresh adapter processes. Throughput counts legacy-equivalent actions so an atomic declaration is not credited merely for collapsing sequential prompts.

## Boundary

The prototype is experiment-only. Full games use structured decoding for complete attacker offers and explicitly fall back to the same positional action in both runs for unsupported decisions. Priority pass and Lightning Bolt targeting are covered by the fixed frontier. The production policy network, 32-row observation tensor, legacy ABI, and rules semantics are unchanged.
