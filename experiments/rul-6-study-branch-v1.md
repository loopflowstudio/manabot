# Production Study fork/apply/return evidence

Contract: `etude.study-branch.contract.v1` (`91d522aa824566b5dc32922960ea307370c49232c442c33048d5e7e0bd10fb53`)
Driver: `full_clone/current_game_v1`
Command path: `structured_offers/step_structured_v1`
Run: `2026-07-18T06:22:17.072562Z`; canonical: `true`

## Interactive lifecycle

| Phase | p50 | p95 | p99 | max |
|---|---:|---:|---:|---:|
| Fork | 0.551 ms | 0.624 ms | 0.700 ms | 1.994 ms |
| Publish offers | 0.007 ms | 0.015 ms | 0.022 ms | 0.038 ms |
| Structured apply | 0.153 ms | 0.186 ms | 0.224 ms | 0.272 ms |
| Return | 0.210 ms | 0.250 ms | 0.293 ms | 0.814 ms |
| End to end | 0.921 ms | 1.070 ms | 1.173 ms | 2.701 ms |

Sustained rate: **943.0 cycles/s** across 2000 checked cycles.

## Retained siblings

Retained 512 simultaneous production branches; applied 256 and re-checked 256 untouched siblings.
Peak RSS delta: **54.0 MiB** (peak 147.3 MiB).

## Exactness, privacy, incarnation, and failure

- Source digest: `068c7029a81cf0ee08a0a338b7a6f6c2f1f00d80c442a40f0f62a61029c99fa2`.
- Bound object: entity `16`, incarnation `0`.
- Child-only zone-change checks: 2257; object-ref mismatches: 0.
- Viewer-private observation checks: 2257; opponent-hand exposures: 0.
- Typed failure cases: 9; rejected-branch mutations: 0.
- Study fallbacks: 0; source authority fallback counters remained zero.

## Reproduction and identity

```bash
uv run scripts/bench_study_branch.py measure
uv run scripts/bench_study_branch.py verify
```

Host: `macOS-26.0.1-arm64-arm-64bit`; Python `3.12.12`; logical CPUs: `16`.
Compiled extension SHA-256: `2335165aff82c05cdc53c011d0622626bc72de590c9a7556db32e53a8cc3b6a4`.
Source closure SHA-256: `2c1ad685314cdff3859568990acd939955bd6bb36be4fef27d864dcddd858d7e` over 110 recorded paths.
Artifact SHA-256: `7035661f24124765d61ee7ade63371cb58360a21dc26e227a584af7f1ceaa5bf`.

Raw evidence retains every latency sample, all RSS phase samples, exact source and binary identity, the workload address/object/return digests, failure receipts, and zero-mismatch counters.
