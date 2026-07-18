# RUL-9: Played Release and Training Workloads

**Run:** 2026-07-18  
**Matchup:** UR Lessons versus GW Allies  
**Verdict:** **MISS** — release miss, training pass, capacity pass, fallbacks pass.

## Result

Release missed only the player-facing live gates: WebSocket Command p95 was 150.492 ms against the 100 ms maximum, and live complete-game throughput was 0.347/s against the 1.0/s minimum. The inner semantic Command p95 remained 4.291 ms against its 10 ms maximum; headless and replay remained 588.5 and 599.1 steps/s against the 500/s minimum. Training passed at 4.520 roots/s, 578.5 traversals/s, and 0.0429 games/s against minima of 4.0, 512, and 0.04. The measurement-origin source reproduced one exact 132-Command semantic trace across live, headless, and persisted replay execution. All authority, search, cap, projection, and overflow counters remained zero. The decision remains `retain full_clone/current_game_v1`.

| Workload | Command p50 / p95 | Step throughput | Complete games | Peak RSS |
|---|---:|---:|---:|---:|
| Live release | 31.590 / 150.492 ms | 45.7/s | 0.347/s | 322.2 MiB |
| Headless release | 0.598 / 1.403 ms | 588.5/s | 4.458/s | shared release cell |
| Persisted replay | 0.596 / 1.380 ms | 599.1/s | 4.538/s | shared release cell |
| 4×128 training | 0.047 / 0.093 ms | 4.520 roots/s; 578.5 traversals/s | 0.0429/s | 1026.2 MiB |

Training PUCT decision latency was 648.3 ms p50 and 1398.1 ms p95 across 421 root Commands and 53888 traversals. All 4 games reached terminal.

## Semantic capacity and miss diagnosis

The compiled catalog contains 2088 active tokens. Tokens per admitted definition were 70 p50, 97 p95, and 148 maximum. Acting-view decisions reached 42 visible references with zero projection failure, unadmitted definition, truncation, or overflow.

The final workload did not reproduce the exploratory expanded-token pressure: its counterfactual maximum was 2493, below the 4,096 diagnostic frontier. The selected shared catalog remained 2088 active tokens with definition-row references and zero overflow. At the maximum, zone attribution was `{"battlefield": 851, "exile": 148, "graveyard": 1273, "hand": 221}`. The pre-registered capacity gates pass without clipping; retain `full_clone/current_game_v1` and the shared catalog/reference representation.

## Integrity

- Authority receipt: `57f5e2d1a482831e4e1edeb69632c0b34030d2e8ee701eab9b13b484c6937147`.
- Frozen PR #153 parity provenance: `af198bb3dcac542a6a34f1cbd8250938e4bfe0e79df7d284de4928873cc0f60b`; live/headless/replay parity samples are retained unchanged from the measurement origin.
- Immutable measurement origin: `498df1eda031f1d6ea68f72f792dfd68195bb10eab884b006aa7772b586564da` at `experiments/data/rul-9-played-workloads-v1.measurement.json` (file SHA-256 `9a3933a570772e8d3e04b59526faaf1d51b5fc0e26ba8c02e08eae36599bc951`).
- Measurement source closure: `3f612831df0179a6ce0c1f85e2def15cec6c68ae880ebea5d60d75e8bae13ac2` over 99 files; these files and the bound native extension `c95a85bba1128e6c3afdade5b5cf59dfeb3b1ec464fdd403cdf155a5cf834f8e` produced the samples.
- Canonical raw evidence: `59f323301f71283635b90fc6cdb43eda2c1079cbf5304c5fceb3d0262e8ba906` over 2316408 bytes in the immutable origin; the derived receipt binds this corpus without duplicating it.
- Derivation/report source closure: `1d078f3322aede00af77c3e75bc62e2c886059991e6be4233dc51237b100a411` over 99 files; this source only rederived and rendered the retained samples and did not produce them.
- Derivation native extension identity: `c95a85bba1128e6c3afdade5b5cf59dfeb3b1ec464fdd403cdf155a5cf834f8e` (_managym.cpython-312-darwin.so, release profile).
- Contract: `9593f3ff7e0eed8fe0316ec66bda96296815480fc0d37441831586cdf3555477`.
- Derived evidence artifact: `21fb0d66416ed0f897fb954ced69f71841e136984e2d83869c4806bfc4b707d7`.
- Release RSS samples: 842; training RSS samples: 782.
- Raw evidence retains every Command/PUCT duration, game duration, RSS sample, semantic census row, terminal hash, outcome hash, and fallback/cap counter.

## Strongest confound

These absolute performance gates were measured on one Apple Silicon host. Exact workload, measurement source, toolchain, worker topology, and binary identities are pinned, but another host may move latency and RSS without changing the representation. The derivation source is bound separately and did not rerun the workload. The fail-closed verifier treats measurement-origin or derivation drift as new evidence, not as a continuation of this run.

## Reproduce

```bash
uv run experiments/runners/run_rul9_played_workloads.py \
  --migrate \
  --measurement-origin experiments/data/rul-9-played-workloads-v1.measurement.json \
  --contract experiments/contracts/rul-9-played-workloads-v1.json \
  --out experiments/data/rul-9-played-workloads-v1.json \
  --report experiments/rul-9-played-workloads-v1.md
./scripts/verify-rul9-played-workloads
```
