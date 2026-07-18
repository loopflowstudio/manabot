# RUL-10: Jeong Jeong's Deserters Vertical Slice

**Verdict:** **PASS** required gates; product-budget observation **MISS**.  
**Classification:** content-only card semantics plus one reusable immutable pack-catalog kernel extension.

## Result

Literal seeds 1 and 3 reached terminal through production Etude WebSocket, direct headless Commands, and persisted canonical replay. Each tape contains a Jeong cast followed by a mandatory target Command and an observed +1/+1 counter delta. Revision witnesses and ordered semantic event groups matched on all three surfaces, with zero authority fallback or semantic overflow.

| Surface | Command p50 / p95 | Inner p95 | Steps/s | Games/s | Peak RSS |
|---|---:|---:|---:|---:|---:|
| live | 35.442 / 106.653 ms | 2.477 ms | 41.1 | 0.324 | 335.8 MiB |
| headless | 0.426 / 0.655 ms | 0.655 ms | 1041.4 | 8.200 | 230.1 MiB |
| replay | 0.419 / 0.671 ms | 0.671 ms | 1039.8 | 8.187 | 229.8 MiB |

## Played increment

- Seeds: `[1, 3]`.
- Prompt families: `{"CHOOSE_TARGET": 23, "DECLARE_ATTACKER": 73, "DECLARE_BLOCKER": 18, "DISCARD_THEN_DRAW": 4, "PAY_OR_NOT": 1, "PRIORITY": 132, "SCRY": 3}`.
- Offer families: `{"activate": 1, "cast": 46, "choose": 31, "declare_attackers": 73, "declare_blockers": 18, "pass_priority": 55, "play_land": 30}`.
- Maximum uncapped offer count: `8`.
- Linked cast/target/counter witnesses: `2`.
- Exactness mismatches: `0`.

## Semantic capacity

The admitted programs have 1389 active tokens and Jeong's program uses 39 tokens; played decisions reached 50 visible references and 1760 expanded program tokens. Overflow and unadmitted visible definition counts are zero. Full learning-definition projection is not claimed: Jeong's exact Rebel subtype is outside the frozen v1 categorical vocabulary, and this increment does not migrate schemas.

## Gates

Required gates: `pass` ([]). Product budgets are reported without expanding this Task into optimization: `miss` (['live outer Command p95', 'live games/s']). The live outer p95 is 106.653 ms against 100 ms and live completion is 0.324/s against 1.0/s.

## Integrity

- Evidence artifact: `7783a88676abe170dba7fad6a351beeee1975eb7d2fe41b44ccbf5e82ac2ba9f`.
- Contract: `2c59bfd185c887fbcceb583399b52cf18c6f9b57764b836c351be1956643cd8d`.
- The four RUL-9 files were hash-checked but their workloads were not run.
- Raw evidence retains per-Command timings, RSS samples, semantic token samples, prompt/fallback inventories, both authorities, and both persisted canonical replays.

## Reproduce

```bash
./scripts/verify-tla-jeong-increment
```
