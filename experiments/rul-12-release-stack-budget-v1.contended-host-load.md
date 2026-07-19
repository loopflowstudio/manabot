# RUL-12 Release-Stack Budget Receipt

## Result

Overall admission: **MISS**. The fixed 132-Command UR Lessons versus GW Allies trace was measured through the same TestClient client-send-to-accepted-ack boundary as RUL-9, with no summary cache and synchronous terminal replay persistence.

| Surface | Command p50 | Command p95 | Steps/s | Games/s |
|---|---:|---:|---:|---:|
| live | 37.651 ms | 163.683 ms | 34.8 | 0.263 |
| headless | 1.126 ms | 5.994 ms | 239.2 | 1.812 |
| replay | 1.118 ms | 8.511 ms | 193.0 | 1.462 |

Live inner semantic Command p50/p95 was 4.164/16.969 ms. Release peak RSS was 323.7 MiB across 369 retained samples.

## Training and capacity

The unchanged `full_clone/current_game_v1` 4x128 workload delivered 1.708 roots/s, 218.7 traversals/s, and 0.0162 complete games/s. PUCT p95 was 3745.683 ms, Command p95 was 0.104 ms, and peak RSS was 1031.2 MiB.

The semantic catalog used 2088 active tokens; maximum tokens per definition were 148, maximum visible references were 42, and every overflow, projection-failure, unadmitted-definition, native-mismatch, authority fallback, and training fallback count was zero.

## Exactness and provenance

Live, headless, and persisted replay retained one terminal witness and one ordered logical consequence hash. The current proof reran 798 viewer projections, spectator rejection, `stale_object` and `stale_revision` atomic rejection, 62 public commitments, a materialized revision-29 hypothesis, and zero `RulesProviderGap` without changing the checked historical RUL-11 receipt.

Artifact: `71641a8c10691dbf1fc7f07e819d3e6514b80630a1ea0bb92a5801f0e11d0ed9`. Source closure: `7791160af186a977e0355e005ec42be05576082915d2d53b02f66b62c5cd5fa5`. Native extension: `8398c3d0746f9550bb4ce376cb79cd202d37dc8e939767f04a7cb8c6ab9d1515`. RUL-9 origin artifact `498df1eda031f1d6ea68f72f792dfd68195bb10eab884b006aa7772b586564da` was a byte-checked `rerun: false` input.

WebSocket stack: fastapi 0.135.1, starlette 0.52.1, pydantic 2.12.5, pydantic-core 2.41.5, anyio 4.12.1, httpx 0.28.1, uvicorn 0.41.0, websockets 16.0. Absolute performance remains single-host evidence.

## Reproduce

```bash
./scripts/verify-rul12-release-stack-budget
```
