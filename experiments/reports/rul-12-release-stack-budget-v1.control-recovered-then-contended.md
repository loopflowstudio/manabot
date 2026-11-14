# RUL-12 Release-Stack Budget Receipt

## Result

Overall admission: **MISS**. The fixed 132-Command UR Lessons versus GW Allies trace was measured through the same TestClient client-send-to-accepted-ack boundary as RUL-9, with no summary cache and synchronous terminal replay persistence.

| Surface | Command p50 | Command p95 | Steps/s | Games/s |
|---|---:|---:|---:|---:|
| live | 21.358 ms | 131.500 ms | 49.7 | 0.377 |
| headless | 1.235 ms | 3.913 ms | 255.5 | 1.935 |
| replay | 1.182 ms | 2.846 ms | 280.5 | 2.125 |

Live inner semantic Command p50/p95 was 4.194/12.886 ms. Release peak RSS was 322.0 MiB across 232 retained samples.

## Training and capacity

The unchanged `full_clone/current_game_v1` 4x128 workload delivered 1.928 roots/s, 246.8 traversals/s, and 0.0183 complete games/s. PUCT p95 was 3671.358 ms, Command p95 was 0.099 ms, and peak RSS was 1025.2 MiB.

The semantic catalog used 2088 active tokens; maximum tokens per definition were 148, maximum visible references were 42, and every overflow, projection-failure, unadmitted-definition, native-mismatch, authority fallback, and training fallback count was zero.

## Exactness and provenance

Live, headless, and persisted replay retained one terminal witness and one ordered logical consequence hash. The current proof reran 798 viewer projections, spectator rejection, `stale_object` and `stale_revision` atomic rejection, 62 public commitments, a materialized revision-29 hypothesis, and zero `RulesProviderGap` without changing the checked historical RUL-11 receipt.

Artifact: `c14f3a43b17840217113f9edeaf8e03f9c430e9a725a62222b57ed2fa47b1959`. Source closure: `7791160af186a977e0355e005ec42be05576082915d2d53b02f66b62c5cd5fa5`. Native extension: `8398c3d0746f9550bb4ce376cb79cd202d37dc8e939767f04a7cb8c6ab9d1515`. RUL-9 origin artifact `498df1eda031f1d6ea68f72f792dfd68195bb10eab884b006aa7772b586564da` was a byte-checked `rerun: false` input.

WebSocket stack: fastapi 0.135.1, starlette 0.52.1, pydantic 2.12.5, pydantic-core 2.41.5, anyio 4.12.1, httpx 0.28.1, uvicorn 0.41.0, websockets 16.0. Absolute performance remains single-host evidence.

## Reproduce

```bash
./scripts/verify-rul12-release-stack-budget
```
