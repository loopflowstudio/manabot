# Selected-match public-commitment parity v1

This checked receipt re-executes the immutable 132-Command UR Lessons versus
GW Allies authority tape through Etude live WebSocket play, persisted canonical
replay, and direct managym execution. It binds the viewer-safe
`PublicCommitment` stream, two exact-range `BeliefTracker` histories, one
materialized revision-29 discard hypothesis, and atomic unsupported/mismatched
failure proofs.

Run from the repository root:

```bash
./scripts/verify-public-commitment-parity
```

The verifier rebuilds the local Python extension, runs debug Rust and focused
uv-managed Python tests, and recomputes the receipt. It references the frozen
RUL-9 measurement origin but does not rerun its workloads or rewrite any RUL-9
or INT-12 evidence.
