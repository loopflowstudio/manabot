# MTG-268 — batched local-update rollouts

The Task directive and `manabot/README.md#local-update-search` supplied the plan;
no separate accepted design was present. The implementation follows those docs.
Reversible choice: one live frontier across all sampled worlds and supported root
actions, independent policy RNGs, stable world/action receipt order, and a single
batched predictor that also owns singleton root inference. No batching flag.

Delete — do not maintain (done): the nested production per-branch rollout loop
and singleton-only prediction implementation. Complete private audit tapes,
observation hashes, branch seeds, native Command application, terminal/sign
semantics, sampling identities, deadline errors and historical receipt readers
survive. Compound execution was already removed; it was not restored.
The benchmark loads the old implementation from git only as a comparison oracle.
Compression also removed the redundant singleton-prediction test adapter and
full Observation parsing for materialized-root hash checks. No deletion remains.

No product question is open. The docs' 16-world, depth-20, 30-second example is an
illustrative budget, not a latency promise. Docs now explain float32 batching
rounding and possible trajectory changes at sampling thresholds. Runtime replay
still requires the retained implementation and allocation identities.

## Measurement (2026-11-08)

`uv run --extra dev python scripts/bench_local_update.py --reference 841b1c50c`
passed discrete receipt comparison and numeric tolerance checks for every cell.
Raw result: `scratch/mtg-268-timing.json`. Apple M4 Max, macOS 26.0.1, Torch 2.10.0,
one CPU/inter-op thread, initialized 64-wide/two-layer scalar policy, current
observation defaults. Two seeded opening positions (42 and 43), search seed 17,
depth 20; four/five supported root actions. One timed decision per cell; warm
singleton inference, then before/after on the same root. Separate profiling pass.
These are bounded fixed-position searches, not full games or evaluations.

| Worlds | Position 42 before → after (s) | Position 43 before → after (s) |
| --- | --- | --- |
| 1 | 0.266 → 0.195 | 0.312 → 0.240 |
| 4 | 0.940 → 0.679 | 1.215 → 0.860 |
| 16 | 3.833 → 2.647 | 4.699 → 3.321 |

Speedup 1.30–1.45x. At 16 worlds/position 42, network forwards fell from 1,281
to 21 (one root plus 20 plies); profiled inclusive network time 2.436 → 1.130 s.
All RolloutReceipt fields except numeric values matched exactly. Largest value
error 5.96e-8, largest target error 7.09e-9; tests use atol 1e-6 / rtol 1e-5.

Outside-network costs at the same position/allocation:
- Native semantic JSON plus viewer hashing: 1,298 calls, 0.302 → 0.259 s
  (~233 → 200 μs/call). This canonical witness remains required.
- Full Python Observation parsing: 1,298 → 18 calls, 0.149 → 0.0011 s.
  Per-ply code now reads the native hash from JSON without rebuilding and
  reserializing the unused DecisionFrame. Ordinary JSON decoding remains; the
  latter figure measures full-model parsing only, not all JSON decoding.
- Session snapshot: 64 calls, 0.0103 → 0.0070 s (~161 → 109 μs/rollout).
  Under 1% of elapsed decision time; kept unchanged, with complete audit content.

Limit: 16 worlds are **not** near the old one-world cost on this fixture. The
study's MTG-183/w13 trained checkpoint was not measured or rebound to this world.
There is no trained strength or production-latency claim. Wider rollout batches
retain all native branches and tensors together, so memory scales with worlds
and supported root actions. Native stepping, encoding and per-row network work
remain; further optimization would need its own measured choice.

## Validation and handoff

Focused pytest selections: 10 distinct checks passed (seven new batching,
terminal/sign, support, nonfinite and deadline cases; existing exact-belief replay
and direct/learned lifecycle cases). Hash-path change rechecked the four affected
parity/frontier cases. Ruff check/format and `git diff --check` passed.
Gate owns the broader affected suites. No new test module or Rust change.
No games, evaluations, training, servers or watchers were started.

A prior docs presentation failed because Desktop had no window for this checkout;
the reviewable files remain `manabot/README.md` and `README.md`.
Inherited b015d5 and older Task notes remain at
`841b1c50c:scratch/questions.md`; they select no work here.

## Compression (2026-11-08)

Materialized roots and rollout plies now share the native hash reader; root
support is computed once. Singleton rollout execution alone supplies the parity
oracle, and the configurable test fixture avoids loading a second teacher.
The timings above predate these cuts and were not rerun.

`uv run --extra dev pytest tests/sim/test_local_update_sampling.py -q -k 'batched or frontier or batch_'`: 7 passed. Separate exact-belief replay and direct/learned lifecycle selection: 3 passed. Ruff check/format and `git diff --check` passed.
