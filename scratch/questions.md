# MTG-268 — batched local-update rollouts

The Task directive is the design; no separate accepted design was present.
It says: "Batching should bring 8-16 worlds near today's one-world cost."
That latency outcome remains unmet. Neither implementation progress nor the
measurements below relax it. `manabot/README.md#local-update-search` describes
the current behavior.

Iteration direction: "extend scripts/bench_local_update.py to distinguish
observation encoding, tensor assembly, native stepping and model costs; optimize
the dominant measured cost while preserving complete receipts and legal support."

## Design and deletion cut

One live frontier spans all sampled worlds and supported root actions, with
independent policy RNGs and stable world/action receipt order. Singleton root
inference uses the same batched predictor. No batching flag. Complete private
audits, native Commands, hashes, terminal/sign semantics, sampling identities,
deadline failures and historical receipt readers survive. Compound execution
remains removed.

Reversible choice for this iteration: CPU inference attention removes only
columns unused by every row in the batch, then restores the original coordinates
before action joins and value pooling. Historical mean keeps its padded
denominator; the value token stays last and neutral. Training retains its dense
autograd path; device inference avoids synchronization for a dynamic shape.
Checkpoint parameters and observation schemas remain unchanged.

### Delete — do not maintain

- Production sequential rollout loop and singleton-only predictor: replaced by
  the shared frontier; old source exists only as the benchmark oracle in git.
- Full semantic Observation parsing at every ply/materialized root: native
  canonical hashes are read from JSON; complete receipt content survives.
- CPU inference attention over globally empty object columns: compact, attend,
  restore. Keep all legal actions and original object joins.
- Benchmark loading only the old teacher: load the old ActorCritic too, so the
  baseline does not silently inherit optimized attention.
- Duplicate root decision-frame projection and registration config parsing:
  reuse the parsed frame and config. Restore attention objects and the optional
  value token in one scatter.

## Implementation measurement (2026-11-08)

`uv run --extra dev python scripts/bench_local_update.py` passed all six cells.
Raw output: `scratch/mtg-268-timing-final.json`; reference `841b1c50c` supplies
both teacher and model. Apple M4 Max, macOS 26.0.1, Torch 2.10.0, one CPU/inter-op
thread. Initialized 64-wide/two-layer scalar policy, current observation defaults,
checkpoint SHA256 `5e3e03fb91c9fba2ea75bb7736d797505ea5385cf9dd538a9d0bab6b582899a6`.
Two seeded opening positions (42/43), search seed 17, depth 20, four/five supported
root actions. One timed decision per cell, warm singleton inference, then
before/after on the same root; separate profiling pass at position 42/16 worlds.
Bounded fixed-position searches only; no games or evaluations.

| Worlds | Position 42 before → after (s) | Position 43 before → after (s) |
| --- | --- | --- |
| 1 | 0.243 → 0.149 | 0.302 → 0.175 |
| 4 | 0.968 → 0.514 | 1.211 → 0.647 |
| 16 | 3.775 → 1.950 | 5.658 → 2.466 |

Measured speedup 1.62–2.29x; 16 worlds still cost **8.0–8.2x the old one-world
search**. These single samples are not a variance estimate or production claim.
All discrete RolloutReceipt fields (including complete audit JSON and every
observation hash), support and allocation matched. Maximum base-probability error
1.49e-10, rollout value error 5.96e-8, target error 6.93e-9; enforced tolerances
atol 1e-6 / rtol 1e-5. Float32 batching and attention shapes can change rounding;
a sampling-threshold crossing can change trajectories elsewhere. Replay still
requires the retained source, runtime, checkpoint and allocation identities.

Expanded initial profile (`scratch/mtg-268-profile-initial.json`, same position
and allocation before this iteration) identified model cost 1.162 s, of which
attention was 0.770 s. Native step/audit was 0.940 s, Python encoding 0.280 s,
semantic JSON/hash 0.262 s and tensor assembly 0.014 s. This selected attention
as the optimization target. Final profiles:

| Inclusive category | Original sequential (s) | Final (s) |
| --- | --- | --- |
| Model | 2.294 | 0.451 |
| Attention (inside model) | 0.985 | 0.076 |
| Object embeddings (inside model) | 1.074 | 0.357 |
| Native Command step and audit | 0.941 | 0.916 |
| Observation encoding | 0.331 | 0.273 |
| Native observation extraction | 0.005 | 0.007 |
| Tensor assembly | 0.017 | 0.013 |
| Semantic JSON and native viewer hash | 0.274 | 0.255 |
| Full Python Observation parsing | 0.134 | 0.00014 |
| Session snapshot | 0.0084 | 0.0069 |

Model forwards fell 1,281 → 21. Native steps remain 1,280; canonical semantic
witnesses 1,298; snapshots 64. Full Python parsing fell 1,298 → 2. Nested model
categories must not be added to their parent. JSON/hash still includes canonical
serialization; the parsing category does not count ordinary JSON decoding.
Native stepping includes Command checks, retained audit and returned observation.
The initial attention profile and final profile are separate runs, not paired
microbenchmarks. Prior batching evidence remains in `scratch/mtg-268-timing.json`
and `63f7d6a:scratch/questions.md` (1.30–1.45x, before final root parsing removal).

## Validation and remaining work

`uv run --extra dev pytest tests/model/test_value_aggregation.py tests/sim/test_local_update_sampling.py tests/sim/test_local_update.py -q -k 'inference_attention or batched or frontier or batch_ or real_root_replay'`
— 14 passed, 30 deselected after compression. Six attention cases cover all value aggregations,
one/two layers, interior padding, varying per-row support, original coordinates,
and gradients after inference; eight retained cases cover root replay, frontier
receipt parity, terminal/sign handling, full support, nonfinite values and deadlines.
Ruff check/format and `git diff --check` passed; `lf context` is within both limits.

Compression rerun: `uv run --extra dev python scripts/bench_local_update.py`
passed all six cells (`scratch/mtg-268-timing-compress.json`, same checkpoint and
settings). For worlds 1/4/16, position 42 before → after seconds:
0.247 → 0.153 / 0.982 → 0.520 / 3.863 → 1.986; position 43:
0.308 → 0.189 / 1.258 → 0.658 / 4.967 → 2.498. Discrete evidence still matches;
maximum base/value/target errors remain 1.49e-10 / 5.96e-8 / 6.93e-9.
Speedup is 1.62–1.99x and 16 worlds still cost 8.0–8.1x the old singleton.
These independent single samples establish no compression speedup. Native
step/audit is still dominant (0.933 s); forwards remain 1,281 → 21.
The benchmark now checks and reports each numeric channel through one helper;
the attention test runs a fresh autograd forward after inference.

Gate still owns complete local-update suites and affected model suites
(`tests/model/test_value_aggregation.py`, `tests/model/test_inference.py`,
`tests/model/test_agent.py`), plus required model-cost/action-selection checks:
`tests/training/test_active_training.py`, `tests/env/test_observation.py`,
`tests/training/test_action_cutoff.py`, `tests/verify/test_native_buffer_capacities.py`.
No Rust, input shape or world change needs native/admission validation.

The next measured bottleneck is native Command stepping/audit (0.916 s), which
alone exceeds old one-world search time, followed by model object embedding,
Python encoding and semantic hashes. Further latency work needs a native
step/audit design that preserves every witness; simply removing network overhead
cannot meet the current target on this fixture. No acceptance decision relaxing
the target is recorded. The Task outcome remains incomplete pending further work
or an explicit design/review decision.
MTG-183/w13 remains unmeasured and was not rebound to this world; no trained
strength claim. Memory still scales with worlds × supported root actions.

The README's depth-20, 16-world, 30-second example is an illustrative budget,
not a latency promise. Reviewable files are `manabot/README.md` and `README.md`;
an earlier Desktop presentation had no window for this checkout. Older Task
notes remain at `841b1c50c:scratch/questions.md`.
