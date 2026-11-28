# MTG-497: GPU-batched evaluation through a vectorized, asynchronous match runner

## Direction

Jack Heart redirected the Task on 2026-11-26 (steer `ad0ea947`):

> "Not being ambitious enough feel like we should completely separate the play match API, make it fully asynchronous, and allow games to step in whatever way maximizes batching"

The steer's own expansion: step 64–256+ eval games at once through
`VectorEnv`/`RolloutPool` on the v11 CSR `ObservationBatch` -> `Prepared` path,
backfill finished slots at once (no straggler tail), yield completed
`(match, Result)` asynchronously, and share the bf16/MPS/overlap path with
training (MTG-449, MTG-496). This replaces the earlier brief ("Build directly on
Etude's `AGENTS` ... contract", steer `15f42691`) and the 4-leg synchronous
driver planned from it; none of that driver was committed.

Outcome: a managed evaluation on CUDA, Apple Silicon or CPU keeps its accelerator
busy because one game process steps many matches' games together and sends one
pre-collated `Prepared` per checkpoint per round; a report states games/hour and
rows per forward for `w160l8` and a 34M model against CPU batch-1.

Jack Heart added MLX in steer `d440244b`: prefer CUDA, then MLX, then MPS,
then CPU. Measure MLX against the same Apple Silicon MPS and CPU arms while
the Etude binding and its deployment are evaluated separately.

## Upstream state (checked 2026-11-27)

- The dependency pin and latest published [etude-sdk on PyPI](https://pypi.org/project/etude-sdk/)
  are `0.4.7` (verified 2026-11-27). That release has only synchronous
  `EvalRun.play`. The pass 6 checks used the editable 6db832 checkout at
  `154b8db`.
- Etude EXP-77 (`~/src/etude.exp-77-saturate-the-gpu`) committed `play_many`
  (`2469c00`) and merged `main` at `c5923cf` (clean tree, unreleased). Its
  `docs/evaluation.md` states the contract: `EvalJob(play=...)` xor
  `EvalJob(play_many=...)`; synchronous `play` is refused at declaration (so
  every Manabot `play` callback becomes `async def`); `max_delay_seconds` is
  gone (not used in our `Agents(...)` declarations; only frozen `EvalCompute`
  models retain it); game side awaits `Forward(inputs, *, rows=...)` (pass
  `rows=prepared.batch` for pre-collated inputs); `ForwardError` carries
  `.checkpoint`, and catching it in `play_many` drops affected slots and yields
  `(match, error)` while unrelated matches continue; `Matches` lives in
  `etude.eval.game` with `await matches.take(n)` -> `(match, agents)` pairs;
  yield `(match, tables | Result | Exception)` as each finishes; an uncaught
  exception fails every held match; `run.should_stop` stops new grants.
  Reference examples: `examples/vector_game.py` and `examples/vectorized.py`.
- Etude `310ac1` (`~/src/etude.310ac1-enable-shared-gpu`, committed at
  `de0c927`, unreleased) adds shared-GPU eval inference on Apple Silicon with
  `ETUDE_EVAL_SHARED=0|1`.
- `manabot` `origin/main` added three Academy/action-folding commits since our
  base (`080ea64372`, `65dcf2340a`, `b611abb3d7`); none touch eval, batching,
  or `SlotLoad`. (`57d4ea59b5` GRU bf16 fix is already in our base.)
- Jack Heart redirected the release dependency in steer `44a4f9bb`
  (2026-11-27). Build on editable EXP-77 (`c5923cf`) and
  6db832 (`49b085a`) now, while keeping `pyproject.toml` and `uv.lock`
  pinned. The merge gate is a published `etude-sdk 0.5.0`. At release, update
  the dependency bounds and exact play-runtime pin, then run
  `uv lock --upgrade-package etude-sdk` and check the locked environment.

## What exists

- **Loader & precision (`manabot/eval/agent_loader.py`, `manabot/eval/agents.py`,
  `manabot/policy/batching.py`, `manabot/sim/registration.py`,
  `manabot/backends/torch/learner.py`):** device order `cuda -> mlx -> mps -> cpu`
  with `MANABOT_EVAL_DEVICE` override (out of identity), and MLX requires the
  optional extra plus Metal; `load`/`load_tf32`/`load_bf16`
  accept any-batch `Prepared` and return `[batch, A]` float32 masked logits;
  `concatenate` rebases `MemoryRows` (rejects `fold is not None`); `AGENTS`,
  `AGENTS_TF32`, `AGENTS_BF16`, `agents_for(precision)`, and
  `etude_compute_class(precision)` (float32 keeps `etude-policy-f32-auto-v1`);
  `tf32` device guard in `validate_precision`.
- **Native & vector runner (`managym/`, `manabot/eval/vector.py`,
  `manabot/eval/pairing.py`, `manabot/eval/match.py`,
  `manabot/sim/native_game.py`):** seeded `SlotLoad.fresh(..., seed=)`,
  `RecordedVectorEnv.record_in_memory`/`take_slot_record`, shared per-leg setup
  (`resolve_game_match`, `game_runtime_identities`, `arena_source_sha256`,
  `set_execution_limits`), shared pairing admission/finalization, and Torch-free
  async `play_pairings` stepping up to `slots` legs on v11 CSR
  `ObservationBatch -> Prepared` with immediate backfill and per-match yield.
  `scripts/public_api.py check` and `tests/test_package_boundaries.py` pass.
- **Apple Silicon measurement (`experiments/eval_throughput.py`,
  `experiments/reports/eval_throughput.md`):** 12 arms on M4 Max across 32
  held-out TLA trophy deals (128 games/arm, exact leg-for-leg outcome and
  128/128 replay parity). MPS vector-64 reaches 1,989 games/hr on trained
  `w160l8` (2.66× CPU batch-1; forward 362.5→39.6 s, 17.1% of wall) and 1,882
  games/hr on seeded `w384l8` (2.41× CPU batch-1; forward 346.8→52.0 s, 21.3%);
  single-process `other_arena` + `replay` take ~79–83% of wall time, confirming
  multi-process game workers sharing a GPU agent (`play_many`) are needed to
  scale past ~2,000 games/hr. MLX vector-64 (float32) matches MPS within
  2% with 17–23% less forward time; managed `play_many` on MLX reaches
  ~4,700–5,100 games/hr. RunPod L4 (`966e8ff296`): CUDA vector-64 3,380
  (float32) and 3,908 (bf16) games/hr on `w384l8`, 17.3× CPU batch-1,
  limited by `other_arena`.

## Design notes

- **Precision (confirmed in steer `44a4f9bb`):** declare the agent set
  (`AGENTS`, `AGENTS_TF32`, `AGENTS_BF16`) and the tournament precision.
  The float32 compute envelope and the paired seeds stay stable across
  precision comparisons.
- **Vector runner:** TLA trophies adapt Etude's `Matches` to
  `play_pairings`, which passes `rows=prepared.batch` to the awaited forward
  and isolates `ForwardError` to held matches drawing on that checkpoint.
  Each decision round divides its elapsed time among live slots. The game
  limit and recorded execution time use the same charge.

## Review record

The superseded 4-leg review (F1–F6 of the earlier page) is fully settled in
passes 1–3. `scratch/concept-review.html` was rewritten at `c1df3f02c2`
(after pass 5); every claim below was checked against the code. Placement:

- **Async seats, loader, game budget and ownership (F2–F5).** Pass 6 resolved
  these findings. The vector runner uses one async sampler forward per
  checkpoint, shares its preparation and sampling with scalar play, and keeps
  checkpoint admission within the game worker. Managym owns slot record
  detachment, seed continuity and opt-in decision events. Scalar and vector
  runners share attempt retention. The trophy job threads one precision
  through the loader, environment and pairing. CUDA profiling remains deferred.

- **Host work and live frames (F1, Q1/Q2) -> deferred.** 1a changes stored
  trace annotations and breaks the "same traces as serial play" acceptance
  below, so it is not taken headless; the runner keeps live frames (1b).
- **Q4 (transport).** Ship review moved the remaining input copies into Pass 7.

## Passes

1–5. Done (`0286257840`, `dd11bd0a3f`, `08a854747f`/`7e79ad34ae`,
`d2012147d4`/`c4924ee4ed`, `3d685ce0aa`/`966e8ff296`, `a27d8c0264`/`92a578b417`);
see "What exists".
6. Done (`bb911a150`). Refactored sampling, loading, slot budgets and retention with exact trace parity. Verified 104 focused Python tests, 28 debug vector tests, the rebuilt CPython 3.13 extension, the public API snapshot, and a local Etude MPS bf16 trophy deal with four complete records.
7. Done (`f3b00a846`, `43d5ef0da`, `06389e498`). Made strided inputs contiguous,
   separated batch preparation from seat sampling and reproduced and fixed a
   negative grant after 10 failed matches filled a 9-slot runner's error queue.
   Live Etude resume reused a stored named trophy benchmark row and played only
   the new checkpoint. All 512 Gate 1 games replayed with zero array copies.
   The MPS compilation probe favored eager execution. See `scratch/qa.html`
   and `experiments/reports/eval_throughput.md`.
8. **Release (blocked on `etude-sdk 0.5.0` on PyPI; 0.4.7 on 2026-11-27).**
   Update the three `etude-sdk` bounds and the `==0.4.7` play-runtime pin in
   `pyproject.toml`, run `uv lock --upgrade-package etude-sdk`, then a locked
   sync and the managed eval tests.

## Docs

Managed loading, precision and Apple device selection are documented in
`docs/evaluation.md#etude-agents`. The same section describes managed `play_many`.

## Deferred findings

- **Vector host work (review F1).** Native per-slot heuristic policies in
  `SlotLoad.fresh` (L4 fill 9.9–18.6 of 64 rows), replay and retention off the
  event loop, and the open trace choice: keep live viewer frames (1b, current)
  or record compact annotations verified at replay (1a, changes stored
  traces). Host arena + replay is 72–89% of L4 wall. Needs a Task and the
  trace decision.
- **L4 `w160l8` bf16 slower than float32** (forward 24.0 -> 40.5 s). Needs a
  CUDA profile on a rental.
- **Scalar host overhead (S2)** in `scratch/questions.md`.
- **Etude (S3):** `etude-sdk 0.5.0` release (merge gate), Apple shared GPU
  (`310ac1`, `de0c927`), never evicting the only checkpoint in
  `Forwards._evict`.

## Done when

- The runner plays paired deals with the same Commands and traces as serial
  play under a row-pure forward, and every game replays.
- One experiment file evaluates on NVIDIA, Apple Silicon and CPU hosts
  without edits; a declared `bf16` shows in the players' compute class.
- The report gives games/hour and rows per forward for w160l8 and the 34M
  model on CUDA, MLX and MPS beside CPU batch-1, naming each arm's limiting stage.
- `tests/test_package_boundaries.py` and `scripts/public_api.py check` pass.
