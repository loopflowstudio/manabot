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

- Installed and PyPI `etude-sdk` is still `0.4.7`: `EvalRun` has only
  synchronous `play`.
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
  `RecordedVectorEnv.record_in_memory`/`slot_record`, shared per-leg setup
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
  scale past ~2,000 games/hr. (Prior L4 reference from MTG-328: 0.85 s/game
  host work, 21% L4 busy below ~34M.)

## Design notes

- **Precision (confirmed in steer `44a4f9bb`):** declare the agent set
  (`AGENTS`, `AGENTS_TF32`, `AGENTS_BF16`) and the tournament precision.
  The float32 compute envelope and the paired seeds stay stable across
  precision comparisons.
- **Vector runner:** TLA trophies adapt Etude's `Matches` to
  `play_pairings`, which passes `rows=prepared.batch` to the awaited forward
  and isolates `ForwardError` to held matches drawing on that checkpoint.
  The per-slot budget advances with live decision rounds.

## Review record

The superseded 4-leg review (F1–F6 of the earlier page) is fully settled in
passes 1–3. `scratch/concept-review.html` was rewritten at `c1df3f02c2`
(after pass 5); every claim below was checked against the code. Placement:

- **Async seat (review F2, Q3) -> pass 6.** `agents.py:43` types
  `Sampler.forward: Forward | ModelForward`; `choose()` casts to the sync form
  (`agents.py:81`), so an Etude-built `Sampler` fails there; `play_scalar`
  swaps `.forward` through `blocking`; `vector.py:827` groups batches by
  `id(agent.forward)`. Adopted 3a, made for review: `Sampler` holds only the
  async `Forward` plus `checkpoint_key`, one sync adapter serves
  `play_scalar`, vector groups by `checkpoint_key` and reuses the
  `Sampler`'s prepare/sample. Passed over 3b (key only) because the union stays.
- **Loader (F3, Q5) -> pass 6 for code; L4 diagnosis deferred.** Probing
  `cuda` imports Torch on macOS before MLX is chosen; MLX bf16 casts weights
  every forward and copies critic values; Torch `cache_programs` runs outside
  `forward_precision`. Q5 adopted 5a, made for review: one
  `MANABOT_EVAL_DEVICE` knob documented as a runtime target
  (`cuda|mlx|mps|cpu`); 5b (separate backend/device vars) passed over as two
  knobs for one choice.
- **Game budget and admission (F4, Q6) -> pass 6.** `vector.py:984` adds each
  whole round to every active slot's `gameplay_seconds`, so 64-256 slots on
  CPU hit `game_seconds=120` together; `vector.py:399` `execution_seconds`
  includes other matches' replay pauses. Adopted 6a, made for review: charge
  each round per active slot (`dt / active`) and record the same time as
  execution time. Also verify/hash each checkpoint once per worker, create
  temporary directories at finalization, bound the deferred queue.
- **Ownership (F5) -> pass 6.** `vector.py:35` imports the training
  `manabot.env.failures.RecordedVectorEnv` (`_TrainingCapture`), not
  `managym.recording.RecordedVectorEnv`; an explicit `SlotLoad.fresh` seed
  becomes the slot's later auto-seed (`slots.rs:155-167,144`); five sites
  patch `slot_record` fields by hand; attempt retention duplicates
  `evaluation._execute_attempt`; precision is declared at three unchecked
  sites; `env.rs:729,737` now clones `recent_events` on every step, training
  included.
- **Host work and live frames (F1, Q1/Q2) -> deferred.** 1a changes stored
  trace annotations and breaks the "same traces as serial play" acceptance
  below, so it is not taken headless; the runner keeps live frames (1b).
- **Q4 (transport) -> 4a**, deferred below.

## Passes

1–5. Done (`0286257840`, `dd11bd0a3f`, `08a854747f`/`7e79ad34ae`,
`d2012147d4`/`c4924ee4ed`, `3d685ce0aa`/`966e8ff296`, `a27d8c0264`/`92a578b417`);
see "What exists".
6. **Seat, loader, budget and ownership fix pass (next).** Everything placed
   "pass 6" above, in one pass. Delivers: one async `Sampler` keyed by
   checkpoint; no Torch import on macOS MLX selection; MLX bf16 cast once at
   load, logits only; Torch program cache in the forward precision; per-slot
   game budget; cached admission; eval on `managym.recording`; explicit seeds
   that leave the slot's sequence alone (Rust: rebuild, debug nextest of
   `vector_env`); one `take_slot_record`-style detach and one shared attempt
   helper; one precision helper producing `Agents`, `Environment.config` and
   `PairingSpec.precision`; the `recent_events` clone limited to envs that
   build decision views. Teaches: whether the runner keeps trace parity
   (`tests/eval/test_vector_eval.py`, `test_pairing_play.py`,
   `test_etude_agents.py`) with the smaller surface. No new rental;
   `experiments/eval_throughput.py` stays as measured.
7. **Release (blocked on `etude-sdk 0.5.0` on PyPI; 0.4.7 on 2026-11-27).**
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
- **Zero-copy compact `Prepared` transport (S1, Q4 4a).** Gate 1 pickled
  4.13 GB in-band, 0 arrays by pointer. Waits on Etude's array-container
  interface.
- **Scalar host overhead (S2)** in `scratch/questions.md`.
- **Etude (S3):** `etude-sdk 0.5.0` release (merge gate), Apple shared GPU
  (`310ac1`, `de0c927`), never evicting the only checkpoint in
  `Forwards._evict`.
- `torch.compile` (optional per the Task).

## Done when

- The runner plays paired deals with the same Commands and traces as serial
  play under a row-pure forward, and every game replays.
- One experiment file evaluates on NVIDIA, Apple Silicon and CPU hosts
  without edits; a declared `bf16` shows in the players' compute class.
- The report gives games/hour and rows per forward for w160l8 and the 34M
  model on CUDA, MLX and MPS beside CPU batch-1, naming each arm's limiting stage.
- `tests/test_package_boundaries.py` and `scripts/public_api.py check` pass.
