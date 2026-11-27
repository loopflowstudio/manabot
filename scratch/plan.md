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

Outcome: a managed evaluation on CUDA, Apple MPS or CPU keeps its accelerator
busy because one game process steps many matches' games together and sends one
pre-collated `Prepared` per checkpoint per round; a report states games/hour and
rows per forward for `w160l8` and a 34M model against CPU batch-1.

## Upstream state (read 2026-11-26)

- Installed `etude-sdk` is 0.4.7: `EvalRun` has only synchronous `play`.
- Etude EXP-77 (`~/src/etude.exp-77-saturate-the-gpu`, HEAD `5947da0` =
  v0.4.7) is a plan with no code. Its contract sketch (`scratch/plan.md` there):
  `async def play(match, agents)` with awaited `Forward`; `EvalJob(play_many=...)`
  taking an async generator over `Matches`, with `await matches.take(n)` ->
  `(match, agents)` pairs and yields of `(match, Tables | Result | Exception)`;
  a yielded exception fails one match, an escaping one fails all held matches;
  `max_batch` counts inputs (a pre-collated batch counts once);
  `max_delay_seconds` deleted; synchronous `play` refused. Its open readings
  (sync `play` removed outright, `max_delay_seconds` gone) are unconfirmed.
- Consequence if EXP-77 lands as planned: every Manabot `play` callback
  (`play_pairing`, `play_cell`, the experiment files) must become `async def`,
  not only the vectorized path.
- Unblocking artifact for pass 3: an etude release with `play_many`. First
  command: `uv lock --upgrade-package etude-sdk && uv sync --extra dev --extra artifacts --extra evaluation`,
  then read `etude.eval.run.EvalRun` and `etude.jobs.EvalJob`.

## What exists

- `manabot/eval/agent_loader.py::load`: `cuda` or `cpu`, float32, rejects any
  input that is not `Prepared(batch=1)`, joins memory-free rows with
  `policy/batching.py::concatenate`, runs memory rows one per batch.
- `manabot/eval/agents.py::Sampler.choose` prepares one dense row and calls
  `forward([row])`; `eval/match.py::play_cell` plays legs serially through
  `sim/native_game.py::execute_game` on a scalar `Env` with capture journal,
  viewer frames, digests and replay receipts.
- Native `VectorEnv.load_slots_into_buffers([SlotLoad.fresh(slot, setup)])`
  replaces one slot's game; `SlotLoad.fresh` takes no seed, and
  `reset_each_into_buffers(setups, seeds=)` seeds only a whole-vector reset.
  `RecordedVectorEnv` (`managym/recording.py:1254`) records vector games.
- Probe findings that still hold (M4 Max, `c7d6c628d1`, seeded random weights,
  real `play_pairing` rows): CPU 1-thread batch-1 forward 35 rows/s at w160l8,
  24 at w384l8; MPS saturates by batch 8 (w160l8 ~410 rows/s, w384l8 ~210),
  bf16 gives nothing on MPS; 12 processes sharing MPS reach ~478 rows/s. On an
  L4 (MTG-328) game-side host work, 0.85 s/game against the heuristic, set the
  rate below ~34M (21% GPU busy). bf16 on CUDA runs for memory-free policies
  (PR #451); MTG-449 covers only the GRU memory fold. Evaluation memory rows
  have `MemoryRows.fold is None`, so `concatenate` can rebase them.

## Design

### Loader (independent of Etude)

- Device `cuda`, then `mps`, then `cpu`; `MANABOT_EVAL_DEVICE` names one;
  unknown values fail with the accepted list. Device stays out of identity.
- Inputs are `Prepared` of any `batch`; output per input is `[batch, A]`
  float32 masked logits. So a vector driver sends one pre-collated input per
  checkpoint, and per-row callers keep working.
- Memory rows join in `concatenate` (rebase `detail_references`,
  `member_references`, `member_groups`, `record_batches`; reject
  `fold is not None`).
- Precision (made for review, see `questions.md`; irreversible once rows are
  stored): declared per agent set (`AGENTS`, `AGENTS_BF16`), compute class from
  `sim/registration.py::etude_compute_class(precision)`, float32 keeps
  `etude-policy-f32-auto-v1`; `tf32` only on CUDA, `bf16` on CUDA/MPS (fix the
  missing `tf32` check in `backends/torch/learner.py::validate_precision`).
  Fallback if rejected: float32 only.

### Vector match runner

`manabot/eval/vector.py` (name for review) owns a slot pool over one
`RecordedVectorEnv` of S slots. Each slot holds one leg of one match; a match
is done when all its legs are.

- Load: a free slot takes the next leg (or the next match from the source),
  dealt at the leg's exact seed through a seeded per-slot load (native
  addition: `SlotLoad.fresh(..., seed=)`, if no existing path seeds one slot).
- Step: rows needing a checkpoint are written as one CSR `ObservationBatch`,
  prepared once per checkpoint into one `Prepared`, forwarded, sampled per row
  with that decision's `policy_rng_seed` (NumPy PCG64, as now). Heuristic seats
  act natively in the same step.
- Finish: a finished leg frees its slot at once; a finished match is finalized
  (shards, `rows.json`, replay receipts, card blobs, as `play_cell` writes) and
  yielded. A native failure fails that match's leg as now; a forward failure
  escapes.
- The runner is an async generator over an abstract match source and awaited
  forwards, so pass 2 tests it with a local source before Etude ships; pass 3
  binds it to `Matches`. Torch-free (`tests/test_package_boundaries.py`).
- Shares preparation with training collection
  (`manabot/training/rollout/collection.py`); MTG-496 owns the overlap core, and
  this runner reuses what exists rather than forking it.

The risk that can sink this: a vector-stepped game must produce the same
Commands and `game_trace_sha256` as `execute_game` on a scalar `Env` for the
same deal under a row-pure forward, including viewer frames and digests the
eval trace records. If `RecordedVectorEnv` traces differ from eval traces, the
evidence format decides whether eval adopts vector traces (frozen tournaments
keep theirs) or the recorder changes.

## Passes

1. **Loader (done, `0286257840`).** `cuda -> mps -> cpu` + `MANABOT_EVAL_DEVICE`; any-batch `Prepared` inputs returning `[batch, A]` float32 masked logits (no `concatenate` on single pre-collated inputs); `MemoryRows` joined in `concatenate` (rejecting `fold is not None`); `etude_compute_class`, `AGENTS`/`AGENTS_TF32`/`AGENTS_BF16` (1-arg top-level loaders required by Etude `Agents`), `validate_precision` `tf32` guard, GRU `summaries.new_zeros` bf16 fix; verified on CPU + Apple MPS (float32 parity and bf16).
2. **Vector runner, locally.** Slot pool, seeded per-slot load (native if
   needed), per-checkpoint pre-collated forward, backfill, per-match
   finalization and yield, against a local match source. Acceptance: a fixed
   set of paired deals played through the runner and through `play_pairing`
   give equal `game_trace_sha256`, scores and Commands; every game replays; a
   native crash on one leg leaves the others valid; no slot is idle while legs
   are pending; forward inputs carry more than one row. This is the pass that
   can prove the plan wrong (trace parity).
3. **Etude binding** (blocked on the EXP-77 release above). `play_many` over
   the runner; existing `play` callbacks to `async def` if sync `play` is
   removed; experiment files and `AGENTS` settings (no `max_delay_seconds`).
4. **Measure.** `experiments/eval_throughput.py` +
   `experiments/reports/eval_throughput.md`: 32 held-out trophy deals per arm
   against the heuristic, trained w160l8 and seeded w384l8; arms CPU batch-1
   (0.4.7 path), CPU vector, MPS vector, L4 float32/bf16 at several slot counts;
   games/hour, rows per forward, GPU busy, cost. Exploratory, `--max-usd 5`.
   Then the docs (`docs/evaluation.md#etude-agents`, `docs/training.md`,
   `docs/experiments.md`, `manabot/README.md`).

## Docs

`6e803136af` documented the superseded design (`AGENTS_BF16`, `play_pairing`
`precision`, grouped deal stepping) as if built; it was reverted in this
realign so docs describe only what exists. Pass 4 writes them for what ships.

## Review record

`scratch/concept-review.html` reviewed the superseded 4-leg design. F1
(stepping boundary) and F6 (deal window) are superseded by the vector runner.
F2 (declared precision) and F3 (memory rows in `concatenate`) carry into pass
1. F4 (opt-in MPS) stays passed over: MPS default, opt-out by
`MANABOT_EVAL_DEVICE=cpu`, revisited after pass 4. F5 (`max_batch` 32) is
moot once EXP-77 counts pre-collated inputs.

## Deferred findings

- Etude: Apple GPU detection for the shared agent process (310ac1); never
  evicting the only loaded checkpoint in `Forwards._evict`.
- Game-side recording cost (`questions.md`): the limit on CUDA below ~34M.
- Compacting padded joined batches in `agent_loader` if profiling shows it.
- `torch.compile` (optional per the Task).

## Done when

- The runner plays paired deals with the same Commands and traces as serial
  play under a row-pure forward, and every game replays.
- One experiment file evaluates on NVIDIA, Apple Silicon and CPU hosts
  without edits; a declared `bf16` shows in the players' compute class.
- The report gives games/hour and rows per forward for w160l8 and the 34M
  model on CUDA and MPS beside CPU batch-1, naming each arm's limiting stage.
- `tests/test_package_boundaries.py` and `scripts/public_api.py check` pass.
