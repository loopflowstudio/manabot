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

## Upstream state (read 2026-11-27, third realign)

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
- **Unblocking artifact for pass 4:** an `etude-sdk` release `> 0.4.7` shipping
  EXP-77 `play_many` and async `play`. First command once available:
  `uv lock --upgrade-package etude-sdk && uv sync --extra dev --extra artifacts --extra evaluation`,
  then read `etude.eval.run.EvalRun`, `etude.jobs.EvalJob`,
  `etude.eval.game.Matches`, and `etude.eval.agents.Forward`.

## What exists

- **Loader & precision (`manabot/eval/agent_loader.py`, `manabot/eval/agents.py`,
  `manabot/policy/batching.py`, `manabot/sim/registration.py`,
  `manabot/backends/torch/learner.py`):** device order `cuda -> mps -> cpu` with
  `MANABOT_EVAL_DEVICE` override (out of identity); `load`/`load_tf32`/`load_bf16`
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

## Design notes (open / still shaping pass 4)

- **Precision (made for review, see `scratch/questions.md`; irreversible once
  rows are stored):** declared per agent set (`AGENTS`, `AGENTS_TF32`,
  `AGENTS_BF16`), compute class from `etude_compute_class(precision)`, float32
  keeps `etude-policy-f32-auto-v1`; seeds pinned to the float32 envelope.
  Fallback if rejected before L4 bf16 rows are recorded: float32 only.
- **Vector runner (`manabot/eval/vector.py`, name for review):** `play_pairings`
  stays Torch-free and async over `MatchSource`; pass 4 binds it to Etude's
  `Matches`, passes `rows=prepared.batch` on awaited `Forward`, and isolates
  `ForwardError` per affected checkpoint slots rather than letting it abort
  unrelated held matches.

## Passes

1. **Loader (done, `0286257840`).** `cuda -> mps -> cpu` + `MANABOT_EVAL_DEVICE`; any-batch `Prepared` -> `[batch, A]` float32 logits; `MemoryRows` in `concatenate`; `etude_compute_class`, `AGENTS`/`AGENTS_TF32`/`AGENTS_BF16`; `tf32` guard; GRU bf16 fix.
2. **Vector runner, locally (done, `dd11bd0a3f`).** Seeded `SlotLoad.fresh(..., seed=)`; `RecordedVectorEnv.record_in_memory`/`slot_record`; shared admission/finalization (`eval/pairing.py`, `eval/match.py`); Torch-free async `play_pairings` (`eval/vector.py`). Trace parity with `play_pairing` proved field by field and by `replay_game`/`audit`/`replay_failure` (heuristic and GRU memory); raw `game_trace_sha256` differs only through wall-clock timings and `mode="vector"`.
3. **Local measurement and runner cleanup (done, `08a854747f`, `7e79ad34ae`).** Consolidated per-leg setup (`resolve_game_match`, `game_runtime_identities`, `arena_source_sha256`, `set_execution_limits`), dropped `_` prefixes on `eval/vector.py` helpers, added `experiments/eval_throughput.py`, and measured all 12 Apple M4 Max arms in `experiments/reports/eval_throughput.md` (128/128 outcome and replay parity; MPS vector-64 2.41–2.66× CPU batch-1, host arena + replay now 79–83% of single-process wall).
4. **Etude binding, L4 arm, cleanup, docs (blocked on `etude-sdk > 0.4.7`; PyPI still `0.4.7` as of 2026-11-27, EXP-77 committed at `c5923cf` and `310ac1` at `de0c927`, both unreleased).**
   - Upgrade `etude-sdk` and bind `play_many` over `play_pairings` (pass
     `rows=prepared.batch` to awaited `Forward`; catch `ForwardError` to drop
     affected slots and yield `(match, error)` per match without failing
     unrelated held matches).
   - Convert `play_pairing` and experiment `EvalJob(play=...)` callbacks
     (`experiments/{elemental,llm_claude,llm_qwen,mana_taps,opponent_memory,tla_frontier,tla_trophies}.py`)
     to `async def` (or `play_many=` where appropriate).
   - Deduplicate `manabot/eval/match.py`: collapse `play_cell`'s ~165-line copy
     of `replay_block`/trace/stats retention onto `finalize_settled_cell`
     (accepting an `Iterable[dict[str, Any]]` with `try/finally` trace write).
   - Run and add the CUDA L4 float32 and bf16 arms (`--max-usd 5`) to
     `experiments/reports/eval_throughput.md`.
   - Write docs for what ships: `docs/evaluation.md#etude-agents`,
     `docs/training.md`, `docs/experiments.md`, `manabot/README.md`.

## Docs

`6e803136af` documented the superseded design (`AGENTS_BF16`, `play_pairing`
`precision`, grouped deal stepping) as if built; it was reverted so docs
describe only what exists. Pass 4 writes them for what ships.

## Review record

`scratch/concept-review.html` reviewed the superseded 4-leg design. F1
(stepping boundary) and F6 (deal window) are superseded by the vector runner.
F2 (declared precision) and F3 (memory rows in `concatenate`) landed in pass 1.
F4 (opt-in MPS) stays passed over: MPS default, opt-out by
`MANABOT_EVAL_DEVICE=cpu`, revisited after pass 4. F5 (`max_batch` 32) is
moot once EXP-77 counts `rows=` on pre-collated inputs.

## Deferred findings

- Etude: Apple GPU detection for the shared agent process (`310ac1`, committed
  at `de0c927`, unreleased); never evicting the only loaded checkpoint in
  `Forwards._evict`.
- Game-side recording cost (`scratch/questions.md`): the limit on CUDA below
  ~34M and on single-process MPS (~79–83% of wall time in pass 3).
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
