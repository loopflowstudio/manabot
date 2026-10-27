# MTG-193 — Batch shared evaluation across matchups; light, bounded game processes

Status: **accepted** — Jack Heart approved this design as written ("keep going", 2026-10-26), including `shared-eval-v1` unchanged. Kickoff plan 2026-10-26. Direction accepted by Jack Heart,
2026-10-26: fix all three serving limits from MTG-183's rented calibration
(Etude `d0955f2a44b6`, `307c3a8630af`, `00119d1c9211`). Extended by Jack Heart,
2026-10-27 (concept review 4, "1a 2b 3b"): evaluation checkpoint players play
served only, and code anchors are Torch-free. MTG-193 runs its own rental
calibration under $5 with a spend ledger.

## Problem (from the Task)

1. Matchup is in the batching key → 180 model loads, 214,201/214,216 forwards
   single-row, decisions wait p50 0.34 s, the serial service ~80% busy.
2. Every native game child imports Torch (~225 MB) → 256 active games OOM on 32 GB.
3. Thread oversubscription on CPU hosts whose `os.cpu_count()` is the host's (384).

## Demo

On the A4500 rental, the MTG-183 runner's `gpu` mode plays the same 1,000-game
pair cohort (update 5,950 vs 5,600): `serving.json` shows `model_loads: 2`
(one per checkpoint), median batch rows ≫ 1, `host_to_device_calls/decisions`
≤ 0.1, games/hour well above 2,600, and 256 active games complete without OOM.
On `cpu32` the same pair is played served on CPU at least as fast as the
scalar baseline (about 1,700 games/hour), within the declared thread budgets.
Locally: `uv run --extra dev pytest tests/eval/test_batched_eval.py tests/sim/test_child_surface.py -q`
plays a served game whose child reports `torch` absent from `sys.modules` and a
thread count within its declared bound.

## Risk findings (2026-10-26, this checkout at `558e0186b`)

- **Matchup is not a true per-input dependency.** `PolicyKey.input_binding`
  hashes `matchup_sha256` *and* `content_suite`; `AuthoredMatchup.content_suite`
  is itself per role pair, so it also splits. The only matchup-varying model
  input would be the card catalog, but `SemanticCardEncoder` is keyed by the
  checkpoint's pack and `checkpoint_world` already binds the complete authored
  roster. Per-setup legality stays checked by `PolicyStore.admit` →
  `validate_agent_setup`. So: drop both from the key; no in-batch catalog work.
- **Single-row batches have a second cause:** the service flushes only the FIFO
  head's group. With one key per checkpoint this resolves naturally (pair games
  have two keys); no scheduler rewrite needed.
- **Copies:** `_infer` → `PolicyInference.prepare` moves each of the 20
  observation keys plus 3 `CardReferences` tensors separately. All keys are
  float32 after `_infer`'s `.float()` cast, so one packed float block plus one
  int64 reference block = 2 copies per batch, bitwise-identical data.
- **Torch import surface (measured with a sys.modules probe):** importing
  `manabot.sim.match` = 216 MiB RSS with Torch; `managym+numpy+pydantic+psutil`
  = 47 MiB; `etude.run.storage` = 22 MiB, no Torch. Torch enters through:
  eager package re-exports (`manabot/infra/__init__` → `runtime`,
  `manabot/env/__init__` → `vector_env`, `manabot/model/__init__` →
  `actor_critic`); mixed-responsibility modules (`sim/search/teacher1_evidence`
  holds the frame/Command builders but imports `puct`; `verify/util` holds
  `INTERACTIVE_DECK` beside Torch helpers; `verify/competency` scenarios import
  `flat_mc` players); and `sim/match`/`sim/supervision` importing `serving`,
  `guidance`, `local_update`, `puct` at top level plus `import torch` /
  `torch.manual_seed` inside `_execute_game` even for served players.
  A probe with lazy infra/env exports confirmed `env.replay`, `registration` and
  `semantic.learning` are then Torch-free; the rest needs the cuts below.
- **Spawn re-imports the parent's `__main__`** (probe: a `-m` launcher module
  appeared as `__mp_main__` in its spawned child). The MTG-183 evaluator's
  main imports `etude.eval.match` and more, so library cuts alone cannot
  guarantee light children. Children must not use `multiprocessing` spawn.
- **Threads:** on macOS the probe shows 1 thread after Torch import/forward
  (Accelerate BLAS; lazy pools), so the 384-CPU oversubscription cannot be
  reproduced locally. Candidate Linux pools: OpenBLAS (numpy wheels; sized at
  load from env), Torch interop (default = cores, lazy), libgomp spin-wait,
  plus the parent's `ThreadPoolExecutor(max_workers=max_active_games)`.
  `os.cpu_count()` ignores affinity and cgroup quota. The exact culprit is
  established by the per-role thread census on the rental (below), not assumed.

## Chosen approach

### A. Matchup-independent policy key
- `PolicyKey.from_registration` binds checkpoint, observation ABI, action ABI,
  world, device, dtype, deterministic, implementation. Matchup and content suite
  leave the key and remain in registrations, admission and evidence.
- `PolicyStore.admit` keeps the first admission per key (its path loads the
  policy). Every registration is checked on arrival against its own checkpoint
  bytes/SHA, parameter count, world/action ABI, and each of its setups by
  `validate_agent_setup`; setups are not accumulated.
- **Identity:** `EvalCompute` stays `shared-eval-v1`. v1 already promises no
  batch composition (it depends on arrival order within the row/delay caps), so
  grouping across matchups is within the contract; `docs/shared-evaluation.md`
  says so. Bump `PolicyKey.implementation` only if numerics change (they don't).

### B. Two transfers per batch
- `PolicyInference.prepare` packs all float32 keys into one contiguous (pinned
  on CUDA) host block, transfers once (`non_blocking`), and returns views; the
  CPU-admitted `CardReferences` pack into one int64 block. Training callers use
  the same path (one implementation). Counters report actual calls.

### C. Light native children via an explicit child launcher
- New `manabot/sim/child.py` (Torch-free): launches `python -m <module>` with
  `subprocess.Popen(pass_fds=...)` over a socketpair wrapped in
  `multiprocessing.connection.Connection`, an explicit environment (see D), and
  kill/exit-code semantics matching today's supervision. Game, replay and
  serving processes all use it; no parent `__main__` is imported.
- New `manabot/sim/native_game.py` (Torch-free child entry): `_execute_game`
  and `_game_worker` move here from `sim/match.py`. Scalar players are built
  Torch-free by `players.build_code_player` (random, scripted, heuristic);
  only search players (flat-MC, PUCT, local update) import Torch, lazily in
  `_ScalarPlayers`, and only they call `torch.manual_seed`. Plain checkpoint
  players never build in-child (slice 6).
- Split `serving.py`: Torch-free `sim/serving_protocol.py` (`PolicyKey`,
  `DecisionRequest`, `DecisionReply`, `PolicyAdmission`, `InferenceFailure`)
  and `InferenceRuntime` (parent supervisor); Torch-bearing `sim/serving.py`
  keeps `PolicyStore`, `sample_rows`, `_infer`, `_service` as the service child
  entry. `supervision.py` imports only the protocol/runtime.
- Import cuts: drop eager Torch re-exports from `manabot.infra`, `manabot.env`,
  `manabot.model` packages and update consumers to explicit module imports;
  move `build_viewer_frame`/`build_command` (+ helpers) to Torch-free
  `sim/frames.py`; move `INTERACTIVE_DECK`/`winner_from_info_or_obs` out of
  `verify/util` to a Torch-free module; separate competency scenario
  definitions/builders from `flat_mc` player construction so `sim/replay.py`
  imports scenarios without Torch.
- Replay children (`replay_retained`, card-evidence capture) launch the same way.

### D. Declared, bounded threads, proven by census
- New Torch-free `manabot/sim/threads.py`: `effective_cpus()` = min of
  `os.cpu_count()`, `len(os.sched_getaffinity(0))` where available, and cgroup
  v2 `cpu.max` / v1 quota; `ThreadBudget` per role (game child, replay child,
  serving, group worker) with a declared max thread count; `child_environment()`
  sets `OMP/MKL/OPENBLAS/VECLIB_MAXIMUM/NUMEXPR/RAYON_NUM_THREADS` before the
  child interpreter starts. Torch processes also call `torch.set_num_threads`
  and `torch.set_num_interop_threads(1)` at entry.
- Census: each child reports `psutil.Process().num_threads()` (and, on Linux,
  thread names from `/proc/self/task/*/comm`) in its result; the service
  records its own; the parent records its own. `serving.json` / game
  `worker_performance` carry `threads_max` by role. Exceeding the declared
  bound is an infrastructure failure, never a forfeit.
- The group worker (the process running `play_cell`) is allowed
  `max_active_games + 7` with a shared runtime, otherwise 8, and recorded as
  `threads/group_worker` in each cell's `performance.json`, not enforced (it
  also hosts Etude callbacks and checkpoint preparation).

## Acceptance (headless)

- `uv run --extra dev pytest tests/eval/test_batched_eval.py tests/eval/test_local_evaluation.py tests/sim/test_child_surface.py tests/test_package_boundaries.py -q` passes, including:
  - two registrations of one checkpoint with different matchups share one
    `PolicyKey`, one `model_loads`, and one mixed-matchup batch whose per-row
    actions equal single-row inference with the same seeds (inverts today's
    `test_batched_eval.py` assertion that matchups split keys);
  - `host_to_device_calls == 2 × forward_calls` on the fake/CPU-counted path;
  - a served game child and a replay child, launched under a parent `__main__`
    that imports Torch, report `"torch" not in sys.modules` and
    `num_threads ≤` their declared budget; a scalar child stays within budget;
  - `effective_cpus()` honors injected affinity and cgroup `cpu.max` fixtures.
- Same seeds → same Commands for a CPU served panel before/after (fixture
  panel in `test_batched_eval.py`); replay verification and evidence formats
  unchanged (`tests/eval/test_mirror_admission.py`, card evidence tests).
- Remote calibration (slice 7, under the $5 ledger): A4500 `gpu` after and
  `cpu32` served CPU after on the same 1,000-game pair; before = the Task's
  retained baselines (A4500 2,600, `cpu32` scalar 1,700 games/h). Record
  games/hour, load, threads by role, RSS per active game, `model_loads`,
  batch-row histogram, `host_to_device_calls/decisions`. Targets: A4500
  > 2,600 games/h with median batch > 8 and 256 active games without OOM;
  served child RSS ≤ 90 MiB (local 76 MB); `cpu32` served ≥ 1,700 games/h,
  threads ≤ declared total and load ≤ vCPUs. **Landing gate:** #389 lands
  after `cpu32` served CPU at least matches scalar, or Jack Heart decides
  otherwise on that evidence.

## Alternatives rejected

- In-batch per-matchup catalogs: unnecessary (catalog is checkpoint/pack-bound).
- `forkserver`/`fork` contexts: forkserver still prepares `__main__`; fork is
  unsafe with Torch/threads and on macOS.
- Lazy `__getattr__` package exports: hides import cost; explicit module
  imports match repository style.
- Bumping `EvalCompute` to v2: the contract does not change (see A).

## Exclusions

Multi-process or multi-device serving; vectorized sampling that would change
Commands; removing the parent's request hop; making group-worker parents
Torch-free (they load checkpoints for registration); CPU/CUDA parity (stays a
stated limit); serving search players' model calls (follow-up MTG-205); a
second CPU engine before rental counters show a saturated CPU service.

## Delete — do not maintain

- `PolicyKey` matchup/content-suite fields in `input_binding`.
- Per-key `.to(device)` loops in `PolicyInference.prepare` and `_infer`'s
  non-catalog branch (one packed transfer replaces both).
- `mp.get_context("spawn").Process` in `supervision.supervise_game`,
  `replay_retained` and `InferenceRuntime.__init__`.
- `_execute_game`/`_game_worker` in `sim/match.py` (moved, not copied);
  `import torch` and unconditional `torch.manual_seed` in the served path;
  top-level imports of `LocalUpdatePlayer`/`DeterminizedPuctPlayer`.
- Eager Torch re-exports in `manabot/{infra,env,model}/__init__.py`.
- Frame builders in `teacher1_evidence` (moved to `sim/frames.py`).
- `players.build_player` (folded into `guidance`), `ScriptedGreedyPlayer` in
  `players.py` and `RandomMatchupPlayer` in `flat_mc.py` (moved to `anchors.py`),
  the in-child plain-checkpoint build, and `bind_execution`'s anchors-only rule.
- Ad-hoc `OMP_NUM_THREADS`/`MKL_NUM_THREADS` env dicts in
  `experiment/evaluation_queue.py`, `experiment/runner.py` and the
  `experiments/` launchers, replaced by `threads.child_environment()`.

Forbidden: a second worker module kept "for scalar"; lazy-export shims; a
fallback to spawn; a v2 key beside v1.

## Internal slices (one PR)

1–4 implemented 2026-10-26 (local): import cuts, `sim/frames.py`,
`verify/games.py` (decks + `winner_from_info_or_obs`), `verify/scenarios.py`
(competency positions split from runners), `serving_protocol.py`, `child.py`,
`native_game.py`, `threads.py`, matchup-free `PolicyKey` with per-registration
setup validation, packed `transfer`, env-dict cutover, docs and AGENTS.md.
5. Folded into slice 7's rental calibration.

6. Torch-free code anchors and served-only checkpoint players (implemented
   2026-10-27, local; see concept review 4). `sim/anchors.py` holds the v1
   scripted and random players (their source identity); `sim/players.py` is the
   Torch-free `build_code_player` for random/scripted/heuristic/demo search;
   `guidance.build_registered_player` stays the one entry point and owns search
   and checkpoint construction. `eval.execution.is_served` marks flat
   checkpoints; when none is declared, `play_cell` and `evaluate` serve them
   through `eval.execution.resolve_execution` (CPU, one game at a time), so arena callers need
   no cutover.
7. Engine boundary (decided 2026-10-27, implement; documented in
   `docs/shared-evaluation.md`): the engine is the serving child's flush loop
   and `serving._infer`; everything before it stays shared. Both devices keep
   the one serial service per allocation. A CPU host runs several allocations
   side by side (MTG-183's `batched` mode: 6 groups × 12 active games, one
   CPU service each), so "several inference workers per host" already exists
   as layout. An in-process engine saves only the IPC hop and loses
   killability. Multi-worker allocations, intra-op threads and a CPU delay
   are not written until rental counters show a saturated CPU service (YAGNI;
   no second engine without evidence). **Remaining:** the rental calibration
   in Acceptance. MTG-183's runner (`experiments/elemental_long_rent.py`) lives
   on its branch and imports that branch's `elemental_long_eval` family, so the
   calibration runs from a calibration branch merging both, with its own eval
   run names (not `mtg183-eval-*`) and no `collect` into MTG-183's database.
   As of MTG-183 `a75df8b5e` (2026-10-27) the merge brings 31 files and
   etude-sdk 0.2.9; both branches change `experiments/elemental_grid.py`,
   `experiments/elemental_long_eval.py`, `manabot/training/execution.py` and
   `manabot/training/state/learning_state.py`. On this code the runner's modes
   mean: `gpu` is the A4500 arm, but its `GPU` contract caps
   `max_active_games=64` for Torch children, so the calibration branch raises
   it to 256 for the OOM check; `batched` (6 groups × 12 active games, one CPU
   service each) is the `cpu32` served arm; `scalar` (no `EvalCompute`) now
   serves one game at a time per group and no longer measures scalar play, so
   the scalar baseline is the retained run `00119d1c9211`.
   The calibration also settles the RSS target, the thread budgets and the oversubscription
   culprit. If `cpu32` served is slower, iterate the CPU engine (several
   workers per allocation, then intra-op threads, which need a compute-contract
   decision) before Jack Heart decides on retaining scalar play. Ledger:
   $0.00 of $5; no rental launched.

### Implementation decisions (2026-10-26)

- Scalar hooks keep exact `isinstance` dispatch, with the classes imported
  lazily in `native_game._ScalarPlayers`, instead of a Protocol: a structural
  check would newly call `flat_mc` players' `start_game` (compound reset) and
  could change Commands.
- Census is `ProcessCensus` (threads, Linux thread names, RSS, Torch loaded),
  recorded by role. Named budgets `GAME_BUDGET`/`REPLAY_BUDGET` (4) and
  `SERVING_BUDGET` (16) are enforced with `ThreadBudget.account(counters)`;
  `group_worker_budget` (`max_active_games + 7`, or 8 without a runtime) is recorded only,
  in `play_cell`'s `performance.json` on every path.
- Child modules are `module:function` targets, so tests swap
  `match.GAME_WORKER` for fixture workers; the child imports the target module
  inside the game budget, so fixture targets live in Torch-free modules.
- Local measure: model-vs-model served game child 76 MB, 1 thread, no Torch;
  replay child 75 MB; former scalar checkpoint child 285 MB, 1 thread.

## Check result

Realign, 2026-10-27, at `46d52b1e0`: `uv run --extra dev pytest tests/eval/test_batched_eval.py tests/eval/test_local_evaluation.py tests/sim/test_child_surface.py tests/test_package_boundaries.py -q` → 50 passed, 1 skipped (the stale `cold_scalar` argument was already removed in `fbbbd517a`). Full suite belongs to gate/CI.

Implement, 2026-10-27: `test_arena_lost_worker_retains_acknowledged_command`
timed out because its fixture child imported the Torch-loading test module
(~7 s) inside the 5 s game budget. The fixture workers now live in Torch-free
`tests/training/game_error_fixtures.py` (collector adapters moved to
`test_game_error_recovery.py`): `pytest tests/training/test_game_errors.py
tests/training/test_game_error_recovery.py tests/test_package_boundaries.py -q`
→ 32 passed.

Open for gate: the two
`test_two_policies_never_share_a_forward_and_hung_service_is_bounded` cases
failed only under concurrent load. Three `tests/sim` failures
(conditional-search fixture digest, rollout card admission, seed-197 world
binding) also fail at `ea8d0e874`. Remaining `spawn` uses (`eval/local.py`,
`verify/competency`, `llm/*`, `_process.py`) are not game/replay/serving
children.

## Concept reviews 1–3 (headless, 2026-10-26)

The three concepts hold: **policy key** (checkpoint + ABIs + world + numerics;
matchup only in admission/evidence), **child process** (`module:function`
target, no parent `__main__`, pinned pools) and **thread budget by role**.
Applied: `play_cell` records `ProcessCensus` under `group_worker` on every path
(budget `max_active_games + 7`, or 8 without a runtime; recorded, not enforced);
`verify/scenarios` helpers are public. Kept: `InferenceRuntime` in
`serving_protocol.py`; `supervision._failure` imported by `native_game`
(inherited from `main` at `f5e7289b1`). Full text at `ea8d0e874`.

## Concept review 4 (with Jack Heart, 2026-10-27, at `ea8d0e874`, PR #389)

Finding: evaluation had two ways a checkpoint plays, served (`EvalCompute`
set) and scalar (Torch and the checkpoint in every game child), and the PR
improved only the served one, leaving the worst baseline (`cpu32` scalar)
heavy. **Decided by Jack Heart, 2026-10-27 ("1a 2b 3b"):** 1a evaluation
checkpoint players play served only, with scalar play retired once `cpu32`
served CPU is not slower; 2b code anchors become Torch-free in #389; 3b both
extend MTG-193. **Engine boundary (Jack Heart, 2026-10-27):** share everything
up to the engine (registration, `PolicyKey`, admission, request/reply,
children, per-row seeds, receipts); each device's engine owns process layout,
batching, threads and transfer, and CPU performance is iterated on rental
evidence rather than only gated. Implemented as slices 6–7. Proposal text at
`303e65204`.

## Concept review 5 (headless, 2026-10-27, at `302204fd7`)

Against decisions 1a/2b/3b and the engine boundary, the concepts hold:
a checkpoint has one way to play in evaluation (served; `is_served` →
`bind_execution`, local CPU envelope from `resolve_execution`); code and
search players play in their game child; one engine serves both devices, with
the boundary written in `docs/shared-evaluation.md`. No open product choice.

**Next implement direction (implied by 1a, a name only):** "scalar" no longer
names an evaluation way of playing, but the child still calls its in-child
players that. Rename `native_game._ScalarPlayers` → `_InChildPlayers` (local
`scalar` → `in_child`) and say "served and in-child players" in the
`supervision.py` header. Keep the `scalar_policy` performance span and the
`cold_scalar` benchmark variant: both are retained receipt keys. MTG-183's
runner `scalar` mode is MTG-183's name, not changed here. Remaining after
that: the rental calibration and landing gate as written in Acceptance.

The unstaged `tests/training/game_error_*` edits (a `ChildProcess` game-worker
fixture) are another contribution in progress; preserve them.
