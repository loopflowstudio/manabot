# MTG-193 — Batch shared evaluation across matchups; light, bounded game processes

Status: **accepted** — Jack Heart approved this design as written ("keep going", 2026-10-26), including `shared-eval-v1` unchanged. Kickoff plan 2026-10-26. Direction accepted by Jack Heart,
2026-10-26: fix all three serving limits from MTG-183's rented calibration
(Etude `d0955f2a44b6`, `307c3a8630af`, `00119d1c9211`). No design choice below
needs product approval; the remote calibration is coordinated through MTG-183.

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
- `PolicyStore.admit` keeps one admission per key; a later registration for the
  same key must have the same checkpoint path/bytes and only adds setups, each
  validated by `validate_agent_setup` against the resident policy.
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
  through `guidance.build_registered_player` imported only when a registration
  lacks `execution` (the one justified local import: scalar policies are
  Torch's only consumer in the child). Scalar hooks keep `isinstance`
  dispatch with the search classes imported lazily (see implementation
  decisions). `torch.manual_seed` runs only for scalar checkpoint players.
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
- Remote calibration (MTG-183 rental runner `experiments/elemental_long_rent.py`,
  one before/after each on A4500 `gpu` and `cpu32` `scalar`, total < $1.50 of
  MTG-183's $10): record games/hour, load, threads by role, RSS per active game,
  `model_loads`, batch-row histogram, `host_to_device_calls/decisions`.
  Targets: A4500 > 2,600 games/h with median batch > 8 and 256 active games
  without OOM; served child RSS ≤ 90 MiB (local import measure 216 → ~50 MiB);
  `cpu32` threads ≤ declared total and load ≤ vCPUs.

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
stated limit); switching the `cpu32` scalar path to shared CPU inference (may
be measured as an extra calibration arm if budget allows).

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
5. Rental before/after through MTG-183 — not run here (needs MTG-183's runner,
   which is not on this branch, and its allowance). It also settles the
   served-child RSS target, the thread budgets and the oversubscription
   culprit (`scratch/questions.md`).

Known limit (draft follow-up, not accepted scope): code-anchor games
(heuristic/random/scripted) still import Torch in-child through
`flat_mc`/`players`.

### Implementation decisions (2026-10-26)

- Scalar hooks keep exact `isinstance` dispatch, with the classes imported
  lazily in `native_game._ScalarPlayers`, instead of a Protocol: a structural
  check would newly call `flat_mc` players' `start_game` (compound reset) and
  could change Commands.
- Census is `ProcessCensus` (threads, Linux thread names, RSS, Torch loaded),
  recorded by role. Named budgets `GAME_BUDGET`/`REPLAY_BUDGET` (4) and
  `SERVING_BUDGET` (16) are enforced with `ThreadBudget.account(counters)`;
  `group_worker_budget` (`max_active_games + 7`, or 8 scalar) is recorded only,
  in `play_cell`'s `performance.json` on every path.
- Child modules are `module:function` targets, so tests swap
  `match.GAME_WORKER` for fixture workers.
- Local measure: model-vs-model served game child 76 MB, 1 thread, no Torch;
  replay child 75 MB; scalar child 285 MB, 1 thread.

## Check result

Focused suite (batched/local/mirror/match eval, child surface, boundaries,
game errors, puct/local-update, teacher1, verify, inference, academy deck
parity): see final line below. Remote calibration (slice 5) pending MTG-183.

Compress 2026-10-26: `uv run --extra dev pytest tests/sim/test_child_surface.py tests/eval/test_batched_eval.py tests/eval/test_local_evaluation.py tests/eval/test_evaluation_match.py tests/test_package_boundaries.py tests/training/test_game_errors.py -q` → 80 passed, 1 skipped; broader suites remain with gate.

## Concept review (headless, 2026-10-26)

Reviewed the implementation against the accepted design and the Task outcome.
The three concepts hold: **policy key** (checkpoint + ABIs + world + numerics;
matchup only in admission/evidence), **child process** (`module:function`
target, no parent `__main__`, pinned pools) and **thread budget by role**
(census recorded, child roles enforced). No open product choice.

Next implement direction (settled by the accepted outcome "every evaluation
process (group workers, …) runs with a declared, bounded thread count … prove
it by counting threads"):

- **Census the group worker on both paths.** Today only `InferenceRuntime.close`
  records the parent (`supervisor`, shared path). The scalar path (the
  `cpu32` arm, 30 group workers, ~1,000 threads) writes no parent census, so
  the calibration cannot attribute threads to it. Record
  `ProcessCensus.take()` into `play_cell`'s `performance.json` on every path,
  under one role.
- **One name for that role: `group_worker`** (the Task's term), replacing
  `supervisor`. Budget `max_active_games + 7` with a runtime; without one the
  worker plays games sequentially, so a constant allowance (main + replay/
  evidence helpers, 8). Still recorded, not enforced, per the accepted
  assumption. These counters are new on this branch, so no frozen evidence
  is renamed; update `docs/shared-evaluation.md`.

Considered and kept: `InferenceRuntime` living in `serving_protocol.py`
(accepted split; the module is "the Torch-free side of serving"); code-anchor
games still loading Torch (known limit, follow-up).
