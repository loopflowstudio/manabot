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
  and `_game_worker` move here from `sim/match.py`. Code players are built
  Torch-free by `players.build_code_player` (random, scripted, heuristic);
  only search players (flat-MC, PUCT, local update) import Torch, lazily in
  `_InChildPlayers`, and only they call `torch.manual_seed`. Plain checkpoint
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
Commands; making group-worker parents
Torch-free (they load checkpoints for registration); CPU/CUDA parity (stays a
stated limit); serving search players' model calls (follow-up MTG-205); a
second CPU engine before rental counters show a saturated CPU service.

## Delete — do not maintain

All removed, including the parent request relay; no deletion target remains
(list at `a8da83e82`). Forbidden: a second worker module kept "for scalar";
lazy-export shims; a fallback to spawn; a v2 key beside v1.

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
   the one serial service per allocation; after the parent hop, `cpu32` runs
   one shared CPU allocation per machine. An in-process engine saves only the
   IPC hop and loses killability. Multi-worker allocations, intra-op threads
   and a CPU delay are not written until rental counters show a saturated CPU
   service (YAGNI; no second engine without evidence). **Remaining:** the
   rental calibration in Acceptance, run from a calibration branch that merges
   MTG-183's runner (`experiments/elemental_long_rent.py`), with its own run
   names and no `collect` into MTG-183's database. Its `scalar` mode now
   serves, so the scalar baseline is the retained run `00119d1c9211`. If
   `cpu32` served is slower, iterate the CPU engine (several workers per
   allocation, then intra-op threads, which need a compute-contract decision)
   before Jack Heart decides on retaining scalar play.

### Calibration (implement, 2026-10-27)

Runner edits on the calibration branch (first merge `950e2066e`; current
branch below): runs `mtg193-cal-{rent,batched,gpu}`, `GPU`
`max_active_games=256` (`max_batch_rows` 64), fixed units skip `stored()`,
receipts returned as `receipts.tar` (per-group `performance.json`/
`serving.json`) and `progress.jsonl` (per-minute load, threads/RSS by child
target). Unit `pair-005950-v-005600` (1,000 games).
Launch path: etude-sdk 0.2.9 refuses the shared launcher DB (schema 10, held
by MTG-183's 0.2.6 controller), so runs are built with the branch
(`~/.manabot/mtg193-calibration/build_runs.py`) and queued with the
controller's 0.2.6 client (`queue_runs.py`), under
`doppler run -p manabot -c prd -- doppler run -p etude -c prd --`.

Ledger (cap $5; committed = caps until settled):

| Etude run | arm | rental | cap | settled |
|---|---|---|---|---|
| `091840ecd017` | `cpu32` `batched` (6 × 12) at `950e2066e` | cpu3c 32 vCPU, $0.96/h | $2.50 | ~$0.07, 0 valid |
| `a8182c3ea633` | A4500 `gpu` (256 active) at `950e2066e` | a4500 ≤ $0.40/h | $0.40 | ~$0.01, 0 valid |
| `a43579fb6172` | A4500 `gpu` at `014868990` (pre-ABI-fix) | a4500 | $0.40 | $0.00, lost (capacity_refused) |
| `ac309c613577` | `cpu32` `batched` (6 × 12) at `625283822` | cpu3c 32 vCPU, $0.96/h | $2.50 (quote $1.20) | ~$0.68, 1,000/1,000 valid |
| `2e066580c458` | A4500 `gpu` (256 active) at `625283822` | a4500 | $0.40 | ~$0.10, 999/1,000 valid (1 hit 600 s at 818 decisions) |
| `6ed5413f5292` | `cpu32` `batched` at `7616dd740` (+ fsync fix) | cpu3c 32 vCPU | $2.50 | lost: pod create HTTP 400, no machine; Etude estimate ~$1.20, likely $0 actual |
| `ccc8a29160c0` | A4500 `gpu` (256 active) at `7616dd740` (+ fsync fix) | a4500 | $0.40 | ~$0.03, 1,000/1,000 valid |
| `afb7b8581189` | A4500 `gpu` (256 active) at `41102d8c3` (parent hop) | a4500 $0.25/h | $0.40 | ~$0.05, 1,000/1,000 valid |

Settled ~$0.94 measured, ~$2.14 counting `6ed5413f5292`'s estimate; $2.86 of $5 left. Nothing running (2026-10-28 03:50 UTC).

Pre-hop arms (full narrative, tables and the yappi profile at `a8da83e82`):
`950e2066e` arms were all invalid (MTG-183's w13 checkpoints vs main's w14
`e742e8bff`; the calibration branch reverts it). Experiment runners
registered checkpoints with the default observation space, failing v1 ABI
admission (fixed in `c3bd35895`/`064d2395c`). At `625283822`: `cpu32` 1,508
games/h (layout: 200 cells × 5 games, ~13 active), A4500 2,557 (batches all
1 row). Met then and since: `model_loads` 2, `host_to_device_calls` = 2 ×
forwards, served game child 1 thread, no Torch, ≤ 88 MiB, serving child 5
threads ≤ 16. Over budget (recorded, not enforced): A4500 group workers
267–268 threads vs 263; on `cpu32` Etude's `python3` callbacks (~330 each)
and the runner parent (772), which are not this PR's processes. Per-Command
fsync gained nothing (reverted `1a032e95b`). The binding limit was the
allocation parent relaying every request (~8 ms CPU per decision, ~150
decisions/s), which led to the parent-hop decision below.

### Implementation decisions (2026-10-26)

- In-child hooks keep exact `isinstance` dispatch, with the classes imported
  lazily in `native_game._InChildPlayers`, instead of a Protocol: a structural
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

Implement, 2026-10-27 (parent hop): `uv run --extra dev pytest tests/eval/test_batched_eval.py tests/eval/test_local_evaluation.py tests/sim/test_child_surface.py tests/test_package_boundaries.py tests/eval/test_evaluation_match.py tests/training/test_game_errors.py tests/eval/test_mirror_admission.py tests/eval/test_inference_receipts.py -q` → 146 passed, 1 skipped (11 min). Host load average ~370 from other sessions during the run.

Compress (2026-10-27/28, detail in git log; serving and execution modules settled, no
further cut found): `uv run --extra dev pytest tests/eval/test_batched_eval.py tests/sim/test_child_surface.py tests/training/test_game_errors.py -q` → 50 passed, 1 skipped.

Compress 2026-10-28: `tests/academy/test_trigger_order.py` imported `ObservationSpace` from the bare `manabot.env` package (inherited from `main`'s `e742e8bff`); now imports it from `env.observation` → 3 passed. Dropped `verify/games.MOUNTAIN_DECK` (unused since `main`) → `tests/verify/test_util.py tests/test_package_boundaries.py` 16 passed.
`InferenceRuntime` writes `allocation.json` through one `_write_allocation` → `tests/eval/test_batched_eval.py tests/sim/test_child_surface.py` 26 passed, 1 skipped.
Dropped the resolved concept-review 5 page (`scratch/concept-review.html`, at `b81336386`); `players.build_code_player` documents that demo search also takes `seed` → `tests/sim/test_child_surface.py tests/test_package_boundaries.py` 14 passed.
Compress 2026-10-28 (later): no code cut in `threads`, `child`, `serving_protocol`, `serving`, `native_game`, `supervision`, `players`, `match`, `anchors`, `frames`, `eval.execution`, `verify.games`, `model.inference.transfer`, `verify.{scenarios,competency,scenario_validation,checkpoint_scenarios}`, `eval.{match,evaluation,serving_benchmark}`, `experiment.checkpoint_evaluation`, `search.flat_mc`; trimmed resolved notes from `scratch/questions.md`. No code edits, no rerun.

Open for gate: three `tests/sim` failures (conditional-search fixture digest,
rollout card admission, seed-197 world binding) also fail at `ea8d0e874`.
Remaining `spawn` uses (`eval/local.py`, `verify/competency`, `llm/*`,
`_process.py`) are not game/replay/serving children.

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

## Concept reviews 5–6 (headless, 2026-10-27, at `302204fd7`, `c6c8f4c87`)

Concepts hold: a checkpoint plays served only (`is_served` → `bind_execution`,
local CPU envelope from `resolve_execution`); code and search players play in
their game child (`native_game._InChildPlayers`); one engine serves both
devices. The `scalar_policy` span and `cold_scalar` benchmark variant keep their
names as retained receipt keys. No open product choice.

## Parent hop decided (Jack Heart, 2026-10-27)

Jack Heart lifted the Exclusion "removing the parent's request hop" ("I dont
have a reason for that") and chose option **b**, accepting a bigger change for
the faster long-run design: **game children talk to the inference service
directly.** The allocation parent keeps game lifecycle, deadlines, failure
classification and the durable journal, but it is no longer on the decision
path.

Requirements carried forward:
- Same seeds give the same Commands; replay verification and evidence formats
  are unchanged; identities change only if the compute contract does.
- Durability: a Command is journaled at the same boundary as today before it
  is treated as committed. Journal writes must not sit on the decision path.
- Failure handling: a dead or hung service, a dead child, and deadline expiry
  remain infrastructure failures, never forfeits, with the same receipts.
- One service, many direct client connections. The service loop's
  per-request cost becomes the next ceiling: measure it (receipt counters),
  and keep per-request work minimal (no per-request `asdict`/JSON).
- CPU and GPU engines share this layout up to the engine (Jack Heart's
  engine direction). `cpu32` uses one shared CPU service per machine instead
  of a service per cell.
- Calibrate on rentals within the remaining $5 ledger.

Implemented (2026-10-27, local):
- `InferenceRuntime.connect` opens a socket pair per served game and passes
  one end to the service over a handoff socket (`socket.send_fds`); the
  service acknowledges each descriptor before the parent closes its copy
  (closing an in-flight socket's last descriptor disconnects it on macOS).
  `supervise_game` passes the other end to the game child (`ChildProcess`
  `pass_fds`) and closes its own copy; a failed handoff is that game's
  infrastructure failure, retained in its journal.
- `InferenceClient` (Torch-free) speaks `Connection` framing over a plain
  socket so send and reply share one deadline (`SO_SNDTIMEO` bounds each
  write, not the send). `game_worker` clamps each request to the game's
  wall deadline; a failure is reported as `("inference_failure", message)`,
  and the parent fails the allocation and records the game as an
  infrastructure failure with no failed player.
- `serve` watches control, handoff and game connections with one
  `selectors.DefaultSelector`; one read per ready connection per wake-up
  (each game has at most one request in flight) fills a due batch from any
  backlog. `service_receive` (`EvalPerformance.measure`, no span) and
  `reply_send` measure the service's own per-request work.
- The parent's journal is opened once per game, still fsynced per event.
- Not done here: `cpu32` one shared CPU service per machine is a runner
  layout (MTG-183's `batched` mode) and goes on the calibration branch.

Calibration after the parent hop (implement, 2026-10-27): worktree
`~/src/manabot.mtg193-calibration`, branch `jack-heart/mtg193-calibration-2`
(unpushed) = `recovery/mtg193-calibration-20261028` + this branch at
`10c91a433` (`d61ae13fd`; supervision takes this branch, so the fsync change
stays reverted; both experiment ABI hunks take this branch), then
`41102d8c3`: the `batched` arm is one shared CPU allocation per machine
(`EvalCompute(max_active_games=64, max_batch_rows=64)`, 10 × 4 groups, as
`gpu`). Runner tests pass (22). Local smoke (`smoke_shared.py`, shared CPU,
4 active, 100 games): 100/100 valid, `model_loads` 2, batch rows 1/2/3/4 =
8,130/3,380/925/33 of 12,468 forwards (before the hop: all 1), request p50
26 ms, serving 1 thread. Arms built as `batched-hop.json` / `gpu-hop.json`.

A4500 `afb7b8581189` (parent hop, receipts in
`~/.manabot/mtg193-calibration/out/afb7b8581189/`): **5,247 games/h**
(686 s; before 2,557–2,608), 1,000/1,000 valid; `model_loads` 2; median
batch 5 rows, mean 5.6 (38,577 forwards for 214,231 decisions; before: 1);
`host_to_device_calls` = 2 × forwards (0.36 per decision); request p50 92 ms,
p95 212 ms (before: p50 1.14 s); service CPU 366 s of 683 s (~54% busy);
serving 5 threads, 1.1 GiB; game children 1 thread, ≤ 77 MiB, no Torch;
load 7–20 on 48 vCPUs. Peak active children 88 of 256, so the batch target
(median > 8) and the 256-active OOM check are bound by how many games the
10 × 4 groups keep pending, not by the service. Group-worker parents
(`elemental_long_rent`) hold up to 135 threads each; Etude `play` callbacks
(`python3`) ≤ 14 threads, ≤ 609 MiB.

Remaining: the `cpu32` arm (`batched-hop.json`, built at `41102d8c3`, cap
$2.50) and the landing gate. It is not queued: the `cpu32` on-demand (limit 1)
reports "at its machine limit" because lost run `6ed5413f5292`'s session
(`cpu32-b937fa5a80b7`, no pod at RunPod, window ended 00:08 UTC) is never
ended. Etude bug (0.2.6 and 0.2.9, `Scheduler._observe`): a session with no
receipt past its deadline has its attempts finished as lost
(`create_unconfirmed`) but returns without `_end_session`, so
`unsettled_sessions` keeps counting it; `etude terminate` does not clear it.
Once Etude ends that session, queue with `queue_runs.py batched-hop.json --go`
from the 0.2.6 client as in Calibration. Last rechecked 2026-10-28 11:09 UTC:
still blocked, nothing queued, no new spend. Unfixed on Etude main v0.2.11
(`scheduler.py:1898-1902`); tracked as Etude EXP-13 (comment added).
Compress 2026-10-28 (third): `guidance._build_policy_player` now checks its leaf evaluator by class identity instead of the name string it inherited from `players` → `tests/sim/test_puct_compatibility.py tests/sim/test_mcts.py` 21 passed. No other cut in `guidance`, `teacher1_evidence`, `verify.util`.
Compress 2026-10-28 (fourth): `docs/shared-evaluation.md` no longer describes the retired one-allocation-per-group CPU layout; no cut in `model.inference`, `performance`, `replay`, `target_practice`, `decision_profile`, experiment runners. Docs only, no rerun.
Compress 2026-10-28 (fifth): tests only. `test_batched_eval` imports `os`/`signal` at the top; the missing-target `ChildProcess` check is its own test → `tests/eval/test_batched_eval.py tests/sim/test_child_surface.py` 27 passed, 1 skipped. No cut in the small import-cutover diffs (`env`/`infra`/`model` packages, `heuristic`, `rollout`, `sim`, `belief.demo`) or the game-error fixtures.
Compress 2026-10-28 (sixth): no cut in the remaining import-cutover files (`academy`, `model`, `config`, `eval.{admission,evaluate,cards,endpoint}`, `experiment.{runner,evaluation_queue,execution,study}`, `search.{puct,value,local_update}`, AGENTS.md); `ruff check` on all changed Python passed. No edits, no rerun.
