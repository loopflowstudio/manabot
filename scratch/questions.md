# MTG-193 open questions and assumptions — 2026-10-26

Design: `scratch/19d240-batch-shared-evaluation.md`. MTG-184's landed notes
live at `ce43a5cf7:scratch/questions.md`.

- **Assumption — compute identity unchanged:** grouping across matchups stays
  under `shared-eval-v1` because v1 never fixed batch composition (it already
  varies with arrival order). If a reviewer reads the contract as fixing
  per-matchup grouping, bump `EvalCompute.version`; that is the only change.
- **Assumption — served-child RSS target ≤ 90 MiB:** derived from local import
  measurements (47 MiB light set vs 216 MiB today); confirm on Linux.
- **Open — oversubscription culprit:** not reproducible on macOS. The rental
  thread census by role and thread name decides which pool caused ~1,000
  threads on `cpu32`; the bound is enforced regardless.
- **Rentals (Jack Heart, 2026-10-27):** MTG-193 may launch its own machines,
  under $5 total, with a spend ledger. MTG-183's runner is on its branch.
- **Decided (Jack Heart, 2026-10-27):** code anchors become Torch-free in
  #389, and evaluation checkpoint players become served-only, gated on a
  `cpu32` served-CPU calibration arm against `cpu32` scalar. See concept review
  4 in the design.
- **Assumption — supervisor census recorded, not enforced:** the group-worker
  parent hosts Etude callbacks and checkpoint preparation whose threads this
  change does not own; failing an allocation at close would discard results.
- **Assumption — budgets:** game/replay 4, serving 16 (CUDA driver/profiler
  threads). Local counts are 1; the rental census confirms or revises them.
- **Stacked on MTG-188 (Jack Heart, 2026-10-26):** this branch merges
  `jack/mtg-188-measure-inference-startup-decision-latency-and-game` (PR #381).
  Its design notes are the parent Task's context, not direction for MTG-193;
  they live at `f5e7289b1:scratch/mtg188-context.md` (removed here 2026-10-26). MTG-188's serving/worker
  timing was ported into the split modules (`serving.py`, `serving_protocol.py`,
  `native_game.py`); its service-start failure test now injects a `ChildProcess`
  launch failure instead of a `multiprocessing` start failure.

## Sync resolution — 2026-10-27

The incoming conflict revision `b8820b0d9e9160086d8acc13075638003c61b3e5` extends the
inherited MTG-188 Qwen runtime and game runner. Retain its runtime fixes,
effective campaign budgets, hosted-model accounting and checkpoint-opponent
support; these paths do not implement MTG-193's shared evaluation changes.
Keep this branch's assumptions above. The target's MTG-188 notes remain at
`b8820b0d9e9160086d8acc13075638003c61b3e5:scratch/questions.md` as historical
context, not new execution direction. No rental is launched by this sync.

## Served-only and Torch-free anchors — implement 2026-10-27

- **Assumption — in-child checkpoint search players keep `torch.manual_seed`:**
  agent PUCT, policy-prior PUCT and local update still play in-child; only the
  plain `checkpoint` kind left the child, so only its seeding went with it.
- **Assumption — local CPU envelope (`resolve_execution`):** callers without an allocation
  (monitor, study, attack, card evaluation, `evaluate.py`, experiment runners)
  serve on CPU with `max_active_games=1` (games stay sequential, as before) and
  default batching; deadline `2 × games × game_seconds` covers play and replay.
  The CPU engine work (Jack Heart's engine-boundary steer) may change these.
- **Changed — the training monitor plays one panel:** it always declares an
  execution (local CPU serving when none is given), so all deals share one
  CPU service instead of one `evaluate` call per deal. Block retention still
  comes from `rows.json` after each completed deal.
- **Changed — `bind_execution` admits code and search players** unchanged into
  served cells (they play in their child); before, any non-anchor was rejected.
- **Changed — `serving_benchmark`'s `cold_scalar` arm is served** (one row,
  not resident, no catalog cache). Frozen experiment runners (MTG-181 grid,
  MTG-186, MTG-183 head-to-head) now serve too; reruns get new identities.

## Engine boundary — implement 2026-10-27

- **Decision — no CPU-specific engine code yet:** a CPU host already gets
  several serial CPU services by running several allocations (one per group).
  exp-07 measured batched CPU inference at about 12× scalar obs/s. If the `cpu32`
  served arm is slower than scalar and `serving.json` shows the service
  saturated, the first candidates are several workers per allocation and then
  intra-op threads. Both are allocation settings; intra-op threads could change
  numerics, so they would need a compute-contract decision.
- **Search players stay in-child (Jack Heart, 2026-10-27):** serving search
  players' model calls is out of scope for MTG-193; follow-up MTG-205.

## Code-anchor source identity moves to `sim/anchors.py` (compress, 2026-10-27)

Scripted-greedy and random registrations now pin `manabot/sim/anchors.py`
bytes (random formerly pinned `search/flat_mc.py`; scripted pinned
`sim/players.py`, whose bytes this slice changes anyway). New anchor
registrations therefore get new `source_sha256` values and do not reuse
earlier deal evidence; frozen evidence is untouched. Assumed acceptable
under decision 2b (Torch-free anchors); revisit if a frozen v1 anchor must
keep matching for resumption.

## Calibration baselines (realign, 2026-10-27)

- **Assumption — "before" arms are the retained MTG-183 runs** (`d0955f2a44b6`
  A4500 about 2,600 games/h; `00119d1c9211` `cpu32` scalar about 1,700) on the
  same 1,000-game pair, so the $5 ledger buys only the two "after" arms. A
  fresh `main` before-arm is optional if the ledger allows and the hosts differ.
- **Assumption — one launching writer (2026-10-27 15:50):** the interactive
  Task session `a35bc11b…` committed `d2f46c62c` and calibration `7616dd740`
  while the implement step held this worktree. The implement step therefore launched
  nothing after `2e066580c458` and left its `serve` drain fix uncommitted for that
  writer and gate, so two writers would not spend the $5 ledger.

## Parent hop (decided by Jack Heart, 2026-10-27: option b; implemented)

- **Assumption — failure text:** a game whose handoff fails reads
  "inference connection handoff failed: …"; a decision timeout in the child
  keeps "inference service request timed out". Both remain infrastructure
  failures with no failed player and fail the allocation.
- **Assumption — `allocation.json` has no request latency:** decisions no
  longer pass the supervisor; latency lives in `worker_performance` (game) and
  `serving.json` (service). No reader used the allocation's request counters.
- **Assumption — the w13 calibration branch** (revert of `e742e8bff`) is
  measurement-only and never lands; MTG-183's checkpoints cannot play w14.

## Calibration after the parent hop (implement, 2026-10-27)

- **Done:** A4500 `afb7b8581189`, 5,247 games/h (was ~2,600), median batch 5.
- **Blocked — `cpu32` arm:** a stale Etude session from MTG-193's lost run
  `6ed5413f5292` holds the `cpu32` on-demand's one slot (bug in
  `Scheduler._observe`, see the design's Calibration section). Proposed Etude
  fix: end a receipt-less session once its deadline passes; tracked as Etude
  EXP-13, with this case commented there (2026-10-28). Not repaired here: the
  shared controller other Tasks use would have to be stopped and its store
  edited. Landing gate still needs
  this arm or Jack Heart's decision on the A4500 evidence.
- **Assumption — batch target:** median > 8 was not reached because only 88 of
  256 slots were active (10 × 4 groups do not keep 256 games pending), not
  because of the service (~54% busy). Raising groups is a runner setting;
  the 256-active OOM check is likewise unexercised.
