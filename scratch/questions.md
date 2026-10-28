# MTG-193 open questions and assumptions — 2026-10-26

Design: `scratch/19d240-batch-shared-evaluation.md`. MTG-184's landed notes
live at `ce43a5cf7:scratch/questions.md`.

- **Assumption — compute identity unchanged:** grouping across matchups stays
  under `shared-eval-v1` because v1 never fixed batch composition (it already
  varies with arrival order). If a reviewer reads the contract as fixing
  per-matchup grouping, bump `EvalCompute.version`; that is the only change.
- **Settled on rentals:** served-child RSS, thread budgets and the
  oversubscription culprit; evidence in the design's Calibration section.
- **Rentals (Jack Heart, 2026-10-27):** MTG-193 may launch its own machines,
  under $5 total, with a spend ledger. MTG-183's runner is on its branch.
- **Decided (Jack Heart, 2026-10-27/28):** code anchors are Torch-free in
  #389. Checkpoint players are served under a declared allocation and keep
  scalar in-child play without one; retiring scalar waits for a CPU
  measurement (CPU work descoped 2026-10-28).
- **Assumption — supervisor census recorded, not enforced:** the group-worker
  parent hosts Etude callbacks and checkpoint preparation whose threads this
  change does not own; failing an allocation at close would discard results.
- **Stacked on MTG-188 (Jack Heart, 2026-10-26):** this branch merges
  `jack/mtg-188-measure-inference-startup-decision-latency-and-game` (PR #381).
  Its design notes are the parent Task's context, not direction for MTG-193;
  they live at `f5e7289b1:scratch/mtg188-context.md` (removed here 2026-10-26). MTG-188's serving/worker
  timing was ported into the split modules (`serving.py`, `serving_protocol.py`,
  `native_game.py`); its service-start failure test now injects a `ChildProcess`
  launch failure instead of a `multiprocessing` start failure.

## Sync resolution — 2026-10-27

Kept the incoming MTG-188 Qwen runtime and game-runner fixes at
`b8820b0d9e9160086d8acc13075638003c61b3e5` (historical notes in that
revision's `scratch/questions.md`); they do not touch shared evaluation.

## Served-only and Torch-free anchors — implement 2026-10-27

- **Assumption — in-child checkpoint search players keep `torch.manual_seed`:**
  agent PUCT, policy-prior PUCT and local update still play in-child; only the
  plain `checkpoint` kind leaves the child when served, and only then is its
  seeding replaced by per-request service seeds.
- **Superseded 2026-10-28 (option b):** `resolve_execution` no longer adds a
  local CPU envelope; without an allocation checkpoints play scalar and the
  monitor keeps one block per deal.
- **Changed — `bind_execution` admits code and search players** unchanged into
  served cells (they play in their child); before, any non-anchor was rejected.
- **Changed — `serving_benchmark`'s `cold_scalar` arm is served** (one row,
  not resident, no catalog cache). Frozen experiment runners that declare no
  allocation keep scalar play and their identities.

## Engine boundary — implement 2026-10-27

- **Decision — no CPU-specific engine code yet:** `cpu32` runs one shared CPU
  allocation per machine (exp-07: batched CPU inference ≈ 12× scalar obs/s).
  The fallback if it is slower than scalar is slice 7's; intra-op threads could
  change numerics, so they would need a compute-contract decision.
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

- **Blocked `cpu32` arm and unmet batch target:** state, cause (Etude EXP-13)
  and the 88-of-256 active-slot explanation live in the design's Calibration
  section; not repaired here (shared controller, runner setting).
