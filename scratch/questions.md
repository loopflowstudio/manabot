# MTG-497 open questions and assumptions

Assumptions taken during planning (headless, 2026-11-26). Each is reversible.

- **Open, irreversible once rows are stored: how precision is declared.**
  The plan adopts the concept review's 1a for review: per-precision
  `AGENTS_*`/`load_*`, precision a registration argument, the envelope id read
  from `candidate.compute_class_id`, precision in `Environment.config`, seeds
  pinned to the float32 envelope. Alternatives: 1b, an environment variable
  mirrored by hand into all call sites and `Environment.config`; 1c,
  float32 only in this Task. Nothing reduced-precision is recorded until pass
  3's rental, so this can still change before then.
- **MPS before the Etude companion.** The loader selects `mps` in Etude's
  direct mode, where every game process loads its own copy. Probe: 12 processes
  sharing MPS reached 478 rows/s at w160l8 against 323 for one. If per-process
  GPU copies of a large checkpoint exhaust unified memory on the 16 GB mini,
  `MANABOT_EVAL_DEVICE=cpu` is the escape until Etude shares one agent process.
- **Default precision stays float32.** Memory says wide models default to half
  precision; `manabot/backends/torch/learner.py` says nothing picks a reduced
  precision from device or width. The plan follows the code. Should evaluation
  default to the precision a checkpoint was trained under? Checkpoints do not
  record it today.
- **The measurement's 10M+ model is seeded random weights (w384l8, 33.8M)**
  unless a trained one is at hand; random play changes game length, which the
  report states.
- **Apple arms run on Apple hardware outside CI** (the mini if reachable, else
  the laptop), 128 games per arm. This is the real workload, not induced load.

- **Runner trace parity (resolved in pass 2, `dd11bd0a3f`).** Eval keeps
  its existing `managym-game-v1` `GameRecord` format with zero changes to
  replay/audit/card consumers. `RecordedVectorEnv` writes `mode="vector"`,
  `player_policies=("agent", "agent")`, `seed_stride=num_envs`, `stream=slot`,
  while reproducing every deterministic decision field (`frame_sha256`,
  `command`, `command_sha256`, `chosen_offer`, `actions`, `digests`, `winner`,
  `score_a`, `plays`/`behaviour`/`cards`/`card_counts` tables) identically to
  `play_pairing`. Note that `game_trace_sha256 == GameRecord.sha256` hashes
  wall-clock timings (`seconds`, `execution_seconds`, `latency_seconds`,
  `worker_performance`) and `mode`, so even two scalar runs differ in raw
  `game_trace_sha256`.
- **Seeded per-slot load (resolved in pass 2, `dd11bd0a3f`).** Added
  optional keyword-only `seed: int | None = None` to `SlotLoad.fresh` in Rust
  and `managym/__init__.pyi` (non-breaking; recorded in `tests/public_api.json`).
- **Sync `play` removal is EXP-77's reading, not confirmed.** If it lands,
  every Manabot `play` callback converts to `async def` in pass 3.

Ideas outside this Task:

- **Game-side recording cost.** Host work is 0.85 s per game against the
  heuristic and 1.5 s in self-play, the limit on CUDA below about 34M (MTG-328
  saw 21% L4 utilization). In a profile of one pairing, `evaluation_metadata`
  deep-copies registrations and identities at every capture snapshot (about
  three per decision, 20% of host time), `state_digest` runs five times per
  decision (10%), and the viewer frame is serialized twice per decision (14%).
  A Task that removes the repeated copies while keeping trace bytes identical
  would raise CUDA games/hour more than batching does at w160l8.
- **Joining memory rows with fold trees** (retained native memory) needs fold
  trees rebuilt in `concatenate`; evaluation rows carry none.
