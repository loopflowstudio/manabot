# MTG-506 open questions and assumptions

Choices taken during planning (headless, 2026-11-27). Each is reversible and
marked for review in `scratch/plan.md`.

- **The built-in heuristic steps natively and its decisions leave the
  record.** Replay recomputes them. Is losing per-decision heuristic evidence
  (latency, chosen offer) acceptable for managed evaluation?
- **The heuristic keeps its turn state for a game.** Today both runners, and
  `main`, rebuild it at every decision, which changes 16 of 7,155 choices in
  10 of 64 profiled games. Fixing it changes heuristic results relative to
  stored tournaments and gives the heuristic a new Etude identity. Should
  stored heuristic baselines (TLA trophies) be replayed against the corrected
  opponent?
- **Vector records drop `frame_sha256` and the per-decision private-hand
  check** (the MTG-497 open choice 1a). Frames are rebuilt from the record
  when wanted.
- **Replay uses two worker threads**, not child processes.
- **Scalar records keep live frames**; only the metadata deep copy is cut
  there.
- **L4 arms run on one rental under `--max-usd 5`**, as MTG-497's did.

Ideas outside this Task:

- Managed scalar evaluation (`play_scalar`) could seat the heuristic natively
  too, with the same record shape.
- `state_digest` costs 0.41 ms on a scalar `Env` against 0.18 ms per slot on
  the vector; nobody has looked at why.

---

# MTG-497 open questions and assumptions

Assumptions taken during planning (headless, 2026-11-26). Each is reversible.

- **Precision declaration confirmed in steer `44a4f9bb`.**
  The selected option is: per-precision
  `AGENTS_*`/`load_*`, precision a registration argument, the envelope id read
  from `candidate.compute_class_id`, precision in `Environment.config`, seeds
  pinned to the float32 envelope.
- **Apple GPU inference before the Etude companion.** The loader selects MLX
  when available, then MPS, in Etude's direct mode, where every game process
  loads its own copy. MPS probe: 12 processes
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
- **MLX comparison.** Reuse the pass 3 CPU scalar and MPS vector-64 receipts
  and checkpoint bytes on the same M4 Max. Run one float32 vector-64 arm per
  checkpoint. The timings come from separate sessions. Verify every replay
  and compare the actions and state digests leg by leg.

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
- **Sync `play` removal and `play_many` committed in EXP-77 (`c5923cf`, unreleased).** When `etude-sdk > 0.4.7` ships,
  every Manabot `play` callback converts to `async def` (or `play_many=`) in pass 5.

Ideas outside this Task:

- **Zero-copy transport for policy inputs.** Gate 1's 6db832 bus passed zero
  arrays by pointer. Its array discovery sees containers and NumPy arrays,
  while `Prepared` arrives as a dataclass and is pickled in-band. The report
  measures the direct bus with the production input shape. What interface
  should expose those buffers without duplicating `Prepared`?
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

Review choices made headless at `c1df3f02c2` (2026-11-27), each reversible
and marked for review in `scratch/plan.md`: 3a (async-only `Sampler` keyed
by checkpoint), 5a (one `MANABOT_EVAL_DEVICE` runtime-target knob), 6a
(per-active-slot round charge), 4a (transport deferred). Open and not taken:
Q1 (live viewer frames vs compact replay-verified annotations) changes
stored traces; the runner keeps live frames until someone chooses.

Pass 6 implementation choices (2026-11-27):

- Cache metadata by local artifact path, SHA256 and size for the vector run,
  using Etude's immutable artifact contract. Up to twice the slot count of
  checkpoint entries are cached. The deferred layout queue holds up to the
  slot count of matches.
- Apply memory ablation to individual native row coordinates before their
  checkpoint's one batch, including when both seats use the same weights.
- Store a non-float32 precision in the trophy environment config; float32
  omits it, like the float32 deal's seating key, and paired game seeds stay
  stable across precisions. QA (2026-11-27) found that writing
  `"precision": "float32"` raised Etude `tournament_conflict` for a stored
  benchmark of the same name rather than starting a new tournament.
- `tla_trophies_floor.final_environment()` now names its files by the
  digest the stored colab and floor final tournaments recorded
  (`a6a2f68b…`), like `tla_trophies.ENVIRONMENT_FILES`. It hashed the live
  `tla_trophies.py` and `pairing.py`, so this branch's edits made
  `etude eval experiments/tla_trophies_colab.py` raise `tournament_conflict`
  against all 126 stored colab tournaments.
- Not this branch: `experiments/tla_trophies.py` alone already conflicts
  with its stored `tla-trophies-held-out-32` runs on `main` (stored files
  `54cef888…`, `ENVIRONMENT_FILES` `d7dab787…`).

During compression I kept the scalar `play_pairing` keyword arguments used by
experiments. Both runners pass the same `PairingSpec` through admission.
