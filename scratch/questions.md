# MTG-506 open questions and assumptions

Choices taken headless (planning and the realign after concept review,
2026-11-27). Each is marked for review in `scratch/plan.md`.

- **Heuristic identity (review Q2, local v3 taken).**
  Fixing per-game turn state changes 16 of 7,155 heuristic choices in 10 of
  64 profiled games. The plan adds `player_policy.rs` to
  `heuristic_source_paths()` and bumps the Manabot anchor to `native-v3`.
  The Manabot registration changes the cohort and its derived seeds. Etude
  keeps the same `Code.key` and trophy `Seating.definition` when only source
  bytes change. Setting `config.version` to `native-v3` changes both, so the
  standalone probe uses that config. Existing trophy declarations and report
  selections still use the stored tournament identities. Resuming under those
  identities could mix opponents, and changing the existing seatings in place
  would cause a seating conflict. Review the new tournament names and which
  baselines to rerun before publishing shared evaluations.
- **Native heuristic decisions leave the record.** Replay recomputes them; no
  per-decision heuristic latency or chosen offer is stored.
- **Compact vector annotations (review Q1, 1a)** drop `frame_sha256`,
  `chosen_offer`, `action_space_kind` and the per-decision private-hand check.
- **Academy Watch native-seat import is in this Task** (ship-review choice
  `3b`, Pass 7). Pass 8 takes the Pass 7 review's Q1 answer `1a` (a native
  committed-choice log rather than reading card-exposure capture), for review.
- **Opening-settlement isolation is in this Task (review Q4, 4a).**
- **Replay uses two worker threads** with the GIL released in `PyEnv`.
- **Scalar records keep live frames.**
- **L4 arms run on one rental under `--max-usd 5`.**
- **Opening failures use opt-in isolation.** The coordinator asks for
  `isolated=True` on reset and slot loads. The engine reports the actual
  failing seat. An opening failure has an empty commitment prefix and its
  diagnostic replay runs the retained native policies.

- **Post-build review Q1 (1a, for review).** The built-in heuristic refuses a
  declaration without `version: native-v3`. Maintained experiments with stored
  tournaments (`tla_trophies*`, `opponent_memory`, `mana_taps`,
  `tla_frontier`) are left unversioned, so they refuse rather than mix.
  Open: which of them get a new `native-v3` tournament name, and which
  baselines rerun. Irreversible once scored.
- **Post-build review Q2 (2a, for review).** A failed reset walks an empty
  prefix at index 0. MTG-500 can fail before even an agent-only opening
  reaches a position, so ordinary walking skips reset entirely. Diagnostic
  replay runs the recorded policies. In a mixed vector snapshot, the opening
  error is stored on the failed game's ending and projects to a reset failure
  when the stream is read on its own.

- **Pass 4 review Q1 (1a, for review).** One `GameEnd.failure` with
  `operation` replaces the reset-only `OpeningFailure`; a `CodeSeat` decision
  error is a player crash, never a native root failure.
- **Pass 4 review Q2 (2a, for review).** No maintained experiment moves to a
  `native-v3` tournament name in this Task, including `tla_trophies.py`'s
  unscored bf16/tf32 branches.

Ideas outside this Task:

- Managed scalar evaluation (`play_scalar`) could seat the heuristic natively
  too, with the same record shape.
- `state_digest` costs 0.41 ms on a scalar `Env` against 0.18 ms per slot on
  the vector; nobody has looked at why.

The sync onto `85a3a5fb48` keeps MTG-497's match-owned queues and layout
compatibility checks with MTG-506's isolated native seating. This checkout's
plan and tour cover MTG-506. The MTG-497 versions are at the synced commit.

For compression, the comparison base combines the synced MTG-497 commit
`0f885b1f43` and main commit `606e79e90a`. This isolates MTG-506's changes.

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

Scalar and vector callers apply memory blanking before shared host preparation
(2026-11-27). The eval paths use ordinary decision-memory rows, which can be
joined across forwards. Training's retained
prefixes select fold trees separately. Job declarations supply evaluation
precision because saved policy checkpoints have no training-precision field.

Post-pass 7 review choices made headless at `bf5bb9c70d` (2026-11-27), marked
for review in `scratch/plan.md`: 1a (suffix non-float32 trophy benchmark
names; irreversible once scored), 3a (one `vector_layout_key`), 4b (drain
errors before backfill, `GAME_SECONDS = 120.0`, no extra per-slot ceiling).
Q2 (live frames vs compact annotations) stays open with F4 deferred.

Pass 8 uses a keyword-only precision on the trophy report. Etude's default
report call selects float32. The mixed-layout backlog fixture now queues an
unrelated layout: a checkpoint's admitted vector layout is the same in every
deal that uses it, and a difference in seat mana settings shares that layout.

During compression I kept admitted deals in grant order. When a vector drains,
the oldest waiting deal chooses the next layout. The standalone throughput
arms keep the same Match key across precisions and use the trophy pairing's
game budget.

MTG-506 Pass 2 uses the single-step walker for integer-only committed
trajectories with no interventions or issued object-candidate frames and no
consumer apply callback. This covers vector evaluation and keeps semantic
boundary replay on its fork-based path. After a divergent step the walker
rebuilds the verified position, including card capture when requested.

MTG-506 Pass 3 used the retained `w384l8` checkpoint bytes across the panel.
The first trained MPS timing began before formatting was applied. The report
uses the 32-deal rerun whose source fingerprint matches the other nine arms.
Both timings and both 16-deal profiles are under
`.runs/eval-throughput/mtg506/`.

MTG-506 Pass 5 uses the failed game's own operation and frame when projecting
it from a vector journal. The latest batch boundary can describe a subsequent
step by a healthy slot, so a single-game projection leaves that boundary on
the full journal record. The slot-1 step regression advances slot 0 to
exercise this case.

MTG-506 QA (2026-11-27): `ScalarCapture` records no seat policies, and
`managym.Env` exposes no getter for them, so a scalar capture of an Env built
with `player_policies=("agent", "heuristic")` writes an `("agent", "agent")`
record that diverges at index 0 on replay. This predates MTG-506. The vector
capture records its pairs. Left as is; a fix needs an `Env.player_policies`
getter (Rust plus public API) or a `ScalarCapture(player_policies=...)`
argument checked against the Env.

MTG-506 Pass 7 uses native exposure capture to obtain the committed choices
for Academy imports, then binds those choices through Academy's Command path.
A divergence at the first recorded boundary with no verified actions keeps
the opening deal with no reconstructed choices.

All four legs in `.runs/qa506/mps-v64/arena/*/games/deals/candidate__opponent.40900.jsonl`
imported. Leg 0 took 1.44 s to launch with 92 Academy choices from 66 recorded
checkpoint actions. Chromium followed both native seats with **Show both hands**
off. Source bytes matched after launch. The earlier `.runs/eval-throughput/mtg506/w160-mps-matched`
sample diverged at the first digest in the standard verifier on this engine
build. Its original measurement evidence remains tied to its recorded build.
