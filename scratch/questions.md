# MTG-507 assumptions (headless kickoff, 2026-11-27)

Each is reversible.

- **Profile by replaying captured batches**, not by instrumenting the live
  arena; replay runs first on the L4 and sizes the arena repeats.
- The trained checkpoint has older saved readings. Seeded grid controls
  inherit them so cross-width comparisons can hold preparation constant.
  The fresh-reading `w384l8` arena checkpoint has its own capture.
- **Cross-width replay re-tags `Prepared.binding` in the harness** after
  asserting equal observation config, readings and decision memory
  (concept review, 2026-11-27). Alternative: prepare per width from raw
  observations.
- **`bf16-linear-once` is the only candidate production fix**;
  `bf16-cached-autocast` is diagnostic (review Q2a).
- **`agent_loader.py` gains `load_torch_policy` and a shared per-call
  context helper now** (review Q4a), Torch types under `TYPE_CHECKING`.
- **One L4 rental under `--max-usd 5`**, no CPU baseline arms.
  `mtg324-l4-step` was rentable at USD 0.69/hour during the local pass.
- **A fix that changes bf16 logits is not taken**: it would be a new compute
  class beside `etude-policy-bf16-auto-v1`. The report documents the
  threshold instead.
- **float32 stays the default** whatever the profile shows.
- Local MPS timings on 2026-11-27 were taken at load 30-40 and are not
  evidence; only the op counts are used.
- The MTG-497 sync keeps one shared `prepare_batch` for both arena batches and
  captured single rows. The capture hook receives the selected coordinates
  after per-seat memory ablation.

Ideas outside this Task:

- Evaluation loads the padded forward (`bucketed_shapes` and `live_tokens`
  off). Training measured bucketed shapes 2.4-2.6x faster on CUDA; should the
  eval loader use them?

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
