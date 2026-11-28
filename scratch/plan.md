# MTG-506: cut game-side host, replay and heuristic-stepping overhead in vectorized eval

Stacked on MTG-497 (`fb1580b472`, its PR lands first; the `etude-sdk 0.5.0`
merge gate is inherited). The MTG-497 plan is in that branch's history.

## Outcome

A vectorized evaluation against the built-in heuristic spends its wall time
in checkpoint forwards. Every live slot is waiting on a checkpoint decision in
every round, replay no longer stops the stepping loop, and the host does no
per-decision Python projection. The report states games/hour, rows per
forward and replay parity on the 32-deal TLA trophy panel for `w160l8` and
`w384l8`, beside the MTG-497 arms.

Targets on the M4 Max, MPS, 64 slots, trained `w160l8` (today 1,989 games/hr,
15.8 rows/forward, forward 17% of wall): at least 4,000 games/hr, at least 32
mean rows per forward, forward the largest stage. These are targets to judge
the result against, not a protocol; the run is exploratory.

## Where the time goes (profile, 2026-11-27)

`cProfile` over `experiments.eval_throughput --device mps --runner vector
--slots 64 --deals 16` at `fb1580b472` (64 games, 12,700 decisions, 5,545 of
them the checkpoint's; profile and evidence in `/tmp/mtg506-f56n/`). Seconds
are under the profiler, so read them as shares.

| Stage | Seconds | What it is |
| --- | ---: | --- |
| Forward | 32.3 | 465 calls, 11.9 rows mean |
| `finalize_match` (blocks the loop) | 27.4 | replay 22.4 |
| `step_round` host work | 16.1 | |
| Admission | 2.3 | `runtime_fingerprints` 2.1 |

- **The Python viewer frame is the largest host cost, live and in replay.**
  `build_viewer_frame` runs 25,400 times (once live, once in replay per
  decision) for 16.8 s: `serialize_observation` 7.1 s, the
  `ExperienceFrame` validate and dump about 4 s, canonical JSON hashing 5.4 s.
  Live, `arena_decision` is 9.2 of `step_round`'s 16.1 s.
- **Replay digests twice per decision.** `RecordWalker.run`
  (`managym/replay.py:176`) takes a pre-decision digest, clones, steps the
  clone and digests it: 25,464 `state_digest` calls, 10.4 s. A walk of the
  same records without frames costs 0.335 ms per decision (4.25 s for all 64
  games, `/tmp/mtg506-f56n/probe.py`).
- **The brief's scalar findings are not the vector path's costs.** The vector
  runner binds metadata once per leg (`bind_record` 0.8 s total) and digests
  once per decision live (`state_digests` 2.2 s). The per-snapshot
  `evaluation_metadata` deep copy is `ScalarCapture`'s `context_provider`
  (`manabot/sim/native_game.py:292`); in the vector path the deep copies are
  `audit_view`, called five times per game in replay (2.5 s).
- **Heuristic decisions are 56% of all decisions** and each pays the frame,
  annotation, digest and round overhead while its slot contributes no row.

## Findings that shape the design

- **The eval heuristic forgets its turn state at every decision.**
  `manabot/eval/agents.py:248` builds `Heuristic(lambda seed:
  HeuristicPlayer(...))` and both runners call `factory(seed)` per decision
  (`Heuristic.choose`, `choose_code_action`). `main` does the same. The native
  selector is stateful within a turn (`managym/src/agent/heuristic/mod.rs`:
  attack pool, held cards, block plan, kicker memory), and training's native
  heuristic seat keeps that state. Replaying the 64 profiled games through one
  stateful `managym.Heuristic(max_actions=64)` per game changes 16 of 7,155
  heuristic choices, in 10 of 64 games.
- **A native heuristic seat reproduces today's games exactly where that state
  did not matter.** Replaying each record on `managym.Env(player_policies=...)`
  with the opponent seat `heuristic` and only the checkpoint's recorded
  actions reaches the recorded final state digest in 54 of 64 games, the same
  54 (`/tmp/mtg506-f56n/probe2.py`). Playing all 64 games that way takes
  0.40 s of native time.
- **The native policy has no action cap.** `PlayerPolicy::Heuristic` calls
  `select_action(obs, None)` (`managym/src/agent/player_policy.rs`); the
  registered heuristic declares `max_actions: 64` with `first-legal`
  (`experiments/tla_frontier.py:137`). No game in the probe needed the cap,
  but the registration promises it.
- **One policy pair covers the whole vector.** `VectorEnv` holds one
  `[PlayerPolicy; 2]`; the candidate changes seat between a deal's legs, so
  slots need their own pair.
- **The viewer frame is shown to nobody in the vector runner.** Checkpoints
  read native observation rows and the heuristic reads the native
  `Observation`. The frame supplies `frame_sha256`, the chosen offer and the
  private-hand check. `describe_actions` alone yields the offers (0.05 ms per
  decision), and `RecordedVectorEnv._step` already fetches the native semantic
  frame for every stepped slot.
- **`Play` counts need only surfaced decisions of the checkpoint's seat.**
  `checkpoint_counts` (`manabot/eval/behaviour.py:45`) selects the
  checkpoint's native seat; combat and mana facts come from native exposure
  capture, which runs for natively stepped turns too. Build confirms no
  consumer reads the heuristic seat's `PlayCounts` row.

## Design

Choices marked **(for review)** were made with nobody present.

1. **Built-in heuristic seats step natively. (for review)** A seat whose Etude
   `Code` player is the built-in heuristic (no `registration`) becomes a
   native policy seat for its leg; the vector loop surfaces only checkpoint
   decisions. `SlotLoad.fresh` and `reset_each_into_buffers` take a per-slot
   policy pair. The policy spelling gains an optional cap, `heuristic` or
   `heuristic:64`, parsed into `PlayerPolicy::Heuristic { max_actions }`, so a
   record's existing `player_policies` strings replay it with no new field.
   `take_slot_record` writes the slot's pair. A `Code` player carrying a
   `registration`, and any other custom opponent, stays a Python seat with
   both seats `agent`, as today.
   - Consequence: the record of such a leg holds the checkpoint's actions,
     digests and annotations only. Heuristic choices are recomputed at replay
     and bound by the next surfaced digest and the final digest.
     `decisions` counts, per-player latency and the Command cap
     (`max_commands`) then count checkpoint decisions.
   - Passed over: keeping the heuristic in Python behind a faster path. It
     leaves 56% of rounds without a row for that slot, which is the fill
     ceiling the Task names.
2. **The heuristic keeps its turn state for a game in both runners.
   (for review)** The scalar `Heuristic` seat builds one player per game seat
   instead of one per decision, so scalar and vector play the same games.
   This changes heuristic play in about one game in six relative to stored
   results. The edit lands in `manabot/sim/heuristic.py`, which
   `heuristic_source_paths` hashes, so the heuristic's Etude identity changes
   with its behavior and stored matches keep their old player. Frozen
   experiments that pin heuristic source digests keep their literals; build
   checks `tla_trophies*.py` still read their stored tournaments.
   - Passed over: a native policy that resets per decision to preserve
     today's bytes. It would make the lost state permanent and still differ
     from training's opponent.
3. **Vector records carry compact decision evidence; viewer frames are
   reconstructed from the record on demand. (for review; this is the Task's
   open trace choice, 1a)** A surfaced decision's annotation keeps actor,
   action-space kind, the chosen offer (built from `describe_actions`),
   player id, seeds, ordinal and latency, and omits `frame_sha256`. Replay of
   a record without `frame_sha256` builds no viewer frame; it checks actor,
   offer and action-space kind from the walked observation. A frame for any
   decision remains a pure function of the verified state, so nothing is lost
   from stored evidence. Scalar records keep live frames and their hash.
   - Consequence: the vector runner's `private_exposures` counter is no longer
     computed per decision; nothing in that runner receives the frame.
   - Passed over: 1b, live frames with byte-identical traces (the frame is
     52% of live host work and 37% of replay), and a Rust port of the
     projection (byte-equality with the Python projection is as hard as the
     problem).
4. **Replay runs once per transition and off the stepping loop.** The walker
   verifies a plain integer action on the owning env when no consumer needs
   the clone-first guarantee (Academy's apply callback and semantic Commands
   keep it), so a decision costs one digest. `finalize_match` runs in a
   bounded pool of two worker threads; the loop keeps stepping and yields each
   `(match, Result)` as its replay finishes, holding at most `slots` settled
   matches before it stops seating new legs. Admission and replay errors
   still yield `(match, exc)` for that match alone.
   - Why threads: after 1 and 3, replay is about 1.9 s for the 64 profiled
     games (5,300 surfaced decisions at the measured 0.335 ms), against 32 s
     of forward waits in which the loop is idle. Passed over: a pool of
     replay child processes, which costs an import per worker and a pickled
     `ActiveMatch`; it is the fallback if the measurement shows the loop
     starved.
5. **Per-run identity work is computed once.** `runtime_fingerprints` and
   `game_runtime_identities` are cached per match configuration and space for
   the run; replay calls `audit_view` once per game. The scalar capture stops
   deep-copying `evaluation_metadata` at every snapshot with unchanged bytes.

## Source of truth

The native `GameRecord` (`managym-game-v1`): setup, per-game
`player_policies`, surfaced actions, digests and annotations. Results, card
evidence, `Play` counts and any viewer frame derive from walking it.

## Affected surfaces

- `managym`: `PlayerPolicy`, `SlotLoad.fresh`, `reset_each_into_buffers`,
  `managym/__init__.pyi`, `RecordedVectorEnv` per-slot policies,
  `RecordWalker`. Additive; `tests/public_api.json` recorded with plain
  `record`. Snapshot and record strings `agent`/`random`/`heuristic` keep
  their meaning. A changed step path may move the `Engine Budget` digest.
- `manabot/eval/vector.py`, `agents.py`, `match.py`;
  `manabot/sim/native_game.py`, `replay.py`, `evaluation_records.py`,
  `heuristic.py`.
- Readers of decision evidence that must accept both shapes:
  `audit_view`, `_replay_record`, `manabot/eval/match.py` game rows,
  `manabot/reports/evaluation.py`, card evidence, `experiments/eval_throughput.py`.
- Docs: `docs/evaluation.md#etude-agents`, `experiments/reports/eval_throughput.md`.

## Absent and error states

- A record without `frame_sha256` is a compact record; one with it is checked
  as today. A reader never treats absence as a mismatch.
- A native heuristic failure (`AgentError`, panic) arrives in the isolated
  step receipt and fails that leg as a crash attributed to the opponent.
- A leg whose checkpoint never acts before the game ends settles from the
  opening state (`settle_openings`), now also after any step.
- A replay worker's exception yields `(match, exc)`; closing the generator
  waits for running replays and drops none silently.

## Exclusions

- Scalar records keep live frames, their hash and the root-mutation digest
  check.
- Zero-copy `Prepared` transport, the L4 `w160l8` bf16 slowdown and
  `torch.compile` stay deferred from MTG-497.
- No change to observations, action spaces or policy input.

## What goes

- `choose_code_action`'s per-decision `factory(seed)` and the scalar
  `Heuristic.choose` rebuild.
- Live `build_viewer_frame` and `offer_evidence`'s frame hash in the vector
  runner; `ArenaDecision.frame` there.
- The synchronous `finalize_match` drain at the top of `play_pairings`' loop.
- The single vector-wide policy pair as the only way to seat a native policy.
- Must survive: stored records and their replay, scalar trace bytes other
  than the heuristic's corrected choices, per-match error isolation, paired
  seeds, card evidence, the `ForwardError` contract.

## Passes

1. **Native heuristic seats (could prove the plan wrong).** Rust per-slot
   policies and the capped heuristic spelling, one heuristic per game seat in
   the scalar runner, per-slot `player_policies` in records, vector seating
   and settlement. Shows: on four trophy deals, vector and scalar runs agree
   leg for leg on the checkpoint's actions, winner, score and final state
   digest; every vector record replays; rows per forward against MTG-497's
   15.8. If replay of natively stepped records is not deterministic, or
   scalar and vector disagree, the plan stops here.
2. **Compact evidence and background replay.** Annotations without live
   frames, frame-free replay for them, single-digest walk, the bounded replay
   pool. Shows: a test that reconstructs a vector record's frames and matches
   the scalar record's `frame_sha256` at each checkpoint decision; stepping
   continues while a deal replays.
3. **Leftovers and measurement.** Identity and `audit_view` caching, the
   scalar metadata copy, a re-profile to confirm what remains, then the
   panel: Apple arms (CPU scalar, MPS-64, MLX-64) for both checkpoints on the
   M4 Max, and CUDA-64 float32 and bf16 on one L4 rental under `--max-usd 5`.
   Report and docs updated.

## Acceptance check

```bash
uv run --no-sync python -m experiments.eval_throughput \
  --output-dir .runs/eval-throughput/w160-mps-64-native --device mps \
  --runner vector --slots 64 \
  --checkpoint .runs/eval-throughput/policy/00002000-raw.safetensors
```

prints a summary with 128 games, 128 replay passes, mean rows per forward of
at least 32 and `largest_stage: forward`; the scalar arm on the same deals
gives the same winners, scores and checkpoint forward rows.

## Build notes

- This checkout has no `.venv`. MTG-497's
  (`~/src/manabot.mtg-497-support-gpu-batched`) has the editable EXP-77 Etude
  SDK; this one needs the same (`docs/evaluation.md#etude-agents`), and
  `uv run --no-sync`.
- `managym/_managym.cpython-313-darwin.so` here was copied from MTG-497 for
  the profile and is stale after the first Rust edit.
- Rust tests run in debug with `CARGO_BUILD_JOBS=2`.

## Done when

- Against the built-in heuristic, every live slot contributes a row to each
  round's forward, and the report shows the rows-per-forward and games/hour
  change on the trophy panel for both checkpoints on Apple Silicon and L4.
- Other slots keep stepping while a finished deal replays.
- Every game replays, and scalar and vector runs of the same deals agree on
  checkpoint actions, winners and final digests.
- A custom Python opponent still plays through the vector runner.
- `tests/test_package_boundaries.py`, `scripts/public_api.py check` and the
  debug Rust suite pass.
