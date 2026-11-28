# MTG-506: cut game-side host, replay and heuristic-stepping overhead in vectorized eval

Stacked on MTG-497, which lands first and is gated on `etude-sdk 0.5.0`.
Its Pass 7 enabled zero-copy inputs, separated preparation from sampling,
and fixed the failure backlog's slot grant. All 512 Gate 1 games replayed,
and the MPS compilation probe favored eager execution. Pass 8 aligned
observation layout checks with slot reuse, delivered checkpoint errors before
requesting replacement legs, and shared the trophy pairing's 120-second game
budget with the throughput probe. See the MTG-497 plan at `85a3a5fb48` and
`experiments/reports/eval_throughput.md`.

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

Choices marked **(for review)** were made with nobody present. The concept
review (`scratch/concept-review.html`, at `2ee58b5d2d`) reshaped items 1–5;
its recommended answers to its four questions are adopted here, each for
review.

1. **Built-in heuristic seats step natively; each game owns its policy pair.
   (for review)** A seat whose Etude `Code` player is the built-in heuristic
   (no `registration`) becomes a native policy seat for its leg; the vector
   loop surfaces only checkpoint decisions.
   - `PlayerPolicy::Heuristic { max_actions: Option<u32> }` with hand-written
     string `Serialize`/`Deserialize`/`Display`/`FromStr` (`agent`, `random`,
     `heuristic`, `heuristic:64`), so `GameProvenance`, snapshot replay and
     record strings keep their encoding. `PlayerPolicy::is_heuristic()`
     replaces the `contains(&PlayerPolicy::Heuristic)` checks in `env.rs`.
     The native seat honours the cap (`select_action(obs, max_actions)`).
   - `SlotLoad.fresh(..., player_policies=None)` and
     `reset_each_into_buffers(..., player_policies=None)` take a per-slot pair;
     an omitted pair is the `VectorEnv`'s default, never the slot's previous
     occupant (`previous.replacement_root()` must not carry it over).
     `SlotLoad::Snapshot` restores its own pair.
   - `GameInputs` gains optional `player_policies` (omitted when equal to
     `GameRecord.player_policies`), so multi-slot snapshots, journals and
     `replay_failure_inputs` keep swapped seats. Single-game records keep
     `GameRecord.player_policies`; `take_slot_record` writes the slot's pair.
   - Python: split `manabot/eval/agents.py:Heuristic` into
     `NativeHeuristic(max_actions)` (the vector runner seats it natively, the
     scalar runner drives a `HeuristicPlayer`) and `CodeSeat(factory)` for
     registered/custom code players, which stay Python seats with both seats
     `agent`. Only custom seats call `vector.decision_observations`.
   - Consequence: such a record holds only the checkpoint's actions, digests
     and annotations; heuristic choices are recomputed at replay, bound by the
     next surfaced digest and the final digest. `decisions`, per-player
     latency and `max_commands` count checkpoint decisions.
   - Consequence (reopened in ship-review, steer `0b2a547d`, choice `3b`): Academy Watch
     (`managym/academy/watch.py`) originally refused non-`("agent", "agent")`
     records; Pass 7 adds native-seat import inside MTG-506.
   - Passed over: keeping the heuristic in Python behind a faster path (leaves
     56% of rounds rowless for that slot).
2. **Code players keep their state for a game in both runners. (for review;
   review Q2, 2a)** State is lost because `GamePlayer` has no per-game
   lifecycle and `agents.py` calls `factory(seed)` inside `choose`;
   `HeuristicPlayer` is already stateful. Give `GamePlayer` a per-game open
   (`open_game(seed)` returning the game's player) used by both runners, for
   every code player. `heuristic_source_paths()` adds
   `managym/src/agent/player_policy.rs` (the native seat's code) and the
   anchor version moves to `native-v3`. Manabot's registration feeds
   `EvaluationKey.anchor_cohort_sha256` and changes the derived game seeds.
   Etude treats source as provenance, so the standalone probe uses an explicit
   `Code.config.version` to change its player key. The stored trophy tournament
   declarations keep their identities. Review the new tournament names and
   which baselines to rerun before publishing shared evaluations. See
   `scratch/questions.md` for the seating compatibility check.
3. **Compact vector annotations carry only runner-owned facts. (for review;
   review Q1, 1a)** A surfaced vector decision records actor, revision,
   `player_id`, `player_seed`, `player_decision_ordinal`, `policy_rng_seed`
   and `latency_seconds`; no `frame_sha256`, `chosen_offer` or
   `action_space_kind` (`GameInputs.digests[i]` binds the action space,
   `actions[i]` the choice). No live `describe_actions` or observation clone
   for checkpoint or native seats.
   - Readers: `audit_view` takes the offer id from `inputs["actions"][index]`
     when `chosen_offer` is absent (semantic Command strings keep
     `chosen_offer["id"]`); `_replay_record` checks `frame_sha256`,
     `chosen_offer` and `action_space_kind` when present and always actor and
     `player_id`. Absence is never a mismatch.
   - `RecordedVectorEnv._step` skips `semantic_decision_frames_json(active)`
     when isolated, with no journal and no annotation probabilities; on an
     isolated game error it takes `game.frame` from `info["game_error_frame"]`
     so failure replay still binds.
   - The vector runner's `private_exposures` counter goes. Scalar records keep
     live frames and their hash.
   - Passed over: 1b (keep offer and kind: still clones 64 observations and
     runs `describe_actions` per round, read by no consumer), 1c (live frames).
4. **Native failures isolate per leg and name the acting seat. (for review;
   review Q4, 4a)** Isolated step receipts carry `game_error_seat`, captured
   from the chooser in `step_once` or `settle_players`; `vector.py` attributes
   the crash to `leg.game["seat_players"][seat]`, not
   `round_result.players[slot]`. Opening settlement (a seat-0 heuristic's
   first turns inside `reset_each_into_buffers` / `load_slots_into_buffers`)
   is isolated per slot into a poisoned slot and receipt, so one bad leg does
   not abort the batch.
5. **Replay checks each transition once, off the stepping loop.** Fast path
   in `RecordWalker.run` (no `apply`, integer action, no next-boundary
   intervention): step `self.env` directly and digest once (`digests[0]` once
   after reset). On divergence with card capture on, rebuild `self.env`
   through the verified prefix before stopping, so `cards()` keeps excluding
   the diverged step (`tests/env/test_record_reducers.py:105-113`).
   `PyEnv::reset`, `step` and `state_digest` release the GIL
   (`py.allow_threads`, `managym/src/python/bindings.rs`) so replay threads
   run beside the loop. `finalize_match` runs on two worker threads; the loop
   keeps stepping, yields each `(match, Result)` as it finishes, and stops
   seating legs while `slots` settled matches are held. Errors still yield
   `(match, exc)` for that match alone; closing waits for running replays.
   `experiments/eval_throughput.py:237` stops subtracting `replay_seconds`
   from wall time: main-loop stages and background replay time are reported
   separately. Fallback if the loop starves: replay child processes.
6. **Process-constant identity is computed once where it is produced.**
   `manabot/infra/provenance.py` caches its engine/source/content fingerprints
   per process (like `managym/recording.py:runtime_identities`), so scalar
   play, admission and registration gain too. Replay projects the audit once
   per game. Card evidence reads the setup and outcome directly. Scalar
   capture freezes runner context at setup and reads the current context again
   for cutoff diagnostics and the closing journal.

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
  `manabot/sim/agent.py` (`GamePlayer`), `native_game.py`, `replay.py`,
  `evaluation_records.py`, `heuristic.py` (source paths only),
  `manabot/infra/provenance.py`; `managym/src/python/bindings.rs` (GIL).
- Readers of decision evidence that must accept both shapes:
  `audit_view`, `_replay_record`, `manabot/eval/match.py` game rows,
  `manabot/reports/evaluation.py`, card evidence, `experiments/eval_throughput.py`.
- Docs (written ahead of build; `experiments/reports/eval_throughput.md` updates
  in Pass 3 after measurements run):
  - `docs/engine.md` (`#per-seat-players`, `#built-in-heuristic`)
  - `docs/engine-python.md` (`#record-games`, `#read-and-re-execute-a-record`, `#batched-slot-loading-and-snapshots`)
  - `docs/evaluation.md` (`#anchors-and-ratings`, `#etude-agents`, `#ownership`)
  - `docs/academy.md` (`#open-a-replay-directly`, `#watch-a-game-from-an-evaluation`)

## Absent and error states

- A record without `frame_sha256`, `chosen_offer` or `action_space_kind` is
  compact; fields present are checked as today. A reader never treats absence as a mismatch.
- A native heuristic failure (`AgentError`, panic), in a step or in opening
  settlement, arrives in an isolated receipt with `game_error_seat` and fails
  that leg as a crash attributed to the seat that was acting.
- A leg whose checkpoint never acts before the game ends settles from the
  opening state (`settle_openings`), now also after any step.
- A replay worker's exception yields `(match, exc)`; closing the generator
  waits for running replays and drops none silently.

## Exclusions

- Scalar records keep live frames, their hash and the root-mutation digest
  check.
- Profiling the L4 `w160l8` bf16 slowdown remains separate work.
- No change to observations, action spaces or policy input.

## What goes

- `choose_code_action`'s per-decision `factory(seed)` and the scalar
  `Heuristic.choose` rebuild.
- The per-call `runtime_fingerprints` rehash and the per-snapshot metadata
  deep copy.
- Live `build_viewer_frame`, `describe_actions`, `decision_observations` (for
  non-custom seats) and `offer_evidence` in the vector runner;
  `ArenaDecision.frame` there; the duplicate `semantic_decision_frames_json`
  per isolated step.
- `agents.py:Heuristic` (replaced by `NativeHeuristic` and `CodeSeat`).
- The synchronous `finalize_match` drain at the top of `play_pairings`' loop.
- The single vector-wide policy pair as the only way to seat a native policy.
- Must survive: stored records and their replay, scalar trace bytes other
  than the heuristic's corrected choices, per-match error isolation, paired
  seeds, card evidence, the `ForwardError` contract.

## Passes

0. **Sync with main** (`6a48ef1c68`) brought in MTG-502 while keeping the MTG-497 stack.
1. **Native heuristic seats** (`bc03df8508`) added capped policies, per-game code players and isolated opening failures. All 64 MPS games replayed and the first four deals matched scalar on 1,585 checkpoint actions and all endings.
2. **Compact evidence and background replay** (`4071284d29`) removed live viewer projection and moved verification to two bounded workers. All 64 MPS games matched Pass 1 on 5,964 actions, pre-decision digests and endings. The two passes used different build profiles.
3. **Process provenance and capture metadata** (`ad8b9dbaab`, `c2db934c5`) cached build and catalog identity, froze scalar capture context during play and removed redundant replay projection.

The [32-deal panel](../experiments/reports/eval_throughput.md#native-heuristic-and-background-replay-mtg-506)
used the same source across ten arms and one release build per host. All
1,280 games replayed. The trained MPS arm reached 11,959 games/hour and
36.3 mean rows, with forwards taking 85% of wall. Every float32 arm matched
scalar on each leg. The trained L4 bf16 arm disagreed on 19 checkpoint
action traces and matched every score. The matched 16-deal profile reduced
fingerprinting from 1.782 to 0.027 seconds across 80 calls and preserved the
5,964 recorded checkpoint actions. The profiled total wall time rose from
24.7 to 28.6 seconds. The broad Rust suite is deferred to CI.

4. **Post-build review fixes** (`5ddf49e6c`, after MTG-500 sync `793880c852`) enforced the heuristic version, retained isolated openings, isolated registered code failures, exposed per-slot policies, and kept the idle vector while replay was full.

5. **Per-game failure records and player crashes** (`708ce8e777`, review Q1 1a) unified isolated reset and step evidence, routed replay from the whole record and kept CodeSeat crashes in evaluation metadata.

6. **Replay agreement for isolated failures** (realign after the Pass 5
   review, `d49a901555`): `_replay_game_inputs` reports `original_outcome:
   failed` for any stream whose ending is `failed` (isolated reset or step);
   strict batch exceptions stay `execution_outcome_unknown`. A boundaryless
   snapshot with a per-slot failure replays healthy siblings with
   `at_retained_failure: False` and detached slots as `setup_unavailable`.
   Both new assertions fail at `e9e5440e8a`; 165 replay, journal and vector
   eval tests pass.

Pass 5 passed 25 focused tests and the public API check. Slot 1's opening
failure used an injected native receipt. Its step failure used a real native
rejection that the bounded replay CLI matched after slot 0 took another step.
The shared editable Etude checkout acquired merge conflicts during verification.
The 13 eval and training checks passed with a clean archive of `d2dd820a`.
Ship review is next.

Pass 4 passed 32 debug Rust vector tests, 16 focused Python tests,
42 package-boundary tests and the public API check. The four historical
tournament-declaration tests passed with Etude source from `1f7496e9^`.
The editable SDK at `d2dd820a` instead fails collection because Etude
`1f7496e9` renamed `etude.eval.game.Matches` to `MatchStream`.
MTG-497 owns SDK 0.5.0 admission and the trophy callback import. After
that lands, run `lf sync` and repeat
`uv run --no-sync python -m pytest tests/eval/test_experiment_files.py -k 'trophy or frontier_eval_jobs or opponent_memory_evaluation'`
with the released SDK. The broad Rust suite remains with CI.

## Ship-review decisions (2026-11-27, steer `0b2a547d`)

Recorded verbatim in `scratch/ship-review-notes.md`:

> (1) `scratch/ship-review.html` § 1 · Maintained experiment migration to native-v3: 1a (2) `scratch/ship-review.html` § 2 · Compact vector decision records: 2a (3) `scratch/ship-review.html` § 3 · Academy Watch native-seat import (F1): 3b (4) `scratch/ship-review.html` § 4 · Pre-existing ScalarCapture seat-policy gap (F3): 4a

Outcome: return to `build` for Pass 7.

7. **Academy Watch native-seat import (`3b`).**
   - Support importing native-seat `GameRecord`s (e.g. `("agent", "heuristic:64")`,
     `("heuristic:64", "agent")`, `heuristic`, `random`) in `managym/academy/watch.py`
     and configured launch (`managym/academy/launch.py`), replacing the
     `NativeSeatRecord` refusal.
   - Preserve Academy's two-seat acting-view contract and shared decision
     surface: verify against the stored record's surfaced actions, digests, and
     ending while surfacing/reconstructing the native seat's turns cleanly (with
     exact determinism and no hidden-information leaks).
   - Update `docs/academy.md` (`#open-a-replay-directly`,
     `#watch-a-game-from-an-evaluation`) and tests
     (`tests/academy/test_configured_launch.py` and Watch import tests), and
     verify opening a real vector heuristic evaluation record in Academy Replay
     during QA.

## Deferred findings

- None filed from ship-review (`1a`, `2a`, `4a` accepted/dropped; `3b` moved into Pass 7 above; `etude-sdk 0.5.0` stays owned by stacked parent MTG-497).

## Acceptance check

```bash
uv run --no-sync python -m experiments.eval_throughput \
  --output-dir .runs/eval-throughput/w160-mps-64-native --device mps \
  --runner vector --slots 64 \
  --checkpoint .runs/eval-throughput/policy/00002000-raw.safetensors
```

prints a summary with 128 games, mean rows per forward of at least 32 and
`largest_stage: forward`. The per-deal replay receipts show 128 passes. The
scalar arm on the same deals gives the same winners, scores and checkpoint
forward rows.

## Build notes

- This checkout has a CPython 3.13 `.venv` with editable Etude. Use
  `uv run --no-sync`. The corpus is in `/tmp/mtg497-probe/inputs` and the
  trained checkpoint is `/tmp/mtg497-local-policy/00002000-raw.safetensors`.
- The cp313 extension was rebuilt in release for the Pass 3 measurements.
- Rust tests run in debug with `CARGO_BUILD_JOBS=2`.

## Done when

- Against the built-in heuristic, every live slot contributes a row to each
  round's forward, and the report shows the rows-per-forward and games/hour
  change on the trophy panel for both checkpoints on Apple Silicon and L4.
- Other slots keep stepping while a finished deal replays.
- Every game replays, and scalar and vector runs of the same deals agree on
  checkpoint actions, winners and final digests.
- A custom Python opponent still plays through the vector runner.
- Native-seat evaluation records (`("agent", "heuristic:64")` / `("heuristic:64", "agent")`) import and open cleanly in Academy Watch / Replay while preserving Academy's two-seat acting-view contract.
- `tests/test_package_boundaries.py`, `scripts/public_api.py check` and the
  debug Rust suite pass.
