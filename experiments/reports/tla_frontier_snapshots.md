# Exploratory TLA frontier w64 learnability trial: PLR + Vine-RLOO snapshots vs baseline

## Exploratory

**No training or evaluation runs executed; USD 0.00 was spent.** A zero-cost local pre-rental check of the planned four-worker snapshot arm failed before the first game because native multi-worker snapshot-vector wiring is not yet connected from the rules engine into the self-play collector. Catching that integration gap locally avoided spending L4 rental budget on an unrunnable setup or falling back to slow single-worker prefix replay. Next, hold all GPU and CPU rentals until parent Task `6b4a1a` completes collector integration, then run a two-update timing smoke on the four-worker native path before launching the full trial.

This is an exploratory report. Its observations are operational only and are not cited as findings.

## Question

On the 134-deck TLA frontier `w64` setup, how does adding prioritized level replay (PLR) and 4-branch Vine-RLOO snapshot self-play compare with ordinary fresh self-play at matched updates? Specifically, the trial was designed to observe:

1. **Sample efficiency at matched updates:** whether snapshot self-play climbs faster against the first-64 heuristic and beats fresh self-play head-to-head across updates 25–400.
2. **Early-training bootstrap behavior:** whether 20-decision horizon rollouts bootstrapping from an early uncalibrated critic lag baseline initially or help throughout.
3. **Snapshot bank dynamics:** whether root critic-residual prioritization keeps healthy turnover and decay on tactical forks or gets stuck on high-variance topdeck outliers.
4. **Stability and accounting:** whether per-source advantage retention rates, KL and value losses, snapshot counters, and fresh-only `decisions_per_game` isolation stay well-behaved across hundreds of updates.

Because this is an exploratory trial, it makes no committed prediction.

## Method

Both arms are declared in [`experiments/tla_frontier_snapshots.py`](../tla_frontier_snapshots.py) and share one training seed (`21501`), the 134-deck TLA trophy corpus (`tla_frontier`), policy input v11 (`max_actions=64`, first-legal action overflow), the `w64` object-attention architecture (width 64, 2 attention layers, 4 heads, scalar masked-mean critic), damped policy learning (`tla_frontier.learning("w64")`), both learning seats, complete-game batch closure, 256 streams, 64 transitions per update, 4 CPU producer workers, and float32 CUDA execution with bucketed shapes and live-token attention. Both arms evaluate raw checkpoints while writing EMA siblings beside them, because per-update `ema=0.999` barely leaves initialization within 400 updates ([`tla_trophies.md`](tla_trophies.md), [`mana_taps.md`](mana_taps.md)).

| Arm | Episode mix and horizon | Priority and capture | Root actor weight |
| --- | --- | --- | ---: |
| Baseline (fresh self-play) | 100% fresh complete games | None | 1.0 |
| PLR + Vine-RLOO snapshots (4 branches, horizon 20) | 80% snapshot replay, 20% fresh floor (~50/50 row mix); 4 sibling branches per root, capped at 20 decisions | 512-slot bank; root critic-residual priority (half-life 8); fresh-only capture (`min_legal_actions=3`, probability 0.05, up to 64/batch); equal fresh and snapshot weights | 2.0 |

Evaluation scores policy-only play (no search) on `tla_frontier`'s 16 paired trophy deals crossed across all 4 seat/deck legs (64 replay-verified games per comparison) at updates 25, 50, 100, 150, 200, 300, and 400, plus update 0 against the first-64 heuristic. Paired crossed deals separate learned skill from deck and seat strength. The planned budget sits under the USD 25 exploratory cap: 400 updates per L4 arm (11.0 hours, USD 9.50 cap each including startup allowance) plus CPU32 evaluation (1.5 hours, USD 2.00 combined), totaling USD 21.00 before retries or a two-update timing smoke.

Library-shuffle common random numbers across siblings (not yet exposed on `SnapshotConfig`) and a fresh-only warmup were deferred from the initial two-arm run.

## Results

**All four learnability questions remain unanswered because zero training updates and zero evaluation games ran.** Before renting any machine, a local four-stream, four-worker CPU check opened the snapshot arm's collector after syncing parent Task `6b4a1a` and failed before the first game with `ValueError: snapshot bank requires a snapshot vector`. No machine was rented, no checkpoint was written, and no evaluation match was played.

| Stage | Planned work | Completed work | Paid cost (USD) | Status |
| --- | --- | --- | ---: | --- |
| Baseline (fresh self-play), L4 training | 400 updates (1 seed) | 0 updates (0/400) | 0.00 | Not launched (held with snapshot arm) |
| PLR + Vine-RLOO snapshots (4 branches, horizon 20), L4 training | 400 updates (1 seed) | 0 updates (0/400) | 0.00 | Blocked before rental in pre-flight collector check |
| Heuristic milestone evaluation (16 paired deals, 4 legs), CPU32 | 16 checkpoints × 64 games | 0 games | 0.00 | Pending checkpoints |
| Matched-update head-to-head evaluation (16 paired deals, 4 legs), CPU32 | 7 milestones × 64 games | 0 games | 0.00 | Pending checkpoints |

Inspecting the synced codebase explains why the pre-rental check failed:

1. **Native engine layer is present, collector adapter is not.** Parent Task `6b4a1a` (still active in implementation) added native deferred auto-reset (`VectorEnv(..., auto_reset=False)`), batched slot loading (`SlotLoad` and `load_slots_into_buffers`), and `ShardedVectorEnv` capture reads (`snapshot_slots`, `slot_metadata`). However, `ShardedVectorEnv` does not yet implement the `SnapshotVector` protocol (`capture_slots`, `install_slots`, `release_handles`) or forward `auto_reset=False` and `load_slots_into_buffers`.
2. **Collector rejects four-worker snapshot execution.** `open_collector` requires an explicit `snapshot_vector` whenever a `snapshot_bank` is present (`ValueError: snapshot bank requires a snapshot vector`) and rejects any `snapshot_vector` when `workers != 1` (`ValueError: snapshot collection requires workers=1`).
3. **Self-play loop constructs no default snapshot vector.** `self_play_collector` passes `snapshot_vector=None` through unless a caller supplies one explicitly. The only existing `SnapshotVector` implementation (`InMemorySnapshotVector`) is a serial, single-worker prefix-replaying reference vector rather than a four-worker native `GameSnapshot` vector.

## What this supports

- **Operational readiness:** The synced snapshot stack cannot yet run the planned four-worker native `GameSnapshot` training loop without the remaining collector integration from `6b4a1a`. Running a zero-cost local collector check before launching Etude rentals prevented wasted GPU spend.
- **Learnability claims:** None. Because 0 updates and 0 games ran, this report provides zero evidence for or against PLR + Vine-RLOO sample efficiency, early horizon-20 bootstrapping, critic-residual bank dynamics, or multi-hundred-update training stability.

## Recommendation

1. **Keep rentals paused** until parent Task `6b4a1a` (or a follow-up change) delivers three pieces: a native `GameSnapshot`-backed `SnapshotVector` adapter over `VectorEnv` and `ShardedVectorEnv` (with `auto_reset=False`, `load_slots_into_buffers`, and `MemoryStore` lineage safety), multi-worker snapshot admission (`workers=4`) in `open_collector`, and default native snapshot-vector construction in `self_play_collector` when `settings.snapshot` is configured.
2. **Do not substitute `InMemorySnapshotVector` (`workers=1`)** for the GPU trial. Serial CPU prefix replay does not test the intended native restore path and would burn L4 budget waiting on single-threaded replay.
3. **Run a two-update timing smoke first** once synced onto the completed integration, measuring wall seconds per update and decisions per second on four workers so the 400-update two-arm budget and CPU32 evaluations are confirmed under USD 25 before full launch.

## Caveats

- **Missing runs stay missing:** All 400 planned updates per arm and all planned 64-game evaluations are unexecuted; nothing is estimated or filled with zeros.
- **Exploratory single-seed design:** Even after the blocker is cleared, one shared training seed (`21501`) can reveal failure modes, bank behavior, and rough curve shape, but cannot establish a replicated method claim (training seeds are the unit of evidence).
- **Hidden-draw variance across siblings:** Until library-shuffle common random numbers (`ShuffleLibraries`) are exposed on `SnapshotConfig`, sibling rollouts can diverge on hidden library draws rather than solely on policy choices.
- **Raw checkpoint noise:** Evaluating raw weights avoids EMA lag under `ema=0.999` over 400 updates, at the cost of higher checkpoint-to-checkpoint variance than a faster-decaying EMA.

## Appendix: runs, changes, and admission state

### Run ledger

Etude (`etude runs`) lists zero runs for `tla_frontier_snapshots`.

| Job | Commit | Etude run ID | Updates / matches | Launcher cost (USD) | State |
| --- | --- | --- | --- | ---: | --- |
| `baseline` | `c21ade43ac` / `e91265b012` | None | 0 / 400 updates | 0.00 | Not launched |
| `plr-vine-rloo` | `c21ade43ac` / `e91265b012` | None | 0 / 400 updates | 0.00 | Blocked in pre-rental CPU check |
| `tla-frontier-snapshots-heuristic-16` | `c21ade43ac` / `e91265b012` | None | 0 matches | 0.00 | Not launched |
| `tla-frontier-snapshots-vs-baseline-16` | `c21ade43ac` / `e91265b012` | None | 0 matches | 0.00 | Not launched |

### Dated change log

- 2026-11-23 (`c61219fba3577301ddd1a7377bdca4af70ba280f`) — Added [`experiments/tla_frontier_snapshots.py`](../tla_frontier_snapshots.py), initial report notes, and discovery registration in `tests/eval/test_experiment_files.py`.
- 2026-11-23 (`c21ade43acc106ece8b3110a75acbdf1eddc84fe`) — Merged parent Task `6b4a1a` through `a4af50f3f8e12d223c0b17ac998f4914fa5c83e3`.
- 2026-11-23 (`e91265b012acd6309b5a83288848ba014d34fd42`) — The two-update timing smoke is blocked before rental. At `c21ade43acc106ece8b3110a75acbdf1eddc84fe`, after syncing parent Task `6b4a1a` through `a4af50f3f8`, a four-stream CPU check opened the snapshot arm's collector with the planned four workers and failed before the first game with `ValueError: snapshot bank requires a snapshot vector`. The parent supplies native slot restoration, but `self_play_collector` still passes no vector by default, and `open_collector` accepts an explicit snapshot vector only with one worker. Etude listed no runs for `tla_frontier_snapshots`. Neither arm rented a machine or produced a checkpoint. The four-worker timing smoke and all milestone evaluations remain pending the parent's collector integration.
- 2026-11-23 — Completed exploratory report from the blocked pre-rental state; no new run started.

### Admission checks and verified references

- **Experiment declaration tests:** `uv run pytest tests/eval/test_experiment_files.py` passed (30/30).
- **Parent task state at report time:** Task `6b4a1a` (`Wire GameSnapshot into VectorEnv/ShardedVectorEnv with deferred auto-reset and MemoryEncoding/MemoryStore lineage safety`) was still active at `implement · loop 1 pass 2`.
- **Verified blocking code paths:**
  - `managym/__init__.pyi` (`SlotLoad`, `SlotLoadResult`, `VectorEnv.load_slots_into_buffers`, `VectorEnv.snapshot_slots`)
  - `manabot/training/rollout/producers.py` (`ShardedVectorEnv` exposes `snapshot_slots` and `slot_metadata`, but not `SnapshotVector` methods or `load_slots_into_buffers`)
  - `manabot/training/rollout/collection.py:756-759` (`open_collector` raises `ValueError: snapshot bank requires a snapshot vector` and `ValueError: snapshot collection requires workers=1`)
  - `manabot/training/loops/self_play.py:333-407` (`self_play_collector` forwards `snapshot_vector=None` by default)
- **Artifacts and manifest:** No run files were produced or uploaded; `experiments/data.manifest.json` is unchanged.
