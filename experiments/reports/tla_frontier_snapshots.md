# Exploratory TLA frontier w64 learnability trial: PLR + Vine-RLOO snapshots vs baseline

Exploratory run comparing PLR + Vine-RLOO snapshot self-play (`plr-vine-rloo`) against ordinary fresh self-play (`baseline`) on the 134-deck TLA frontier `w64` setup to observe sample efficiency at matched updates, early-training bootstrap behavior, snapshot bank dynamics, and training/accounting stability. It makes no committed prediction.

## Design and assumptions

- **Module.** [`experiments/tla_frontier_snapshots.py`](../tla_frontier_snapshots.py).
- **Arms.** Single shared training seed (`21501`), identical `tla_frontier` `w64` model, learning schedule, 134-deck trophy corpus, `policy_input_version=11`, `streams=256`, `transitions=64`, `workers=4`, float32 with CUDA `bucketed_shapes=True` and `live_tokens=True`:
  - `baseline`: `snapshot=None`, `root_actor_weight=1.0`.
  - `plr-vine-rloo`: `SnapshotConfig(capacity=512, replay_fraction=0.8, fresh_floor=0.2, group_size=4, horizon=20, signal="critic_residual", signal_scope="root", score_half_life=8, fresh_weight=1.0, snapshot_weight=1.0, capture=CapturePolicy(min_legal_actions=3, probability=0.05, max_captures_per_batch=64, fresh_only=True))` and `root_actor_weight=2.0`.
- **Update budget and cap (<= USD25).** Default target is 400 updates (`max_hours=11.0`, `max_usd=9.50` per L4 arm including startup allowance), plus CPU32 evaluation (`max_hours=1.5`, `max_usd=2.00` combined), totaling USD21.00 before retries or a 2-update timing smoke.
- **Raw checkpoints evaluated.** Both arms set `checkpoint_weights="raw"` (still saving EMA siblings beside each export). Under the current loop's per-update `ema=0.999`, EMA checkpoints barely move off initialization within 400 updates (`experiments/reports/tla_trophies.md`, `experiments/reports/mana_taps.md`).
- **Evaluation grid.** `tla_frontier`'s 16 paired trophy deals × 4 crossed legs (64 replay-verified games per comparison) at milestones `25, 50, 100, 150, 200, 300, 400` (plus `0` against the first-64 heuristic), for both heuristic and matched-update head-to-head curves, plus behaviour metrics.
- **Native snapshot vector dependency.** `train()` relies on `6b4a1a`'s native deferred-reset `GameSnapshot` vector (`workers=4`) rather than `InMemorySnapshotVector`. Sync `6b4a1a` (or `main` once merged) and run a 2-update timing smoke before launching the full L4 rentals.
- **Deferred options.** `ShuffleLibraries`/CRN across siblings (not yet exposed on `SnapshotConfig` at design time) and a fresh-only warmup are deferred unless enabled before launch or tested as a follow-up after seeing Arm 1's early curve.

## Runs and results

2026-11-23 — The two-update timing smoke is blocked before rental. At `c21ade43acc106ece8b3110a75acbdf1eddc84fe`, after syncing parent Task `6b4a1a` through `a4af50f3f8`, a four-stream CPU check opened the snapshot arm's collector with the planned four workers and failed before the first game with `ValueError: snapshot bank requires a snapshot vector`. The parent supplies native slot restoration, but `self_play_collector` still passes no vector by default, and `open_collector` accepts an explicit snapshot vector only with one worker. Etude listed no runs for `tla_frontier_snapshots`. Neither arm rented a machine or produced a checkpoint. The four-worker timing smoke and all milestone evaluations remain pending the parent's collector integration.
