# v0.2.6

<!-- loopflow:release-notes=narrative;gate=safe -->

This patch release restores drop-in compatibility for external worker-process collectors (such as `jace`) by making native `ObservationBatch` rollouts picklable without densifying and restoring pre-0.2.4 `ObservationEncoder` shortcut attributes, publishes an explicit CI-enforced `manabot` library API contract, unifies battlefield forms and persistent ability sources in `managym`, and adds opponent-move read-outs to Academy.

## Worker-process rollout transport and published library API

Downstream training loops that run `manabot`'s collector across spawned worker processes can now pass `RolloutBatch` directly over `multiprocessing.Queue` and merge worker batches without touching internal fields, backed by a checked-in public API contract.

- **Picklable `ObservationBatch` and `RolloutBatch.concatenate`:** `managym.ObservationBatch` now implements pickle reconstruction (`managym._restore_observation_batch`) with full layout, span-offset and byte-length validation, keeping CSR rollouts compact across `spawn` queues (~8.3 MB for 960 rows). `ObservationSpace`, `ObservationSelection`, `PackedActions` and decision-memory prefix chains also pickle cleanly. `RolloutBatch.concatenate` and `RecordedObservations.concatenate` merge packed worker batches while shifting recorded indexes, trajectory spans, bootstrap rows and `branch_group_id`s. ([#604](https://github.com/loopflowstudio/manabot/pull/604))
- **Restored `0.2.x` `ObservationEncoder` shortcuts:** `ObservationSpace` restores the pre-#566 `ObservationEncoder` capacity and dimension shortcut attributes (`max_actions`, `cards_per_player`, `perms_per_player`, `*_dim`, etc.) with `DeprecationWarning` targeting `0.3.0`, fixing callers reading `obs_space.encoder.max_actions`. ([#604](https://github.com/loopflowstudio/manabot/pull/604))
- **Explicit `manabot` library API contract:** Declared `__all__` across `manabot`'s 15 external library modules (configs, `ObservationSpace`, `Match`/`Reward`, `UniformDecks`, Torch policy/checkpoints/rollout/collection, `prepare`, `game_errors`), added a checked-in structural contract snapshot (`tests/public_api.json` / `tests/test_public_api_contract.py`) that fails CI on un-shimmed patch breaks, added an in-repo external RL consumer test (`tests/infra/test_external_consumer.py`), and documented the patch-clean `main` release rule. ([#608](https://github.com/loopflowstudio/manabot/pull/608))

## Unified battlefield forms, copiable values and ability sources

`managym` now preserves a card's physical identity across transforms, melds, face-down entries and copies, and resolves abilities against live characteristics or the exact departed source.

- **Transforms, melds, Manifest Dread and modal faces:** Transforming permanents turn over in place, melded permanents preserve both card identities and owner-ordered zone moves, Manifest Dread gives hidden face-down creatures a stable public reference through turn-up and replay, and modal permanent faces transform in place while keeping their chosen face. ([#593](https://github.com/loopflowstudio/manabot/pull/593))
- **Persistent ability sources and new Foundations coverage:** Copied abilities retain their programs and source faces, attached-permanent effects read the attachment at resolution, library moves preserve viewer-known positions, and new card definitions cover linked exile, graveyard choices and per-object activation limits (22 of 23 Foundations object-lifetime ledger rows). Policy input v11 appends face-down state and graveyard-cast status while staying backward-compatible with saved v11 checkpoints. ([#593](https://github.com/loopflowstudio/manabot/pull/593))

## Academy opponent read-outs and board polish

- **Opponent-move read-out before acting:** Academy now shows what your opponent did before accepting your next decision; the first click or action key dismisses the read-out and brings the board up to date without committing a move. Long hands shrink before overlapping, payment previews hold steady while previewing alternatives, and the stack rail keeps the top two entries in view. ([#563](https://github.com/loopflowstudio/manabot/pull/563))
- **Recent Beats and keyboard focus fixes:** Fixed duplicate spell targets in Recent Beats (reading spell names from move narration and targets from presentation events), counted each cast in compound moves accurately, and tracked keyboard focus by control position so identical cards in hand navigate cleanly. ([#564](https://github.com/loopflowstudio/manabot/pull/564), [#565](https://github.com/loopflowstudio/manabot/pull/565))

## Operational and compatibility notes

- **No breaking changes from `0.2.5`:** All `0.2.x` deprecation shims remain intact ahead of the upcoming `0.3.0` cleanup release, and `tests/public_api.json` enforces patch-line compatibility on `"line": "0.2"`.
- **Dependencies unchanged:** `manabot==0.2.6` requires `managym==0.2.6`, `torch>=2.15.0` and `etude-sdk>=0.4.7,<0.5`.
