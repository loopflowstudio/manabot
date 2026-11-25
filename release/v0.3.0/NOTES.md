# v0.3.0

Policy input v11 is required for training, checkpoint evaluation and replay.
Use v0.2.6 for v9 and v10 checkpoints, including frontier Small u1952.
For v1 through v8, use revision `ac601834d`. Existing v11 checkpoints and
packed observation archives retain their input layout.

The Torch vector environment wrapper has been removed. Use
`managym.recording.RecordedVectorEnv` for vector games with recording, or
`managym.VectorEnv` for native stepping. Both expose each terminal result
until the caller refills its slot. [Batched games](../../docs/engine.md#batched-games)

Observations reach a policy only as native `managym.ObservationBatch` rows.
The dense dictionary transport deprecated in v0.2.5 has been removed:
`encode_observation_into`, `encode_observations_into`,
`Env.encode_observation(_into)`, `VectorEnv.set_buffers` and
`write_observations_for_players`, `RolloutPool.set_buffers`, `encode_active`
and `step_active`, and `ObservationLayout.allocate`, `axis_extents` and
`object_references`. Encode observations with
`managym.encode_observation_batch`; register vectors and rollout pools with
`set_observation_layout`, read their rows with `observation_rows()`,
`observations_for_players(players)`, `encode_active_batch()` and
`step_active_batch(actions)`. Policies, admission and preparation accept only
these batches, and offline captures store them with
`ObservationBatch.to_bytes()`. Search receipts recorded with v9 or v10 pools
replay at v0.2.6.

Deprecated `managym` and Academy compatibility surfaces have been removed:
`StepKind` / `StepEnum` now follow rules order (`FirstStrikeCombatDamage = 7`,
`CombatDamage = 8`, …, `Cleanup = 12`); viewer `Observation` JSON uses named
`phase`/`step` strings; flat `Card` and `Permanent` characteristic getters,
`Permanent.card`, `StackObject.card`, `Card.cast_permission`, `PlayerConfig.disclosure`,
`Env.enable_behavior_tracking`, `VectorEnv.opponent_policy`, `WORLD_VERSION`,
`learning_catalog`, `learning_record_names`, `learning_token_value`,
`LearningCatalog`, `managym.recording.CapturedVector`, `RecordedGame.game_trace_sha256`,
and the top-level `academy` package shim are removed (use `managym.academy` and
`card.characteristics` / `permanent.characteristics` / `spell`).

Deprecated `manabot` shims, aliases, and legacy training/sampler loaders have
been removed: `manabot.sim.search.flat`, `manabot.env.replay`,
`manabot.sim.table_bots.CheckpointVillain`, `manabot.training.export`, and
legacy `.pt` tracked-training saves (`learner.pt`/`optimizer.pt`/`ema.pt`) and
`.pt` belief samplers (use portable `.safetensors` saves and samplers; published
policy `.pt` checkpoint loading remains supported).
