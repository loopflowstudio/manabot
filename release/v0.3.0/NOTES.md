# v0.3.0

Policy input v11 is required for training, checkpoint evaluation and replay.
Use v0.2.5 for v9 and v10 checkpoints, including frontier Small u1952.
For v1 through v8, use revision `ac601834d`. Existing v11 checkpoints and
packed observation archives retain their input layout.

The Torch vector environment wrapper has been removed. Use
`managym.recording.RecordedVectorEnv` for vector games with recording, or
`managym.VectorEnv` for native stepping. Both expose each terminal result
until the caller refills its slot. [Batched games](../../docs/engine.md#batched-games)
