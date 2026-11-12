# MTG-289 — cancelled draft removed

The filed-in-error steer is quoted in `scratch/questions.md`. Existing
`Env.hand_snapshot` and `VectorEnv.hand_snapshots` remain the supported path
(`managym/src/agent/env.rs`, `managym/src/agent/vector_env.rs`). Library counts,
both seats in one call and definition-indexed buffers await a consumer request.

Delete — do not maintain: removed the unimplemented
`hidden_zone_counts_into_buffers` declarations and their exclusive import from
`manabot/managym/__init__.pyi`, plus the guide section, example, navigation link
and API table row from `docs/engine.md`. The draft implementation/tests were
already removed. No deletion targets remain and no runtime behavior changed.

Validation: `git diff --exit-code HEAD -- docs/engine.md manabot/managym/__init__.pyi`
— passed; both files match the existing implementation's base documentation.
No build or runtime test needed for the reverted draft.
