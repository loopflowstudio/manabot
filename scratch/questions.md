# MTG-312 assumptions (reversible, taken in write-docs)

- Field: `Execution.bucketed_shapes: bool = False`, excluded from serialization when False (same `exclude_if` pattern as `profile_collection`).
- CLI: `--bucketed-shapes` applies only to `--recipe local|baseline`, like `--device`; regime runs set the field in the regime JSON. Not an override on `--regime`/`--regime-name`, since that would change a named regime's identity silently.
- Scope: the stage's collection and learner policies only; evaluation/search/monitoring stay on the default path.
- `ActorCritic(..., bucketed_shapes: bool = False)` constructor argument replaces the env reader.
- `token_stream="pooled"` + `bucketed_shapes=True` fails regime validation (earlier than the current ActorCritic construction error).
- Receipt: the regime in `run.json` carries the field; no separate Etude param documented.
- `MANABOT_BUCKETED_SHAPES` is no longer read and no longer mentioned anywhere outside scratch; a leftover setting gets no warning.
