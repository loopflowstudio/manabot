# MTG-327 plan

## Docs (written ahead of implementation)

- `docs/observation-and-model.md`: the pooled stream's description, its
  bucketed-shapes guard sentence and its live-tokens guard sentence are gone;
  "Mix" now points to the token-stream report and states that a saved
  `token_stream="pooled"` configuration fails to load with an error naming the
  removal. Implementation must make that error true (ModelConfig validator).
- `manabot/infra/config.py` field comment still describes pooled; rewrite it
  with the code change.

## Status

Implemented: token_stream.py and its tests deleted; `_attend_pooled`,
token types, both guards removed; `ModelConfig` refuses `pooled` by name.
Check: tests/model/{live_tokens,value_aggregation,agent,bucketed_shapes,inference,policy_film}, tests/training/test_bucketed_execution, experiment discovery — 102 passed.
