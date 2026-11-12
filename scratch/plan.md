# MTG-327 plan

## Status

Implemented: `manabot/model/token_stream.py`, its tests, `_attend_pooled`,
token types and both pooled guards removed. `ModelConfig.token_stream` is no
longer a field: a before-validator (`reject_pooled_token_stream`, like
`reject_compound_decisions`) admits only `"tables"` and refuses anything else
with `POOLED_TOKEN_STREAM_REMOVED`. `experiments/token_stream.py` (imported the
deleted module) removed; its report points at `7ac8c9b6f`.

Check: tests/model/{value_aggregation,live_tokens,agent},
tests/training/test_bucketed_execution, tests/eval/test_experiment_files — 76 passed.
