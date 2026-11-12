# MTG-327 plan

## Done

`manabot/model/token_stream.py`, its tests, `_attend_pooled`, token types and
both pooled guards (regime bucketed shapes, `ActorCritic` live tokens) removed.
`ModelConfig.token_stream` is no longer a field: a before-validator
(`reject_pooled_token_stream`, like `reject_compound_decisions`) admits only
`"tables"` and refuses anything else with `POOLED_TOKEN_STREAM_REMOVED`
(test: `test_saved_pooled_token_stream_names_its_removal`). Docs: the only
pooled mention in `docs/observation-and-model.md` points at the report.

Departure from the brief: the brief said keep `experiments/token_stream.py`,
but it imports the deleted module, so Etude discovery broke; it was removed
with its `test_experiment_files.py` entry. `experiments/token_stream.md` and
its two manifest entries stay; the report names `439379869` (last main
revision with the pooled stream) as where to run it.

Check: tests/model/{value_aggregation,live_tokens,agent},
tests/training/test_bucketed_execution, tests/eval/test_experiment_files — 76 passed.

## Remaining

- PR #495 (MLX backend) still open on 2026-11-12. After it merges, rebase and
  drop `"pooled"` from `manabot/policy/checkpoint.py`'s refused list. Not a
  blocker for publishing this PR.
- Pre-land: delete `scratch/` in its own commit; PR notes the 0.x plain
  removal (no alias).

## Compress (2026-11-12)

No code cuts left: no `token_stream`/pooled-stream references remain outside
the refusal validator, its message and test; `encoded_relations` still feeds
`shared_context`/`frame_context`; ruff clean on touched files.

## Concept review (2026-11-12)

No change. One attention layout means no `token_stream` concept: removing the
field and refusing a saved non-`tables` value by name (as compound decisions
are) meets the brief's load/refuse requirement more plainly than a one-value
literal. No open product choice.
