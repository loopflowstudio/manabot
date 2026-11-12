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
its two manifest entries stay; the report names `7ac8c9b6f` (its pilot
revision) as where to run it.

Check: tests/model/{value_aggregation,live_tokens,agent},
tests/training/test_bucketed_execution, tests/eval/test_experiment_files — 76 passed.

## Remaining

- PR #495 (MLX backend) still open on 2026-11-12. After it merges, rebase and
  drop `"pooled"` from `manabot/policy/checkpoint.py`'s refused list. Not a
  blocker for publishing this PR.
- Pre-land: delete `scratch/` in its own commit; PR notes the 0.x plain
  removal (no alias).

## Concept review (2026-11-12): applied

Report's "run them from" line names `439379869` (last main revision with the
pooled stream; the load error names it too); results table keeps `7ac8c9b6f`.
Docs Mix step is one sentence: live-token attention, linking the report.
Check: `grep -n pooled docs/observation-and-model.md` shows only that pointer
and unrelated mean-pooling uses.
