# MTG-319 plan

## Design (built)

- Switch: `Execution.live_tokens: bool`, omitted when False; train_self_play only;
  continuation must preserve it; rejected with `token_stream="pooled"` (regime and
  `ActorCritic`). `ActorCritic(..., live_tokens=)`, `load_checkpoint(..., live_tokens=)`.
- `manabot/model/live_tokens.py` `LiveTokens`: pack `validity != 0` (plus one value
  token per observation) row-major into [L,D]; offsets = cumsum of per-row counts on
  device. Every `GameObjectAttention` layer runs `forward_live` on [L,D]: QKV/out
  projections from the module's own `nn.MultiheadAttention` weights, SDPA over jagged
  NJTs (`nested_tensor_from_jagged(..., min_seqlen=1, max_seqlen=S)` so dispatch reads
  nothing), norms and FFN on packed rows. Scatter back to [B,S,D] with zeros.
- Embeddings, links, facts, shared effects, catalog, focus joins and critic stay on the
  (optionally bucket-trimmed) table. Composes with `bucketed_shapes`.
- Host reads: collection counts live tokens on the CPU batch in `PolicyInference.prepare`
  (`PreparedObservation.live_token_count`) and packs with `nonzero_static`; learning uses
  `nonzero` (one read per forward).
- `historical_mean` works unchanged (padding comes back zero).
- CPU: NJT SDPA falls back to per-observation math (correct, slow). Docs say so; no
  validator.

## Pilot (done)

- `experiments/live_token_attention.py` launches `l4_step_profile` candidate `live`
  (baseline, bucketed, live, bucketed-live). Rule: bucketed-live learner step at w384l8
  <= bucketed / 1.5, equal outputs.
- Etude run `308602b435fe`, source `53cf1357d`, on-demand `mtg319-l4` (l4-etude was
  held by another Task; no L4 stock until 16:45 UTC). USD 0.24, job 1,037 s.
  Bucketed 386 ms vs bucketed-live 202 ms (1.91x): PASS; numerics equal within CUDA
  tolerance. Report `experiments/live_token_attention.md` has Results, Decision
  ("Build out live tokens") and appendix; README index line; data published (75
  manifest entries, untracked in git).

## Checks

- `tests/model`: 310 passed, 4 skipped. `test_live_tokens.py` +
  `test_bucketed_execution.py`: 13 passed. `test_experiment_data_untracked.py`: 1 passed.
- Training suite run locally hit the 1 h limit at ~415/772 under load from another
  worktree. Failures outside this diff's files: `test_game_errors` resource_limit replay
  x3 (`binding_mismatch`), `test_inspect_batch_script` (`RolloutBatch` has no `batch`),
  `test_head_to_head` x2 (needs `--extra evaluation`); `test_active_recovery` and
  `test_learning_state` pass alone. The full suite is left to CI.

## Concept review (2026-11-12): next implement direction

The code and the pilot match the design. The report does not yet answer every
part of the Task's Outcome, and one name doesn't match what it counts.

1. **Report: what the default for larger models would take** (Outcome 3, not
   yet answered). Add a short section after Decision. Facts already
   established: the layout runs only where a stage's collection/learner
   policies use it; evaluation and search don't; on CPU it falls back to
   per-observation math; it composes with `bucketed_shapes`; at w160l8,
   collection on full tables beat bucketed-live (2.4 s vs 2.9 s), which this
   pilot did not separate. State what a default flip requires (CUDA-only or a
   CPU path that refuses, evaluation and search adoption, the w160 collection
   question) as a recommendation. Don't build it here.
2. **Report: what MTG-306's pooled stream becomes** (Outcome 3, not yet
   answered). The evidence points toward stripping it: its only benefit was
   attending over fewer tokens, which live tokens deliver per observation
   without its common-width padding; the two are mutually exclusive.
   Recommend strip-or-keep with that reason, and leave the removal to its own
   Task.
3. **Report: primitives and host reads** (Outcome 1). The design reasoning is
   in `live_tokens.py` and the docs, but not in the report. Add two or three
   sentences to Method: jagged-NJT `scaled_dot_product_attention` (fused
   varlen kernels on CUDA, `min_seqlen`/`max_seqlen` given so dispatch reads
   nothing) against `flex_attention` (block masks, compile required; say
   why it wasn't chosen only if that was actually established, otherwise just
   say it wasn't measured), and the one host read per learning forward.
4. **Name the count for what it counts.** `PreparedObservation.live_objects` and
   `ActorCritic.live_object_count` count live *tokens* (valid objects, belief
   rows plus their global row, value tokens), not objects. Rename them to
   `live_token_count` (field and method), and update the docstrings in
   `inference.py`/`live_tokens.py` and `tests/model/test_live_tokens.py`.
   These are internal names, not published ones, so no alias is needed.

No open product choice: whether to flip the default or strip pooled happens in
later Tasks; this Task only reports a recommendation for each.

## Remaining

- Items 1–4 done: report has Method primitives/host-read sentences, "What a default
  for larger models would take" and "What MTG-306's pooled stream becomes" (strip, own
  Task); count renamed to `live_token_count` (field and method, no alias).
  `pytest tests/model/test_live_tokens.py tests/training/test_bucketed_execution.py`:
  13 passed (the bucketed test lives under `tests/training/`).
- Publishing and CI belong to the caller.
