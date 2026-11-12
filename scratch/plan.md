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
- Figures: `uv run --extra reports python -m experiments.live_token_attention
  experiments/data/live_token_attention` writes `learner_step.svg` (Results) and
  `equivalence.svg` (Numerics) from the receipts; both rendered and checked against
  the tables, published (77 manifest entries).

## Checks

- `tests/model`: 310 passed, 4 skipped. `tests/model/test_live_tokens.py`,
  `tests/training/test_bucketed_execution.py`, `tests/model/test_bucketed_shapes.py`:
  22 passed (after the `live_token_count` rename). `test_experiment_data_untracked.py`: 1 passed.
- Training suite run locally hit the 1 h limit at ~415/772 under load from another
  worktree. Failures outside this diff's files: `test_game_errors` resource_limit replay
  x3 (`binding_mismatch`), `test_inspect_batch_script` (`RolloutBatch` has no `batch`),
  `test_head_to_head` x2 (needs `--extra evaluation`); `test_active_recovery` and
  `test_learning_state` pass alone. The full suite is left to CI.

## Report coverage (Outcome 3)

The report's Method states the primitives (jagged-NJT SDPA with host
`min_seqlen`/`max_seqlen`; `flex_attention` not measured) and the one host read per
learning forward. "What a default for larger models would take" and "What MTG-306's
pooled stream becomes" (strip, in its own Task) are recommendations only; neither is
built here. The count is `live_token_count` (field and method, no alias).

## Remaining

- Publishing and CI belong to the caller.
