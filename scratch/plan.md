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
  (`PreparedObservation.live_objects`) and packs with `nonzero_static`; learning uses
  `nonzero` (one read per forward).
- `historical_mean` works unchanged (padding comes back zero, as padded layers leave it);
  the drafted doc rejection was dropped.
- CPU: NJT SDPA falls back to per-observation math (correct, slow). Docs say so; no
  validator.

## Pilot

- `experiments/l4_step_profile.py` arms now select layouts by constructor flags
  (`LAYOUTS`); its old `MANABOT_BUCKETED_SHAPES` env arm and `make_actor` signature were
  stale since MTG-312. Candidate `live`: baseline, bucketed, live, bucketed-live.
- `experiments/live_token_attention.py` launches it; report
  `experiments/live_token_attention.md` (not yet written).
- Decision rule: bucketed-live learner step at w384l8 <= bucketed / 1.5, equal outputs.

## Checks

- `pytest tests/model/test_live_tokens.py tests/training/test_bucketed_execution.py`: 13 passed.
- `pytest tests/model`: 310 passed, 4 skipped.
- `MTG301_CANDIDATE=live l4_step_profile job --smoke` (CPU): every stage exit 0; numerics
  ok for bucketed, live, bucketed-live (live grads within 8.4e-8, logits 4e-9). The smoke
  found and fixed harness drift: `observation_tensors` now takes no device (learner
  admits on CPU, then `transfer_admitted`), and hyphenated arm names in numerics pairing.
- Committed `2cf2389f3`. Launch: `etude launch experiments/live_token_attention.py
  --rent l4-etude --revision <sha>` through the 0.2.15 overlay (`~/.etude/launcher.db`
  is schema 13; the locked 0.2.16 is 14). 2026-11-12 15:49 UTC: l4-etude, mtg281-l4,
  b015d5-l4 all "none available".

- Polled `etude availability l4-etude` every 2 min, 15:49-16:28 UTC: since 15:51 its only
  slot is held by machine `l4-etude-9249bc075105` (not this Task's; rented until 18:05
  UTC). Not launched; other Tasks' on-demands (mtg281-l4, b015d5-l4) left alone.

- Compress (`fe7043665`): dropped the unused `Arm.environment` (no arm sets env since
  `LAYOUTS`); test reads `OBJECT_VALIDITY_KEYS`; doc serialization paragraph trimmed.
  `test_live_tokens.py`: 5 passed. Merged main `4c9ba1957` (no conflict with this diff).

- 16:33 UTC: l4-etude still 1/1 held (run f71ee505f7cc, MTG-317). Added task on-demand
  `mtg319-l4` (l4-etude's template: L4, 6 vCPU, 48 GB, 50 GB disk, same image, <= $0.60/h);
  it reads "none available, 0 of 1 in use", so RunPod L4 stock is out, not just the slot.
  `etude launch ... --rent mtg319-l4 --revision 53cf1357d` refused (no machines, no run
  started). `--max-usd` on a single-run launch needs `--cohort`; the Job's own `max_usd`
  ($3) is the cap. The shared controller's last loop was 2026-11-11, so `etude queue`
  would never start; a bounded poll (10 min, until 18:30 UTC) launches on first stock.
- 16:45 UTC: stock appeared; launched run `308602b435fe` (mtg319-live-token-attention-live)
  on machine `mtg319-l4-161dd1d181d2`, source 53cf1357d. `etude wait 308602b435fe` drives it.

- Run succeeded (USD 0.24 launcher, job 1,037 s); machine terminated by hand after
  (window ran to 19:01). w384l8 learner step median: bucketed 386 ms, bucketed-live
  202 ms (1.91x, bar 1.5x: PASS); numerics equal to bucketed's CUDA tolerance. Report,
  README index, docs claim updated; data published (75 manifest entries).
- `tests/infra/test_experiment_data_untracked.py`: 1 passed. `tests/model`: 310 passed, 4 skipped.

## Remaining

- Training suite (`--extra dev,artifacts,notebook,reports`, deselect test_etude_evaluation)
  hit the 1 h limit at ~415/772 (in `test_monitor_evaluation`), with another worktree's
  pytest loading the CPU. Failures, none in this diff's files: `test_game_errors`
  resource_limit replay x3 (`binding_mismatch`: native `semantic_decision_frame_json`
  differs from the retained frame); `test_inspect_batch_script` (`damped_policy._targets`:
  `RolloutBatch` has no `batch`); `test_head_to_head` x2 (duckdb, needs `--extra
  evaluation`). `test_active_recovery` and `test_learning_state` failed in the full run
  but pass alone (17 passed). Leave the rest of the suite to CI.
