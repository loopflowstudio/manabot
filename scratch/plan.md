# MTG-325 plan: torch as the first matched backend

The MTG-304 plan this branch inherited is in git (`05c1203e9:scratch/plan.md`).

## Built

- `fbb0062b4`: `manabot/backends/torch/` with `policy.py` (`ActorCritic`, moved
  from `model/actor_critic.py`), `checkpoint.py` (moved from
  `model/checkpoint.py`), `rl.py` (`manabot.rl` numerics, returns, losses,
  selection), `learner.py` (`rl/step.py` + `training/precision.py` + `Learner`
  from `training/updates/learner.py`; the policy and `Precision` are
  TYPE_CHECKING-only imports so the step/EMA stay generic), `arrays.py`.
  Aliases (0.3.0): `model.actor_critic.*`, `model.checkpoint.*`,
  `training.learning.learner.Learner`; unreleased names (`manabot.rl`,
  `training.precision`, `updates.learner.Learner`, `model.buckets`) moved
  without aliases. Every call site in manabot/academy/scripts/tests/paper uses
  the new paths; experiments keep theirs except `cuda_precision.py`
  (`training.precision` had no alias). `manabot/policy/{__init__,prepare,
  checkpoint,protocol}.py`, `tests/model/test_policy_container.py`,
  `manabot/backends/__init__.py` and `policy/README.md` are MTG-321's files
  verbatim as of its 88c97936b (re-synced at realign; it had added
  `NotAContainer`, made `METADATA_KEYS` private and folded `dimensions()` into
  `PolicyCheckpoint`), except one `buckets.py` bullet in the README, so they
  merge cleanly. Re-sync whenever #495 moves. `backends/README.md` is
  MTG-321's plus a torch row and section, keeping "the device" in the
  `arrays.py` row that MTG-321 dropped (add/add conflict expected when the
  second of #495/#this lands).
  Boundary tests: policy imports no framework (static + subprocess),
  backends never import each other, `_TORCH_NOT_YET_MOVED` lists the 73
  modules outside the backend that import torch (both directions checked),
  `rl.py`/`learner.py`/`arrays.py` import nothing else from manabot at runtime.
- `4746f4c9f`: `manabot/policy/buckets.py` (NumPy). `admit_observation` and
  `bind_admitted` measure buckets for a bucketed policy and carry them on
  `AdmittedObservation.buckets`; `trim_observation`/`trim_admitted` stay torch
  in `model/inference.py`; `PreparedObservation` no longer takes `buckets`.
  Behaviour change on CUDA learning: buckets are measured on the host at
  admission instead of by one device read per forward (sizes identical).

- compress pass: `backends/torch/learner.py` drops section comments that
  repeated its docstring and `__init__` comments (`Learner` gains a docstring);
  `model/admission.measure_buckets` and `model/inference` use
  `arrays.to_numpy`/`device_of`; the generic-pieces boundary test compares
  module names without line numbers. `arrays.from_numpy`/`flat_numpy` stay
  unused until the torch parity `BACKEND` (Remaining 3), matching MLX's file.

- `61b87f84a`: `load_checkpoint` reads `.pt` (zip magic) or the container
  (`read_container`, then world/policy-input/architecture checks); save stays
  `.pt`.
- `3b9581475`, `e143082bb`: `assemble(PolicyCheckpoint)` and `BACKEND` in
  `backends/torch/checkpoint.py` (`TorchPolicy`/`TorchLearner`, NumPy
  protocol, canonical names; `Retained` carries the admitted torch
  observation; the damped loss is a local import because
  `training.updates.damped_policy` reaches collection → search → this
  module). Registration in `backends/parity.py` waits for #495 on this
  branch (parity.py/convert.py are not here).

## Coordination (2026-11-12)

- Extensions requested on MTG-321 at 19:41Z (prepare: FiLM, belief, history
  beyond w160l8, pooled token stream, v10, categorical value,
  historical_mean, attention off; container: belief binding, `bc`). No reply
  yet. #495 head 0c7cd1419 changes no shared file since 88c97936b (only its
  scratch), so the copies here are current. #474 still 05c1203e9.
- Rechecked 20:05Z: #495 still 0c7cd1419, #474 still 05c1203e9, no reply on
  MTG-321 to 8d8dc862. #497 merged to main (439379869), so Remaining 5 is done.
- Rechecked after the L4 run: #495 at 90cc33d98 (merge of main/#497, already
  here, plus one `test_experiment_files.py` line); shared `policy/` and
  `backends/__init__.py` files identical to this branch apart from the
  `buckets.py` README bullet, so no re-sync. #474 still 05c1203e9; no reply
  on MTG-321 to 8d8dc862. Remaining 1-3 wait on #495 landing via #474/main.

## Checks

- Boundaries, learning pieces (incl. new alias rows), minimal loop, learner,
  precision, numerical health, returns, objectives, policy container, agent,
  admission, credit assignment, puct/local-update digests, artifact storage,
  experiments, two-seat, bucketed shapes/execution, inference, mps, collection:
  all passed (several runs). After compress: boundaries, learner, bucketed
  shapes, inference, admission, learning pieces: 78 passed in 39 s.
- CPU equivalence, 05c1203e9 vs HEAD (`scratch/equivalence/run.py`,
  `compare.py`; base as a detached worktree with the same `.so`, one thread):
  1,676 tensors bitwise identical, 0 differ. Covers logits, values, a
  log-likelihood + value loss and every parameter gradient for 11 configs
  (default historical_mean, masked_mean, history v1, categorical WDL, FiLM,
  pooled token stream, attention off, 2 layers, hashed records, no world,
  input v10), each padded and bucketed (pooled padded only; base refuses
  bucketed pooled), on 24 real Elemental observations; plus every weight
  after `minimal.train(2)` for two configs (covers rl, gradient_step,
  update_ema, collection). HEAD vs HEAD rerun also 0 differ.
- Belief inputs added to the harness: the belief demo match's generated belief
  and its has/lacks-Bolt conditionings (`encode_belief`), cycled over 24 of
  that match's own observations, 460-id card vocabulary, 13 count buckets;
  padded and bucketed (bucketing turns off with belief rows). CPU 05c1203e9
  vs 4f54559f: 1,846 tensors, 0 differ; again after the main merge (b62cd6c8e):
  0 differ. Belief embedding gradients non-zero.
- CUDA L4, Etude run 15d399dab9ae (on-demand `mtg325-l4`, machine
  mtg325-l4-44bd32217822, terminated after; ~$0.04, Job cap $2.50, cohort
  `--max-usd 3`), revision 4f54559f, torch 2.15.0+cu132, TF32 matmul and
  cuDNN on, every forward/backward config on CUDA (bucketed included),
  `minimal.train` on CPU; one venv/native build, only `manabot/` swapped
  (`base.patch` reversed gives 05c1203e9's tree). Base vs head: 1,827 of
  1,846 tensors bitwise identical; all logits, values, losses and trained
  weights identical; 19 belief-config gradients differ, max abs 3.3e-9
  (belief owner/zone embeddings, attention in_proj). Head vs head repeat:
  13 differ, same tensors, max abs 2.8e-9. The differences are CUDA's
  run-to-run accumulation order in the belief path's backward, not the
  change; the stated tolerance is that floor (≤ 1e-8 abs on gradients,
  bitwise elsewhere).
- After the main merge: live tokens, bucketed execution, boundaries,
  inference, container load, backend protocol: 45 passed. Losses distinct per config; 1,152/1,439
  gradients non-zero, none non-finite.
- Second compress (protocol, container, boundary tests: 17 passed): `TorchLearner` value term uses `rl.squared_value_loss`
  (same ops); `gradient_layout = parameter_layout`.
- New: `test_torch_container_load.py` (2), `test_torch_backend_protocol.py`
  (1); with boundaries, policy container, capacity, minimal loop: 37 passed.
- `pytest --collect-only` over every changed test file: 1,116 collected; only
  errors are duckdb-dependent files (extra not installed).
- Bucket parity: `/tmp/mtg325buckets/compare.py` runs the old torch
  `bucket_sizes` (fbb0062b4) and the NumPy one on 1,960 real batches (history
  off/on, NaN/inf/huge/fractional/negative poison, objects on/off): identical.

## Remaining

1. ActorCritic over `manabot.policy.prepare.Prepared`. Blocked on #495
   landing and MTG-321's answer to the request above: MTG-321's `prepare.py` covers only the
   w160l8 subset (`require_supported` refuses FiLM, belief, history, pooled
   token stream, v10, categorical value, historical_mean, attention off);
   torch must keep running all of them bitwise. Extending `prepare.py` edits
   MTG-321's file; do it after #495 lands, then move each component's inline
   lookups (semantic_cards, facts, shared_effects, name_candidates,
   continuation, recent_events, token_stream, belief) onto prepared indices.
2. `save_checkpoint` writing the container (`load_checkpoint` reads both,
   by zip magic): needs the container to carry every config and the `.pt`
   extras (belief binding, `bc`), which `require_supported`/`_METADATA_KEYS`
   do not yet.
3. Register `"torch": "torch"` in `backends/parity.py` once #495 is on this
   branch (via #474/main), and run parity on a w160l8 capture.
4. Done (Etude 15d399dab9ae, see Checks). Re-run CPU and the L4 launch
   (`scratch/equivalence/mtg325_cuda_equivalence.py`, regenerate
   `base.patch`) after item 1 changes the forward.
5. Done (b62cd6c8e): main merged, `live_tokens.py` in `backends/torch/`
   (unreleased, no alias), `Execution.live_tokens` unchanged; prepared
   observations carry `live_token_count` beside admitted buckets.
6. Rebase/merge onto #474 when it moves (unchanged at 05c1203e9 on
   2026-11-12); delete `scratch/` before landing.

## Concept review items (2026-11-12): built

- `fetch_etude_checkpoint`/`local_checkpoint_file` live in
  `manabot/infra/artifacts.py`; `load_etude_checkpoint` stays in the torch
  backend. `eval/benchmark.py`/`card_evaluation.py` were never on
  `_TORCH_NOT_YET_MOVED` (they import the backend, not torch), so the list
  is unchanged.
- `model/architecture.py` imports no backend: `architecture_receipt(spec,
  space, {name: (numel, trainable)})`, `validate_architecture_receipt(actual,
  saved)`; torch calls `ActorCritic.architecture_receipt()`. Receipts written
  by 51f204ac4 (default, FiLM, 2-layer WDL, three params frozen) verify with
  identical canonical JSON under the new code; `test_capacity.py` pins the
  stored-vs-recomputed bytes.
- `backends/README.md`: the five files are a minimum; private modules allowed.
- Alias cut (steer 8332334d, quoting Jack Heart: "i think were being too
  ambitious withh the deprecations"; "if theyre not explicitly calling
  someting, we dont need to worry about maintaining it"): deleted
  `manabot/model/actor_critic.py` and `manabot/model/checkpoint.py` aliases,
  their `_MOVED` rows and doc mentions; 11 experiments import the new homes.
  The `training.learning`/`model.numerics`/`policy_distribution` aliases are
  #474's and #474 deletes them. `policy/README.md` and `policy/checkpoint.py`
  (MTG-321 copies) now name `manabot.backends.torch.checkpoint.load_checkpoint`:
  expect a one-line conflict when #495 lands.
- Check: boundaries, capacity, torch container load, policy container, recent
  events, record vocabulary, learning pieces, artifact storage, etude job,
  mirror admission, evaluation models, card evaluation, learning signal: 176
  passed, 1 skipped; after the alias cut, boundaries + learning pieces +
  learning signal 53 passed; every changed experiment imports.

## Scope change (steer 7258182e, 2026-11-12): MTG-326 folded in

Jack Heart: "it doesnt make sense to do the framework without a client or
vice versa". Same PR (or two stacked PRs from this Task if too large:
backend first, client second). Next passes:

7. `training/updates`: `update_damped_iteration(backend, learner, batch,
   learning, *, iteration)` and PPO `update_iteration(backend, learner,
   batch, learning, progress, rng, ...)` take the backend module
   (`import manabot.backends.torch as backend`); array math through
   `backend.rl`/`backend.learner`, NumPy for host bookkeeping. One owner each
   for `damped_policy_loss`, `legal_mask`, `policy_logs`,
   `reference_distribution` (`manabot/policy` if NumPy/index logic, else
   `backend.rl`). Executor and `training/loops/minimal.py` pass
   `manabot.backends.torch`; recipes, regimes, Etude entry points unchanged.
   MTG-326's names (steer 8be1c72a): `backend.rl` gets legal_mask,
   policy_logs, damped_policy_loss, clipped_surrogate, reverse_kl,
   squared_value_loss, categorical_value_loss, entropy, transition_gae,
   standardized_advantages, categorical_lambda_returns, selected_rows,
   selection_mask, selected_mean, require, float64_operand;
   `backend.learner` gradient_step, update_ema, forward_precision;
   reference_distribution in `manabot.policy.prepare`. MTG-326's contract text:
   `manabot/backends/README.md` "Algorithms are written once" on branch
   jack/mtg-326-write-the-damped-policy-and-ppo-updates-once.
8. Delete MLX's copied loss in `manabot/backends/mlx/learner.py` for the
   shared update (needs #495's mlx backend here); show agreement with the
   MTG-321 parity capture on the rented M2 (`~/.manabot-mps-bench/access.md`,
   ours until 23:39 UTC 2026-11-12; after that skip and say so).
9. Equivalence, pre-registered: torch backend, one collected batch, PPO with
   KL terms, damped scalar, damped categorical: every diagnostic and every
   per-parameter sum after one update byte-identical to pre-change code;
   name and bound any summation-order difference.
10. Tests: `test_learning_pieces`, `test_minimal_loop` on torch; one
    parametrized test running the update on every installed backend on the
    same captured batch within MTG-321 tolerances (skip absent backends).
11. Docs: `docs/training-loop.md` shows the backend argument and the one-line
    swap; `backends/README.md` gains "algorithms are written once".
