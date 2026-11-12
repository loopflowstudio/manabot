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

## Coordination (rechecked 20:53Z 2026-11-12)

- #495 (MTG-321) merged to main 20:33Z and is merged here (9a728aca2).
  #474 moved to 9b2aeae33 (deletes every deprecated alias; contains main up
  to #497, not #495) and is merged here (e56150ece). This branch's diff
  against #474 therefore includes #495 until #474 merges main again.
- MTG-321 never answered the 19:41Z extension request (steer 8d8dc862);
  its PR is merged, so the extensions to `policy/prepare.py` and
  `policy/checkpoint.py` fall to this Task (the questions.md assumption).

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

1. ActorCritic over `manabot.policy.prepare.Prepared`. Unblocked (#495
   is here); this Task extends MTG-321's files in place. `require_supported`
   refuses FiLM, belief, history beyond w160l8, pooled token stream, v10,
   categorical value, historical_mean and attention off; torch must keep
   running all of them bitwise. Extend `prepare.py`, then move each
   component's inline lookups (semantic_cards, facts, shared_effects,
   name_candidates, continuation, recent_events, token_stream, belief) onto
   prepared indices. Substantial; candidate for the second stacked PR.
2. `save_checkpoint` writing the container (`load_checkpoint` reads both,
   by zip magic): needs the container to carry every config and the `.pt`
   extras (belief binding, `bc`), which `require_supported`/`_METADATA_KEYS`
   do not yet. Follows item 1.
3. Register `"torch": "torch"` in `backends/parity.py` `BACKENDS` (today
   `{"mlx": "mlx"}`) and run parity on a w160l8 capture. Unblocked.
4. Done (Etude 15d399dab9ae, see Checks). Re-run CPU and the L4 launch
   (`scratch/equivalence/mtg325_cuda_equivalence.py`, regenerate
   `base.patch`) after item 1 changes the forward.
5. Done (b62cd6c8e).
6. Done: #474 at 9b2aeae33 merged (e56150ece). Merge again if #474 moves;
   delete `scratch/` before landing.

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
  Compress pass: `model/agent.md` names `ActorCritic.architecture_receipt()`
  instead of the old `architecture_receipt(agent)`; nothing else to cut.
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

7. Done (a522c22ba). `update_damped_iteration(backend, ...)` /
   `ppo.update_iteration(backend, ...)` import no framework or backend at
   runtime (`test_updates_take_the_backend_as_an_argument`); torch types are
   TYPE_CHECKING annotations. Moved to `backends/torch/rl.py`: legal_mask,
   policy_logs, check_probabilities/check_logs/check_behavior,
   damped_policy_loss, clip_fraction; new there: action_logs, reference_logs,
   require_finite. `backends/torch/learner.py`: `no_grad()`,
   `Learner.set_learning_rate`. Deleted `training/updates/references.py` and
   its `training/learning/references.py` alias (#474 later deleted
   `model/policy_distribution.py` too). `reference_distribution` is
   `manabot.policy.prepare`'s NumPy function (added the empty-row refusal,
   vectorized counts with one `bincount`: 0.6 ms action-type / 0.05 ms uniform
   per [256,128] minibatch). Host bookkeeping in NumPy: trained rows,
   minibatch indexes, PPO epoch order, action-type bincounts, tail fraction
   (float32 mean). Still torch inside the updates: tensor methods
   (`.float()`, `.flatten()`, `.abs().mean()`, `.detach()`, `.softmax()`),
   `AdmittedRollout`, `OptimizerHealth`, `selection_diagnostics`,
   `value_scale` (named in `backends/README.md`; item 8 moves what MLX lacks).
   Executor, `learning/objectives.py` dispatcher, scripts and tests pass
   `manabot.backends.torch`; `minimal.py` reads `rl.`/`learner.` from the
   torch backend; docs/training-loop.md and backends/README.md
   "Algorithms are written once" updated.
8. Delete MLX's copied loss (`damped_loss`, `MLXLearner._loss` in
   `manabot/backends/mlx/learner.py`) for the shared update; the mlx backend
   is on this branch now. MLX is not installed on this laptop. Show agreement with the
   MTG-321 parity capture on the rented M2 (`~/.manabot-mps-bench/access.md`,
   ours until 23:39 UTC 2026-11-12; after that skip and say so).
9. Done on CPU (pre-registered: byte-identical). `scratch/equivalence/updates.py`
   at e0ad728a0 (before) vs a522c22ba and the main merge 9a728aca2 (after),
   one thread, one collected batch (2 streams, 6 decisions, hidden 16): PPO
   with KL (uniform and action-type references), damped scalar, damped
   scalar actor scope, damped categorical. Every diagnostic repr, every
   per-parameter float64 sum and the target residual bytes: the JSON files
   are byte-identical (`cmp`). Named summation-order difference: the NumPy
   action-type reference sums its row normalizer in NumPy order, not torch's;
   over 400 random [B<40, A<200] masks it differs from the deleted torch
   function by at most 5 ulp (uniform: identical); it did not change any
   byte on the captured batch.
   Compress pass: the damped update computes its target residual once;
   the boundary test's runtime-import helper drops top-level
   `TYPE_CHECKING` blocks and reuses `_FRAMEWORKS`. Boundaries, learning
   pieces, minimal loop, credit assignment: 66 passed; `updates.py` JSON
   still `cmp`-identical to after.json.
10. Torch half done (`test_learning_pieces`, `test_minimal_loop` pass).
    Remaining: one
    parametrized test running the update on every installed backend on the
    same captured batch within MTG-321 tolerances (skip absent backends).
11. Done in a522c22ba: `docs/training-loop.md` shows the backend argument
    and the one-line swap; `backends/README.md` "Algorithms are written once".

## Main merge (9a728aca2, 2026-11-12)

#495 landed on main 20:33Z; merged main. Conflicts: `policy/{README,checkpoint,
prepare}.py` and `backends/README.md` kept this branch's side (prepare carries
this branch's reference changes). #495's files repointed to this branch's
homes: `parity.py` (torch reference loss from `backends.torch.rl`, reference
from the capture's NumPy targets; the numpy-vs-torch reference cross-check is
gone with the torch function), `convert.py`, `tests/model/test_mlx_backend.py`,
`experiments/mlx_pilot.py` (its torch step now reads the NumPy reference).
Boundary test: kept one copy of main's duplicated block; the MLX rule allows
`manabot.backends.torch` outside `backends/`; `convert.py`/`parity.py` listed
as torch importers. MLX is not installed here: `test_mlx_backend.py` skips.
Checks after merge: boundaries, policy container, experiment files, learning
pieces, minimal loop, torch protocol/container: 55 passed, 1 skipped;
equivalence JSON byte-identical. Before the merge: touched update tests
(objectives, regimes, attack choice, numerical health, packed rollouts, credit
assignment, precision, phase timing, admission, ataraxos gradients, MPS,
two-seat, naming, recovery, active training): all pass except the 3
`test_game_errors` `binding_mismatch` cases already failing on main (MTG-304
plan).
