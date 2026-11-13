# MTG-325 plan: torch as the first matched backend

The MTG-304 plan this branch inherited is in git (`05c1203e9:scratch/plan.md`).

## Built

- `fbb0062b4`: `manabot/backends/torch/` with `policy.py` (`ActorCritic`, moved
  from `model/actor_critic.py`), `checkpoint.py` (moved from
  `model/checkpoint.py`), `rl.py` (`manabot.rl` numerics, returns, losses,
  selection), `learner.py` (`rl/step.py` + `training/precision.py` + `Learner`
  from `training/updates/learner.py`; the policy and `Precision` are
  TYPE_CHECKING-only imports so the step/EMA stay generic), `arrays.py`.
  No aliases remain (cut under steer 8332334d; #474 deleted its own). Every call site in manabot/academy/scripts/tests/paper uses
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

## Coordination (rechecked at realign, 2026-11-12)

- #495 (MTG-321) is on main and merged here; the `policy/prepare.py` and
  `policy/checkpoint.py` extensions fall to this Task or its follow-up.
- #474 is still open at e0e99309a, merged here (acf319cca). Merge again if
  it moves; rebase onto main once it lands.
- Main has one commit past this branch, e38518093 (engine-only: typed mana
  spending restrictions, Rust plus two Python tests); `git merge-tree` shows
  it merges cleanly. Taking it needs the extension rebuilt; leave it to gate
  or the post-#474 rebase.
- The scope decision is posted on the Task (steer aeac5366, 21:39Z): ship
  this PR as it stands and stack Remaining 1 and 2, or build them here.
  Unanswered at realign.

## Checks

- Boundaries, learning pieces, minimal loop, learner,
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

1. ActorCritic over `manabot.policy.prepare.Prepared`: not built; sized in
   "Pass of 21:30Z" below. Recommended as its own stacked Task.
2. `save_checkpoint` writing the container: follows item 1 (the container
   must first carry every configuration and the `.pt` extras).
3. Done (f61ed7dcc): `BACKENDS = {"torch", "mlx"}`; torch passes its own
   parity on the w160l8 capture (see "Pass of 21:00Z").
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
8. Done; see "Pass of 21:00Z".
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
10. Done: `tests/training/test_update_backends.py`; see "Pass of 21:00Z".
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

## Pass of 21:00Z 2026-11-12: items 3, 8, 10

Built:
- `learner.loss(objective)` / `learner.step(loss, max_grad_norm, recorder=)`
  in both backends. The update writes its minibatch loss as a closure over
  the forward (`minibatch_objective` in `training/updates/damped_policy.py`);
  torch runs it once, MLX evaluates it with `nn.value_and_grad`. Torch's
  `gradient_step(loss, ...)` is unchanged and `Learner.step` calls it.
- `backend.rollout.AdmittedRollout` in both (`training/updates/learner.py`
  moved to `backends/torch/rollout.py`; MLX's prepares rows), with `legal`
  and `evidence`. Backend `__init__`s resolve their files as attributes on
  first use.
- `manabot/policy/returns.py`: NumPy `terminal_distances`, `reverse_waves`,
  `transition_gae`, `quantile`, `selection_mask`. Against torch on 2,000
  random [T,E] batches and 28,000 quantiles: bitwise identical
  (`torch.quantile` takes a float64 rank from a float32 q and one fused
  multiply-add from the nearer end; the NumPy form reproduces that).
- `selection_diagnostics`, `value_scale`, `batch_health` are NumPy on host
  rows (`selection.py` no longer imports torch). `OptimizerHealth(...,
  backend=)` and `failure_tensors(..., refusal)` take the backend's check and
  `NumericalError`.
- MLX: `rl.py` has torch's functions (no categorical critic), `learner.py`
  is `Learner` (`damped_loss`, `MLXLearner` deleted), `rollout.py`,
  `arrays.from_numpy/float32`; `MLXActorCritic(dims, tables, config)` and
  `forward_distribution`.
- Protocol: `BackendLearner` is the learner (`loss`, `step`, `gradients`,
  `parameters`, `set_learning_rate`); `BackendPolicy.forward_distribution`.
  `parity.minibatch_loss` runs `minibatch_objective` through a backend, for
  the torch reference and every backend. `TorchLearner` (protocol copy of the
  loss) deleted; torch `BACKEND` parameters keep torch names (`torch_name`).
- `experiments/mlx_pilot.py` calls `Learner` + `minibatch_loss`.
- PPO update: helper signatures only; it calls tensor methods and
  `get_value` (torch-only). Third compress: it steps through `learner.step`,
  passes `backend=` to `OptimizerHealth` and `rl.NumericalError` to
  `failure_tensors` (now required) like the damped update. Check: update
  backends, learning pieces, numerical health, objectives, minimal loop,
  boundaries 85 passed; `updates.py` vs `after.json` differs only in the
  known 1-ulp `value_scale` std.

Checks:
- Torch update equivalence (`scratch/equivalence/updates.py` vs `after.json`):
  parameters, target residual bytes, every PPO diagnostic and every damped
  diagnostic identical except one float64 `value_scale` statistic per damped
  configuration, 1 ulp (relative 1.4e-16 / 1.7e-16): NumPy's summation order
  for the float64 std, not torch's.
- MLX is installed in this worktree's venv now (`uv pip install mlx==0.32.4`,
  the locked version).
- Local (laptop, MLX 0.32.4): parity on 8 learner minibatches and 8
  collection calls of the MTG-321 capture (copied to /tmp/mtg325capture):
  pass for torch and mlx, before and after the MLX loss deletion (mlx: logits
  7.7e-09, values 3.7e-07, min cosine 1.000000, one step 9.4e-05 / 0.944018).
- Rented M2 (21:30Z, `~/mtg325/manabot` = this branch's files and this
  checkout's cp313 extension, `~/mlx-pilot/venv`): `python -m
  manabot.backends.parity ~/mlx-pilot/capture/initial.pt --capture
  ~/mlx-pilot/capture`: pass (torch, mlx). mlx: forward logits 7.0e-09,
  values 3.6e-07, collection logits 6.7e-09, values 3.6e-07, loss 0.0000%,
  min cosine 0.999940 over 201 tensors, one step 9.4e-05 / 0.944017: the
  figures MTG-321 recorded in `backends/README.md` (0.999940 over 201;
  9.4e-05, 0.944017). torch against its own reference: logits 9.5e-09,
  cosine 1.000000 over 169.
- M2 pytest: update backends, mlx backend, boundaries, learning pieces,
  minimal loop, torch protocol: 37 passed. (A first run failed 3 boundary
  tests only because the copy lacked `academy/`; with it, all pass.)
- Laptop pytest: learning pieces, minimal loop, numerical health, credit
  assignment, phase timing, rl objectives, regimes, precision, mps training,
  two-seat, packed rollouts, ataraxos gradients, naming, recovery, active
  training, attack execution, ci selection, conditional search, admission:
  277 passed, 1 failed (`test_conditional_search` frozen fixture:
  `experiments/data/int-13-conditional-strategy-fixture-v1.json` is not in
  this worktree; artifact-store data, unrelated). Then update backends,
  boundaries, torch protocol, mlx backend, learning pieces, minimal loop,
  numerical health, credit assignment, phase timing: 97 passed; experiment
  files: 4 passed.
- The M2 is no longer needed by this Task after this pass.

## Pass of 21:30Z 2026-11-12: main merge, item 1 sized

- Merged main (71ab00215): #499 removed the pooled token stream. Kept this
  branch's `backends/torch/policy.py` homes; `policy/checkpoint.py`
  `require_supported` no longer reads `token_stream` (steer 793944f0);
  `backends/README.md` and the boundary list drop the stream. Checks:
  boundaries, value aggregation, live tokens, bucketed execution, policy
  container, torch container/protocol, update backends: 83 passed; agent,
  bucketed shapes, inference, learning pieces, minimal loop, experiment
  files, regimes: 72 passed.
- `scratch/equivalence/run.py` now runs at HEAD (10 configs; the pooled one
  is gone) and also captures the serving path (`PolicyInference`, no grad,
  B=24 and B=1) and `live_tokens=True` for every attention config: 3,248
  tensors, 13 s, HEAD vs HEAD 0 differ. This is the oracle for item 1.
- Item 1 was not started. What the reading showed:
  - Size. Seven torch components do their lookups inline (`policy.py`
    actions/links/belief, `model/{semantic_cards,facts,shared_effects,
    continuation,name_candidates,recent_events}.py`), each with branches on
    device, autograd, `bucketed` and vocabulary. About 20 test, script and
    experiment files call those encoders or `_gather_*`/`_attend_objects`
    with raw tensors (`test_agent`, `test_value_aggregation`,
    `test_record_vocabulary`, `test_shared_effects`, `naming_reference`,
    `bench_local_update`, `l4_step_profile`, ...). Comparable to everything
    this PR already holds.
  - `prepare` lacks, for torch's configurations: hashed record ids (word
    arrays, no `searchsorted`), no-world definition ids, v10 turn-state
    columns, belief rows (object count N + rows + 1 in every flat index),
    recent events, and the referenced-definition subset learning encodes
    without buckets. `Tables` assumes a catalog world.
  - Flat indices depend on the object count, so bucket trimming has to run
    on the host before `prepare` (today `trim_admitted` trims device
    tensors in the forward).
  - Bitwise hazards: `program_number` (`np.log1p` against `torch.log1p`),
    `/60` and `/FACT_TIMESTAMP_SCALE` in NumPy, `index_put` scatter against
    masked lookups in embedding gradients.
  - The carrier exists already: `AdmittedObservation.references`
    (`CardReferences`) is host-admitted rows travelling with the tensors.
    A slice order that keeps each commit coherent: replace `references`
    with `Prepared` rows; legal-action coordinates and focus; permanent
    links; relations; facts; shared effects; then drop the tensor
    dictionary from the forward. Host cost recorded by MTG-321: 6 ms at
    B=256.
  - Expected CUDA effect: the forward loses its device reads (`nonzero`
    for legal rows and programs, `unique` for referenced definitions) and
    gains one host preparation and transfer per minibatch; rerun item 4.
- Why it is not in this PR: a partly converted forward would ship a second
  input representation beside the dictionary, and would void the L4 and M2
  results recorded above. The PR as it stands is complete for the matched
  torch backend and the updates written once, and its docs state the
  interim (`.pt` save, torch does its own lookups).

Remaining for this PR (if steer aeac5366 is answered "ship"): gate; delete `scratch/` before landing. Items 1 and
2 and the item 4 rerun belong to the stacked follow-up.

## Pass of 21:45Z 2026-11-12: main and #474 merged again

- Main adf601acb (#499's squash) merged cleanly.
- #474 moved to e0e99309a and is merged. Its 64dff0c5f repoints the MLX
  backend, parity, `mlx_pilot.py` and `prepare.py` at `manabot.rl` and
  `manabot.training.updates`; those seven files keep this branch's side.
  Its 73eb69848 edits `model/policy_distribution.py` and `rl/selection.py`,
  both deleted here: `backends/torch/rl.py` already imports `require`
  plainly and now takes the `selection_mask` change (it calls
  `selected_rows(..., kind=kind)`). Its guide test
  (`test_guide_shows_the_loop_as_written`) passes against this branch's
  `minimal.py` and `docs/training-loop.md`.
- Check: minimal loop, boundaries, container support, learning pieces, update
  backends, mlx backend, returns, experiment files: 80 passed.
  `scratch/equivalence/updates.py` against `after.json`: identical except the
  known 1-ulp `value_scale` std per damped configuration.
- Item 1 not started in this pass; the stacked follow-up still needs a
  person's yes (questions.md).

## Docs (write-docs pass, 2026-11-12)

The docs that ship with this change: `manabot/backends/README.md` (contract,
written-once updates, torch backend, convert, parity),
`manabot/policy/README.md`, `manabot/backends/mlx.md`,
`docs/training-loop.md`, and the backend paragraphs of `manabot/README.md`
and `docs/training.md`. This pass corrected: `policy/README.md` said the
torch model does not consume the package; `training-loop.md` linked a
`policy/README.md` anchor that does not exist; `manabot/README.md` listed
rollout admission and the reference magnet under `training/updates/`;
"PPO" where the parity loss is the damped update's. They describe the
branch as it stands (`.pt` save, torch forward not over `Prepared`, PPO on
torch only); rewrite those sentences when Remaining 1 and 2 land.

Second write-docs pass (2026-11-12, checked every cited name and signature
against the code): `backends/README.md`, `backends/mlx.md` and
`docs/training-loop.md` no longer say MLX's `rl` has all of torch's functions
(it has what the scalar damped update calls; `standardized_advantages` and
`selected_rows` are PPO-only and absent, as are the categorical ones); the
parity sample shows the torch lines and `parity: pass (torch, mlx)` (torch
figures from a laptop run of the MTG-321 capture: forward logits 9.7e-09,
values 2.7e-07, collection 0, cosine 1.000000 over 169, one step 7.5e-09);
the updates return `UpdateDiagnostics`, a typed dictionary.

## Concept review (2026-11-12, after steer 449b1a8e)

State reviewed: HEAD 8420fb69d. The three layers hold: `manabot.policy`
(NumPy), `manabot.backends.{torch,mlx}` (matched files), updates taking the
backend module. One mismatch and one new direction; the scope question stays
open.

Next implement direction, in order:

1. One owner for the host recursions. Torch's `rl.terminal_distances`,
   `reverse_waves` and `transition_gae` already run on the host (`.cpu()`),
   duplicating `manabot.policy.returns` bit for bit, while MLX's `rl` calls
   `policy.returns`. The matched rule is that a backend may import
   `manabot.policy` (framework-free) and never another backend; torch's `rl`
   should follow it like MLX. Make torch's `transition_gae` a wrapper over
   `returns.transition_gae` (`arrays.to_numpy` in, `from_numpy` back to the
   rewards' device), delete torch's `terminal_distances`/`reverse_waves`, and
   add `manabot.policy.returns` to `_GENERIC_ALLOWED` in
   `test_generic_learning_pieces_import_only_themselves` (and the runtime
   probe's allowed set). Keep torch's device-side selection
   (`selected_rows`, `torch.quantile`): it runs on the minibatch's device,
   and moving it to the host adds a device read per PPO minibatch on CUDA.
   Check: `scratch/equivalence/updates.py` against `after.json` (only the
   known 1-ulp `value_scale` std), returns, update backends, minimal loop,
   boundaries. Update `backends/README.md`/`mlx.md` sentences that say torch
   keeps its own recursions, and drop the questions.md follow-up line.
2. Speed check, steer 449b1a8e (Jack Heart: "lets give a $10 budget for
   making sure there is no performance regression on pytorch due to this
   change"). Run it after item 1 so it measures the code that ships.
   - Base: `manabot/` at `git merge-base HEAD origin/main` (adf601acb at
     review time; it excludes #474, so a regression found there is bisected
     against #474's tip e0e99309a before blaming this branch). Head: HEAD.
     One L4, one venv and native build, only `manabot/` swapped
     (regenerate `base.patch` from that merge-base).
   - Measure, ABAB interleaved, at least 3 repeats each, medians with spread:
     damped and PPO learner steps/s at a current-recipe size (Elemental),
     padded, `bucketed_shapes`, `live_tokens`, `precision="tf32"`;
     `PolicyInference` obs/s for collection; `save_checkpoint` /
     `load_checkpoint` time.
   - Suspects to time separately if a gap shows: buckets measured on the
     host at admission; NumPy `reference_distribution` plus its upload per
     minibatch; host reads for `selection_diagnostics`, `value_scale`,
     `batch_health`; the `learner.loss` closure; NumPy minibatch indexes and
     PPO epoch order; the zip-magic sniff in `load_checkpoint`.
   - A regression is any median slower than the run-to-run spread; fix it in
     this PR before gate. Launch through Etude with `--max-usd 10` for the
     whole check; record the launcher's cost once here and in the PR body.
     A short targeted CPU run of the same harness on the laptop first, no
     load. Rerun if the forward changes (the Prepared rewrite).

Settled without change: `backend` stays typed `types.ModuleType`; the
`BACKEND` object in `checkpoint.py` (MTG-321's published name) is the
parity loader, the module is the backend the updates take; both read
clearly at their call sites.

Open product choice (unchanged): steer aeac5366, ship this PR with
Remaining 1 and 2 stacked, or build them here. Steer 449b1a8e's "Rerun this
check if the forward changes (e.g. if the Prepared rewrite lands here)"
leaves it open. Items 1 and 2 above are needed on either answer.

## Pass after the concept review (2026-11-12)

- Item 1 built: torch `rl.transition_gae` wraps `policy.returns.transition_gae`;
  `categorical_lambda_returns` walks `returns.reverse_waves`; torch's
  `terminal_distances`/`reverse_waves` deleted. Boundary allow-list gains
  `manabot.policy.returns` and `manabot.backends.torch.arrays`. Check: returns,
  boundaries, update backends, minimal loop 58 passed; `updates.py` vs
  `after.json`: only the known 1-ulp `value_scale` std (3 damped configs).
- Item 2 harness: `scratch/speed/{bench,compare,mtg325_speed}.py`, `base.patch`
  (HEAD→adf601acb, `manabot/` only). One process = one arm: elemental-long-183
  re-pinned (w64l2, 256 streams x 64, both seats, elemental-uniform),
  bucketed + live tokens + tf32 on CUDA; warm-up then 3 damped and 3 PPO
  updates from the same weights (optimizer steps/s via a global step hook),
  50 `PolicyInference` collection forwards at B=256, 5 save/load. ABAB, 3
  processes per arm. Laptop CPU smoke (8 streams x 4, one repeat each): both
  arms collect the same 1,711 rows and 159 damped steps; damped 5.84 vs 5.80
  steps/s, PPO 1.41 vs 1.40, load 0.050 vs 0.049 s (base vs head; single
  samples, no spread).
- Speed check queued: Etude run 47636a275a9e (cohort `mtg325-speed`,
  `--max-usd 10`, revision d035ae750) waiting for `mtg325-l4` stock; the
  first launch was refused (no machines, no cost).
- Scope settled (steer bc2e6bd3): ship as is; Remaining 1 and 2 are MTG-333
  (named in the PR body). Remaining for this PR: the speed result, then gate.

## Split (steer 0152f1d6, 2026-11-12)

Jack Heart: "PR #501 is at 21,600 added lines as a draft. That is past review
size; apply the split the brief allows now". Measured: 17,312 of #501's added
lines were `scratch/` (two generated `base.patch` files, equivalence JSON);
the code diff against #474 was 4,332 / 2,500.
- PR A = #503 (branch `jack/mtg-325-torch-backend`, worktree
  `../manabot.mtg-325-a`, base main a343904ee): e0ad728a0 (the commit before
  the updates took the backend) merged with #474's tip and main. Updates are
  #474's with backend import paths; `manabot.rl` changes from #474 ported
  into `backends/torch/rl.py`; boundary test from this branch minus the
  written-once parts. 2,137 / 1,415 lines plus the harness scripts.
  CPU `run.py` A vs this branch: 3,248 tensors, 0 differ. Pytest: 269 + 47
  passed.
- PR B = #501 (this branch), merged A with `-s ours` (its tree already holds
  A): 64 files, 2,344 / 1,234 against A. Generated scratch files untracked
  (kept on disk).
- No deprecation aliases in A despite the steer's wording: steer 8332334d
  removed them; noted as an open point in #503's body.

## Speed check (steer 449b1a8e), pass 1: regression found and fixed

- Etude 47636a275a9e (revision d035ae750, base adf601acb, one L4): head was
  slower on the learner: damped 6.16 → 4.57 steps/s, PPO 0.219 → 0.157
  steps/s (first pair; spread ~1%); `PolicyInference` faster, 823 → 1,061
  obs/s. Cause: buckets measured on the host in NumPy (4746f4c9f, in #503),
  twice per learner row (admission in `admit_rows`, then `bind_admitted`),
  ~20 ms per 256-row minibatch on the laptop each; base measured on the GPU.
  Learner observations are ~165 KB/row, almost all zero padding.
- Fix (#503 d92b38b61): `bucket_sizes` reduces over the batch axis first with
  no full-size temporaries and names objects with one boolean table (203 → 69
  ms at 3,300 rows; one `any` pass over the data is 38 ms); `admit_rows` only
  checks (`check_observation`). Learner host bucket work per 256-row step:
  41 → 5.4 ms on the laptop. Sizes identical on the 1,960-batch parity
  harness (poison included); CPU `run.py` 0 differ. Ruled out: per-row needs
  cached at admission (per-row reduction costs 118 ms/3,300 rows, slower for
  the damped update's single pass), and trimming on the host before transfer
  (the updates and collection read full-capacity `actions_valid`).
- Updates reference regenerated after the engine change merged with #474
  (e38518093 changes games): `before.json` from #503's tree, `after.json`
  from this branch, same extension: parameters and residuals identical;
  `value_scale` float64 stds differ by 1 ulp (3 stats, categorical).
- Pass 1 complete (3 ABAB pairs, `scratch/speed/run1/compare.txt`), medians
  [min, max] base vs head: damped 6.16 [6.04, 6.19] vs 4.57 [4.56, 4.68]
  steps/s (0.742); PPO 0.2196 vs 0.157 (0.715); inference 822 vs 1,087 obs/s
  (1.32); save 0.032 vs 0.032 s; load 0.169 vs 0.168 s. Same 52,226 learner
  rows and 204 damped steps in every process. Launcher cost $0.58 (final).
- Pass 2 queued as Etude e978886d15e5: revision bb7c3e6a0 (base main a343904ee), cohort
  `mtg325-speed-2`, `--max-usd 9` (pass 1 counts against the USD10).

## Speed check pass 2 (Etude e978886d15e5, revision bb7c3e6a0, base main a343904ee)

3 ABAB pairs, `scratch/speed/run2/compare.txt`; same 53,341 learner rows and
208 damped steps everywhere. Medians [min, max], base vs head:
damped 6.05 [5.96, 6.37] vs 6.38 [6.20, 6.39] steps/s (1.055); PPO 0.2199
[0.2194, 0.2210] vs 0.2108 [0.2105, 0.2116] (0.959, REGRESSION);
`PolicyInference` 831 vs 1,257 obs/s (1.51); save 0.033 vs 0.032 s; load
0.165 vs 0.173 s [0.168, 0.188] (within spread). Launcher cost ~$0.27; the
machine stayed rented after success and was terminated by hand.
PPO is +0.19 s per step (16 steps of ~3,300 rows: 4 epochs x 4). Host bucket
measurement explains about half (67 ms per 3,300 rows on the laptop, every
epoch). Pass 3 (Etude cohort `mtg325-speed-3`, revision f3b7fbaa4) adds a
`backend` arm (#503's tree) and the learner's profiler sections, to place
the rest in #503 or #501.

## PPO fix (#503 ee01ec9df, "bucket needs measured once and regrouped")

`row_needs` measures each row once (reading only the prefix the batch uses;
97.5 ms per 3,300 rows on the laptop vs 69 ms for one batch measurement);
`AdmittedRollout.admit_rows` stores the needs and `bind_admitted(needs=)`
buckets every epoch's minibatches from them (`sizes_of`, 0.015 ms). PPO's 4
epochs measure once instead of 4 times; the damped update pays ~2 ms more per
256-row step. Sizes identical on the parity harness; new
`test_row_needs_regroup_to_the_measured_buckets`; forward 0 differ; updates
byte-identical to `before.json` (#503) and `after.json` (#501). Pass 3
(4668724056dc) measures the code before this fix; pass 4 validates it.

## Speed check pass 3 (Etude 4668724056dc, revision f3b7fbaa4; another, faster L4 host)

Three arms x2 (`scratch/speed/run3/compare.txt`; the job's last `compare`
line failed on a quoting typo, compared locally from the fetched files).
Same 53,341 rows everywhere. base / #503 tree / #501 head: damped 9.46 / 8.93
/ 9.16 steps/s; PPO 0.2539 / 0.2448 / 0.2439; inference 1,284 / 2,382 /
2,373 obs/s; save and load within spread. The gap is all in the learner's
"Forward and loss" section (damped 8.26 / 9.92 / 9.67 s, PPO 12.2 / 14.6 /
15.0 s per update; backward and optimizer equal), and #503 already shows it:
host bucket measurement, not the written-once updates. Launcher cost ~$0.56;
the machine was terminated by hand.

Fix (#503): collection records each row's needs (admission keeps them on
`AdmittedObservation.needs`, `_ObservationRows` stores them beside the rows,
`RecordedObservations.gather_needs` reads them) and `AdmittedRollout`
regroups them, so a learner update measures only the bootstrap rows (1 of
6,199 rows in a CPU check). `row_needs` always counts object references, so
recorded needs serve a learner with or without object trimming. Tests:
`test_collection_records_each_rows_measured_needs`,
`test_row_needs_regroup_to_the_measured_buckets`; forward 0 differ; updates
byte-identical (#503 and #501). Pass 4 (cohort `mtg325-speed-4`, revision
7820d1e8d, three arms x3) validates it. 

## Speed check pass 4 (Etude 8a376219e430, revision 7820d1e8d): no regression

Three arms x3 (`scratch/speed/run4/compare.txt`); same 53,341 rows and 208
damped steps everywhere. Medians [min, max], base main a343904ee / #503 /
#501: damped 9.63 [9.55, 9.64] / 9.90 [9.88, 9.93] / 9.94 [9.70, 10.01]
steps/s; PPO 0.2558 [0.2547, 0.2560] / 0.2575 [0.2566, 0.2589] / 0.2534
[0.2110, 0.2564]; `PolicyInference` 1,282 / 2,350 / 2,296 obs/s; save 0.022 /
0.022 / 0.023 s; load 0.079 / 0.080 / 0.082 s (all within spread); collection
73.6 / 62.9 / 61.9 s. #501's PPO median is 0.9% under base's: one of its nine
updates stalled (0.070 steps/s in process 3; the others 0.253-0.257), and its
learner sections are no slower (PPO minibatches 61.45 vs 61.67 s, forward and
loss 11.77 vs 12.03 s). Machine terminated by hand.

Speed-check cost (launcher, final): $0.58 + $0.46 + $0.34 + $0.66 = $2.04 of
the USD10 cap.
