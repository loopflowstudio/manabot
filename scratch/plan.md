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
  `manabot/backends/__init__.py` and `policy/README.md` are MTG-321's
  c42eee948 files verbatim (except one `buckets.py` bullet in the README), so
  they merge cleanly. `backends/README.md` is MTG-321's plus a torch row and
  section (add/add conflict expected when the second of #495/#this lands).
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

## Checks

- Boundaries, learning pieces (incl. new alias rows), minimal loop, learner,
  precision, numerical health, returns, objectives, policy container, agent,
  admission, credit assignment, puct/local-update digests, artifact storage,
  experiments, two-seat, bucketed shapes/execution, inference, mps, collection:
  all passed (several runs; last 78 passed in 20 s).
- `pytest --collect-only` over every changed test file: 1,116 collected; only
  errors are duckdb-dependent files (extra not installed).
- Bucket parity: `/tmp/mtg325buckets/compare.py` runs the old torch
  `bucket_sizes` (fbb0062b4) and the NumPy one on 1,960 real batches (history
  off/on, NaN/inf/huge/fractional/negative poison, objects on/off): identical.

## Remaining

1. ActorCritic over `manabot.policy.prepare.Prepared`. Blocked on a design
   point (`scratch/questions.md`): MTG-321's `prepare.py` covers only the
   w160l8 subset (`require_supported` refuses FiLM, belief, history, pooled
   token stream, v10, categorical value, historical_mean, attention off);
   torch must keep running all of them bitwise. Extending `prepare.py` edits
   MTG-321's file; do it after #495 lands, then move each component's inline
   lookups (semantic_cards, facts, shared_effects, name_candidates,
   continuation, recent_events, token_stream, belief) onto prepared indices.
2. `save_checkpoint` writing the container and `load_checkpoint` reading it:
   needs the container to carry every config and the `.pt` extras (belief
   binding, `bc`), which `require_supported`/`METADATA_KEYS` do not yet.
   Docs (training-loop.md checkpoint example, manabot/README.md) already
   describe this and are ahead of the code.
3. Torch `BACKEND` in `backends/torch/checkpoint.py` + registration in
   `backends/parity.py` (MTG-321's), retained-reference mode; needs 1.
4. Equivalence proof: MTG-299's harness / 617-tensor comparison against the
   pre-change torch path; CUDA L4 check (under USD 3) with `bucketed_shapes`
   and `precision="tf32"`.
5. After #497 (MTG-319) merges to main: merge main in, move
   `manabot/model/live_tokens.py` into `manabot/backends/torch/` with
   ActorCritic, keep `Execution.live_tokens` where it is (steer cc1dfb64).
6. Rebase/merge onto #474 when it moves; delete `scratch/` before landing.
