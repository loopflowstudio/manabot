# MTG-321 plan notes

Built: manabot/model/mlx/ (prepare, model, checkpoint, learner, parity, convert), mlx extra, boundary test,
tests/model/test_mlx_port.py (MLX-only; 3 passed on the M2), experiments/mlx_pilot.py (capture/bench/soak/report),
experiments/mlx_pilot.md (report), manabot/model/mlx.md (guide), docs/training.md link. Data published
(experiments/data/mlx_pilot, 39 manifest entries).

Result (accepted under the pre-registered rule): whole w160l8 update 755.7 s CPU8 vs 452.1 s MLX = 1.67x;
w384l8 learner median 7.84 vs 5.15 s = 1.57x; collection 2.2-2.7x; MPS learner 1.21x (w160l8), 0.93x (w384l8).
Parity: logits 7.7e-9, values 3.6e-7, loss 3.3e-7 rel, min grad cosine 0.99994 (33 structural zeros excluded).
MLX memory: 13.5 GiB w160, 17.7 w256, 20.5 w320, 23.3 GiB w384 (learner phase).

Steers (quoted in full in .lf/prompts): comment e43e505b (MPS learner arm at w160l8/w384l8, after parity, before soak; done),
comment f6ed5b35 ("Stop running MLX or torch on the laptop ... the rented M2 ... is the only place this Task executes code").
Before that steer, early parity development and one smoke run ran on the laptop; everything after ran on the M2.

Single-file conversion (2026-11-12): done as directed below; `convert` returns (path, tensors), one output path;
two new tests (round-trip, refusal names original revision). M2: 5 passed; local boundary test: 10 passed.

Remaining: delivery only (publish/review/land). Branch is merged with origin/main (0 behind on 2026-11-12), so the
report's link to experiments/cuda_precision.md resolves.
Remote state: ~/mlx-pilot/{manabot,venv,capture (12 GB),bench,report} on the M2; owner releases the host.

Compress (2026-11-12): one parameter mapping `mlx_parameters` (checkpoint.py) now serves conversion, parity's
one-step comparison and the GRU test; `require_supported` raises; `attend`/`packed_steps` shared by model and test;
dead `PackedGRU.compiled` and `prepare` leftovers removed. On the M2: 3 tests passed; parity pass with identical
numbers (step max |Δ| 9.43e-5, update cosine 0.944017); convert + load_mlx_policy round-trip ok.
experiments/mlx_pilot.py left as measured.

## Concept review (2026-11-12, headless)

People see three actions: `convert` a torch checkpoint, `load_mlx_policy` the result, run `parity` on a capture.
The concepts behind them are right (a converted checkpoint is a derived artifact of one torch checkpoint;
parity is the gate; MLXLearner measures and never writes). One mismatch, settled by the existing design:

- **The converted checkpoint lost its provenance** (sidecar `"source"` held a local path, so a world-binding
  refusal said "records no source revision"). Resolved by the single-file conversion above: `hypers`,
  `world_binding`, `architecture`, `source` are JSON in the safetensors metadata (`_METADATA_KEYS`,
  manabot/model/mlx/checkpoint.py), the torch path under `converted_from`; test
  `test_a_refused_converted_checkpoint_names_the_original_revision`. No re-measurement; published capture
  metadata stays as written.

Noted, not changed: `parity` reads a capture format written by `experiments/mlx_pilot.py capture`; the format is
documented in parity's docstring. Moving the writer into the package would edit the measured experiment file.

Compress after single-file conversion (2026-11-12): inlined the one-use `converted_path`; `_METADATA_KEYS` private.
ruff clean; local boundary test 10 passed (MLX tests not rerun: no behavior change, M2-only per steer f6ed5b35).

Concept review, second pass (2026-11-12, headless): checked the built change against the design. Public surface is
still the three actions (`convert`, `load_mlx_policy`, `parity`); provenance mismatch is resolved in code and guide.
Nothing to change, no open product choice. scratch/concept-review.html marked as built.

## Backend layout (steer 2d4fb551, 2026-11-12)

Built as directed: `manabot/policy/` (prepare.py moved; checkpoint.py = container with canonical names,
`PolicyCheckpoint`, `read_container`/`write_container`, `from_torch_state`, `require_supported`/`UnsupportedPolicy`,
`Dimensions`; protocol.py = `BackendPolicy`, `BackendLearner`, `Backend`, `Targets`, `Settings`; README),
`manabot/backends/mlx/{__init__,arrays,policy,rl,learner,checkpoint}.py`, `manabot/backends/{parity,convert}.py`,
`manabot/backends/README.md` (matched layout), `manabot/backends/mlx.md`. Canonical names = torch names with GRU leaves
`input_weight/hidden_weight/input_bias/hidden_bias` and attention `qkv_weight/qkv_bias`; MLX split/fold lives in
backends/mlx/checkpoint.py. Container written/read by numpy (no safetensors dependency); `mx.load` reads it too.
Converted file is now `<stem>.safetensors`. Parity compares in each backend's own layout so numbers are unchanged.
Names aligned with MTG-325's draft (steer 310e6b5f) except metadata keys: kept `hypers`, `world_binding`,
`architecture`, `source`, `converted_from` (the torch record's own fields, which `validate_checkpoint_world` reads).
`policy/buckets.py` not moved: moving `model/buckets.py` would change the torch path.

M2: 19 passed (test_mlx_backend, test_policy_container, test_package_boundaries); parity on the full capture
reproduced the ledger's parity record exactly except step_update_min_cosine (informative) at 1e-10.
Local: boundary + container tests 14 passed; ruff clean.
