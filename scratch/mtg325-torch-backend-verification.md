# MTG-325 baseline evidence for MTG-333

PR #501 merged as `ef28f3a5814978b7a48e1161033b25672bf0870b`.
Full uncommitted verification text is preserved in
`/tmp/mtg333-compress-before/mtg325-torch-backend-verification.md`; earlier
plan: `65fa8c7aa:scratch/plan.md`; pre-land receipts:
`/tmp/mtg325-gate-notes/`.

## Decisions retained from the Task thread

- “it doesnt make sense to do the framework without a client or vice versa”
  folded the backend client into MTG-325.
- Comment `80a5ff79-6850-4125-8d6a-fd5bec5527f5`: “thats fine” to the interim
  implementation. Prepared-driven Torch and container writes remain MTG-333.
- Review `07a4ebc9-871d-4d1d-aa95-b67728c60b80` required one PR, independent
  MLX GRU biases, shared admission, live-token container loading and clipped
  Adam comparison. No aliases: callers move.
- Documented limits: converted scalar containers omit `value_semantic`;
  non-ZIP legacy `.pt` is `NotAContainer`; MLX updates still need Torch via
  health/rollout storage. PPO and categorical critics remain Torch-only.

## Observed after merge, 2026-11-13 UTC

`HEAD` and `origin/main` had tree `145cb1111f76eb28c4acacde7658dd1667ff0dc0`;
`git diff --exit-code HEAD origin/main` returned 0. Diff from gate `c958ae2f1`
under `manabot tests experiments` was empty.

`uv run --extra dev --extra mlx --extra artifacts pytest tests/test_package_boundaries.py tests/model/test_torch_backend_protocol.py tests/model/test_torch_container_load.py tests/training/test_update_backend_protocols.py tests/training/test_update_backends.py -q`
→ **32 passed, one existing nested-tensor warning, 126.10s**, no skips.

`uv run --no-sync python -m manabot.backends.parity /tmp/mtg325-gate-capture/initial.pt --capture /tmp/mtg325-gate-capture --json /tmp/mtg325-verify-parity.json`
→ **parity: pass (torch, mlx)**, eight learner minibatches and eight collection
calls each, unchanged capture binding. Torch/MLX maxima: post-Adam weights
7.45e-9 / 1.19e-7 (bar 1e-6), logits 4.83e-9 / 6.52e-9, values 1.79e-7 /
3.87e-7 (bar 1e-4), relative losses 0 / 1.20e-5 (bar 1e-3), minimum gradient
cosines ≈1 / 0.999999999758 (bar 0.999). Pre-clip norms 3.4389486 /
3.4389496 exceeded 0.267. A tensor's MLX update-direction diagnostic was -1;
that is not the gate, and every post-step weight passed. Tolerance evidence,
not identical trajectories.

Retained A4/gate captures: **3,248 tensors byte-identical**; B4/gate **five
update JSON records byte-identical**. Earlier A4 diagnostics include NumPy
float64 reduction rounding; default action-type reference normalization can
differ by five ulp; earlier real recipe trained weights were within 2.4e-7.

L4 Etude `15d399dab9ae`: bucketed TF32 outputs, losses and trained weights
matched; belief gradients differed ≤1e-8 absolute, also on same-revision
repeats. Speed `8a376219e430`: damped 9.63 → 9.94 steps/s, inference 1,282 →
2,296 obs/s; PPO -0.9% within observed variation. These predate the final
admission/MLX-GRU review corrections; no fresh GPU result followed those fixes.

CI succeeded:
- PR: https://github.com/loopflowstudio/manabot/actions/runs/43366252838
- Linux/macOS CPython 3.12/3.13 package installs:
  https://github.com/loopflowstudio/manabot/actions/runs/43366252747
- Exact-merge push `ef28f3a58`:
  https://github.com/loopflowstudio/manabot/actions/runs/43367019244
  (`gh run watch` exit 0). Selected integration, frontend, clean-machine and
  protocol jobs passed; Rust, conformance and Engine Budget skipped by selection.

`lf task complete MTG-325` returned `state: completed` at
`2026-11-13T04:48:06.406183Z`. Planning writeback returned `pending`:
“Saved locally; pending planning synchronization”. Tracker synchronization
was not claimed complete.
