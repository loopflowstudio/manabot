# MTG-293 Torch 2.15 default, MPS minimum

Built: torch 2.15.0 pins (core floors, training-runtime `+cu132` on Linux,
play-runtime, CI/release workflows, release validator and its test fixture),
cu132 index, regenerated uv.lock. `manabot/training/mps.py` owns
`MPS_MINIMUM_TORCH` and `mps_torch_error()` (compared through `TorchVersion`, so
2.15 pre-releases are rejected); `accelerator()` skips MPS on older Torch,
baseline names the minimum when MPS is present but not admitted, and
`_runtime_identities` rejects MPS stages on older Torch (it runs before every
stage loop, so the loop does not repeat the check). Docs updated earlier
(docs/training.md, docs/development.md, README.md, manabot/README.md).

Not changed: frozen fixtures, experiment reports, benchmarks HTML and
configs/llm vLLM lock that record Torch 2.10 as evidence.

Checks: test_mps_training 24 + test_baseline_recipe 34 passed after compress; isolated artifacts/
Etude check 31 passed; test_release + test_play_launcher + package boundaries
passed after the release fixture floor moved to 2.15.0. CUDA under 2.15/cu132
not re-qualified on rented GPUs (say so in the PR).

## Concept review (2026-11-11): next implement direction

One fact, one message: `mps_torch_error()` currently says "installed Torch
{version} computes incorrect MPS gradients (seen on 2.10)", which asserts a
defect for 2.11–2.14 that no run measured. Reword so the claim matches the
evidence: Torch {version} is below the MPS minimum {MPS_MINIMUM_TORCH}; Torch
2.10 computed incorrect MPS gradients on learner minibatches. Keep the remedy
generic to every caller (upgrade Torch with `uv sync --locked`, or train on
CPU): the same text is raised by `execute_regime`, where `--recipe local` is
not the caller's interface. Update the `match=` patterns in
`tests/training/test_mps_training.py` accordingly. Nothing else changes: the
concepts (one constant, selection skips, admission rejects, baseline explains)
already match the Task, and the cu132 move is the Task's stated fallback.
