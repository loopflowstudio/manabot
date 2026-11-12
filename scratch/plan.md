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
