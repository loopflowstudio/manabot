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

Delete — do not maintain: nothing; this replaces pins and adds an admission check, no parallel path.

Not changed: frozen fixtures, experiment reports, benchmarks HTML and
configs/llm vLLM lock that record Torch 2.10 as evidence.

Checks: test_mps_training 24 + test_baseline_recipe 34 passed after compress; isolated artifacts/
Etude check 31 passed; test_release + test_play_launcher + package boundaries
passed after the release fixture floor moved to 2.15.0. CUDA under 2.15/cu132
not re-qualified on rented GPUs (say so in the PR).

## Concept review (2026-11-11): done

`mps_torch_error()` claims only what was measured (wording below). The remedy
is generic (`uv sync --locked`, or train on CPU) because `execute_regime` raises
the same text. Committed in 6b64a6bb3, wording refined in 86b00b538.

## Root cause (for the PR description; steer 67f27f5c)

On an Apple M2 (macOS 26.7) the MPS weight gradient of `F.linear` with a large
reduced dimension (`[160, 44550]^T @ [44550, 160]`, float32, seeded random
tensors, no manabot code) is wrong on Torch 2.10, 2.11, 2.12 and 2.13 (0/5
fresh-process repetitions pass, relative L2 error 2.25) and correct on 2.14 and
2.15 (5/5, relative error 5e-7). In manabot it shows up as wrong or nonfinite
gradients in the w160l8 attention output projection on real learner
minibatches. An M4 Max does not reproduce it. 2.14 would suffice; 2.15 stays
the pin. Recorded in the `manabot/training/mps.py` docstring; the error now
says "Torch 2.10–2.13 compute incorrect MPS linear-layer weight gradients on
Apple M2". test_mps_training + test_baseline_recipe: 58 passed.

## Compress (2026-11-12)

Code needed no cuts. docs/training.md MPS paragraph reflowed and aligned with the
measured cause (M2, 2.10–2.13, linear weight gradients); the remedy lives in the error text.

After merging main at dac7f9005 (managym published as its own package, release
workflow rewritten): workflows still pin 2.15.0, no 2.10 pins remain outside
evidence; test_release + test_mps_training 57 passed.

## Remaining work

None in implementation. QA (scratch/qa.html) found and fixed explicit `baseline --device mps` on old Torch (6fab820bf). Left for gate/CI: full suites, CUDA cu132 on Linux.
