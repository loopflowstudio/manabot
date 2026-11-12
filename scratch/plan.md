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

Remaining: rebase onto main before landing (report links experiments/cuda_precision.md, which is on origin/main only).
Remote state: ~/mlx-pilot/{manabot,venv,capture (12 GB),bench,report} on the M2; owner releases the host.

Compress (2026-11-12): one parameter mapping `mlx_parameters` (checkpoint.py) now serves conversion, parity's
one-step comparison and the GRU test; `require_supported` raises; `attend`/`packed_steps` shared by model and test;
dead `PackedGRU.compiled` and `prepare` leftovers removed. On the M2: 3 tests passed; parity pass with identical
numbers (step max |Δ| 9.43e-5, update cosine 0.944017); convert + load_mlx_policy round-trip ok.
experiments/mlx_pilot.py left as measured.
