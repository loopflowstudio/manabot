# MTG-325 baseline evidence for MTG-333

Full verification, exact commands and receipts:
`9fe9eb1e7:scratch/mtg325-torch-backend-verification.md`.
PR #501 merged as `ef28f3a5814978b7a48e1161033b25672bf0870b`;
MTG-325 completed 2026-11-13T04:48:06Z (tracker writeback was pending).

The baseline oracle matched 3,248 tensors and five update records exactly.
Fresh CPU Torch/MLX parity passed on eight learner and eight collection calls;
32 focused tests passed. L4 reference `8a376219e430`: damped 9.94 steps/s,
inference 2,296 obs/s, PPO within variation. These are MTG-325 receipts,
never MTG-333 measurements. MTG-333's plan holds its fresh evidence and bars.

MTG-325 deliberately left Prepared-driven Torch and container writes for
MTG-333 (Task comment 80a5ff79: “thats fine”). MLX still excludes PPO and
categorical critics. Converted scalar containers omit value_semantic; old
non-ZIP .pt is NotAContainer. No aliases for moved callers.
