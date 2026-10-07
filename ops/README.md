# Remote hardware declarations

Machine allocation requests (`JobSpec`) live in `jobs/`; deployment runs through
`uv run manabot deploy --spec ...`. `mixes/` retains historical input bytes only.
See [remote training](../docs/remote-training.md) for compilation, deadlines,
retrieval, cleanup and the explicitly invoked paid gate.

Current observed inventory, 2026-10-06: zero rented RunPod pods. Query `uv run manabot deploy
status` for a fresh observation. The parked AWS sandbox/job system was removed;
it is retained in Git history, not maintained alongside the RunPod path.
