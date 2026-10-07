# Remote hardware declarations

Hardware mixes live in `mixes/`; deployment runs through `uv run manabot deploy`.
See [remote training](../docs/remote-training.md) for compilation, deadlines,
retrieval, cleanup and the explicitly invoked paid gate.

Current observed inventory, 2026-10-06: zero rented RunPod pods. Query `remote
status` for a fresh observation. The parked AWS sandbox/job system was removed;
it is retained in Git history, not maintained alongside the RunPod path.
