# Supplied research guide — retained source

Jack Heart supplied *Understanding Strategic Learning in Manabot* on 2026-10-14,
published in PR287 at `68eba6104e3049b0feb73e8d243e8a686eb1abf6`.
The exact original is preserved in [cpu-probes/research-guide-source.txt](cpu-probes/research-guide-source.txt).
It is scientific direction, not approval of each mechanism.

The guide connects regularization, exploration schedules, strategic cycling,
useful simulation throughput, and information-safe policy/search improvement.
For this Task the relevant systems question is strength per total compute:
measure simulation, observations, inference, crossings, storage and branching
separately; compare whole workloads with equal semantic coverage. Batched APIs
need not execute concurrently. Clone/undo, immutable sharing and GPU suitability
must be checked against actual code and retained negative evidence.

Ataraxos integrates simulator/history storage, reconstructs observations and
masks, preallocates memory, and supports nonterminal resets. Its historical
system cost does not isolate a CUDA benefit. Source-check specific paper claims.
Managym remains rules authority; manabot owns learning and inference. Any fast
path needs deterministic replay, branch isolation and viewer-safe observations.

The initial research findings and raw probes are in [cpu-findings.md](cpu-findings.md).
The subsequent active user goal asks for a laptop benchmark that exposes the
conditions important at GPU scale and supports continued optimization.
