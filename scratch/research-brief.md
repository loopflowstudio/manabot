Requested by Jack Heart on 2026-10-14 from Understanding Strategic Learning in Manabot, published in PR287. Source: https://github.com/loopflowstudio/fantasia/blob/68eba6104e3049b0feb73e8d243e8a686eb1abf6/manabot/LEARNING_RESEARCH_GUIDE.md

This is a scoped research follow-up, not approval of the guide's proposed mechanisms. Verify current code and retained evidence before drawing conclusions; source-check paper-specific claims. Distinguish implemented instruments, measured benefits, hypotheses and unknowns. Preserve frozen experiment identities and all negative/inconclusive evidence. Existing experiments and their checkouts must not be interrupted or modified. Use existing research, arena and reporting instruments rather than a second runner.

The first deliverable is a decision-bearing research report and a prospective bounded protocol. Use available code, saved artifacts, and small local correctness or profiling probes with explicit limits recorded before execution. No paid resources or new training campaign is authorized by capture or the initial analysis launch. Record source/world/schema/checkpoint identities, actual cost, uncertainty and failures; propose additional empirical work separately. Questions may remain open with explicit missing evidence. Keep plans and analysis in topic-named scratch notes until accepted for durable publication.

Problem and outcome
Raw transitions per second can hide bottlenecks in observation building, policy inference, branching or Python/Rust crossings. Produce current, reproducible end-to-end training and search profiles that identify the highest-value CPU optimization before a GPU rewrite.

Acceptance
- Inspect current VectorEnv, rollout pools, branching driver, immutable content sharing and observation/inference interfaces. Verify which paths are actually parallel rather than assuming batched APIs execute concurrently.
- Reuse MTG-41 and MTG-48/51-54/70/73 benchmarks and receipts. Explain what changed since those results; do not repeat settled clone/undo implementation merely because the guide mentions it.
- Under a declared small local profiling cap, separate simulation, observation, inference, boundary, clone/undo, storage and memory costs for representative training and shared-root search workloads. Record configuration, contention, counters and raw artifacts.
- Compare relevant serial/batched/multicore possibilities with an Amdahl bound and prioritize a measurable next optimization. Require semantic parity, viewer safety and replay/rollback witnesses for any experimental fast path.

Boundary
Managym owns rules and possible-world meaning; Manabot owns inference and learning. This first task delivers measurements and an optimization recommendation, not a parallel rules engine or distributed learner. Coordinate with MTG-108 rather than duplicating distributed RL.
