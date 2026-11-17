# Optional vector statistics cost

Exploratory implementation check, 2026-11-17. Jack Heart selected "1a and 2b":
independent optional-analysis failure and custom Python questions on every vector
game. The intended claim is optional-hook cost, including no-hook framework
cost, on a bounded real buffered workload; no inference or training claim.

Compare kept `76510eaf3f` without hooks, edited without hooks, and edited with
CardPerformance, Behaviour and a per-game decision counter. Same release build,
interpreter, dependencies and supplied observation buffers. Four vector streams
play sixteen complete Mountain-only games each at seeds starting 7, action 0,
with native trivial-decision skipping. Completed streams are masked while
others finish. No recordings, native summary files, policy inference, network
or artificial contention. Five rotating repetitions, one fresh subprocess per
cell, no concurrent cells. Time reset, buffered stepping, stream synchronization,
close and final aggregation; exclude imports, vector/buffer allocation and
post-run output hashing. Process CPU and elapsed time are separate metrics.
All arms must agree on game/decision count, winner sequence and final native
state digests. Report errors, medians and full ranges; overlapping ranges do
not establish a difference. This fixture is cheaper than a real neural training
loop and overstates hooks' share of such a loop. It does not exercise rich
combat boards or expensive user code. No GPU overhead claim.

Results pending the frozen comparison. Script: `experiments/vector_statistics_cost.py`.
