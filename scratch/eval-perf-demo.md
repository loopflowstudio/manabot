# Evaluation performance demo — 2026-10-24

Jack Heart: “im having trouble interpreting the flame charts. dont know what cell
or workers mean”; “i basically want like inference vs game engine vs ???”.

The report now defaults to activities during games: Inference, Game engine,
Evidence and replay, and Other and unmeasured. Clicking a category exposes its
retained phases. Raw resource keys and receipts stay unchanged; visible controls
say During games, Comparison elapsed, Game CPU usage and Inference service.
Each view explains its clock and overlap. Inference round-trip includes queueing
and communication, not pure GPU computation. Replay is measured in the comparison
view, not invented in the game-process view. Grouping follows coverage-aware seed
aggregation so missing phases do not become zero. No overlapping resource sums
are combined into one wall-time partition.

Demo: http://127.0.0.1:8767/report.html, built at .runs/eval-perf-demo/report.html
from the retained offline four-game CPU proof and deliberately Command-capped
checkpoint-pair fixture under .runs/local-eval-proof. These are not matched
speedup evidence. Jack Heart accepted the revised overview: “Yes, keep this overview”. Activity categories remain the default; detailed phases stay behind drill-down.

Other observed report rough edges remain: missing timestamps render epoch dates,
training-oriented axis controls appear for evaluation-only reports, empty eval
sections say vs undefined, and mixed complete/incomplete input says Completed.
These are findings, not a selected report redesign.

Check: ./scripts/verify-eval-performance-report passed with activity overview,
phase drill-down, coverage, normalization, timeline and evidence-link checks.
Design: [shared evaluation](batched-eval-inference.md). The new pursue continues
independently with evaluation continuation; this demo does not restart that Graph.
