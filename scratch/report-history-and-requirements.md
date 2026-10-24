# Two-seat report review — 2026-10-22

Requested here: explain the absent graphs; review prior Manabot reports and Etude prototypes; consider W&B, Neptune, ClearML, Tinker and Prime; restore filterable views, training performance, heuristic curves with a baseline and matchup breakdowns.

## Finding

The current live_evaluation.report constructs tables and embeds etude.eval.report.render, itself a table/matrix report. It calculates curves and writes curves.json, but draws none. head_to_head.report independently draws only tables. This is a composition failure: the richer renderers survive and were bypassed, not deleted by the experiment-data migration.

The deployed Etude _training_order already sorts recorded checkpoint ordinals, then natural checkpoint names. Here every point has ordinal=null, clocks={}, run=null, and a digest filename. Natural sorting numeric chunks inside a hash produces the bizarre order. It is not chronological and is not strictly alphabetical. Manabot retained coordinates in its separate observer ledger but did not supply/link the training checkpoint records to Etude's source.

Immediate correction: sorted the live heuristic table by numeric update, verified HTTP order 250,500,750,1000,1250,1500,1750,2000,2053. Original page retained outside public/. Local source fix in manabot/remote/live_evaluation.py; Ruff and the ten Etude adapter checks passed. This is a presentation-only correction; no evaluator, controller, worker, recipe, rental or score was changed. New graphs are not yet deployed.

## Existing versions and useful parts

| Date | Version/evidence | What existed |
| --- | --- | --- |
| Oct 6 | Manabot editable report notebooks; c5c455263, 923595631, 6a0dfef4e; docs/evidence/history-report-learning-2026-10-06.md | Checkpoint learning curves, per-seed histories, matched-transition/cost comparisons, explicit missing initialization and diagnostic limitations; create-once notebook generating read-only HTML. |
| Oct 6 | History-input W&B report in the same evidence note | Native per-seed charts, recorded-cost axes, paired effect and bounds, selectable W&B views. Explicit publication; original reports preserved. |
| Oct 7–8 | Deck-specific Manabot learning dashboard, 27f724a06; docs/evidence/learning-dashboard-2026-10-08.md | Strength-first page, SVG deck curves, training-step and hours axes, freshness/progress, checkpoint table and expandable diagnostics/evidence. Core still in manabot/training/learning_dashboard.py. |
| Oct 8–11 | Etude EXP-2 hosted research prototype, etude-studio commit 64886dd | Experiment/research map, jobs/progress, evidence provenance and counters. Useful navigation/model context, not itself a replacement for metric plots. Prototype explicitly retired/minimized; preserved in git. |
| Oct 11 | Etude experiment-api-prototype-full tag 4131e08; scratch/report-demo/capacity-pilot.html | Offline interactive uPlot page, clock switch, run visibility, drag zoom, seed aggregates and evaluation intervals. Accepted controls recorded in decisions 70–75; missing-clock behavior in 76–77. |
| Oct 13–15 | Successor Etude report, now src/etude/train/report.{py,html} | Self-contained vendored uPlot, stable run colors, URL-preserved clock/visibility/zoom-related view, regenerate/reload; multi-attempt/continuation history. Current source also links evaluation results to exact recorded training checkpoints via etude/reports.py. |
| Oct 15–16 | Manabot training phase/cohort reports, ae810205b, 941079fcc, edf6bb0d7 | Setup/collection/learning/export/diagnostic/unattributed accounting, attempt timelines, cohort selection, non-overlapping recovery accounting. Core time_profile.py and time_cohort.py survive. |
| Oct 21–22 | Two-seat live_evaluation.py and head_to_head.py | Sound retained evaluation/attempt records and score intervals, but table-only pages without the previous interactive/chart/performance composition. |

Local Etude design record: /Users/jack/src/etude-studio/docs/design/decisions.md. Historical prototypes are evidence, not current implementation instructions.

## External prior art consulted

- W&B: run filters and numeric axis selection. https://docs.wandb.ai/guides/runs/filter-runs and https://docs.wandb.ai/guides/app/features/panels/line-plot/reference
- Neptune: shared chart controls, metric/widget filtering, selected runs, x-series choice, consistent color/line conventions. https://docs.neptune.ai/configure_report and https://docs.neptune.ai/select_runs
- ClearML: scalar/plot overlays and task comparison. https://clear.ml/docs/latest/docs/webapp/webapp_exp_comparing/
- Tinker: training utilization, sampling/training throughput, request latency percentiles, in-flight work, trace drilldown. https://tinker-docs.thinkingmachines.ai/tinker/session-metrics/
- Prime RL: separate progress, stability and performance metrics; per-environment rewards/evals; trainer wait-for-batch versus orchestrator wait-for-policy to identify the bottleneck. https://github.com/PrimeIntellect-ai/prime-rl/blob/main/docs/training.md

These are design references; they do not establish that our frozen run recorded every corresponding metric.

## Report requirements from this conversation

1. A shared selection state for runs/control, matchup, candidate deck, play/draw, checkpoint range and chart/metric search; stable colors and URL-persisted choices across refresh. Chart filters and table filters must agree.
2. Strength first: two-seat and the retained one-seat control against the SAME heuristic on one plot, sorted by training update; switch to learner decisions and recorded training hours. Matchup small multiples: Lessons vs Allies, Allies vs Lessons, Lessons mirror, Allies mirror, plus an explicit aggregate. Whole-deal intervals, coverage and source checkpoint per point. Missing baseline milestones are gaps, never fabricated points or a substituted 50% line.
3. Keep direct head-to-head separate: same matchup controls, 50% reference appropriate here, W/D/L, uncertainty and both arms' actual cost/data coordinates.
4. Training-performance section: phase totals and per-update trends, collection/learning/export/diagnostic time, throughput and resource utilization where actually recorded. Drill into environment, inference, transfers and optimizer only where non-overlapping measurements exist. Distinguish training throughput from playing strength. Show per-matchup game counts/lengths/decisions and available throughput/timing; explicitly label historical per-matchup timing unavailable when not recorded. Never divide aggregate GPU/learning time evenly among matchups.
5. Status/freshness: distinguish page generation, source heartbeat and last completed evaluation; paused/uncertain continuation must remain obvious. Separate reserved budget, incurred estimates and settled costs.
6. Human labels (updates-2053) separate from immutable digest identity. Link the exact checkpoint declaration/coordinates into Etude; do not infer chronology from artifact names or digests. Natural sorting already exists in Etude, so preserve it and repair the missing join rather than duplicate an upstream sorter.
7. Keep familiar typed data/report owners: reuse Etude interaction/curve capabilities and Manabot's game-domain matchup/time accounting. Avoid another independent training/evaluation/controller. Test reverse digest order, chronological order, filters, baseline gaps, all four candidate-relative matchups, continuation accounting, freshness and browser behavior.

Implementation of the full report remains separate from this research and the immediate table hotfix. No new training/evaluation or paid work is needed to render retained evidence.

## Experiment prototype delivered

Built `.runs/two-seat-report-prototype/` from the Mini's retained native rows, dashboard observations, direct comparison records and the digest-verified published one-seat baseline. Review URL: http://mini-heart.local:8766/prototype/ . The page has actual uPlot curves, independent run toggles, all four matchup panels, play/draw filters, three clock axes, update range, chart search, linked drag zoom, sparse baseline expansion, training phase accounting/trends and a sorted evidence table. It is explicitly a snapshot. Per-matchup training timing and GPU utilization remain unavailable, while per-matchup CPU evaluation length/time are labeled separately.

Verified the served page's chart values, separate baseline endpoint, filters, URL persistence, linked zoom, mobile layout and zero browser errors with headless Chrome. Existing training/evaluation/controller processes and original report are unchanged. No PR was created; the next conversation determines which pieces belong in Manabot and Etude.
