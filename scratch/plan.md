# MTG-318 plan

## Docs (write-docs step)

- `experiments/collection_tail_graphs.md`: pre-registered question, prediction,
  refutation, USD 5 cap and decision rule, plus the arms and measures. Commit it
  before any scored run; results, chart, cost, limits and the appendix go below it.
- No user-facing doc edits yet: an `Execution` switch exists only if the decision
  rule accepts a path. If one is accepted, document it in
  `docs/observation-and-model.md` beside "Bucketed padding (profiling)" and in
  `docs/training.md` beside `execution.bucketed_shapes`.
