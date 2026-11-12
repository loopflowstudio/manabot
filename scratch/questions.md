# Open questions (MTG-318)

- Decision rule thresholds were not specified beyond the brief's example
  ("at least 1.5x faster at equal outputs within 1e-5"). Assumed: 1.5x on
  calls of <= 8 rows at both sizes, no >5% regression for 9-32 rows, 1e-5 max
  abs on logits/values/log-probs, and lower projected whole-collection time
  including capture/compile cost over 100 updates. Reversible before scored runs.
- If a path is accepted, the `Execution` field name and whether it may combine
  with `bucketed_shapes=False` are undecided (assumed: it requires bucketed shapes).
