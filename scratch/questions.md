# Open questions (MTG-318)

- Decision rule thresholds were not specified beyond the brief's example
  ("at least 1.5x faster at equal outputs within 1e-5"). Assumed: 1.5x on
  calls of <= 8 rows at both sizes, no >5% regression for 9-32 rows, 1e-5 max
  abs on logits/values/log-probs, and lower projected whole-collection time
  including capture/compile cost over 100 updates. Reversible before scored runs.
- If a path is accepted, the `Execution` field name and whether it may combine
  with `bucketed_shapes=False` are undecided (assumed: it requires bucketed shapes).
- The `l4-etude` template is 6 vCPUs / 48 GB; MTG-301 ran on a 48-CPU EPYC
  host. Host-bound timings may not compare across the two; the report reads
  every arm against its own job's baseline only.
- The compiled arm uses plain `torch.compile(mode="reduce-overhead")` with
  default dynamo flags (as the brief names it); the probe's explain sets
  `capture_scalar_outputs`/`capture_dynamic_output_shape_ops` to show breaks
  beyond the first.
- The retained MTG-301 weights no longer load (the card pool grew), so the run
  re-captures inputs and initial weights on the current world.
