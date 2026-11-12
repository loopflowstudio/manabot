# Live-token attention against bucketed shapes on an L4 — MTG-319

**Committed before any scored run (2026-11-12).** Question: does attending over
valid objects only (`Execution.live_tokens`, packed jagged attention) beat
bucketed shapes enough to build out as the layout for larger models? Prediction:
at w384l8, with bucketed shapes on in both arms, live tokens take the learner
step to about 1/1.3 of bucketed's time (bucket padding is roughly half the
object slots, and attention blocks are most but not all of a step), so the
pre-registered bar is **not** met. The result that refutes the prediction and
passes the bar: `bucketed-live` learner step ≤ 1/1.5 of `bucketed` at w384l8,
with logits, values, losses and gradients equal to the padded path within float
tolerance. Cost cap: USD 3 for the L4 job (Task budget USD 10).

## Method

`experiments/live_token_attention.py` launches MTG-301's profile
(`experiments/l4_step_profile.py`, candidate `live`) on one rented L4: one
in-situ w160l8 update of the frozen MTG-267 regime keeps collection observations
and learner minibatches; they replay through `baseline`, `bucketed`, `live` and
`bucketed-live` in separate processes at w160l8 and w384l8, interleaved with
baseline repeats. Each arm writes fixed-weight numerics compared with baseline.

The layout is described in
[live-token attention](../docs/observation-and-model.md#live-token-attention-profiling).
On CPU, `tests/model/test_live_tokens.py` checks logits, values and gradients
against the padded path within 1e-5, alone and over buckets, with masked-mean
and value-token critics, and that collection packs with its host-measured count.

## Results

Not yet run.
