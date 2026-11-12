# MTG-300 plan

Result: measured, no code change. The Task's stop condition holds, so the
README paragraph on a per-weights-version learning cache was reverted.

Measurement (`scratch/measure_encodes.py`, CPU, 1 thread, world
`managym.WORLD_SETUP`, 360 catalog programs, default `Match`, 8 streams,
1584-row batch, torch CPU profiler share of one `update_iteration`):

| path | optimizer steps | grad encodes (referenced rows) | no-grad encodes (full) | encode fwd share |
|---|---|---|---|---|
| damped policy | 198 | 198 (792 rows) | 1 (360 rows, bootstrap) | 6.5% grad + 0.8% no-grad |
| PPO surrogate (2 epochs x 4 minibatches) | 8 | 8 (32 rows) | 1 (360 rows, bootstrap) | 0.3% grad + 0.7% no-grad |

- Every gradient-enabled forward is followed by exactly one optimizer step,
  so each weights version gets exactly one gradient encode; it cannot be shared.
- The only no-grad learner forward is the bootstrap/targets forward, once per
  update (`_targets` / `get_value`). No second no-grad forward runs under the
  same weights, so there is nothing to share.
- Batch health (`batch_health`) reads collection's packed action data, and
  minibatch diagnostics reuse the step's logits: neither runs a forward.
- Collection already shares one table per `PolicyInference` scope.
- Default `Match` decks reference ~4 rows per minibatch; larger decks raise the
  gradient-encode share but not the counts.

Remaining overlap, out of this Task's scope: the bootstrap's full encode
(~0.7-0.8% of an update) recomputes the table that collection's
`PolicyInference` scope built under the same behavior weights and then
dropped on scope exit.
