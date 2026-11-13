# manabot.policy

The policy contract every framework backend shares, and the host work that
does not depend on a framework. **This package never imports an array
framework** (PyTorch, MLX, JAX); it uses NumPy only
(`tests/test_package_boundaries.py` enforces it).

```python
from manabot.backends.torch.checkpoint import load_checkpoint

policy, space = load_checkpoint("policy.safetensors")
batch = policy.prepare(obs)              # check, trim, index: a Prepared batch
logits, values = policy.evaluate(batch)  # NumPy: logits [B,A], values [B]
```

`Prepared` is the one policy input. A policy's `prepare` runs this package's
host work in order (admission, bucket trimming, indexing) and every backend's
forward reads the result; no forward reads an observation dictionary.

- `admission.py` owns the checks an observation batch passes before a policy
  reads it: shapes and binary validity, definition and vocabulary references,
  fact programs, shared-effect partitions, continuation, naming and action
  joins, permanent links, recent events and belief rows. `check(obs, spec,
  legal)` runs them all, and every backend calls it on the raw rows.
  Malformed input raises `ValueError` naming the check; nothing is truncated
  or repaired.
- `buckets.py` chooses the sizes bucketed shapes trim a batch's variable
  axes to (`bucket_sizes`) and trims the NumPy rows (`trim_observation`),
  remapping object references to the trimmed table.
- `prepare.py` turns one admitted observation batch into a `Prepared` batch:
  definition rows, record-vocabulary lookups, legal-action coordinates, packed
  GRU schedules, shared-effect routes, belief and recent-event rows, live-token
  offsets and inference column compaction. A backend's graph then holds only
  gathers, scatters and arithmetic. `Prepared` is generic over its array type:
  `prepare` returns NumPy arrays and a backend moves them to its device
  (torch's `on_device`). Optional groups are `None` when unused: a fact
  table's `programs` holds its indexes together, and each `Packed` schedule
  owns the original sequence lengths. The batch records the
  model and observation configuration it was prepared for, and a forward
  refuses a batch prepared for another.
  `reference_distribution` is the magnet a reverse KL pulls toward, `[B,A]`
  float32 from the legal-action buffers.
- `checkpoint.py` is the checkpoint container: one safetensors file of
  canonical float32 parameters, lookup buffers and the checkpoint's
  configuration, world binding and source record, for every configuration
  the torch policy runs. `read_container` validates the world binding and
  refuses files that are not containers (`NotAContainer`); `read_metadata`
  reads only the header. `UnsupportedPolicy` is what a backend raises for a
  configuration it does not implement.
- `returns.py` holds the returns and advantage-selection recursions over host
  NumPy rows (`transition_gae`, `selection_mask`, `quantile`,
  `terminal_distances`). The update diagnostics read them and every backend's
  `transition_gae` wraps them; torch's device-side selection matches them bit
  for bit.
- `protocol.py` states what a backend provides (`BackendPolicy`,
  `BackendLearner`, `Backend`) and one captured minibatch's constants and
  coefficients (`Targets`, `Settings`). Values are `[B]`.

Two type aliases name the NumPy mappings that cross this boundary:
`ObservationBuffers` (`manabot.env.observation`), an observation batch by
buffer name, and `ParameterArrays` (`manabot.policy.checkpoint`), parameters
by canonical name.

What stays in a backend's graph is every float operation whose rounding
differs between NumPy and the framework. `prepare` does integer bookkeeping,
masks, casts and division by a constant; `Prepared` carries raw known-card
counts and signed program integers, so the product with a count and the
`log1p` of a program number are computed by the backend.

Backends live in [`manabot/backends/`](../backends/README.md), which also
covers [saving and loading a container](../backends/README.md#policy-checkpoints).
