# manabot.policy

This package defines the policy contract shared by every backend. It prepares
host inputs with NumPy and managym. `tests/test_package_boundaries.py` checks
that it stays independent of PyTorch, MLX and JAX.

```python
from manabot.backends import resolve

backend = resolve("torch")
policy, space = backend.load_policy("policy.safetensors")
batch = policy.prepare(obs, inference=True)  # check, trim, index: a Prepared batch
logits, values = policy.evaluate(batch)      # NumPy: logits [B,A], values [B]
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
  remapping object references to the trimmed table. Native scans measure the
  last occupied row and furthest referenced row in each object segment for
  each batch item. The policy supplies the columns, bases and segment bounds.
- `prepare.py` turns one admitted observation batch into a `Prepared` batch:
  definition rows, record-vocabulary lookups, legal-action coordinates, packed
  GRU schedules, shared-effect routes, belief and recent-event rows, live-token
  offsets and inference column compaction. Object joins and ownership follow
  the batch's trimmed object axes. A backend's graph then holds only
  gathers, scatters and arithmetic. `Prepared` is generic over its array type:
  `prepare` returns NumPy arrays and a backend moves them to its device
  (torch's `on_device`). Optional groups are `None` when unused: a fact
  table's `programs` holds its indexes together, and each `Packed` schedule
  owns the original sequence lengths. The batch records the
  model and observation configuration it was prepared for, and a forward
  refuses a batch prepared for another.
  `reference_distribution` is the magnet a reverse KL pulls toward, `[B,A]`
  float32 from the legal-action buffers.
  `Layout` holds the reading of every column of every buffer the model
  reads: `as_is`, `log`, `rank`, `gather`, `lookup`, `mask`, `fact_value`,
  `fact_id`, `duration`, `own` (decoded by the buffer's own preparation), or
  one of the sized readings `one_hot:n` (n + 1 flags) and `steps:n` (the value
  and n threshold flags), and for stored checkpoints only `fact_value_raw` and
  `duration_two`. A reading states how many inputs its column becomes, and a
  buffer's numeric width is the sum over its columns. A fresh
  model derives them from the kinds `managym.observation_layout()` declares; a
  loaded one takes them from its checkpoint's `input_readings`, or
  `legacy_readings(space)` when the checkpoint predates the record. `prepare`
  applies each numeric reading once, so a backend never scales a column, and
  admission keys include the readings, so two models that differ in one
  reading never share a prepared batch. A column whose kind has no reading
  raises when the layout is built; joins and ids never reach a projection. A
  reading may write more than one input (`duration`: four flags), so
  projection widths come from `Layout` (`numeric_width`, `fact_width`)
  ([how the network reads a column](../../docs/observation-and-model.md#how-the-network-reads-a-column)).
- `checkpoint.py` is the checkpoint container: one safetensors file of
  canonical float32 parameters, lookup buffers and the checkpoint's
  configuration, world binding and source record, for every configuration
  the torch policy runs. `read_container` validates the model interface and
  binds world policies through `binding.py`. Stored card programs keep their
  learned tokens while new definitions read the running catalog. The loader
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
buffer name, and `ParameterArrays` (`manabot.policy.checkpoint`), named
parameters or gradients in canonical or backend layout.

What stays in a backend's graph is every float operation whose rounding
differs between NumPy and the framework. `prepare` does integer bookkeeping,
masks, casts and the column readings; `Prepared` carries raw known-card
counts and signed program integers, so the product with a count and the
`log1p` of a program number are computed by the backend. How those counts
are scaled and pooled belongs to the model (`ModelConfig.stream_norm`), not to
a reading.

Backends live in [`manabot/backends/`](../backends/README.md), which also
covers [saving and loading a container](../backends/README.md#policy-checkpoints).

`decision_memory.py` validates dense history through record-aligned chunks and a
five-integer predecessor cursor. Retained prefixes reuse these content-bound
certificates and vocabulary-bound immutable feature fragments. Record features
are prefix-independent; preparation adds sequence coordinates and batch routes without gathering dense history. The
cache owns no trainable output; unreachable game nodes release their features.
