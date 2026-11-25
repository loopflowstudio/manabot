# manabot.policy

This package defines the policy contract shared by every backend. It prepares
host inputs with NumPy and managym. `tests/test_package_boundaries.py` checks
that it stays independent of PyTorch, MLX and JAX.

```python
import managym
from manabot.backends import resolve

backend = resolve("torch")
policy, space = backend.load_policy("policy.safetensors")
obs_batch = managym.encode_observation_batch([obs], space.layout)
batch = policy.prepare(obs_batch, inference=True)  # admitted and indexed on the host
logits, values = policy.evaluate(batch)            # NumPy: logits [B,A], values [B]
```

Every backend's forward reads a `Prepared` batch produced from a native
`managym.ObservationBatch` or an `ObservationSelection` (with optional
`RetainedMemory`). Belief-enabled policies pass explicit
`belief=belief_view.rows()` alongside `obs_batch`. With bucketed shapes, Torch
and MLX read exact per-axis counts from the batch's CSR offsets before
admission. Host preparation uses those extents to translate object joins and
restore the full action width.

- `admission.py` owns the checks an observation batch passes before a policy
  reads it (`admit(obs, spec, *, bucketed, belief=None) -> ObservationSpans`):
  layout binding, definition and vocabulary references, decision-memory
  contracts (`check_memory_contract`), and explicit `BeliefRows` validation
  (`beliefs`). Every `ObservationBatch` already certifies its v11 CSR row
  offsets, capacities and native joins on construction (live encoding or
  validated `ObservationBatch.from_bytes`). Both backends call `admit` on the
  host and prepare the `ObservationSpans` it returns. Malformed input raises
  `ValueError` naming the check; nothing is truncated or repaired.
- `spans.py` defines `type PolicyObservation = managym.ObservationBatch | ObservationSelection`
  and reads immutable packed values and offsets from native batches.
  `ObservationSelection` references rows across one or more batches (with
  optional `RetainedMemory`) without masquerading as a dict mapping; use
  `.project(key)` for inspection or tests. Preparation reads directly from
  `RowSpans` and restores the model's padding where its arithmetic depends on
  shape.
- `buckets.py` chooses the sizes bucketed shapes trim a batch's variable
  axes to (`sizes_of`, `bucket`, `BucketSizes`, `object_trimming`), reading
  per-axis extents directly from CSR offsets. `object_map` supplies the lookup
  for joins into the trimmed object table.
- `prepare.py` turns admitted `ObservationSpans` (plus optional `BeliefRows`)
  into a `Prepared` batch: definition rows, record-vocabulary lookups,
  legal-action coordinates, packed GRU schedules, shared-effect routes, belief
  and recent-event rows, live-token offsets and attention/fact compaction.
  Object joins and ownership follow the batch's trimmed object axes (read from
  `spans.buckets`, with no separate `bucketed` flag). A backend's graph then
  holds only gathers, scatters and arithmetic. `Prepared` is generic over its
  array type: `prepare` returns NumPy arrays and a backend moves them to its
  device (torch's `on_device`).

  Backends translate their device and purpose (`inference=True` vs learning)
  into two explicit layout parameters on `prepare(spans, tables, *, attention="padded", facts="padded", belief=None, ...)`:

  ```python
  from manabot.policy.prepare import prepare

  # Admitted `ObservationSpans` and checkpoint `tables`; pad used object columns to a power of two.
  batch = prepare(spans, tables, attention="bucketed_columns", facts="padded")
  batch.attention_columns  # sorted column indices of power-of-two length (>= 8), or None if dense
  ```

  - `attention: AttentionLayout`: `"padded"` (default, full object table),
    `"used_columns"` (columns valid in at least one row; Torch CPU and MLX
    inference), `"bucketed_columns"` (used columns padded with empty columns
    via `buckets.bucket` to a power of two; Torch MPS inference),
    `"packed_groups"` (rows sorted by live count in `packed_attention`, falling
    back to `used_columns`), or `"live_tokens"` (flat `live_index` /
    `live_offsets`, plus MLX's `compact_attention`). When `belief` is present,
    all attention-compaction modes include belief rows and the belief global
    token.
  - `facts: FactLayout`: `"padded"` (default), `"used_columns"` (columns valid
    in at least one row for unbucketed CPU and MLX inference), `"live_rows"`
    (live fact rows for MLX learner compaction), or `"bucketed_live_rows"`
    (live facts padded to a power of two of at least 64 for Torch MPS).

  Unknown layouts raise `ValueError`. All attention-compaction layouts share
  `select_attention_columns`
  ([inference column compaction](../../docs/observation-and-model.md#inference-column-compaction)).
  Optional groups are `None` when unused: a fact
  table's `programs` holds its indexes together, and each `Packed` schedule
  owns the original sequence lengths. The batch records the
  model and observation configuration it was prepared for, and a forward
  refuses a batch prepared for another.
  `reference_distribution(obs, kind)` is the magnet a reverse KL pulls toward,
  `[B,A]` float32 computed directly from CSR `actions` spans and `action_counts`.
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
  skips all-zero card, permanent and stack rows when applying numeric readings.
  It reads relation features at valid slots and restores the original row
  coordinates for every backend. The host applies the readings once. Admission
  keys include them, so two models that differ in one reading never share a
  prepared batch. A column whose kind has no reading raises when the layout
  is built; joins and ids never reach a projection. A
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
- `protocol.py` defines host policy inference, checkpoint readers and loaders,
  and one captured minibatch's constants and
  coefficients (`Targets`, `Settings`). Values are `[B]`.

Two type aliases name the host inputs and parameter mappings that cross this
boundary: `PolicyObservation` (`manabot.policy.spans`, a native
`managym.ObservationBatch` or `ObservationSelection`) and `ParameterArrays`
(`manabot.policy.checkpoint`, named parameters or gradients in canonical or
backend layout).

What stays in a backend's graph is every float operation whose rounding
differs between NumPy and the framework. `prepare` does integer bookkeeping,
masks, casts and the column readings; `Prepared` carries raw known-card
counts and signed program integers, so the product with a count and the
`log1p` of a program number are computed by the backend. How those counts
are scaled and pooled belongs to the model (`ModelConfig.stream_norm`), not to
a reading.

Backends live in [`manabot/backends/`](../backends/README.md), which also
covers [saving and loading a container](../backends/README.md#policy-checkpoints).

`decision_memory.py` validates native decision-memory CSR tables and interned
`RetainedMemory` prefixes through record-aligned chunks and a five-integer
predecessor cursor. Retained prefixes reuse these content-bound certificates
and vocabulary-bound immutable feature fragments. Record features are
prefix-independent; preparation adds sequence coordinates and batch routes
without gathering dense history. The cache owns no trainable output;
unreachable game nodes release their features.
