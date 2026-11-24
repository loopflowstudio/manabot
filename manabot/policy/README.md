# manabot.policy

This package defines the policy contract shared by every backend. It prepares
host inputs with NumPy and managym. `tests/test_package_boundaries.py` checks
that it stays independent of PyTorch, MLX and JAX.

```python
from manabot.backends import resolve

backend = resolve("torch")
policy, space = backend.load_policy("policy.safetensors")
batch = policy.prepare(obs, inference=True)  # admitted and indexed on the host
logits, values = policy.evaluate(batch)      # NumPy: logits [B,A], values [B]
```

Every backend's forward reads a `Prepared` batch. With bucketed shapes,
Torch and MLX read exact counts from a native v11 `ObservationBatch` or measure occupied and referenced rows in dense mappings before admission. The
resulting `BucketedObservation` pairs views of the original rows with their extents.
Host preparation uses those extents to translate object joins and restore
the full action width.

- `admission.py` owns the checks an observation batch passes before a policy
  reads it: shapes and binary validity, definition and vocabulary references,
  fact programs, shared-effect partitions, continuation, naming and action
  joins, permanent links, recent events and belief rows. Both backends call
  `admit` on the host and prepare the views it validated.
  Malformed input raises `ValueError` naming the check; nothing is truncated
  or repaired.
- `spans.py` reads immutable packed values and offsets for the native
  batch. Admission binds the batch layout and checks checkpoint vocabularies.
  Preparation uses the offsets for slot coordinates and restores the model's
  padding where its arithmetic depends on shape. A copied or read-only
  dictionary goes through the full dense checks.
- `buckets.py` chooses the sizes bucketed shapes trim a batch's variable
  axes to (`bucket_sizes`). The native `ObservationLayout.axis_extents`
  measures each item's occupied rows, object references, continuation joins
  and effect spans. `bucket_observation` reads the buffer names and row dimensions
  from `ObservationLayout.axis_buffers` and slices in original coordinates.
  `object_map` supplies the lookup for joins into the trimmed object table.
- `prepare.py` turns one admitted observation batch into a `Prepared` batch:
  definition rows, record-vocabulary lookups, legal-action coordinates, packed
  GRU schedules, shared-effect routes, belief and recent-event rows, live-token
  offsets and attention/fact compaction. Object joins and ownership follow
  the batch's trimmed object axes (read from `obs.buckets` on
  `BucketedObservation` or `ObservationSpans`, with no separate `bucketed`
  flag). A backend's graph then holds only gathers, scatters and arithmetic.
  `Prepared` is generic over its array type: `prepare` returns NumPy arrays and
  a backend moves them to its device (torch's `on_device`).

  Backends translate their device and purpose (`inference=True` vs learning)
  into two explicit layout parameters on `prepare(obs, tables, *, attention="padded", facts="padded", ...)`:

  ```python
  from manabot.policy.prepare import prepare

  # Admitted batch `obs` and checkpoint `tables`; pad used object columns to a power of two.
  batch = prepare(obs, tables, attention="bucketed_columns", facts="padded")
  batch.attention_columns  # sorted column indices of power-of-two length (>= 8), or None if dense
  ```

  - `attention: AttentionLayout`: `"padded"` (default, full object table),
    `"used_columns"` (only object columns valid in at least one row; Torch CPU
    and MLX inference), `"bucketed_columns"` (used object columns padded with
    empty columns via `buckets.bucket` to the next power of two, at least
    `MIN_BUCKET = 8`, collapsing to `None` when empty, when width ≤ 8, or when
    the bucket reaches the object width; Torch MPS inference),
    `"packed_groups"` (groups sorted by live count in `packed_attention`, with
    used columns as the fallback for groups that cannot be packed), or
    `"live_tokens"` (flat `live_index` and `live_offsets` for Torch CUDA and
    MLX, plus `compact_attention` coordinates and a key mask for MLX).
  - `facts: FactLayout`: `"padded"` (default), `"used_columns"` (only fact
    columns valid in at least one row; unbucketed CPU and MLX inference), or
    `"live_rows"` (live fact rows only). MLX learner compaction requires
    bucketed shapes and inert object padding.

  Unknown `attention` or `facts` values raise `ValueError`. Both
  `ObservationSpans` and dense buffers share `select_attention_columns` and
  produce identical column selections
  ([inference column compaction](../../docs/observation-and-model.md#inference-column-compaction)).
  Optional groups are `None` when unused: a fact
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
