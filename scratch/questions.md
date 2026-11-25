# Questions and assumptions (MTG-437)

Choices made with nobody in the conversation (each documented in
`scratch/concept-review.html` and reversible):

- **`ObservationBatch` stays sealed in the public API:** pickle uses
  `__reduce__` -> `managym._restore_observation_batch(layout, batch_size, tables)`
  (with full schema + per-span capacity validation) rather than a public
  `ObservationBatch(layout, batch_size, tables)` `#[new]`. Keeps `admit_native`'s
  native-certificate contract honest and avoids locking a raw-bytes builder into
  `0.2.6`.
- **`ObservationSpace.__reduce__` reconstructs from `self.config`** (627 KB ->
  ~350 B per worker payload). Verified that checkpoints store `ObservationConfig`
  JSON in safetensors metadata rather than pickling `ObservationSpace`, and
  `__reduce__` does not affect loading any historical default-pickled space.
- **Shim attributes live on `ObservationSpace` via a warned `__getattr__`**,
  not on a proxy returned by `encoder`, so `space.encoder is space` stays true
  and `ObservationEncoder(config).max_actions` works too. Covers the whole
  pre-#566 attribute set (`library_shape` from `V11_LAYOUT` so v9/v10 works,
  and all six `num_*` counts).
- **`RolloutBatch.concatenate` joins packed `[rows, 1]` batches only.**
  Multi-column `[time, trajectory]` batches raise (even in a 1-element list).
  Snapshot batches skip zero-bootstrap parts when joining `next_obs` and
  re-offset non-negative `branch_group_id`s across workers to prevent RLOO
  collisions.
- **Prefixes pickle through `Chunk`/`Tail`/`Prefix.__reduce__` (plus
  `PrefixObservation`/`CompactObservation` and `ObservationSelection.__reduce__`)**,
  not through `MemoryArchive` per holder. `MemoryArchive` stays as the explicit
  storage form.

- Reuse the native layout's existing constructor arguments for all pickle
  protocols. The pre-change layout round-tripped with protocols 2–5, but
  protocols 0 and 1 raised `TypeError`. Cache the resolved packed strides on
  that layout so each gathered or restored batch uses the encoder's row size.
- The worker regression launches a clean `ChildProcess` to own the consumer's
  spawn queue. This exercises the actual queue feeder and keeps pytest's
  main module out of the collection worker.

Outside this work (future follow-ups):

- `collect_pair()[0]` retains both learners' recorded rows in `recorded.rows`
  because `_build_complete_batch` shares one `RecordedSpans` across learners.
  Compacting per-learner spans in `collect_pair` would halve IPC and RAM as its
  own Task.
- jace still merges densely (`_merge_recorded`). After `0.2.6` it can call
  `RolloutBatch.concatenate` and drop `pack_batch_for_ipc`; that is a jace
  change.
