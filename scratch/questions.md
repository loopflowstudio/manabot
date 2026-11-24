# Questions (choices made without a person present)

See `scratch/concept-review.html` for full tradeoffs and numbered review questions.

- **Fold `reassign_roots_into*` into `SlotLoad.fresh(slot, player_configs=None)`:**
  Deleted `reassign_roots_into` / `reassign_roots_into_buffers` instead of
  widening it to accept parked slots. Patching `reassign_roots_into` failed
  counterexamples on mixed seed semantics, stale pre-call seeds in
  `RecordedVectorEnv`, and `Playing` slots under `action_limit`.
- **Uniform `BufferedStreams.load_slots_into_buffers` and `open_native_streams`:**
  Added `load_slots_into_buffers` to `BufferedStreams` / `ShardedVectorEnv`
  (using `SlotLoad.with_slot` and `SlotLoadResult.with_slot`) and made
  `open_native_streams` always return `BufferedStreams`. Kept the existing
  `workers != 1` guard for adaptive collection in `open_collector` out of scope
  for this PR.
- **Unify `TargetPracticeVector` on explicit refill:**
  Parked terminal rows in `TargetPracticeVector` and added
  `load_slots_into_buffers` so `Collector` has zero eager-reset branches.
- **Torch `manabot.env.vector_env.VectorEnv`:**
  Keeps Gym-vector automatic reset explicitly inside the wrapper by saving
  step-state tensors and terminal `last_info` before calling
  `load_slots_into_buffers([SlotLoad.fresh(s) ...])`.
- **Readable `VectorEnv.step`:**
  Parks terminals too; readable callers restart with `reset_all`. No readable
  `load_slots` was added.
- **No deprecated shims:**
  No deprecated aliases for `auto_reset`, `replace_failed_into_buffers`, or
  `reassign_roots_into_buffers` (pre-1.0).

- **Sharded load admission:** one native call borrows all participating
  vectors and admits all replacements before publishing any. The coordinator
  runs sparse loads on the calling thread. Independent worker commits could
  spend a seed and publish a recording on the first shard even when the
  second shard rejected a setup or observation.

- **Diagnostic roots:** Target practice accepts only a fresh load of its
  configured match. The read-only `SlotLoad.uses_slot_setup` distinguishes
  that request from a setup override or a snapshot before any root is built.
- **Collector refill:** Plain and adaptive games share one pending refill set.
  The collector finalizes pending learner decisions before checking the active
  mask, since finalization can itself satisfy an exact quota.

- **Benchmark timing:** Count explicit slot loads alongside native steps.
  Hash terminal rewards and flags before loading the next batch.
