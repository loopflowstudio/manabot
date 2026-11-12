# MTG-328 open choices (reversible; simpler option taken in the docs)

1. CLI: the docs replace `--bucketed-shapes` with `--padded-shapes` (both
   switches off on CUDA) and keep `--bucketed-shapes` as a deprecated,
   warning no-op alias removed in 0.3.0. Alternative: drop the flag outright.
2. Evaluation layout is an `EvalAllocation` setting (`bucketed_shapes`,
   `live_tokens`, default `auto`), retained in attempt receipts and kept out of
   `EvalCompute` identity, on the grounds that it changes only summation order,
   like batch composition. Alternative: put it in `EvalCompute` (new CUDA
   player/benchmark identities).
3. `auto` resolves off for non-`train_self_play` stages; explicit
   `bucketed_shapes=True` stays allowed on CPU/MPS (only `live_tokens=True`
   is refused off CUDA, per the steer).
4. `ActorCritic(...)` and `load_checkpoint(...)` keep defaulting both off; only
   regimes and evaluation allocations resolve `auto`.
5. MTG-324 integration remains open. This checkout has explicit precision,
   with no `_from_ladder`. Reuse `TrainingRegime.authored_dump` and
   `_read_saved_stage` when combining precision and layout defaults; layout
   provenance now uses Pydantic's field-set tracking, with no private set.
6. An `EvalExecution` read back without the layout fields (receipts written
   before this change) resolves them from the device, so an old CUDA receipt
   reads as layout on. Kwargs and JSON omission are indistinguishable at the
   validator; left as is since no reader replays old receipts' layout.
7. Fresh CUDA regimes built by `experiments/*.py` recipes now serialize
   `bucketed_shapes`/`live_tokens: true`, so their regime hashes differ from
   earlier runs of the same recipe; saved run.json files keep theirs.
   `baseline-v3` keeps its version label.
