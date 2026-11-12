# MTG-328 open choices (reversible; simpler option taken in the docs)

1. CLI: the docs replace `--bucketed-shapes` with `--padded-shapes` (both
   switches off on CUDA) and keep `--bucketed-shapes` as a deprecated,
   warning no-op alias removed in 0.3.0. Alternative: drop the flag outright.
2. Evaluation layout is an `EvalAllocation` setting (`bucketed_shapes`,
   `live_tokens`, default off), retained in attempt receipts and kept out of
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
7. Fresh CUDA regimes built by `experiments/*.py` recipes now serialize
   `bucketed_shapes`/`live_tokens: true`, so their regime hashes differ from
   earlier runs of the same recipe; saved run.json files keep theirs.
   `baseline-v3` keeps its version label.
8. Outcome 3 asked evaluation and search to adopt the switches so a checkpoint
   "is evaluated at the same speed". Measured on an L4 at w384l8, the layout
   made the served forward 15% slower at ~9 rows per call and left the pass at
   118 s. Taken: evaluation accepts the switches but defaults padded; search
   players and likelihood keep main's padded loads (unmeasured). Alternative
   for review: default on only for allocations with large batches, after
   measuring one (e.g. 256 active games).
