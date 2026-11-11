# Infrastructure Wave memory

## Training throughput and experiment delivery (b015d5, 2026-11-09)

Jack Heart said "keep going" after "I recommend repairing BF16, lengthening the
packed cells, and completing the missing measurements." The bounded continuation
belongs to b015d5's protocol; software delivery does not complete its experiment.
The live plan and spending/admission limits belong there, not in this memory.

- **CPU autocast is not CUDA admission.** The original BF16 size ladder failed
  before training on A4000/L4/A40 at a Float/Half semantic-card join, despite CPU
  checks. Normalize destination/source dtypes without detaching gradients, prove
  float32 parity, and admit the actual CUDA worker plus checkpoint reload before
  multiplying rentals. C1's A40 regression and two-update BF16 worker passed,
  but its post-worker validator rejected `cuda:0` before checkpoint reload.
  Offline reload after repair passed; full admission on the repaired source
  remains unproved. Neither partial execution nor local reload admits expansion.
- **Admission fixtures must reflect worker receipts.** A synthetic `cuda` string
  missed the worker's indexed `cuda:0`. Parse device type and test real-shaped
  receipts plus the ordinary checkpoint loader, retaining malformed/non-CUDA,
  unhealthy-run and corrupt-byte rejection. Audit retained bytes without
  changing their failed execution status; freeze repairs prospectively.
- **Completed packed jobs can yield no valid packing measurement.** All seven
  original packed cells lacked three overlapping post-warmup updates per seed.
  Budget for startup/finish skew, retain exact included ordinals and use matched
  solo ordinals. Longer attempts need prospective identities; insufficient
  overlap is not evidence of a crossover. Keep failed/short attempts in accounting.
- **Hardware labels do not isolate hardware effects.** Earlier small-model
  receipts showed 1.6–1.8x host variation on the same GPU label; later solo size
  measurements favored different GPUs at different sizes. Preserve CPU class,
  source, world, input, precision and host identity. The Elemental ladder's v6
  input uses effective hashed records even when the model setting says catalog;
  separate TLA catalog timings cannot supply its whole-update multiplier.
- **A steady-state rate is not a rental bill.** Reserve setup, idle, failed
  attempts and unresolved liability. Original cleanup encountered provider 429s;
  termination requests needed provider deletion confirmation. Cost estimates
  remain labelled until invoices reconcile.

Evidence: `experiments/training-update-time.md` and the manifest-backed
`experiments/data/training-throughput-b015d5-v1/report.md`. Original step 3 has
no a384l8 result or admissible packed comparison; TF32's short health check did
not establish learning equivalence. Related 5da5f4 v7 batching traces are scoped
in Experiments memory and do not establish a hashed-v6 speedup.

## Standing pipeline measurements (855bff, 2026-11-11)

Jack Heart: "The benchmark should report mean rows per inference call
(environment_decisions / policy-inference calls from collection_timings)
and the idle fraction (idle_stream_steps / (calls x streams)) per size."
Complete-game collection retires streams after their quota and terminal game;
policy batches shrink toward the longest tail. A larger serving batch limit does
not remove this cost. Current batching counters include warmup, while update-time
series can exclude it; preserve that distinction in reported rates.

The bounded CPU proofs retained 1.743 rows/call and 0.1285 idle fraction for two
streams/two w64 updates, then four replayed evaluation games after correcting the
authored deck keys. These dirty-source subset runs preceded the Python/dependency
and MTG-273 sync; they establish neither a full-ladder reference nor a speedup.
Comparison admission now includes Python as well as Torch version, host, workload
and behavior. Reference-host designation, complete measurements and real
interleaving remain open in the Task plan. Short ladders cannot establish MTG-273
long-run flatness; its separate evidence is in
`experiments/self-play-collection-273.md`.

Coverage: filesystem inspection on 2026-11-11 found no immediate Infrastructure
child directories or child memories. Infrastructure memory was read in full;
related Experiments memory was read only in its self-play metric-indexing section
this pass (shared-effect performance/delivery was read in the earlier pass).
Other Experiments sections were not read or curated here.
