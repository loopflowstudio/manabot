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
and behavior. A report completed for a subset is still not a standing reference:
recording and comparison share one completeness rule (all four sizes and every
workload supported by the device), checked before workers start with the error
retained. Controlled tests cover that rule for CPU/CUDA and training-only MPS;
they do not establish device performance or real-source repeatability.
There is no reference host to designate. Jack Heart rejected the offered hosts:
"i dont really thin kany of these options are acceptable tbh". On the
reference-commit alternative he said: "no i think you have it right. this is a
strong reason to have at least one more stable gpu. maybe aws gpu perf is more
consistent?" (comment `262b941c-2188-4426-8ff0-35f7e52eaa10`, 2026-11-11).
The baseline carries only a source commit and suite interface; comparison reruns
both clean sources interleaved in one job on one machine, or one pod on a rented
GPU. Historical rates and hosts cannot supply the denominator or bind future
jobs. CPU readings prove the harness only. Provider selection is outside this
Task, and no spending or launch is authorized.

The implementation and controlled tests follow that direction; no baseline has
been committed. Complete default-ladder runtime, reference adoption and real-source
interleaving remain open in the Task plan. Short ladders cannot establish MTG-273
long-run flatness; its separate evidence is in
`experiments/self-play-collection-273.md`.

Coverage: filesystem inspection on 2026-11-11 found no immediate Infrastructure
child directories or child memories. Infrastructure memory was read in full;
related Experiments memory was read only in its introduction and self-play
metric-indexing section this pass (shared-effect performance/delivery was read
in an earlier pass).
Other Experiments sections were not read or curated here.
