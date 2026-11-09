# Infrastructure Wave memory

## Training throughput and experiment delivery (b015d5, 2026-11-08)

Jack Heart said "keep going" after "I recommend repairing BF16, lengthening the
packed cells, and completing the missing measurements." The bounded continuation
belongs to b015d5's protocol; software delivery does not complete its experiment.
The live plan and spending/admission limits belong there, not in this memory.

- **CPU autocast is not CUDA admission.** The original BF16 size ladder failed
  before training on A4000/L4/A40 at a Float/Half semantic-card join, despite CPU
  checks. Normalize destination/source dtypes without detaching gradients, prove
  float32 parity, and admit the actual CUDA worker plus checkpoint reload before
  multiplying rentals. C1's local repair/tests do not yet establish CUDA success.
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

Coverage: this Graph's reconciliation found no immediate Infrastructure child
directories or child memories on 2026-11-08. Related Experiments memory was read
only in its shared-effect performance/delivery section; other sections were not
curated here.
