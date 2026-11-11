# Infrastructure Wave memory

## Training throughput and experiment delivery (b015d5, reconciled 2026-11-11)

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
  After prospective freeze, repaired-source admission `537b2f764bba` passed the
  regression, two BF16 CUDA updates and ordinary checkpoint reload on 2026-11-10.
  The earlier failed execution remains failed; offline reload did not admit it.
  Successful execution still establishes no reduced-precision learning equivalence.
- **Admission fixtures must reflect worker receipts.** A synthetic `cuda` string
  missed the worker's indexed `cuda:0`. Parse device type and test real-shaped
  receipts plus the ordinary checkpoint loader, retaining malformed/non-CUDA,
  unhealthy-run and corrupt-byte rejection. Audit retained bytes without
  changing their failed execution status; freeze repairs prospectively.
- **Completed packed jobs can yield no valid packing measurement.** All seven
  original packed cells lacked three overlapping post-warmup updates per seed.
  Budget for startup/finish skew, retain exact included ordinals and use matched
  solo ordinals. Longer attempts need prospective identities; insufficient
  overlap is not evidence of a crossover. C1/C2 later supplied valid overlap;
  C2's large-model packed seeds each supplied 6–7 matched updates. Keep failed
  and short attempts in accounting rather than replacing their evidence.
- **Aggregate throughput and one seed's progress are different objectives.**
  C2's one-host A40 comparison measured about 1.30×/1.32× two-seed aggregate
  gains at w160l8/a384l8, with slower per-seed updates. Four w160l8 seeds bought
  about 7% more decisions/hour than two but 1.88× per-seed latency; its admission
  threshold crossing was provisional. Four a384l8 seeds failed the memory rule;
  three were never tested, so two is not an established physical ceiling.
  In the 2026-11-11 closeout, Jack Heart said "but i will say its probably
  reasonable to still do 1x seed sometimes", then "i think that should be the
  default ,but not e taht we can do 2 seeds". The maintained recommendation
  defaults to one seed per GPU, with packing an explicit aggregate-work option.
  Frozen runs and recipes are unchanged.
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
- **Unknown acquisition state must stop retries.** C2's selector treated a failed
  database read as refusal and acquired three extra idle pods. Their $0.504125
  posted cost remains charged to C2 even though the overall cap held. Reconcile
  provider state before another acquisition; command success alone also missed
  refused creates in C1. All four C2 pods have retained deletion evidence.

Evidence: `experiments/training-update-time.md` and the manifest-backed
`experiments/data/training-throughput-b015d5-v1/report.md`, especially repaired C1
and C2. Original step 3 still has no a384l8 result or admissible packed comparison;
later cohorts do not rewrite it. Neither TF32 nor BF16 health establishes learning
equivalence. Exact closeout quotes are retained at
`3c169916e2d642b5ead16aa303b979232635a63f:scratch/questions.md`.

## Training/evaluation performance workflow (4ceb1d, 2026-11-11)

The Task brief quotes Jack Heart's focus as "how fast can we get learning to
happen" and "inference during training time: basically training speed, inference
speed, per size of model". `experiment-performance` addresses collection,
inference, learner and evaluation cost with same-machine controls and behavior
evidence. Native engine work stays in `engine-performance`; Coverage keeps its
shared `engine-optimization` graph. `docs/workflows.md` owns the routing and
commands. Workflow installation grants no campaign or spending authority, and
software delivery cannot complete a parent scientific experiment.

Instrumented CPU/operator probes are attribution, not ordinary CUDA throughput.
Separate scientific evidence is needed when precision, policy lag or batch
composition changes. Task 855bff owns the standing size-ladder benchmark;
use its shipped command once available rather than duplicating it in this workflow.

Coverage: this Graph's reconciliation found no immediate Infrastructure child
directories or child memories on 2026-11-11. Related Experiments memory was read
in the shared-effect introduction and performance/delivery/recovery sections;
the intervening admission/accounting detail and other sections were not read in
this pass or curated. Its v7 batching traces do not establish a hashed-v6 speedup.
