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

Small inference batches need not indicate a serving limit: complete-game
collection idles streams that reach their closure target while other games finish.
Read `StageRecord.idle_stream_steps` alongside inference rows/calls; older receipts
default that counter to zero and cannot establish absence of idle slots. Changing
closure to improve batching also changes sampling and needs learning evidence.

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

Bound a whole invocation from retained runtime evidence, not update counts alone.
The first CPU ladder exhausted its former 3,600-second deadline with six rows;
the remaining size completed separately. Their completed attempts sum to about
5,500 seconds, with w64 perturbed by an accidental overlapping run. That sum is
not a full invocation measurement, and the derived nine-hour comparison remains
an estimate. The repaired 7,200-second standalone and 36,000-second comparison
limits preserve workload identity; controlled deadline tests establish plumbing,
not sufficient runtime. Receipts are located in 855bff's `scratch/plan.md`.

No baseline has been committed. Complete clean default-ladder runtime, reference
adoption and real-source interleaving remain open in the Task plan. Rebuild the
matching native extension after upstream API changes before fresh measurements;
pre-sync receipts do not validate the merged pipeline. Short ladders cannot establish MTG-273
long-run flatness; its separate evidence is in
`experiments/self-play-collection-273.md`.

855bff coverage: realignment on 2026-11-11 listed Infrastructure and found no
immediate child directories or child memories; its memory was read in full.
Related Experiments memory was not re-read in this pass. Earlier passes read its
introduction, self-play metric-indexing and shared-effect performance/delivery;
other Experiments sections were not read or curated.

4ceb1d coverage: the Graph's reconciliation found no immediate Infrastructure child
directories or child memories on 2026-11-11. Related Experiments memory was read
through the shared-effect section (including performance, delivery and accounting)
and the training failure recovery discussion. Other sections were not read in this
pass, and that memory was not curated. Its v7 batching traces do not establish a
hashed-v6 speedup.
