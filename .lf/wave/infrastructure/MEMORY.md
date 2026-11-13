# Infrastructure Wave memory

## CI job budgets (38ef25, reconciled 2026-11-12)

Measure job start to job end on the target runner, including installation and
build. `docs/development.md` owns the ordinary Ubuntu PR budget of eight
minutes; Engine Budget retains its same-runner reference comparison and is
exempt. The macOS nightly has no such budget. In Task 38ef25, main's slowest
Integration job was 1443 s. PR #506 run 43337105047 at `1fc4e1b24` passed
with every ordinary job at or below 349 s. Its exact merged push,
43341403551 at `60664d149`, kept every ordinary job below 480 s but failed
the live experiment/notebook fixture's training resource deadline (129 passed,
one failed). Engine Budget passed in 685 s. The bounded fixture then passed
Ubuntu debug/xdist in run 43345024882 at `4db5cd64b`, but its experiment job
took 549 s, exceeding the target. CPU Torch dispatch 43348710097 at
`a1efea11b`, including main's mana-payment optimization (#504), then passed:
ordinary max 406 s, Integration max 350 s. Green tests and a met job budget
are separate obligations; this dispatch supplies both, but PR-event and
successful post-merge verification of the repair remain outstanding.

- Inspect setup before adding shards: `uv sync` built the engine in release,
  then `maturin develop` built it again in dev. Syncing all needed extras with
  `--no-install-package managym`, building once and retaining `UV_NO_SYNC=1`
  removed the duplicate while keeping native debug assertions. Cache paths
  must follow the action's actual directory; `~/.cache/uv` never hit here.
- Bound complete-game smoke work by streams/games, not just the transition
  floor. Two streams retained the real-worker one-update size check. For an
  orchestration fixture that does not require complete-game training, the
  existing one-seat `exact-quota` collector bounds learner rows while live
  monitoring still completes games. The experiment fixture repair pins one
  update and asserts all four receipts contain 16 learner transitions, with
  120-second resource guards. The comparison passed Ubuntu debug/xdist in
  151.70 s; temporarily restoring the old workload failed the new assertion
  locally (`610 == 16`). Four real attempts and live monitoring remain.
  The failed run does not identify the expired clock or separate collection
  time from scheduling; a larger allowance alone is no timing evidence.
- Match dependencies to the runner before retaining large caches. Run
  43345024882 restored a pruned uv cache but downloaded about 2.8 GiB of
  CUDA Torch/NVIDIA/Triton wheels per Integration job; one install took 181 s
  against peers' 17–18 s. Integration now selects the existing locked CPU
  Torch with `--no-default-groups --group play-runtime` and setup-uv defaults;
  wheel-retention overrides were removed. All six installs in 43348710097
  took 5–6 s with Torch `2.15.0+cpu`, no NVIDIA/Triton log entries and
  unchanged model/evaluation pass and skip counts. This establishes CPU CI
  coverage, not GPU admission. Python unit and Academy Integration still
  use the default runtime, so their download variability remains relevant.
- Parallel tests need isolated files/ports and injected time at scheduling
  boundaries. A 30-second future wait failed beside other workers; the final
  admission fixture releases the gate in its injected sleep instead. The
  five-minute test watchdog belongs in each xdist worker, not its controller.
- Keep timing margin: unchanged evaluation-long tests varied from 153 to
  249 s across two CI runs. The successful six-group layout used four logical
  workers; earlier two-worker groups remained over budget.

Coverage: read Infrastructure's complete objective and memory and listed its
directory on 2026-11-12; there were no immediate child directories or child
memories. No other Wave memory was read in this reconciliation.

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

Coverage: this Graph's reconciliation found no immediate Infrastructure child
directories or child memories on 2026-11-11. Related Experiments memory was read
through the shared-effect section (including performance, delivery and accounting)
and the training failure recovery discussion. Other sections were not read in this
pass, and that memory was not curated. Its v7 batching traces do not establish a
hashed-v6 speedup.
