# Training throughput on the ten-deck Elemental recipe, v1 (draft, not frozen)

Task b015d5. Question: for two-seat ten-deck Elemental self-play, which machine
shape and training regime give the most strength per dollar and per hour? The
candidates are packing several seeds on one machine, not waiting for each
batch's slowest game, and overlapping collection with learning. Multi-producer
collection belongs to Task c5fba4 and is read from its result, not repeated here.

Status: design accepted by Jack Heart at experiment-design review (2026-11-06,
comment 14e6cc54), with fixed gates replaced by step-at-a-time choice. Nothing
has launched. Acceptance confers no spending authority beyond the execution
review below. Step 1 freezes once PR #436 (the control recipe) has landed, or
the probe pins an exact commit containing it. Each later step freezes after the
previous step's readings have been discussed and that step chosen, before any
of its scored runs.

## Authority and budget

- **Execution authority.** Jack Heart's Task comment 9f9d0956 (2026-11-06) says
  to run this experiment under a USD 50 hard cap. The cap covers training,
  evaluation, failures and machine setup, and the comment asks to spend no more
  than needed. The same comment sends design and execution review to Jack Heart
  in the Task conversation, so no launch happens before that review.
- **Proposed spend.** At most $24.40 is reserved below. The remaining $25.60 of
  the cap is not allocated; using it needs an amendment made before the run that
  would spend it.

## Fixed identities

- **Source.** Main after PR #435 (`45cc961a2`) and PR #436 (07c490, the
  `elemental_variants.elemental_recipe` declaration and
  `learning_signal.ELEMENTAL_PAIRS`). If c5fba4's change to
  `manabot/training/rollout/collection.py` lands first, it is included. Each
  launch pins one commit, and the report names it.
- **Recipe.** The control is `elemental_variants.elemental_recipe`. That is
  MTG-181/183's frozen `experiments/regimes/elemental-long-183-v1.json` with
  07c490's two compatibility changes: the world label moves to the current
  world, and capacity recovery is on. Everything else is unchanged: 256 streams,
  two learning seats, `elemental-uniform`, complete games with at least 64
  decisions per stream, damped policy learning, CUDA float32, one producer, the
  recipe's policy input and observation capacities. 07c490 used this exact
  control as its lambda baseline (p.5 v.8), so its strength curve (.258 / .335 /
  .417 at u20 / u40 / u75) is a reference. That curve was trained on a pre-#435
  source, so it is never a scored arm here.
- **Variants.** Every regime change gets its own regime ID
  (`elemental-throughput-<arm>-v1`). Frozen worlds, observation versions and
  recipe files stay unchanged.
- **Hosts.** Every run records `TrainingRun.host`: CPU model, schedulable CPUs
  and quota. The GPU is part of the run identity. Hosts of the same GPU label
  differ 1.6-1.8x, so wall-clock and dollar comparisons are made within one host
  class or on one machine (see "Host confound").

## Step 1: is collection the bottleneck? (measurement only, no strength)

Retained fast-L4 receipts read 17.6 s collecting and 15.1 s learning per update
(54% collection). They predate #435, which cut learner targets 15-27% of an
update and left collection mostly alone, so collection's share should now be
higher. The idle share (.63-.67 of stream slots in two-seat runs) is a bound,
not a measurement. Step 1 measures both on real CUDA.

**Job.** `experiments/training_throughput.py` `probe`: one Etude run per
machine. A driver runs these phases in sequence on the same host, each as
`scripts/bench_training_update.py` child processes (ordinary subprocesses, not
`multiprocessing` spawn) on the control recipe with `--device cuda`.

| Phase | Children | Updates each | Seeds | Reads |
|---|---|---|---|---|
| P1 solo | 1 | 12 | 26001 | phase tree, idle share, CPU cores, GPU busy share, peak host and device memory |
| P2 pack-2 | 2 at once | 12 | 26002-26003 | the same, per child and per machine |
| P3 pack-4 | 4 at once | 8 | 26004-26007 | the same; skipped if P2 peak host memory x 4 exceeds 85% of the machine |
| P4 CUDA counts | 1 | 3 | 26008 | `torch.profiler` kernel launches and `torch.cuda.set_sync_debug_mode("warn")` sync counts per collection forward and per learner minibatch |

P4 runs last and alone, so its instrumentation does not perturb P1-P3. Readings
use updates 2 onward; update 1 includes warm-up. Timing comparisons use median
seconds per update, not means.

**Machines.** Two L4s (Etude's `l4` template: 6 vCPUs, 48 GB host, ceiling
$.60/hour) so both host classes are likely to appear, and one RTX A4000 (16
vCPUs, 48 GB, ceiling $.40/hour) as the more-cores, cheaper-GPU shape. If both
L4s land in one host class, that is reported; no third L4 is rented for it.

**Capture.** Each child writes its full `TrainingRun` receipt: collection and
learning phase trees, `idle_stream_steps`, `cpu_seconds`, host, games,
environment and learner decisions. The driver samples, once a second:
`nvidia-smi` GPU utilization and per-process device memory, each child's
`VmHWM`/`VmRSS` from `/proc`, and the machine's `MemAvailable`. Results go to
`ETUDE_OUTPUT_DIR`, journaled in `/dev/shm` and copied out at phase ends.

**Measures.**
- Collection share = collection / (collection + learning) seconds per update,
  split into native stepping, encoding, inference and the remainder.
- Idle share = `idle_stream_steps` / stream slots. Recoverable collection is
  bounded by (collection - assembly) x idle share; P1 gives the real value on
  the current source.
- Overlap bound = min(collection, learning) / (collection + learning).
- Packing gain = K x s/update(K=1) / s/update(K), per machine. The result is
  dollars per million learner decisions, for each machine and K.
- CPU cores per child (`cpu_seconds` / wall) and GPU busy share. Spinning CUDA
  waits inflate CPU cores, so CPU use alone does not show host-bound work.

**Software before step 1 freezes.** No training code changes.
- `scripts/bench_training_update.py --json OUT` writes the child's whole
  `TrainingRun` receipt, and takes `--recipe` to the control declaration
  instead of re-pinning the frozen file itself. The existing re-pin has no
  capacity recovery, so one overflowing game would end a probe.
- `experiments/training_throughput.py`: the probe `Job` (`etude.call`
  entry, as in `experiments/policy_inference_gpu.py`), its phase driver, the
  1 Hz samplers and a `collect` cleanup that copies outputs from `/dev/shm`.
- Acceptance: a CPU run of the driver (P1 and P2 at 8 streams and 2 updates,
  samplers skip `nvidia-smi` when it is absent) writes receipts and samples
  for every child. `etude launch ... --quote` shows
  one machine per probe under the line's limit. No laptop run measures
  timing.

**Choosing the next step.** There are no fixed pass/fail gates. The experiment
goes one step at a time: after step 1, its readings are reported in the Task
conversation, and Jack Heart chooses the next step with them. That choice is
recorded here as an amendment before that step's runs. The readings that inform it:

| Reading | Points toward |
|---|---|
| Collection share and the measured idle share | Step 3 (stop waiting), sized by (collection - assembly) x idle share |
| Overlap bound, spare cores and GPU busy share | Step 4 (overlap) |
| Packing gain per machine, dollars per M learner decisions, memory headroom | Step 2 (seeds per machine) and its slot count |

Steps are taken in order of expected saving per dollar and per hour of work. A
step with a small measured bound is not built. The experiment may end after
step 1 with its report and the machine recommendation.

## Step 2: seeds per machine

Packing changes no learning rule. Each seed is a separate process with its own
seed, model, optimizer and environments, and the processes share only hardware.
Under a synchronous update, seconds per update do not enter the algorithm. A
packed seed therefore learns as it would alone by construction, apart from
CUDA's ordinary run-to-run nondeterminism. **No strength runs are needed to
judge packing.** Its strength per dollar is the control's strength per update
times updates per dollar, which step 1 measures directly.

Two checks support the construction:
- A CPU digest: 24 streams, 3 updates, run alone and with a second process
  running beside it. Raw, EMA, optimizer and game-attempt digests must be
  identical (`digest2.py` exclusions).
- Packed CUDA seeds' per-update training metrics (loss, entropy, KL, rows per
  update) must fall within the spread of solo seeds at matched updates, with
  one-update watchdog margins intact.

**Etude machine slots.** Jack Heart approved building and shipping this on
2026-11-06 as Etude Task 8ffc3e, independently of step 1's readings. Today a machine runs
its session's attempts in turn ("There is no run-count packing setting",
`etude-sdk/docs/launcher.md`). The change is an on-demand `slots: int` that
places up to that many runs on one machine at once. Each run keeps its own
attempt directory, outputs, receipt and log. Cost is split across concurrent
runs by placed time. Cancelling or failing one run leaves the others running.
The machine's window covers its longest run. Acceptance checks: Etude tests for
placement, cost split, independent cancel and the courier publishing each run's
outputs, plus one two-run CPU canary. This is an Etude PR, not a manabot
workaround. The same change lets steps 3 and 4 place control and arm seeds on
one machine.

## Step 3: stop waiting for the slowest game

At a closed batch, a finished two-seat stream waits until every stream finishes
its game. There are two candidates:

- **Two-seat exact quota (`elemental-throughput-exact2-v1`).** A row is ready
  for training once its successor is known: the same seat's next decision, or
  the end of the game. A batch closes at a fixed number of ready rows per
  stream, so no stream ever idles. Rows still in flight carry into the next
  batch, and their targets bootstrap from the value at the successor
  observation, as one-seat exact quota does. The quota per stream is the
  control's realized rows per stream after the opening updates, set by a
  30-update calibration as 07c490 did (about 190 at 48,555 rows over 256
  streams). Capacity recovery discards a failed game's unlearned rows.
  Software: let `batch_closure: "exact"` accept `learning_seats: "both"`, with
  per-seat carry-over. Build it after c5fba4's `collection.py` change lands.
  Acceptance: with the closure set to complete games, digests are unchanged.
  Unit tests cover carry-over across a batch boundary for both seats, rows
  per stream equal to the quota, and the bootstrap index of carried rows.
- **Refill, rejected unless its bias is small.** Finished streams start new
  games, and games still unfinished at close are discarded. That over-weights
  short games. Before any build, the bias is measured offline from the step-1
  and control receipts' game-length distributions: the decision-weighted
  length distribution kept under discard, compared with the one played. Refill
  is built only if the kept distribution's mean game length is within 5% of
  the played one. Otherwise only exact quota runs.

07c490's one-seat result (complete games .035 below exact quota at matched
decisions, interval including zero; .071 below at matched wall time and cost)
sets no expectation for two seats.

## Step 4: overlap collection and learning

`elemental-throughput-overlap1-v1`: the collector gathers batch n+1 with the
weights from update n-1 while the learner computes update n from batch n. The
behavior policy lags by exactly one update. Importance ratios, clipping and
`collection_kl` already use the behavior log-probabilities recorded at
collection, so the estimator is unchanged apart from that lag, which the report
names. Software: a learner thread on its own CUDA stream, a weight snapshot
handed to the collector at each update boundary, and per-update receipts
recording which weights collected each batch. Acceptance: with overlap off,
digests are unchanged. With overlap on, a CPU test shows that batch n is
collected by update n-2's weights (one update stale relative to the learner's
input), and recovery and finalization still export complete learning state.
The wall-time gain is bounded by step 1's overlap bound.

## Strength arms (steps 3 and 4)

- **Seeds.** Four per arm, seeds 26101-26104. Every arm and the control share
  each seed's initial weights. The control is trained fresh on the frozen
  source, not reused from 07c490.
- **Target.** 75 updates (07c490's lambda length, about 3.6M learner
  decisions), with raw exports every 5 updates so wall-time-matched checkpoints
  exist.
- **Evaluation.** `learning_signal.ELEMENTAL_PAIRS` (50 fixed deck pairings,
  digest `9bff7292…503f`, deals 7070000000-7070000049): the model's four cross
  legs against the fixed heuristic, 200 games per checkpoint. These are
  evaluation deals only; no training, screen or calibration uses them. Rented
  32-vCPU machines play them, never the laptop. Raw exports are scored.
- **Scored checkpoints.** u20, u40 and u75 on the update clock. For each arm,
  the exports nearest the control's median wall time at u40 and u75 are also
  scored (5-update resolution).
- **Clocks.** Primary: strength at matched wall seconds and matched dollars,
  because the question is per hour and per dollar. Secondary: strength at
  matched learner decisions, which separates "learns less per sample" from
  "runs faster". Collection, learning, idle, rows, games, optimizer steps and
  cost are reported per arm.
- **Analysis.** Paired by seed: arm minus control. Seed-level t intervals and
  pairing-resampled intervals are reported separately. Strength between scored
  wall-matched exports is not interpolated; the nearest export is scored.
- **Host confound.** If Etude slots exist, each seed's control and arm run on
  one machine. Otherwise, runs are screened with #435's band: the median of
  `time/collection/environment_step_and_encoding` over updates 2-11, above
  9 s is slow. Slow hosts are cancelled (`etude cancel`, never deleting the pod
  first) and relaunched. The cost of cancelled runs is charged and their
  receipts are kept. Wall-matched comparisons use only fast-host runs, and the
  report states how many were replaced.
- **Inconclusive.** Both intervals include zero and the paired point estimate
  is within ±.03. Otherwise the result is directional, and a claim needs an
  interval that excludes zero.

## Budget

L4 at about $.59/hour, ceiling $.60; A4000 ceiling $.40; 32-vCPU CPU ceiling
$1.00. Each run reserves its window plus 15 minutes of setup at the ceiling.

| Line | Runs | Window | Reserved |
|---|---:|---:|---:|
| Step 1 probe, L4 | 2 | 1.5 h | $2.10 |
| Step 1 probe, A4000 | 1 | 1.5 h | $.70 |
| Step 3 calibration (quota) | 1 | .75 h | $.60 |
| Control seeds | 4 | 1.5 h | $4.20 |
| Step 3 exact quota | 4 | 1.5 h | $4.20 |
| Step 4 overlap | 4 | 1.5 h | $4.20 |
| Evaluation, about 48 checkpoints x 200 games | 2-3 passes | 2 h each | $3.00 |
| Reserve for failed attempts and replaced hosts | | | $5.40 |
| **Proposed total** | | | **$24.40 of $50** |

Lines for steps not chosen are not spent. Step 2 has no strength line. Its Etude
change costs one CPU canary, paid from the reserve. If steps 3 and 4 both
proceed, they share the control seeds.

## Stop and recovery

- Stop launching when billed plus reserved spend would exceed the proposed
  total. Exceeding it needs an amendment; exceeding $50 is never allowed.
- A machine not booted within 15 minutes is terminated and its window charged.
  `runpodctl billing pods` reconciles GPU spend. CPU evaluation machines appear
  in no bill, so their rental windows are the estimate.
- If a probe fails twice on its own errors (not a dud host), step 1 stops and
  the report says why. A run stopped by its watchdog keeps its exports;
  comparisons extend only as far as every run reached.
- Calibration failure, or no quota within 10% of the realized rows, stops
  step 3 before scoring.
- Seeds are cut equally across arms before launch, never after scoring.

## Deliverable

`experiments/data/training-throughput-b015d5-v1/report.md` in the artifact
store. It contains the step-1 table (per machine and K: s/update, phase tree,
idle share, CPU cores, GPU busy share, peak memory, $/M learner decisions), the
step readings and choices, the arms' strength curves on wall, dollar and decision clocks,
cost against this budget, and the machine and regime recommendation that
updates `experiments/training-update-time.md`.
