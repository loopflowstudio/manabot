# Training throughput on the ten-deck Elemental recipe, v1 (step 3 frozen, not run)

Task b015d5. Question: for two-seat ten-deck Elemental self-play, which machine
shape and training regime give the most strength per dollar and per hour? The
candidates were packing several seeds on one machine, not waiting for each
batch's slowest game, and overlapping collection with learning. Step 1 measured
all three. Packing is adopted and validated below, stop-waiting is deferred and
overlap is dropped. Whether several producers learn as one does belongs to Task
c5fba4 and is cited, not retested; how producers combine with packing is measured
for speed in step 2. Step 3 measures how training speed and packing change with
model size, GPU and numeric precision, as the first input to a planner that turns
(hours, dollars) into a training plan.

Status: design accepted by Jack Heart at experiment-design review (2026-11-06,
comment 14e6cc54), with fixed gates replaced by step-at-a-time choice. Step 1 has
run (see its execution record). Jack Heart accepted its evidence and chose step 2
(2026-11-07, comment 9d671e78) and accepted its design, including running from Etude
main `a7718f5` under an isolated `ETUDE_HOME` (comment 850a636c). Jack Heart then
asked this step to own how packing combines with multiple producers (2026-11-07,
comment ceff7d63), which added legs S14, S22 and S44. Step 2's software landed as
`668212567` (PR #442) and step 2 is frozen at that commit (see its freeze receipt).
Jack Heart approved execution (2026-11-07, comment e89d4889), and step 2 has run (see
its execution record). Jack Heart accepted step 2's evidence and set step 3's objective
(2026-11-07, comment 1f4c2872), then refocused it on large-model speed (comments
8ac4eb82, 518e5f09 and b9215b8a). Jack Heart accepted step 3's design (2026-11-07,
comment 17799706). Its software landed as `cb722dc8c` (PR #448), and step 3 is frozen at
that commit (see its freeze receipt). Nothing has been rented for it.
Acceptance confers no spending authority beyond that review. Each step freezes after
the previous step's readings have been discussed and that step chosen, before any of
its runs.

## Authority and budget

- **Execution authority.** Jack Heart's Task comment 9f9d0956 (2026-11-06) says
  to run this experiment under a USD 50 hard cap. The cap covers training,
  evaluation, failures and machine setup, and the comment asks to spend no more
  than needed. The same comment sends design and execution review to Jack Heart
  in the Task conversation, so no launch happens before that review.
- **Proposed spend.** Step 1 cost about $1.06 and step 2 about $.98, so about $2.04
  of the $50 cap is spent. Step 3 proposes a $10.00 reservation (see its budget).
  Anything further needs an amendment made before the run that would spend it.

## Fixed identities

- **Source.** Main after PR #435 (`45cc961a2`) and PR #436 (`ebed0963e`, 07c490's
  `elemental_variants.elemental_recipe` declaration and
  `learning_signal.ELEMENTAL_PAIRS`). Step 1 ran before c5fba4's change to
  `manabot/training/rollout/collection.py` landed; step 2 includes it. Each launch
  pins one commit, and the report names it.
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

**Step 1 freeze receipt (2026-11-06).** Software review passed on main
`70d79ccb8`, so step 1 is bound to that commit. The launch revision is
`70d79ccb82be9a06ee662ae6fd3c7d8140ebe38c`. That commit contains #435 and #436.
c5fba4's `collection.py` change has not landed and is not included. This
receipt is recorded after that commit and changes no executable byte.
- Source bundle: `runs("70d79ccb8")` freezes 1,463 tracked files (bundle JSON
  SHA-256 `394b055b…bb35`). Pod admission re-verifies them through
  `bind_etude_source`. Key files: `experiments/training_throughput.py`
  `e529f50b…b379`, `scripts/bench_training_update.py` `fff75a6a…f40a`,
  `experiments/elemental_variants.py` `eca02a6c…8074`, frozen regime
  `elemental-long-183-v1.json` `be9de376…64ad`, `uv.lock` `b562227a…7c58`.
- Probe config (without the bundle, sorted-key SHA-256 `785c962a…2f3e`):
  control `experiments.elemental_variants:elemental_recipe`, CUDA, 256 streams,
  phases solo 1×12 (26001), pack-2 2×12 (26002-3), pack-4 4×8 (26004-7),
  cuda-counts 1×3 (26008), memory fraction .85, 1 s samples.
- Resolved child regime: `bench-training-update` on w17, both seats, complete
  games, at least 64 decisions, `recover_capacity` true, CUDA, 3,600 s wall.
  Model-dump SHA-256 at 12 / 8 / 3 updates: `d76f17b4…7c23` / `69fa12c0…f705` /
  `46098f8b…dc9b`.
- Job: `training-throughput-probe`, entry
  `etude.call --function experiments.training_throughput:run`, an importable
  module rather than a discovery namespace. Cleanup runs
  `python -m experiments.training_throughput`, with one attempt per run, 1.5 h
  and $1.05 per L4 run (the A4000 run is launched with `--usd-per-run 0.70`).
  Etude discovery read this file and built run seeds 1-2 without executing
  games or training.
- Isolation: step 1 uses no evaluation deals. Its seeds 26001-26008 are disjoint
  from the strength seeds 26101-26104 and from the `ELEMENTAL_PAIRS` deals.
- Admission and quote with the etude-sdk 0.2.14 overlay (the locked 0.2.12
  client rejects the launcher's Vast provider records): the A4000 run quoted
  RTX A4000, at least 16 vCPUs, 48 GB, $.25/hour, planned $.44 within its $.70
  line. **The L4 quote remains open:** the provider reported no available
  machines (cohort `training-throughput-probe-l4`, 2 machines, $2.10). Re-quote
  before launch. This availability gap changes no frozen identity.
- Authority: execution stays under comment 9f9d0956's $50 inclusive cap and the
  $2.80 step-1 lines above. Freezing confers no new spending authority. Launch
  still waits for execution review.

**Step 1 execution (2026-11-07 UTC).** Jack Heart approved launching step 1 as
frozen at execution review (comment 37512749): two L4 probes on a new on-demand
`b015d5-l4` (at most 2 machines, under $.60/hour) and one A4000 probe on `a4000`
with `--usd-per-run 0.70`, within the $2.80 step-1 lines. Steps 2-4 are not
authorized by that approval.
- Re-quote at launch: L4 quoted $.59/hour, planned $2.06 within $2.10. The two
  L4 runs launched at 05:27 as cohort `training-throughput-probe-l4`
  (`--max-usd 2.10`): runs `2b5e4b4a1fb4` (seed 1, machine
  `b015d5-l4-06d85fdd0fde`) and `04c78c2631c6` (seed 2, machine
  `b015d5-l4-f6648404bf33`), revision `70d79ccb8`.
- The A4000 re-quote again found no available machine, so its run was queued:
  `11937fd035f5`, cohort `training-throughput-probe-a4000`, `--max-usd 0.70`. It
  rented `a4000-e28308d569b9` at 05:43.
- Outcome: all three machines completed P1-P3. `04c78c2631c6` succeeded. The other
  two report `failed` only because the P4 child was SIGKILLed at about 61 GB peak,
  with the cause not established. P4 was not retried, under the two-failure stop
  rule. Cost about $1.06 of the $2.80 lines ($.96 posted by RunPod, the rest
  estimated from rental time). All three machines are released. Readings and evidence:
  `experiments/data/training-throughput-b015d5-v1/report.md`. Steps 2-4 are not
  chosen.

**Job.** `experiments/training_throughput.py` `probe`: one Etude run per
machine. A driver runs these phases in sequence on the same host, each as
`scripts/bench_training_update.py` child processes (ordinary subprocesses, not
`multiprocessing` spawn) on the control recipe with `--device cuda`.

| Phase | Children | Updates each | Seeds | Reads |
|---|---|---|---|---|
| P1 solo | 1 | 12 | 26001 | phase tree, idle share, CPU cores, GPU busy share, peak host and device memory |
| P2 pack-2 | 2 at once | 12 | 26002-26003 | the same, per child and per machine |
| P3 pack-4 | 4 at once | 8 | 26004-26007 | the same; skipped if P2 peak host memory x 4 exceeds 85% of the machine |
| P4 CUDA counts | 1 | 3 | 26008 | `torch.profiler` kernel launches and `torch.cuda.set_sync_debug_mode("warn")` sync counts per update (run total / updates); per collection forward or learner minibatch only as upper bounds, since one total covers every phase |

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
| Collection share and the measured idle share | Stop waiting (then numbered step 3), sized by (collection - assembly) x idle share |
| Overlap bound, spare cores and GPU busy share | Step 4 (overlap) |
| Packing gain per machine, dollars per M learner decisions, memory headroom | Step 2 (seeds per machine) and its slot count |

Jack Heart's choice after step 1 is recorded under step 2.

Steps are taken in order of expected saving per dollar and per hour of work. A
step with a small measured bound is not built. The experiment may end after
step 1 with its report and the machine recommendation.

## Step 2: seeds per machine (amended 2026-11-07)

**Choice.** Jack Heart accepted step 1's evidence and chose this step (comment
9d671e78). The standard training shape becomes **four seeds per machine**, through
Etude `--slots 4`, on an A4000 when one is available and an L4 otherwise. Step 1's
reason: at K=4 throughput rose 1.66-2.35x and dollars per million learner decisions
fell to $.023 (A4000) and $.046-.047 (L4). A seed takes 1.7-2.4x longer per update,
so packing buys dollars and total throughput, not time to one result.
Step 1 measured one producer per seed. New recipes default to four producers
(`manabot.recipes.builders.sampled_self_play`, c5fba4), so this step also reads speed
at K seeds of P producers each, and the report recommends a shape per machine class.

**Claim under test.** A packed seed learns as a solo seed does. Each seed is its own
process with its own seed, model, optimizer and environments, and they share only
hardware, so this holds by construction unless something leaks between them: a
shared path, a wall-clock dependency, or Etude's new concurrent worker. No strength
runs are needed, because the learning rule is unchanged. Strength per dollar is the
control's strength per update times updates per dollar. This step plays no
evaluation deals and scores no checkpoint. The speed legs make no learning claim:
whether a run learns the same at any producer count is c5fba4's question, cited and
not retested. Its landed evidence is a CPU check at four streams: under a fixed
policy, observations, actions, game seeds and draws agree at 1, 2 and 4 producers
(`test_uniform_draws_agree_across_producer_counts`, this control's roster). Its
protocol is accepted, not frozen, and plans no trained comparison across producer
counts: its one-producer pod reads speed only.

**Step 2 freeze receipt (2026-11-07).** Jack Heart accepted software review and directed
this freeze at PR #442's commit (comment 258f3c96). The launch revision is
`668212567d8dbe17882f7c857274f6a7d3c494db`, on main, for every leg. It contains #435,
#436, `43150799f` and `405223165`. This receipt is recorded after that commit and changes
no executable byte.
- Source bundle: every leg's job carries one bundle of 1,479 files, sorted-key SHA-256
  `6334a27b…1ff3`: all 1,480 tracked files except `CLAUDE.md`, which is a symlink. Key files:
  `experiments/training_throughput_packing.py` `9d1cd47b…2f4d`,
  `experiments/training_throughput.py` `f36641e3…cd19`,
  `experiments/elemental_variants.py` `0e51b536…0df4`,
  `experiments/learning_signal.py` `836230fd…9cb0`, `manabot/experiment/job.py`
  `62aa4fd3…5aca`, `manabot/infra/host.py` `7cd0dbbd…208b`, frozen regime
  `elemental-long-183-v1.json` `be9de376…64ad` (as step 1), `uv.lock` `b562227a…7c58`
  (as step 1).
- Regimes, on w17, by sorted-key SHA-256 of the job's regime:
  - digest `elemental-throughput-digest-v1` `0366ff8d…1c3e`: 24 streams, 3 updates, CPU,
    600 s watchdog, equal in D1 and D2;
  - metric `elemental-throughput-pack-check-v1` `57823471…728a`: 256 streams, 20 updates,
    CUDA, one producer, 2,400 s watchdog, equal in M1, M2 and M3;
  - `elemental-throughput-producers-2-v1` `53867f36…59c1`: S22;
  - `elemental-throughput-producers-4-v1` `8c414618…5f26`: equal in S14 and S44. Both
    speed regimes run 256 streams, 8 updates, CUDA, 1,500 s watchdog.
- Discovery: Etude `read_job_file`, client `a7718f5`, read all eight legs at this revision
  without executing games or training: 19 runs (1, 2, 4, 4, 1, 1, 2, 4), each leg with its
  own job name and the seeds in the table. Limits match "Run limits".
- Acceptance: `tests/training/test_training_throughput.py`, `tests/infra/test_host_facts.py`
  and `tests/training/test_elemental_closure_lambda.py` pass on this tree (38 tests).
- Isolation: step 2 plays no evaluation deals and scores no checkpoint, so no held-out set
  or anchor is touched. Seeds 26201-26202 and 26211-26214 are disjoint from 26001-26008,
  26101-26104 and the `ELEMENTAL_PAIRS` deals. M3 is the noise reference.
- Client: `ETUDE_HOME=$HOME/.etude-b015d5-step2 doppler run -p etude -c prd -- doppler run
  -p manabot -c prd --preserve-env=RUNPOD_API_KEY -- uv run --with "etude-sdk[s3,eval] @
  git+file:///Users/jack/src/etude@a7718f59a706fdb1e4cac7c28b2a13e5b7afbd11" etude ...`.
  The shared `~/.etude/launcher.db` stays at schema 11. Its isolated home holds both
  on-demands at 4 slots and at most 1 machine, and no rented machine.
- Quote, 2026-11-07 about 19:28 UTC, at this revision, nothing rented:
  - `b015d5-l4x4`: L4, 6 vCPUs, 48 GB, $.59/hour, rentable. D1 alone plans $.44 within
    $.45. The four M1 runs share one L4 at 4 slots, planned $.74 within $3.00. The four
    S44 runs share one at 4 slots, planned $.59 within $2.40.
  - `b015d5-a4000x4`: **open.** RunPod reports no A4000 ($.25/hour offer, ceiling $.40), so
    Etude refuses the quote at every leg. The selection rule's 30-minute wait covers this.
  - These quotes plan a new rental per leg. Etude's `rent` has no quote, so the 5-hour hand
    rental is priced from the rate: at most $2.95 on an L4 and $1.25 at the A4000 offer.
    Placement on a held rental has not been exercised.
- Authority: step 2 stays under comment 9f9d0956's $50 inclusive cap and comment ceff7d63's
  $4.50, about $5.56 reserved with step 1. Freezing confers no spending authority. Nothing
  launches or rents before execution review with Jack Heart.

**Step 2 execution record (2026-11-07/08).** All 8 legs ran at `668212567` on one A4000
rental, `b015d5-a4000x4-27c8c5c7f309` (RunPod `nbc3u0h4ot5rv0`, 16 vCPUs, 62 GB,
$.25/hour), held 20:44-00:36 UTC.
- **Selection.** The second rent attempt got the A4000, so no L4 was rented.
- **Placement.** Legs were placed with `etude queue --rent` (same job bytes): `etude
  launch --rent` refuses a held rental while the controller holds its lease.
- **Runs.** All 19 declared runs succeeded. The digests are equal.
- **Failure.** One attempt hung after update 2 with the GPU idle: M2's seed 26214,
  `91e148fbd392`. Etude's 1-hour limit killed it, and its one retry succeeded. After
  that, a stall watcher, cancelling at 600 s with no new update row, guarded the
  remaining legs and never fired. No stop rule was reached.
- **Cost.** About $.98 against the $4.50 line: $.807 posted, the rest estimated at
  the billed rate. The machine is terminated and the isolated controller stopped.
- **Results.** Readings, recommendation and evidence locations are in
  `experiments/data/training-throughput-b015d5-v1/report.md`, "Step 2".

### What runs

All legs run in sequence on **one rented machine**, so the digest legs share CPU
and the solo and packed legs share a host class. Runs go through `TrainingJob`, the
path real training uses. A leg is launched after the previous one ends; `etude wait`
sequences them, because a free slot would otherwise take the next run early.

| Leg | Regime | Device | Runs on the machine at once | Seeds | Reads |
|---|---|---|---:|---|---|
| D1 digest solo | digest variant | CPU | 1 | 26201 | reference digest |
| D2 digest packed | digest variant | CPU | 2 | 26201, 26202 | 26201 against D1 |
| M1 packed | control, 20 updates | CUDA | 4 | 26211-26214 | per-update metrics, K=4 speed |
| M2 solo | control, 20 updates | CUDA | 1 | 26211-26214 in turn | per-update metrics, K=1 speed |
| M3 solo repeat | control, 20 updates | CUDA | 1 | 26211 | run-to-run noise at a fixed seed |
| S14 | producers-4, 8 updates | CUDA | 1 | 26211 | speed at (K, P) = (1, 4) |
| S22 | producers-2, 8 updates | CUDA | 2 | 26211, 26212 | speed at (2, 2) |
| S44 | producers-4, 8 updates | CUDA | 4 | 26211-26214 | speed at (4, 4) |

- **Digest variant** `elemental-throughput-digest-v1`: the control recipe at 24
  streams, 3 updates, device CPU, one thread. It is a check, not a training shape.
- **Metric regime** `elemental-throughput-pack-check-v1`: the control recipe
  (`elemental_recipe`) to 20 updates at 256 streams, CUDA, with the same regime
  bytes in M1, M2 and M3. Damped-policy rates follow the absolute iteration, so
  the shorter endpoint leaves the learning schedule unchanged.
- **Speed variants** `elemental-throughput-producers-2-v1` and `-4-v1`: the control
  recipe to 8 updates, differing from it only in `execution.workers` (and the watchdog
  the launch sets). M1 is shape (4, 1) and M2 is (1, 1).
- **Seeds** are disjoint from 26001-26008, from strength seeds 26101-26104 and from
  the `ELEMENTAL_PAIRS` deals.
- **Source.** One launch commit containing the software below. Main's `43150799f`
  (fact and shared-effect embeddings) and `405223165` (c5fba4's sampled-roster
  producers) arrive with it. For this one-producer control, the latter's
  `collection.py` change only lifts a restriction. Every leg shares the commit, so
  neither touches this comparison, but step 1's absolute speeds are from `70d79ccb8`
  and are not re-measured against it. Main through `711b9d839` changes only the TLA
  frontier experiment files after those two.

### Digest check (D1, D2)

Packed and solo must give the same training digest at a fixed seed. Whole-file
hashes cannot be that digest: they differ between same-seed runs because of
identity and timing fields. The digest is therefore semantic:

- initial raw, final raw and EMA policy files: every tensor equal, and every other
  field equal except `bc/run_id`;
- the optimizer file: byte-identical;
- `game-attempts.jsonl`: every row equal except `attempt/seconds`, `context/run`, and
  `context/coordinates/{training_seconds, active_training_seconds, host_load_1m,
  rss_bytes, process_cpu_seconds}`;
- the receipt's updates, games, environment decisions and learner transitions.

The comparison lists any other difference by path, so a new volatile field fails
visibly rather than being absorbed. D1 and D2 share a machine because CPU float
results can differ across CPU models. CPU is the only device where equality is
expected: step 1 showed same-seed CUDA runs diverge across machines.

Design-time reading (2026-11-07, laptop, `0e8937f26`, not step evidence): 26201 at
24 streams and 3 updates, once alone and once beside 26202 as plain processes. All
86 tensors in each policy file, the optimizer bytes and all 72 game-attempt rows
matched, with exactly the exclusions above. The check is therefore workable. What the
rented legs add is Etude's concurrent worker, its shared track directory and a Linux
host.

### Metric comparison (M1, M2, M3)

Read from each run's Etude metrics, per update, updates 2-20: policy and value loss,
entropy, approximate KL, `collection_kl`, clip fraction, rows and games per update,
mean game length and capacity-recovered games.

- **Shape.** Per metric and seed, the packed-minus-solo difference of means over
  updates 11-20, next to M3's solo-minus-solo difference at seed 26211. M3 gives the
  scale: CUDA noise alone at a fixed seed.
- **Report.** Each metric's curve for all nine CUDA runs, the four paired differences with a
  seed-level t interval, the first update at which M1 and M2 differ and at which M2
  and M3 differ, and whether any packed seed leaves the range of the solo seeds.
- **Expected reading.** Packed seeds lie within the solo spread, and paired differences
  are about the size of M3's. Four seeds over 20 updates can show a gross leak, not a
  small effect, and the report says so. There is no pass/fail gate; a departure goes
  to evidence review with the curves.
- **Health.** No abandoned games, no replacement debt, and every run reaches 20 updates
  inside its watchdog with at least one update's margin.

### Shape readings from the same runs

These come at no extra cost and are the first measurements of the real launch path:

- Sustained s/update and M decisions/hour at K=4 and K=1 over 20 updates on one host
  (step 1 had 8-12), and $/M decisions at the rental's billed rate.
- Time from placement to first update when four attempts set up at once.
- Peak RSS per run against the memory allowance reported on the pod, and each run's
  peak CUDA bytes (`memory/peak_reserved_cuda_bytes`, new in `405223165`) against the
  GPU's memory.
- Etude's `shared by 4 runs` cost split against `runpodctl billing pods`. Step 1's run
  list understated the bill, so the bill stays the cost of record.
- Whether each run's outputs and metrics stayed separate. Etude gives every run on a
  machine one `ETUDE_TRACK_DIR`.

### Seeds and producers (M1, M2, S14, S22, S44)

Speed only. For each of the five (K, P) shapes, on the one rented machine:

- **Concurrent span.** A packed leg's runs set up separately, so they do not start
  together as step 1's children did, and the first to finish frees the machine for the
  rest. A shape is read only over updates 2 onward that lie wholly inside the span when
  all K runs train. Update rows hold no wall clock, so a run's training start is its
  first `gpu-samples.csv` timestamp, and an update ends at that plus its
  `training_seconds`. The sampler starts a few seconds before training, which is small
  against an update. The report gives each run's count of updates read. Under 3 for any
  run means the shape is **not read**: its table row says so, it gets no recommendation,
  and a longer leg needs an amendment. Solo shapes read updates 2-8.
- **Sustained s/update.** Median over the concurrent span per run, averaged over the
  shape's runs. M1's and M2's full 20 updates are reported beside that. M2 uses its four
  solo runs.
- **M learner decisions per hour** across the machine, summed over concurrent runs
  from those medians, and **per dollar** at the rental's billed rate, excluding setup as step 1 did.
- **GPU busy share.** Every CUDA run samples `nvidia-smi` once a second into
  `gpu-samples.csv` in its output. The sampler reads the whole GPU, and its timestamps
  bound the run. For a shape, take the mean utilization from the last run's first row to
  the first run's last row, dropping the first eighth for start-up, as step 1 did.
- **CPU saturation.** Producers are threads of the run's process, so the update rows'
  `process_cpu_seconds` over `training_seconds`, differenced across the concurrent
  span, is each run's busy cores, as in step 1. Sum over the shape's runs, then divide by the machine's vCPUs. Spinning CUDA waits
  count as busy, so read this beside s/update, not alone.
- **vCPU count.** The provider's figure for the rented pod (`runpodctl`), recorded in
  the report. The pod cannot show it: step 1 read 128 CPUs and no quota on every pod,
  so `HostFacts` does not hold it. Busy cores above that figure would mean the pod can
  use more than it rents, and the report says so.

Each shape is one leg on one machine, with no repeat, and step 1's runs of 8 updates
differed by up to 10% across seeds at K=4. Smaller differences are not read.

**Recommendation.** The report names one seeds x producers shape for each machine
class, A4000 (16 vCPUs) and L4 (6 vCPUs), by dollars per million decisions, and also
gives the fastest shape per seed. Only the rented class is measured. The other's shape
is inferred from busy cores per shape against its vCPUs, labeled as an inference, with
the machine that would confirm it. Expect (4, 4) to oversubscribe an L4: 16 producer
threads and 4 learners on 6 vCPUs. Running it there is deliberate, and a slow reading is
the finding, not a failure.

The two halves of a shape have different owners. K is a launch choice: slots belong to
the on-demand, and no regime byte changes. P is `execution.workers`, a regime field, so
each producer count is a declared variant, and a live continuation must preserve it. The
standard shape picks its machine at rental, A4000 first, after the regime is declared.
So the report also says whether one P serves both classes, and what the worse class pays
for it. If none does, the regime has to be chosen after the machine, and that goes to
Jack Heart with the recommendation. This step adopts no producer count.

### Machine and launch

- **On-demands**, new, `--slots 4`, at most 1 machine: `b015d5-a4000x4` (RTX A4000,
  16 vCPUs, 48 GB, under $.40/hour) and `b015d5-l4x4` (L4, 6 vCPUs, 48 GB, under
  $.60/hour). Both ask for an 80 GB disk, because four attempts each keep their own
  source, venv and native build. The existing `a4000` carries two ghost records
  against its limit. Both are recorded in the isolated home `~/.etude-b015d5-step2`,
  with storage `r2` and provider `runpod` matching the shared launcher.
- **Launch.** `experiments/training_throughput_packing.py --leg
  d1|d2|m1|m2|m3|s14|s22|s44`, with `--cohort` for a leg of several runs, in the order
  of the table. M2 launches one seed at a time with `--seeds`, so a free slot cannot
  take the next seed early.
- **Selection.** Try `etude rent b015d5-a4000x4 --hours 5`. If RunPod has no
  A4000 at launch, rent `b015d5-l4x4` instead. Step 1's A4000 took 16 minutes to
  appear after queueing, so try for up to 30 minutes first. The report names the GPU
  used; A4000 and L4 are not compared in this step.
- **Window.** Expected 3.5 hours: setup about 5 minutes per attempt, D1 and D2 about
  10 minutes each, M1 about 35, 5 solo runs of about 20, S14 and S22 about 10 each and
  S44 about 20. The window is held by hand, so `etude terminate` ends it when S44
  finishes. It does not end itself.
- **Run limits.** Each CUDA run: `max_hours` 1.0, `max_usd` $.75. After
  `training_runs`' 900 s setup and 300 s finalization reserves, that leaves a 2,400 s
  watchdog, against about 1,500 s for 20 packed updates. Each digest run: `max_hours`
  .5, `max_usd` $.45, leaving a 600 s watchdog; one run took 46 s on a laptop through
  the real worker (2026-11-07), so slower rented cores still fit. Each speed run:
  `max_hours` .75, `max_usd` $.60, leaving a 1,500 s watchdog, or 187 s per update
  against step 1's 61-71 s at K=4. A watchdog stop keeps its exports and the updates it
  reached still read. Etude prices each run as if it had the machine alone, and a run
  must fit the rental's remaining time, so S44 needs .75 hours left when it starts, at
  about 3.2 hours. The 5-hour window leaves room for one retry.
- **Setup deadline.** An attempt's `max_hours` starts before setup, and the worker
  refuses to train with less left than its watchdog plus finalization. So setup must
  finish within 900 s or the run fails at admission, having trained nothing. Step 1's
  setups took about 5 minutes alone; four at once have never run, and each builds the
  native extension. D2 is the first reading, so read its `phase setup` times from the
  logs before M1. If two at once took over 7 minutes, launch M1 and S44 as two pairs
  about 8 minutes apart with `--seeds`. That keeps setup inside its deadline at the
  price of the speed reading: pairs overlap only briefly, so K=4 reads few updates at M1
  and probably none at S44, whose first pair can finish 8 updates before the second
  trains (see "Concurrent span"). A run that fails this way is retried staggered, and
  counts as that leg's first failure.
- **Client.** `--slots` is on Etude main (`a7718f5`, PR #85) and unreleased; PyPI's
  latest is 0.2.14. The launcher packages its own client for the pod, so no release is
  needed to run it. But main's launcher schema is version 12, and opening the shared
  `~/.etude/launcher.db` (version 11) upgrades it in place, after which 0.2.14 clients
  in other Tasks refuse it. Step 2 therefore runs from Etude `a7718f5` under an isolated
  `ETUDE_HOME`, as PR #85's canary did, with its own controller. Every Etude command in
  this step sets `ETUDE_HOME=~/.etude-b015d5-step2`, nested Doppler as for step 1, and
  `uv run --with "etude-sdk[s3,eval] @ <Etude at a7718f5>"`. That client must never run
  without `ETUDE_HOME`. Etude main's later `199335c` adds only a checkpoint timestamp,
  so the pin stands. Moving the shared launcher is a release decision outside this step.
  (Since then the shared database has reached schema 12 through MTG-267's client; see step 3's
  "Etude since the design".)

### Software before step 2 freezes

No learning code changes. No frozen world, observation version or recipe changes.

1. **Memory allowance** (`manabot/infra/host.py`). Step 1's gate read
   `psutil.virtual_memory().total`, the 540 GB host, so it could not trip. Add one
   reader of the pod's allowance: the smallest of cgroup v2 `memory.max`, cgroup v1
   `memory/memory.limit_in_bytes`, and `RUNPOD_MEM_GB` if the pod sets it (the name is
   unverified), or nothing if none is readable. `HostFacts` gains an optional
   `memory_limit` holding the bytes and their source, excluded when absent, so retained
   receipts still load. Step 1 read `cpu_quota` as None on every pod, so cgroup v2 at
   that path may not be visible there. Etude's job environment also drops `RUNPOD_*`, so the RunPod figure
   has to come from `/proc/1/environ`. Which source works on a pod is unknown until
   step 2's first receipt.
2. **Probe gate** (`experiments/training_throughput.py`). `run_phase` gates on that
   allowance, falling back to host memory only when none is readable.
   `PhaseResult` records which it used, so a blind gate is visible in the evidence.
   With `--slots` nothing runs this gate, and K=4's 34 GB fits the 48 GB minimum, so
   step 2 does not depend on it. It matters before anyone tries K above 4, and the
   receipts' allowance is what such a choice would read.
3. **Concurrent setup** (`manabot/experiment/job.py`). Etude runs `setup` once per
   attempt, so four attempts would run the `test -x rustup || curl ...` install and
   `rustup toolchain install` at once on one `$HOME/.cargo`. `machine_setup` takes a
   `flock` around those lines where the machine has one, which every rental does. `uv sync` stays per attempt, because each attempt has its own
   source directory and uv and cargo lock their own caches. Four concurrent native
   builds cost minutes per rental, which the shape readings report. Etude's
   `Preparation` builds once per machine, but it does not fit yet: the editable install
   and its `.so` live in the source tree it copies. If setup proves costly, that is a
   later change. The job's setup bytes change for every launch from this commit on,
   which is new job identity, not new behavior.
4. **Launch file** (`experiments/training_throughput_packing.py`, new). Declares the
   four regimes and a `runs(revision, leg, seeds)` with CLI-readable parameters,
   giving each leg its own Etude job identity while the regime bytes stay equal across
   legs, so that repeated seeds are separate runs. It also holds `training_digest` and
   the comparison named above. CUDA legs prepend `GPU_SAMPLER` to the job's command, a
   backgrounded `nvidia-smi -l 1`. The digest legs run without it.
5. **Tests** (`tests/training/test_training_throughput.py` and `tests/infra/test_host_facts.py`).
   Allowance from injected cgroup v1, v2 and environment fixtures, smallest wins,
   absent stays None. Gate trips on an injected allowance below host memory. Digest
   comparison reports an injected change at its path and ignores only the listed fields.
   A two-process CPU test (8 streams, 2 updates, one seed solo then beside a neighbor)
   gives equal digests. Etude's `read_job_file` discovers the launch file and each
   leg's runs; `read_experiment` is not used, because it requires an `eval` and this
   step evaluates nothing. Speed legs differ from M1's regime only in id, updates,
   producers and watchdog. Only CUDA legs sample, and the sampler writes rows, stops
   with the job and leaves its exit status.

Acceptance before freeze: those tests pass, `etude launch ... --quote` under the
isolated home shows the four M1 runs and the four S44 runs each on one machine with 4
slots, and the regime digests of M1, M2 and M3 are equal. Nothing is deleted. `Probe`
and `PHASES` stay as step 1's frozen record.

### Stop rules

- A digest mismatch stops the machine after D2. The legs that follow would have
  nothing to show until it is explained; the difference goes back to Jack Heart.
- A failed run is retried once on the same rental unless it is plainly a dud host.
  A second failure of the same leg ends the step, with evidence kept.
- A slot OOM or SIGKILL at K=4 is itself the finding, and ends the step.
- No machine by the end of the selection wait: nothing is spent, and the wait is
  reported. Both machines are terminated before the report.
- Spend stops at $4.50 against the bill.

### What this does and does not settle

Passing shows that packing changes nothing detectable at this length, on one machine and one
GPU type, and that the launch path works at K=4. It does not measure strength, does not
compare A4000 with L4 as a learning machine, does not show K above 4, and does not
recommend for the long run beyond the 20 updates measured. Long runs' watchdogs must be
sized from K=4's s/update, roughly twice solo.

The digest and metric legs cover one producer per seed. The speed legs add what that
leaves out, at speed only. They read the control's world, and not c5fba4's recipe with
its wider input and roster. They show which shape is cheapest on the rented machine
over 8 updates. They do not show that a packed multi-producer run learns as a solo one
does: the producer half rests on c5fba4's fixed-policy collection check, which has no
trained counterpart, the packing half rests on this step's one-producer legs, and
nothing here runs the two together for learning.

## Step 3: model size, GPU and precision (designed and accepted 2026-11-07)

**Direction.** Jack Heart accepted step 2's evidence (comment 1f4c2872): four seeds per
machine with one producer is the standard cost-efficient shape. He set the objective in
his words: "you have N hours and M dollars, get me the strongest model you can". The
deliverable is a reusable planner that, given hours and dollars, chooses machines, seeds
per machine, host class, the recipe's horizon at the shape's measured seconds per update,
and how the final model is selected, with selection's evaluation counted against the
dollars. He then made model size a planner input (comment 8ac4eb82) and refocused this step
on large-model speed (comment 518e5f09): "i think the bigger models are the ones where speed
is likely more important to figure out", "lets start focus on that". The planner stays the
goal, and this step measures its first input itself rather than waiting for c5fba4's
calibration. Comment b9215b8a adds "where we stop wanting to group seeds": the model size
at which packing stops paying.

**Questions.**
1. Seconds per update, updates per hour and dollars per update for each (model size, GPU,
   precision), with the phase split that explains it.
2. The packing gain, K × s/update(K=1) / s/update(K), against compute parameters for each
   GPU, and the size where K=1 becomes the cheapest shape.
3. Whether TF32 matmul or bf16 autocast buys speed at the larger sizes, and whether a short
   learning-metrics check shows a gross change in learning.
4. What those readings imply for $100 over two weeks at each size.

This step measures speed only. It plays no evaluation deals, scores no checkpoint and makes
no strength claim. Whether a larger model is worth choosing at all is c5fba4's equal-hours
question, which this step does not answer.

### The planner's current reading at w64 ($0, existing evidence)

This is the planner's arithmetic applied to what already exists. The arithmetic is in
`scratch/planner_sketch.py` until the planner is built. Its inputs:

- **Speed.** Seconds per update per seed at w64 (step 2, A4000, EPYC 7702): 37 solo, 50 at
  K=2, 62 at K=4. L4 (step 1): solo 25.4 or 37.0 by host class (31 is used), 36.5 at K=2,
  62 at K=4. Billed $.256/hour (A4000) and $.59 (L4). A run's seconds per update fall as
  its games shorten, and 20 updates cannot show where they settle; MTG-183 ran 34.8 s on
  an L4 over its long windows.
- **Continuation.** RunPod jobs last at most 12 hours, so a two-week seed is about 28
  learner continuations (`manabot.experiment.job.LearnerContinuation`; MTG-183 ran 7).
  Each continuation restarts streams and RNGs and may land on a different host class.
  Taken: 10 minutes of setup and staging per window, and step 2's hang rate (1 of 21
  attempts) at a 600-second stall watcher and one re-setup. That leaves 98.5% of rented
  time training.
- **Curve.** MTG-183's one seed, on world w13 with full heuristic grids (rating, heuristic
  = 1000): 961 at u100, 1031 at u200, 1077 at u400, 1118 at u800, 1166 at u1600, 1206 at
  u3200, 1248 at u5950. Past u400 that is 40-48 Elo per doubling of updates. Nothing is
  measured past u5950, so the late slope is a scenario: 44 per doubling, or 20. The
  current world is w17, and 07c490's baseline (.258 / .335 / .417 at u20 / u40 / u75 on
  `ELEMENTAL_PAIRS`) is the only w17 reading of this recipe's curve.
- **Seed spread.** 07c490 scored 3 baseline seeds at u40: .365, .325, .315. With the three
  arms that sat within .03 of it, 12 seeds have a standard deviation of .027. A
  200-game checkpoint alone has about .033 of game noise, so this spread is consistent with
  no seed effect at all. The seed component's rough 95% upper bound is .023, about 18 Elo
  at that score. There is
  no seed spread past u75 on any world, and 07c490 has no u150 scores: its lambda runs
  stopped at 75 updates and closure's 150-update target was not reached.
- **Selection.** Best of N gains E[max of N] × σ²/√(σ² + σₑ²), with σₑ about 10 Elo at
  2,000 selection games per candidate, discounted by the rank correlation ρ between a cut
  and the end when a plan prunes early. Selection costs a cpu32 pass at $1/hour and about
  13,000 games/hour (07c490), plus 15 minutes of setup: about $.87 to choose among 4.

The plans compared are one seed solo per machine ("solo"), four packed per machine
throughout ("pack-4"), and successive halving on each machine, 4 → 2 → 1 over 3:4:7 of
the window ("halve"). Central scenario: late slope 44, σ = 10 Elo, ρ = .8. Flip marks the
cells where σ = 0 or (slope 20, σ = 18) chooses another plan. "Machines" means the
assumed limit of three A4000s at once; stock has often allowed one.

| Dollars | Hours | Binds | Central plan (A4000, at most 3) | Final updates | Dollars | Flips to (σ = 0; slope 20 with σ = 18) |
|---:|---:|---|---|---:|---:|---|
| 10 | 6 | hours, machines | solo × 3, best of 3 | 575 | $5.32 | solo × 1; none |
| 10 | 24 | hours; dollars allow 1 machine | solo × 1 | 2,300 | $6.14 | none; halve × 1 |
| 10 | 72 | dollars | solo × 1 | 3,743 | $10.00 | none; none |
| 50 | 6 | hours, machines | solo × 3 | 575 | $5.32 | solo × 1; none |
| 50 | 24 | hours, machines | solo × 3 | 2,300 | $19.14 | solo × 1; none |
| 50 | 72 | hours; dollars allow 2 machines | solo × 2 | 6,899 | $37.42 | solo × 1; halve × 2 |
| 100 | 6 | hours, machines | solo × 3 | 575 | $5.32 | solo × 1; none |
| 100 | 24 | hours, machines | solo × 3 | 2,300 | $19.14 | solo × 1; none |
| 100 | 72 | hours, machines | solo × 3 | 6,899 | $56.01 | solo × 1; halve × 3 |
| 100 | 120 | hours, dollars, machines | solo × 3 | 11,498 | $92.87 | solo × 1; halve × 3 |
| 100 | 336 | dollars | solo × 1 | 32,195 | $86.02 | none; halve × 1 |

What the table supports:
- **Pack-4 throughout never wins a strength plan.** It is the cheapest shape per decision,
  but in a fixed budget a final seed's own updates count, and four seeds share them. At
  ($100, 336 h) on one A4000 it reaches 19,213 updates against 32,195 solo, 0.74
  doublings fewer, which best of 4 recovers only if σ is above about 30 Elo.
- **The worked example, ($100, 336 h).** Dollars bind: one A4000 for 14 days is about
  $86. Halving reaches 27,021 updates for its survivor, at $87.44 including three selection
  passes. Solo reaches 32,195. Halving pays when σ exceeds about 13.5 Elo at slope 44, or
  about 8 at slope 20, with ρ = 1 (16 and 9 at ρ = .8). The u40 bound is 18 Elo, and no
  late spread is known. **This choice is decided by the two unknowns, not by the
  evidence.** Halving is the safer choice when the late curve flattens, when late seeds
  spread more than early ones, or when a seed can fail in ways a stall watcher does not
  catch. Solo is the safer choice when the curve keeps its slope and seeds converge.
- **L4 only, ($100, 336 h).** $100 buys about 169 L4 hours, so the plan is dollar-bound in
  a week: solo reaches 19,383 updates, and halving compresses 4 → 2 → 1 into that week.
- **When hours bind,** spare dollars buy more solo seeds on more machines (best of N),
  never packing. Two levers then raise a seed's own speed: renting the faster host class
  (step 1's EPYC 9254 L4 ran 1.46× faster than the EPYC 7702 hosts), and the deferred
  stop-waiting and overlap candidates. Only these cells give those candidates a use.
- **Model size moves every row.** At a larger model each seed's update costs more, so
  every horizon shrinks and the curve's early, steep part matters more. That is why this
  step measures size first.

### Sizes

Declared variants of the control, `elemental_variants.elemental_recipe`, that differ only
in the agent's width, depth, heads and feed-forward width, each with its own regime ID.
Compute parameters exclude embedding tables, counted as c5fba4 counts them, on this
control's input as it resolves, policy input v6 with 512 facts (counted 2026-11-07 at
`98573bdf9`; 07c490's protocol calls the recipe's input v1, and the resolved regime
reads v6):

| Label | Width | Layers | Heads | Feed-forward | Compute parameters | × w64 |
|---|---:|---:|---:|---:|---:|---:|
| w64 (the control) | 64 | 2 | 4 | 256 | 243,842 | 1.0 |
| w128l4 | 128 | 4 | 4 | 512 | 1,334,786 | 5.5 |
| w160l8 | 160 | 8 | 8 | 1,280 | 4,953,602 | 20.3 |
| a384l8 | 384 | 8 | 8 | 1,536 | 18,868,226 | 77.4 |

The ladder steps about 4× in compute. W160l8 has c5fba4's shape, but on this control's
input, so its count differs from c5fba4's 5,341,922 on the TLA input. A384l8 is the
Ataraxos-sized network Jack Heart named (384 × 8, feed-forward 1,536). Feed-forward is the
model's default (heads × width) except for a384l8. Nothing in c5fba4's frozen regimes
changes. If c5fba4's w160l8 calibration reads first, it is a cross-check of the trend
only: its input, roster and producer count differ.

### Precision

Training is float32 only: `Execution.precision` admits only `"float32"`, and nothing in
`manabot/` enables TF32, autocast or bfloat16. This step adds two declared execution
options, used only by step 3's variants:

- `"tf32"`: TF32 tensor-core matmul and convolution (`torch.backends.cuda.matmul` and
  `cudnn` `allow_tf32`) for the run's process. Weights and activations stay float32.
- `"bf16"`: `torch.autocast("cuda", dtype=torch.bfloat16)` around the policy forward in
  collection and the learner's forward and losses. Weights, optimizer state, targets and
  the loss reductions stay float32, and no gradient scaler is used, as bf16 does not need
  one.

Both change numerics, so each is a new regime, never an edit to the control. Omitting the
field still means float32, so retained regimes keep their bytes. CUDA only: a CPU regime
naming either is rejected. A4000 (Ampere), L4 and RTX 4090 (Ada) support both.

**Learning-metrics check.** At w160l8 on the third GPU, seed 26301: float32 twice
(the repeat is the noise scale, as M3 was), TF32 once and bf16 once, 12 updates each, K=1.
Step 2's metrics (losses, entropy, both KLs, clip fraction, rows, games and decisions per
game), compared over updates 5-12 as differences from the first float32 run, beside the
float32 repeat's difference. One seed and 12 updates can show a gross change only. Same-seed
CUDA runs already diverge at update 2, so equal metrics are not expected. A precision whose
differences are several times the repeat's goes to evidence review, and no reading here
approves a precision for training. These four runs are also that size's K=1 cells on that GPU:
the first float32 run replaces the ladder's 4-update cell.

### Machines

One hand-held rental per GPU, legs run in turn as in step 2, so every cell on a GPU shares
a host. Each GPU's report names its host CPU and the provider's vCPU count.

- **A4000** (`b015d5-a4000x4`, $.25/hour, 16 vCPUs, 16 GB): the standard shape's machine.
- **L4** (`b015d5-l4x4`, under $.60/hour, 6 vCPUs, 24 GB): the fallback machine.
- **A third, faster GPU**, chosen at freeze from live RunPod quotes: the card with the most
  dense bf16 tensor throughput per quoted dollar among RTX 4090, RTX 5090, A40 and L40S,
  with at least 24 GB and 8 vCPUs, under $.95/hour. RTX 4090 is the default. A new
  on-demand `b015d5-fast` in the isolated home, `--slots 4`, 80 GB disk.

If one GPU cannot be rented within 30 minutes, its cells are reported as not read, and the
other GPUs run. Nothing is substituted without an amendment.

### Cells

Every cell is the control at 256 streams, both seats, one producer, CUDA, on seeds
26301-26304, with no evaluation. Speed cells run 4 updates and read updates 2-4 inside the
concurrent span, as step 2 read its legs. The packing ladder on each GPU, float32:

| Size | K | Runs |
|---|---|---|
| w64 | 1, 4 | new only on the third GPU; A4000 from step 2 and L4 from step 1 |
| w128l4 | 1, 2, 4 | each GPU |
| w160l8 | 1, 2, 4 | each GPU |
| a384l8 | 1, 2, 4 | each GPU |

Precision cells, K=1: w160l8 and a384l8 at TF32 and bf16 on each GPU, w160l8's on the third
GPU being the learning-metrics check.

Rules, applied in order, smallest size first, so later cells are sized by earlier ones:

- **Memory.** K runs only where K × the size's K=1 peak reserved CUDA bytes
  (`memory/peak_reserved_cuda_bytes`) fits in 85% of the GPU's memory, and K × peak RSS fits
  in 85% of the pod's allowance. A skipped cell says which limit stopped it.
- **Crossover.** Once a size's K=2 gain on a GPU is under 1.10, the larger sizes on that GPU
  skip K=2 and K=4: packing is taken not to pay past its crossover. Step 1 saw 10% spread
  across K=4 seeds, so a gain under 1.10 is not distinguishable from 1. This assumes the
  gain falls with size; the report says each place it rests on the assumption.
- **K=4** runs only where the K=2 gain is at least 1.30.
- **Precision skip.** If TF32 or bf16 at w160l8 is within 10% of float32 on a GPU, a384l8
  still runs it, since larger matmuls are where tensor cores should matter.

### Readings

Per cell, from the run receipts and `gpu-samples.csv`, as step 2:
- s/update (median over the read updates, averaged over the cell's runs), updates per hour
  per seed, machine M learner decisions per hour, dollars per update and per M decisions
  at the billed rate;
- the phase split: native stepping, encoding, inference, learning, and the remainder;
- GPU busy share, busy cores against the pod's vCPUs, peak reserved CUDA bytes and peak RSS;
- the packing gain at K=2 and K=4.

Then per GPU and precision:
- **Packing gain against compute parameters,** on a log axis with the w64 rows, and the
  crossover: the smallest size whose K=2 gain is under 1.10, bracketed by the two sizes on
  either side.
- **Launch-bound or compute-bound.** Learning seconds per update against compute
  parameters at K=1. Learning that grows far slower than parameters, with the GPU less than
  half busy, is launch-bound: the learner takes about 190 optimizer steps of 256 rows
  per update (`minibatch_rows` = streams). If the larger sizes read launch-bound, the
  report names the learning pass's minibatch and launch shape as the next candidate. It is
  not built here. A larger minibatch changes optimization, so it would be a declared
  variant with its own learning check, and kernel fusion or CUDA graphs would need a digest
  check.
- **$100 over two weeks, per size.** On the cheapest measured (GPU, precision, K): updates
  per hour, the hours $100 buys after an evaluation and continuation allowance, whether
  hours or dollars bind, and the final seed's updates under one solo seed, halving and
  pack-4, with a curve for each size marked unknown. Curves for the larger sizes do not
  exist, so this gives updates, not strength.
- **The planner's speed table,** one row per measured (size, GPU, precision, K) with its
  receipts, for the planner that follows this step.

### Software before step 3 freezes

No learning-rule change. No frozen world, observation version or recipe changes.

1. **Size variants** (`experiments/elemental_variants.py`): `sized(regime_id, updates,
   width, layers, heads, feedforward=None)` over `elemental_recipe`, changing only those
   agent fields. Test: each variant's dump equals the control's outside the agent's size
   fields and its ID, and the compute-parameter counts above.
2. **Precision options** (`manabot/training/regime.py`, `manabot/training/execution.py`):
   `Execution.precision` admits `"tf32"` and `"bf16"`; one helper applies the run's
   precision at the collection forward and the learner's forward. Tests: float32 regimes
   keep their bytes and digests (the CPU digest of step 2's D1 unchanged); CPU regimes
   naming either option are rejected; with an injected CUDA flag, `tf32` sets the two
   `allow_tf32` switches and `bf16` enters autocast only around the declared forwards.
   No local test can measure CUDA numerics; the learning-metrics check is that test.
3. **Launch file** (`experiments/training_throughput_sizes.py`, new): the cells above as
   legs `--leg <gpu>-<size>-k<K>-<precision>`, each a declared regime, reusing step 2's
   `GPU_SAMPLER` and speed-leg helpers from `experiments/training_throughput_packing.py`.
   Run limits from each size's expected time: the watchdog is the larger of 1,500 s and 6×
   the previous size's K=1 s/update × the cell's updates, so a large model does not trip
   it. The stall watcher is the larger of 600 s and 3× that s/update.
4. **Readings** (`experiments/data/training-throughput-b015d5-v1/step-3/readings.py`),
   producing every report figure, as step 2's.
5. **On-demand** `b015d5-fast` in the isolated home, and a quote for every GPU.

Acceptance: the tests pass; `read_job_file` discovers each leg; a CPU run of one w128l4
and one a384l8 leg at 8 streams and 1 update finishes through the real worker; and
`etude launch ... --quote` under the isolated home places a K=4 leg on each on-demand.
Etude stays at `a7718f5` under `~/.etude-b015d5-step2`, as in step 2.

**Etude since the design (checked 2026-11-07).** Main's `adf319ee0` (MTG-267) pins the job
project's `etude-sdk` to Etude main `199335c`, whose only change over `a7718f5` is
`Checkpoint.wall_ms`, and this branch carries it through `e8dbf0b02`. Step 3's jobs therefore
run SDK `199335c` under an `a7718f5` client. Older-client-to-newer-job is the direction that
commit found safe; no step 3 leg has run that pairing yet. The shared `~/.etude/launcher.db`
now reads schema 12, as the isolated home does, so the reason step 2 gave for isolation (an
upgrade locking out 0.2.14 clients) has already happened. The freeze keeps the accepted
client: Etude `a7718f5` in `~/.etude-b015d5-step2`, which the acceptance quotes used and
step 2 ran. That is the setting the design accepted, and its remaining reason is a client
that step 2 has exercised on every launch path. Moving the client to `199335c`, or dropping
the isolated home, would be an amendment with its own re-quote. The re-quote at the launch
commit is the first check of an `a7718f5` client placing `199335c` jobs.

**As built (2026-11-07).**
- Sizes: `elemental_variants.sized`; `training_throughput_sizes.recipe(size, precision,
  updates)` names each regime `elemental-throughput-<size>-<precision>[-check]-v1` (the
  12-update check legs carry `-check`). W64 float32 is the control's bytes.
- Precision: `manabot/training/precision.py` owns both options. `apply_matmul_precision`
  sets the TF32 switches at each stage's start and restores the process's own at the run's
  end, so float32 keeps PyTorch's defaults (cuDNN's allows TF32, as before). Collection
  and the learner wrap each policy forward in `forward_precision` and cast its outputs to
  float32, so behavior log-probabilities, targets and losses are computed in float32.
  This covers the damped rule and the clipped-surrogate rule alike.
- **Not anticipated in the design: the network did not run under autocast.** Its
  scatter, `index_add`, `index_put` and `masked_scatter` joins (permanent links, facts,
  shared effects, state records, name candidates, action logits) refuse mixed dtypes,
  and autocast leaves them mixed. Each now converts its source to the destination's dtype.
  For float32 that is a no-op. Checked on CPU: step 2's D1 variant (24 streams, 3 updates,
  seed 26201) through the real worker at `0d3af321e` and on this tree gives digests that
  differ only in `training_source_sha256`, across 72 game attempts, all three policy
  exports and the optimizer. That is the float32-unchanged acceptance, run by hand once.
  It is not a CI test, because a pinned training digest would move with every rules change.
- bf16 is exercised on CPU autocast by a real two-stream collection and damped update
  (`tests/training/test_precision.py`): every agent forward ran under autocast, and
  outputs and parameters stay float32. CPU autocast stands in for CUDA's here. It covers
  the same forwards but says nothing of CUDA numerics.
- Launch file: 42 legs. `--pace` passes the previous size's K=1 s/update (default 37,
  step 2's solo A4000 w64). Hours round up to quarter hours and the watchdog takes the
  slack. `max_usd` is hours plus 15 minutes at the on-demand's ceiling. `limits()` and
  `python -m experiments.training_throughput_sizes LEG PACE` print a cell's stall time.
- Readings: `step-3/readings.py` and `collect.py` (step 2's, for this experiment) read
  `legs.json`, `machines.json` and `runs/`. Dry-run on step 2's runs relabeled as cells:
  every section printed. Unpublished until step 3's evidence exists.
- On-demand `b015d5-fast` in the isolated home: `NVIDIA GeForce RTX 4090`, 8 vCPUs,
  48 GB, 80 GB disk, under $.95/hour, `--slots 4`, storage `r2`. The freeze may swap the
  GPU for the quoted best of the four candidates.
- Quotes, 2026-11-07 about 01:48 UTC, isolated home, `--revision HEAD` at `e3ccbb0d7`,
  leg `<gpu>-w128l4-k4-float32`. Each on-demand places all four seeds on one machine at 4
  slots: A4000 $.25/hour (first-wave $.25), L4 $.59 ($.59), RTX 4090 $.89 ($.89). Nothing
  rented. Stock moves, so re-quote at the launch commit.
- Checks: `pytest tests/training/test_precision.py tests/training/test_training_throughput_sizes.py
  tests/training/test_training_throughput.py tests/infra/test_host_facts.py
  tests/training/test_elemental_closure_lambda.py tests/training/test_two_seat_learning.py
  tests/training/test_rl_objectives.py tests/training/test_regimes.py tests/model` passed,
  after rebuilding the native extension that main's Rust had outdated.

**Step 3 freeze receipt (2026-11-08 UTC).** Jack Heart accepted software review and directed
this freeze at PR #448's merge commit (comment a5890a28). The launch revision is
`cb722dc8c94086cb52b6b39a4ea7b50d7f1f529e`, on main, for every leg. Its tree equals the
reviewed branch head `cb184cb89`. This receipt is recorded after that commit and changes no
executable byte.
- Source bundle: every leg's job carries one bundle of 1,483 files, sorted-key SHA-256
  `dc98b968…6746`: all 1,484 tracked files except the `CLAUDE.md` symlink. The method
  reproduces step 2's `6334a27b…1ff3` at `668212567`. Key files:
  `experiments/training_throughput_sizes.py` `fa8c181e…d603`, `experiments/elemental_variants.py`
  `3a9dd378…1bc2`, `manabot/training/precision.py` `d152bd6f…23c1`,
  `manabot/training/regime.py` `f0856e42…814d`, `manabot/training/execution.py`
  `2c250340…cb94`, `manabot/training/rollout/collection.py` `24eb0b15…9253`,
  `manabot/training/learning/learner.py` `94e51ba2…1447`. Unchanged from step 2:
  `experiments/training_throughput_packing.py` `9d1cd47b…2f4d`, `experiments/training_throughput.py`
  `f36641e3…cd19`, `experiments/learning_signal.py` `836230fd…9cb0`, `manabot/experiment/job.py`
  `62aa4fd3…5aca`, `manabot/infra/host.py` `7cd0dbbd…208b`, and the frozen regime
  `elemental-long-183-v1.json` `be9de376…64ad`. `uv.lock` is now `6e3a1426…be18`, which pins
  jobs to Etude SDK `199335c` (see "Etude since the design").
- Regimes, on w17. Each digest is the sorted-key SHA-256 of the job's regime with
  `wall_seconds` removed, because `--pace` sets the watchdog at launch. With the watchdog
  left in, the digests are recorded at launch.
  - Speed cells, 4 updates: w64 float32 `8cea4f99…5ed2`; w128l4 float32 `851217e0…58fa`;
    w160l8 float32 `c5dc36ea…7e74`, tf32 `1dcdd470…9e93`, bf16 `827d4053…6f13`; a384l8
    float32 `86f74df8…b65b`, tf32 `48ac0bd7…8365`, bf16 `3f8bfbe0…940a`.
  - Learning check, 12 updates: w160l8 float32 `68e24691…f45f` (equal in the check and its
    repeat), tf32 `af6e178b…36cd`, bf16 `56785682…857c`.
  - A regime's bytes are equal across GPUs and K. At the default pace (37 s), every speed
    cell's job reads 0.75 hours and the check legs read 1.25 hours.
- Float32 unchanged: step 2's D1 regime, discovered on this tree, still digests to
  `0366ff8d…1c3e`.
- Discovery: Etude `read_job_file`, client `a7718f5`, read all 42 legs at this revision
  without executing games or training. That gives 81 runs, each leg with its own job name
  and its declared seeds: K of 26301-26304.
- Acceptance: the "As built" checks pass on this tree, `tests/model` included: 380 passed
  and 1 skipped. Through the real worker on CPU at 8 streams and 1 update, the w128l4 and
  a384l8 cells complete
  (`test_a_size_cell_trains_through_the_real_worker_on_cpu`, 2 passed). The hand D1 training
  digest check ran at build and is not repeated here.
- Isolation: step 3 plays no evaluation deals and scores no checkpoint, so no held-out set
  or anchor is touched. Seeds 26301-26304 are disjoint from every earlier step's.
- Client: as step 2's, `ETUDE_HOME=$HOME/.etude-b015d5-step2`, Etude `a7718f5` through
  the home's `e` wrapper. The shared controller is not used. The isolated controller is
  stopped, and the home holds no rented machine.
- **Third GPU: A40.** The machines rule takes the card with the most dense bf16 tensor
  throughput per quoted dollar. Only two candidates fall under $.95/hour at RunPod's
  secure-cloud price, the price the on-demands pay: RTX 4090 at $.89 (165 dense bf16
  TFLOPS with FP32 accumulate, NVIDIA's Ada figure; about 186 per dollar-hour) and A40 at
  $.59 (150, NVIDIA's datasheet; about 254). RTX 5090 at $1.19 and L40S at $1.09 are over the
  ceiling. The rule picks the **A40**, as on-demand `b015d5-a40` in the isolated home: A40,
  at least 8 vCPUs, 48 GB, 80 GB disk, under $.95/hour, `--slots 4`, storage `r2`. Leg names
  keep the `fast-` prefix and its $.95 ceiling. `b015d5-fast` (RTX 4090) stays defined and
  is not used. The A40 has 48 GB, and its memory bandwidth is below the 4090's. The rule
  reads tensor throughput only, so the 4090 may still be faster per hour. The cheaper card
  is the one this rule selects.
- Quote, 2026-11-08 03:30-03:31 UTC, at this revision, leg `<gpu>-w128l4-k4-float32`,
  nothing rented:
  - `b015d5-l4x4`: L4, at least 6 vCPUs, $.59/hour, rentable. All four seeds share one L4
    at 4 slots, with a first wave of $.59.
  - `b015d5-a40`: A40, at least 8 vCPUs, $.59/hour, rentable. All four seeds share one at
    4 slots, with a first wave of $.59.
  - `b015d5-a4000x4`: **open.** RunPod reports no A4000 ($.25/hour, ceiling $.40), and
    Etude refuses the quote. The machines rule's 30-minute wait covers this, as in step 2.
  - For reference, `b015d5-fast` quoted RTX 4090 at $.89/hour, rentable.
  - These quotes are the first placement of `199335c` jobs by the `a7718f5` client. Placement
    is planned, and no leg has run under that pairing.
- Budget at the quoted rates, hand rentals priced from the rate as in step 2: A4000 6 h is
  $1.50, L4 4 h is $2.36, and A40 5 h is $2.95. The $10.00 reservation and its lines stay as
  written. The third-GPU line is now $1.80 under its $4.75.
- Authority: step 3 stays under comment 9f9d0956's $50 inclusive cap, with $10.00 reserved
  beside the $2.04 spent. Freezing confers no spending authority. Nothing launches or rents
  before execution review with Jack Heart.

### Stop rules

- A failed run is retried once on the same rental unless the host is plainly a dud; a
  second failure of the same cell ends that GPU's ladder with its evidence kept.
- A CUDA out-of-memory at a K the memory rule admitted is itself a reading: the cell is
  recorded and the larger K on that size is skipped.
- A precision whose learning-metrics differences exceed several times the float32 repeat's
  is reported, and its speed cells are still read.
- Each rental ends by `etude terminate` after its last cell, and spend stops at the line.

### Budget

| Line | Window | Reserved |
|---|---:|---:|
| A4000 rental, all cells | 6 h | $1.56 |
| L4 rental, all cells | 4 h | $2.40 |
| Third GPU rental, all cells and the learning check | 5 h | $4.75 |
| Failed attempts and a replaced host | | $1.29 |
| **Step 3 total** | | **$10.00** |

Cell times at the larger sizes are what this step measures, so the windows are estimates:
about 3.5 hours of cells on the A4000 if a384l8 runs near 350 s/update at float32. The
crossover rule removes the slowest cells. With the $2.04 spent, the experiment's reserved
total is about $12.04 of the $50 cap.

### What this does and does not settle

It gives the speed of each size per GPU and precision, the size where packing stops paying,
and the updates $100 buys at each size. It does not show that a larger model is stronger,
that a precision learns as float32 does beyond a gross check, or that four updates predict a
long run. The planner after this step takes these speeds, the curves that c5fba4 and later
runs measure, and a measured seed spread; the w64 table above shows the planner's choice
at ($100, 336 h) turns on that spread and on the late curve.

## Stop waiting for the slowest game (deferred candidate)

This was numbered step 3 until comment 1f4c2872 gave that number to the planner. Jack
Heart deferred it on 2026-11-07 because it needs paid strength runs, and comment
1f4c2872 keeps it a candidate only where the time budget binds. Step 1
read its ceiling as 33-38% of an update, and probably loose. It is not authorized
and nothing below is built. The design stands for a later choice.

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
  per-seat carry-over. Acceptance: with the closure set to complete games, digests are unchanged.
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

## Step 4: overlap collection and learning (dropped)

Jack Heart dropped this on 2026-11-07. Step 1's bound was .39-.42 of an update, but
packing already fills the GPU that overlap would use, so the two savings do not
add. The design is at `0e8937f26`.

## Stop-waiting strength arms (if chosen)

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
- **Host confound.** Each seed's control and arm run on one machine through Etude
  slots; if they cannot, runs are screened with #435's band: the median of
  `time/collection/environment_step_and_encoding` over updates 2-11, above
  9 s is slow. Slow hosts are cancelled (`etude cancel`, never deleting the pod
  first) and relaunched. The cost of cancelled runs is charged and their
  receipts are kept. Wall-matched comparisons use only fast-host runs, and the
  report states how many were replaced.
- **Inconclusive.** Both intervals include zero and the paired point estimate
  is within ±.03. Otherwise the result is directional, and a claim needs an
  interval that excludes zero.

## Budget

L4 at about $.59/hour, ceiling $.60; A4000 ceiling $.40. Each line is reserved at
its ceiling.

| Line | Runs | Window | Reserved | Spent |
|---|---:|---:|---:|---:|
| Step 1 probes | 3 | | $2.80 | about $1.06 |
| Step 2 machine, L4 worst case | 19 on one rental | 5 h | $3.00 | about $.98 (A4000, 3.85 h) |
| Step 2 failed attempts and a replaced host | | | $1.50 | in the line above (one hung attempt, about $.25) |
| **Step 2 total** | | | **$4.50** | |

Expected step 2 cost: about $.90 on an A4000, about $2.10 on an L4. Step 1's unspent
$1.74 is released. Stop waiting's lines were $17.40 at most: calibration, control and
exact quota seeds, their evaluation and its reserve. They come back in an amendment if it
is chosen; overlap's are gone. Step 3's lines are in its own budget table. Every line falls under the $50 cap, which includes step 1.

## Stop and recovery

- Stop launching when billed plus reserved spend would exceed the step's line.
  Exceeding it needs an amendment; exceeding $50 is never allowed.
- A machine not booted within 15 minutes is terminated and its window charged.
  `runpodctl billing pods` reconciles GPU spend. CPU evaluation machines appear
  in no bill, so their rental windows are the estimate.
- If a probe fails twice on its own errors (not a dud host), step 1 stops and
  the report says why. A run stopped by its watchdog keeps its exports;
  comparisons extend only as far as every run reached.
- Calibration failure, or no quota within 10% of the realized rows, stops
  stop waiting before scoring.
- Seeds are cut equally across arms before launch, never after scoring.

## Deliverable

`experiments/data/training-throughput-b015d5-v1/report.md` in the artifact
store. It contains the step-1 table (per machine and K: s/update, phase tree,
idle share, CPU cores, GPU busy share, peak memory, $/M learner decisions), the
step readings and choices, step 2's digest result, metric comparison and shape
readings, its seeds x producers table and shape per machine class, step 3's speed table
per (size, GPU, precision, K) with its packing crossover and $100-over-two-weeks reading,
cost against this budget, and the machine recommendation that updates
`experiments/training-update-time.md`.
