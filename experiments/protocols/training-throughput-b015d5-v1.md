# Training throughput on the ten-deck Elemental recipe, v1 (step 2 amendment, not frozen)

Task b015d5. Question: for two-seat ten-deck Elemental self-play, which machine
shape and training regime give the most strength per dollar and per hour? The
candidates were packing several seeds on one machine, not waiting for each
batch's slowest game, and overlapping collection with learning. Step 1 measured
all three. Packing is adopted and validated below, stop-waiting is deferred and
overlap is dropped. Multi-producer collection belongs to Task c5fba4 and is read
from its result, not repeated here.

Status: design accepted by Jack Heart at experiment-design review (2026-11-06,
comment 14e6cc54), with fixed gates replaced by step-at-a-time choice. Step 1 has
run (see its execution record). Jack Heart accepted its evidence and chose step 2
(2026-11-07, comment 9d671e78) and accepted its design, including running from Etude
main `a7718f5` under an isolated `ETUDE_HOME` (comment 850a636c). Step 2's software is
built; it has not frozen or launched, and execution review returns to Jack Heart.
Acceptance confers no spending authority beyond that review. Each step freezes after
the previous step's readings have been discussed and that step chosen, before any of
its runs.

## Authority and budget

- **Execution authority.** Jack Heart's Task comment 9f9d0956 (2026-11-06) says
  to run this experiment under a USD 50 hard cap. The cap covers training,
  evaluation, failures and machine setup, and the comment asks to spend no more
  than needed. The same comment sends design and execution review to Jack Heart
  in the Task conversation, so no launch happens before that review.
- **Proposed spend.** Step 1 cost about $1.06. Step 2 reserves $3.50, under Jack
  Heart's "well under $5". Total reserved: about $4.56 of the $50 cap, which still
  includes step 1. Anything further needs an amendment made before the run that
  would spend it.

## Fixed identities

- **Source.** Main after PR #435 (`45cc961a2`) and PR #436 (`ebed0963e`, 07c490's
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
| Collection share and the measured idle share | Step 3 (stop waiting), sized by (collection - assembly) x idle share |
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

**Claim under test.** A packed seed learns as a solo seed does. Each seed is its own
process with its own seed, model, optimizer and environments, and they share only
hardware, so this holds by construction unless something leaks between them: a
shared path, a wall-clock dependency, or Etude's new concurrent worker. No strength
runs are needed, because the learning rule is unchanged. Strength per dollar is the
control's strength per update times updates per dollar. This step plays no
evaluation deals and scores no checkpoint.

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

- **Digest variant** `elemental-throughput-digest-v1`: the control recipe at 24
  streams, 3 updates, device CPU, one thread. It is a check, not a training shape.
- **Metric regime** `elemental-throughput-pack-check-v1`: the control recipe
  (`elemental_recipe`) to 20 updates at 256 streams, CUDA, with the same regime
  bytes in M1, M2 and M3. Damped-policy rates follow the absolute iteration, so
  the shorter endpoint leaves the learning schedule unchanged.
- **Seeds** are disjoint from 26001-26008, from strength seeds 26101-26104 and from
  the `ELEMENTAL_PAIRS` deals.
- **Source.** One launch commit containing the software below. Main's `43150799f`
  (fact and shared-effect embeddings) arrives with it. That commit's model change
  does not touch this comparison, since every leg shares the commit, but step 1's
  absolute speeds are from `70d79ccb8` and are not re-measured against it.

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
- Peak RSS per run against the memory allowance reported on the pod.
- Etude's `shared by 4 runs` cost split against `runpodctl billing pods`. Step 1's run
  list understated the bill, so the bill stays the cost of record.
- Whether each run's outputs and metrics stayed separate under the shared
  `ETUDE_TRACK_DIR`.

### Machine and launch

- **On-demands**, new, `--slots 4`, at most 1 machine: `b015d5-a4000x4` (RTX A4000,
  16 vCPUs, 48 GB, under $.40/hour) and `b015d5-l4x4` (L4, 6 vCPUs, 48 GB, under
  $.60/hour). Both ask for an 80 GB disk, because four attempts each keep their own
  source, venv and native build. The existing `a4000` carries two ghost records
  against its limit. Both are recorded in the isolated home `~/.etude-b015d5-step2`,
  with storage `r2` and provider `runpod` matching the shared launcher.
- **Launch.** `experiments/training_throughput_packing.py --leg d1|d2|m1|m2|m3`, with
  `--cohort` for a leg of several runs. M2 launches one seed at a time with
  `--seeds`, so a free slot cannot take the next seed early.
- **Selection.** Try `etude rent b015d5-a4000x4 --hours 4`. If RunPod has no
  A4000 at launch, rent `b015d5-l4x4` instead. Step 1's A4000 took 16 minutes to
  appear after queueing, so try for up to 30 minutes first. The report names the GPU
  used; A4000 and L4 are not compared in this step.
- **Window.** Expected 2.7 hours: setup about 5 minutes per attempt, D1 and D2 about
  10 minutes each, M1 about 35, and 5 solo runs of about 20. The window is held by hand,
  so `etude terminate` ends it when M3 finishes. It does not end itself.
- **Run limits.** Each CUDA run: `max_hours` 1.0, `max_usd` $.75. After
  `training_runs`' 900 s setup and 300 s finalization reserves, that leaves a 2,400 s
  watchdog, against about 1,500 s for 20 packed updates. Each digest run: `max_hours`
  .5, `max_usd` $.45. Etude prices each run as if it had the machine alone, and a run
  must fit the rental's remaining time, so M3 needs its whole hour left when it
  starts, at about 2.4 hours. This is why the window is 4 hours.
- **Client.** `--slots` is on Etude main (`a7718f5`, PR #85) and unreleased; PyPI's
  latest is 0.2.14. The launcher packages its own client for the pod, so no release is
  needed to run it. But main's launcher schema is version 12, and opening the shared
  `~/.etude/launcher.db` (version 11) upgrades it in place, after which 0.2.14 clients
  in other Tasks refuse it. Step 2 therefore runs from Etude `a7718f5` under an isolated
  `ETUDE_HOME`, as PR #85's canary did, with its own controller. Moving the shared
  launcher is a release decision outside this step.

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
   `flock` around those lines. `uv sync` stays per attempt, because each attempt has its own
   source directory and uv and cargo lock their own caches. Four concurrent native
   builds cost minutes per rental, which the shape readings report. Etude's
   `Preparation` builds once per machine, but it does not fit yet: the editable install
   and its `.so` live in the source tree it copies. If setup proves costly, that is a
   later change. The job's setup bytes change for every launch from this commit on,
   which is new job identity, not new behavior.
4. **Launch file** (`experiments/training_throughput_packing.py`, new). Declares the
   two regimes and a `runs(revision, leg, seeds)` with CLI-readable parameters,
   giving each leg its own Etude job identity while the regime bytes stay equal across
   legs, so that repeated seeds are separate runs. It also holds `training_digest` and
   the comparison named above.
5. **Tests** (`tests/training/test_training_throughput.py` and `tests/infra/test_host_facts.py`).
   Allowance from injected cgroup v1, v2 and environment fixtures, smallest wins,
   absent stays None. Gate trips on an injected allowance below host memory. Digest
   comparison reports an injected change at its path and ignores only the listed fields.
   A two-process CPU test (8 streams, 2 updates, one seed solo then beside a neighbor)
   gives equal digests. Etude's `read_job_file` discovers the launch file and each
   leg's runs; `read_experiment` is not used, because it requires an `eval` and this
   step evaluates nothing.

Acceptance before freeze: those tests pass, `etude launch ... --quote` under the
isolated home shows the four M1 runs on one machine with 4 slots, and the regime
digests of M1, M2 and M3 are equal. Nothing is deleted. `Probe` and `PHASES` stay as
step 1's frozen record. The `digest2.py` reference is gone: no such file was kept, and
the exclusions are now listed above.

### Stop rules

- A digest mismatch stops the machine after D2. The legs that follow would have
  nothing to show until it is explained; the difference goes back to Jack Heart.
- A failed run is retried once on the same rental unless it is plainly a dud host.
  A second failure of the same leg ends the step, with evidence kept.
- A slot OOM or SIGKILL at K=4 is itself the finding, and ends the step.
- No machine by the end of the selection wait: nothing is spent, and the wait is
  reported. Both machines are terminated before the report.
- Spend stops at $3.50 against the bill.

### What this does and does not settle

Passing shows that packing changes nothing detectable at this length, on one machine and one
GPU type, and that the launch path works at K=4. It does not measure strength, does not
compare A4000 with L4 as a learning machine, does not show K above 4, and does not
recommend for the long run beyond the 20 updates measured. Long runs' watchdogs must be
sized from K=4's s/update, roughly twice solo.

## Step 3: stop waiting for the slowest game (deferred)

Jack Heart deferred this on 2026-11-07 because it needs paid strength runs. Step 1
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

## Step 4: overlap collection and learning (dropped)

Jack Heart dropped this on 2026-11-07. Step 1's bound was .39-.42 of an update, but
packing already fills the GPU that overlap would use, so the two savings do not
add. The design is at `0e8937f26`.

## Strength arms (step 3, if chosen)

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
| Step 2 machine, L4 worst case | 12 on one rental | 4 h | $2.40 | |
| Step 2 failed attempts and a replaced host | | | $1.10 | |
| **Step 2 total** | | | **$3.50** | |

Expected step 2 cost: about $.70 on an A4000, about $1.60 on an L4. Step 1's unspent
$1.74 is released. Step 3's lines were $17.40 at most: calibration, control and exact
quota seeds, their evaluation and its reserve. They come back in an amendment if step 3 is
chosen; overlap's are gone. Every line falls under the $50 cap, which includes step 1.

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
  step 3 before scoring.
- Seeds are cut equally across arms before launch, never after scoring.

## Deliverable

`experiments/data/training-throughput-b015d5-v1/report.md` in the artifact
store. It contains the step-1 table (per machine and K: s/update, phase tree,
idle share, CPU cores, GPU busy share, peak memory, $/M learner decisions), the
step readings and choices, step 2's digest result, metric comparison and shape
readings, cost against this budget, and the machine recommendation that updates
`experiments/training-update-time.md`.
