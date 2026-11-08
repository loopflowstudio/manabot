Inherited MTG-243 naming notes (another branch's handoff) were removed; they are at
`c6746f3ff:scratch/questions.md` and confer nothing on b015d5. Main's 07c490 notes (arrived with PR #436) are at
`ebed0963e:scratch/questions.md` and likewise confer nothing here. Main's c5fba4 notes
(arrived with the sync onto `646cb9821`, updated by the sync onto `86f81c8d7`) are at
`86f81c8d7:scratch/questions.md` and likewise confer nothing here.

# Task b015d5 — training seconds per update (PR #435, merged `45cc961a2`)

History of the measurement and concept reviews: `git show 645b9be21:scratch/questions.md`.
Live items:
- Inherited, unrepaired: `test_resource_limit_receipt_replays_its_retained_encoding_boundary`
  fails because `tests/fixtures/tla-pilot-permanent-capacity.jsonl` retains a schema 7
  frame while native emits `SEMANTIC_DECISION_VERSION` 8 (`binding_mismatch`). The test
  is in no CI selection. It needs its own bug Task; the fixture is retained evidence.
- `HostFacts.cpu_model` reads `/proc/cpuinfo` (null off Linux); the GPU stays in run
  identities, not `TrainingRun.host`.
- Proposed infrastructure Wave memory (not written; `.lf/wave/` is read-only here):
  rented-host CPU class moves update time 1.6-1.8x at a fixed GPU label, so compare
  `host.cpu_model` and native stepping before attributing a slowdown; a packed
  ~50,000-row column makes per-time-step device recursions launch-bound.

## experiment-design b015d5 (2026-11-06): training throughput

Design: `experiments/protocols/training-throughput-b015d5-v1.md` (draft). Reversible
assumptions, for Jack Heart's review:
- Control = 07c490's `elemental_variants.elemental_recipe` (world relabel + capacity
  recovery), so the design waits for PR #436 instead of declaring a second copy of the
  same variant. If #436 stalls, this file can declare the identical two changes under
  its own ID.
- Reuses 07c490's 50 `ELEMENTAL_PAIRS` evaluation deals, so scores read against its
  baseline curve. They stay evaluation-only here; that protocol called them reserved, so
  sharing them is a choice for review.
- Packing is judged without strength runs, because processes share only hardware.
  Spending strength runs on it would buy nothing a digest cannot show.
- Proposed reservation $24.40; the remaining $25.60 of the cap is held back pending an
  amendment.
- An A4000 probe is included as the larger-CPU, cheaper-GPU shape. Etude's `a4000`
  template lists several rented machines that may be ghost records; check
  `etude machines` before launch.

## implement b015d5 (2026-11-06): step-1 probe software

Built: `scripts/bench_training_update.py` `--json` (whole receipt), `--recipe` as a file
or `module:function(regime_id, updates)` declaration, `--cuda-counts`;
`experiments/training_throughput.py` (phase driver, 1 Hz samplers, `/dev/shm` journal,
module-run cleanup, `runs()`); `tests/training/test_training_throughput.py`.
Reversible choices:
- PR #436 is still open, so `elemental_variants` is not on main. The probe names the
  control as the declaration string `experiments.elemental_variants:elemental_recipe`;
  it resolves at any commit containing #436 (main since `ebed0963e`). The CPU test
  keeps the frozen 183 file: it checks driver mechanics (receipts, samples, memory
  skip, cleanup), which the recipe does not change; the control's own 2-update CPU
  bench is recorded in the design status. Launch pins a commit that has both.
  Checked: the declaration loads from #436's file (w17, 256 streams, both seats,
  `recover_capacity` true).
- Children drop every `ETUDE_*` variable and track locally; only the driver is the
  launched run. Each child's `TrainingRun` receipt is the evidence.
- A phase is skipped when the previous completed phase's largest child peak x its child
  count exceeds `memory_fraction` (.85) of machine memory. Peak is Linux VmHWM, else the
  largest sampled RSS.
- Any failed phase fails the run after all phases have run; evidence is kept.
- Quote 2026-11-06 at `8d178f40e`: `--rent l4 --machines 2` plans $2.06 (reservation
  $2.10); `--rent a4000` refused, no stock at quote time. Re-quote before launch.


## experiment-freeze b015d5 (2026-11-06): step 1

Bound to main `70d79ccb8`. Receipt is in the protocol. The A4000 quote
succeeded ($.44); the L4 quote is still open because RunPod reported no
available L4 machines. Quoting needs the 0.2.14 overlay and nested Doppler
(see the launch-client memory). Re-quote L4 at execution review.

## pool-validation b015d5 (2026-11-07): step 1 executed

Report: `experiments/data/training-throughput-b015d5-v1/report.md` (published).
Open for later, none blocking step 1:
- The cause of the K=4 recording stall on L4-a is unknown. The P4 SIGKILL is also
  unproven, but both peaks are within 13 MB of 61.05 GB, which points to a pod limit.
- `run_phase` reads host memory, not the pod's allowance, so the memory gate cannot
  trip on rented pods. Fix before relying on it for K above 4.
- The bill had posted $.96 by 07:16 UTC; about $.10 of the $1.06 total is still
  estimated. Etude's run list shows about $.44, which is below the bill.

## experiment-design b015d5 (2026-11-07): step 2 amendment

Design: protocol section "Step 2: seeds per machine (amended 2026-11-07)". Nothing
frozen or launched. Route: needs software (protocol's five items), then freeze, then
execution review with Jack Heart. Reversible assumptions, for review:
- "Two-run digest test" runs on CPU. Same-seed CUDA runs are not expected to match, so
  the CUDA legs are compared by metrics, with one repeated solo seed (M3) as the noise
  scale. M3 is my addition, at about $.10-.20.
- All legs share one hand-rented machine, so CPU digests and the K=1/K=4 readings share
  hardware. That needs `etude wait` between legs and an explicit `etude terminate`.
- Metric legs run 20 updates and four seeds. Enough to show a gross leak, not a small
  effect. Any leak would appear early, and 20 matches a scored coordinate of 07c490.
- Step 2 uses Etude main under an isolated `ETUDE_HOME`, because schema 12 would lock
  0.2.14 clients out of the shared launcher database. Moving the shared launcher to a
  release is a separate decision, and "standard shape" needs it before other Tasks can
  pack. Etude's 0.2.15 release is not requested here.
- Setup stays per attempt with a `flock` on the rustup lines; `Preparation` does not
  fit the editable, in-tree native build yet.
- The A4000 selection wait is 30 minutes, then L4.
Open: whether cgroup or `RUNPOD_MEM_GB` shows a pod's allowance is unknown until the
first step 2 receipt. Step 1's `cpu_quota` was None on every pod.

## implement b015d5 (2026-11-07): step 2 software

Built the protocol's five items. `uv run --extra dev pytest
tests/training/test_training_throughput.py tests/infra/test_host_facts.py -q`: 26 passed,
including the two-process digest through the real worker on a shared track folder.
Open or assumed:
- Acceptance quote, 2026-11-07 about 13:20 UTC, isolated home, `--revision HEAD`: on
  `b015d5-l4x4` the four M1 runs share one L4 at 4 slots, $.59/hour, planned first-wave
  rental $.74, cohort budget $3.00, nothing rented. `b015d5-a4000x4` refused: RunPod
  had no A4000 (offer $.25/hour when asked at 13:14). L4 had also refused minutes
  earlier, so stock moves. Re-quote both at the launch commit for execution review.
- Isolated home is `~/.etude-b015d5-step2`; run the client as
  `ETUDE_HOME=... doppler ... uv run --with "etude-sdk[s3,eval] @
  git+file:///Users/jack/src/etude@a7718f59a706fdb1e4cac7c28b2a13e5b7afbd11" etude ...`.
  Never run that client without `ETUDE_HOME`: it would upgrade the shared database.
- 80 GB disks are my choice (the shared on-demands use 40-50), for four venvs and builds.
- `RUNPOD_MEM_GB` is read as decimal GB, the smaller reading if it means GiB.
- Main's `405223165` (c5fba4's sampled-roster producers) is synced here through `646cb9821`;
  the protocol says the launch commit includes it.
- `flock` is not on macOS, so `machine_setup` takes the lock only where it exists (gate);
  every rented machine has it. The script's order and failure handling are tested with
  stand-in tools; real lock contention first happens on the rental.

## compress b015d5 (2026-11-07): step 2 software

`uv run --extra dev pytest tests/training/test_training_throughput.py
tests/infra/test_host_facts.py -q`: 26 passed. `HostFacts.memory_limit` is now one nested
field, replacing `memory_limit_bytes`/`memory_limit_source`; the protocol says so.
`machine_setup` bytes changed after the 13:20 acceptance quote, so the launch-commit
re-quote also covers that.

## realign b015d5 (2026-11-07): step 2 plan against code and upstream

Code matches the protocol's five items; no code changed. Protocol corrected:
- New rule "Setup deadline". Etude's `max_hours` clock starts before setup, and
  `run_worker` refuses to train with less than watchdog plus finalization left, so setup
  over 900 s fails admission. Four concurrent native builds are unmeasured. The 7-minute
  D2 trigger and staggering M1 in two pairs are my reversible choice, for execution review.
- Exact isolated client invocation moved from these notes into the protocol, since
  scratch is removed before landing. Source lines and step 3's stale c5fba4 dependency updated.
- Digest watchdog checked: 46 s of 600 s, laptop, 24 streams, 3 updates, real worker. It
  shows margin only; it is no timing evidence.

Before freeze: re-quote both on-demands at the launch commit. Etude main is at `199335c`;
the pin stays `a7718f5`.

## concept-review b015d5 (2026-11-07): step 2 against the accepted design

Walked one person's path: rent a four-slot machine by hand, run the legs in turn, compare,
terminate. No design change and no code direction; implement need not iterate.
- Normal path holds at Etude `a7718f5`: a hand rental stays until its deadline or
  `terminate` (`docs/launcher.md`), and placement fills machines with free slots first
  (`scheduler.py`), so cohorts land on the held machine and `--max 1` is not hit. Not yet
  exercised on a held rental: the acceptance quote placed runs on a new one.
- Failure and recovery hold: the booted pod's guardian ends it at its deadline without a
  controller, bounding a lost session at the 5-hour window ($3.00 at the L4 ceiling, under
  the $4.50 stop). Failing one run leaves its neighbors running. An unbooted pod still needs
  the isolated controller's tick.
- Shape stays out of the regime. Slots belong to the on-demand, so packing changes no
  recipe and no frozen identity. That is the right owner.
- `PhaseResult.machine_bytes` can now hold a pod allowance. The name is retained step 1
  evidence, and `memory_source` says which it is; not worth a rename.

Separate and already recorded: others cannot pack until an Etude release moves the shared
launcher, so "standard" is unavailable to other Tasks until that decision.

## implement b015d5 (2026-11-07): seeds x producers legs (comment ceff7d63)

Resolves the four-producer question above: speed legs S14, S22, S44 run after M3 in
`experiments/training_throughput_packing.py`; protocol and budget amended (reservation
$4.50, window 5 h). Still not frozen; review, land, freeze and execution review follow.
Checks: focused launch and sampler tests pass; a four-producer control variant (8 streams,
1 update, CPU) finished through the real worker; the quote at `a1c83043e` with the new
launch file: M1 and S44 each on one L4 at 4 slots, $.59/hour, nothing rented; A4000
refused for stock. Re-quote at the launch commit.
Reversible choices, for review:
- GPU busy share needed a sampler that `TrainingJob` runs lack, so every CUDA leg prepends
  a backgrounded `nvidia-smi -l 1` writing `gpu-samples.csv` to its output. This changes
  job bytes, not regime bytes. CPU needs nothing: producers are threads, so
  `process_cpu_seconds` already counts them.
- Step 1 read 128 CPUs and no quota on every pod, so the vCPU count comes from the provider
  (`runpodctl`), not `HostFacts`.
- Speed legs: `max_hours` .75, `max_usd` $.60 (the hours plus 15 minutes at the ceiling),
  seeds are the first K of 26211-26214.
- Only one machine class is rented, so the recommendation for the other is an inference
  from busy cores, labeled as one. Measuring both classes would need a second rental that
  comment ceff7d63 does not ask for.

## compress b015d5 (2026-11-07): seeds x producers legs

`Leg.updates` is gone (read from the regime); `_speed_leg(seeds=, producers=)` names the
shape; `elemental_variants.self_play_stage` is now public and replaces the local assert.
Launch docstring corrected to `--hours 5`. `pytest tests/training/test_training_throughput.py
tests/infra/test_host_facts.py tests/training/test_elemental_closure_lambda.py`: passed.

## realign b015d5 (2026-11-07): seeds x producers legs against code and upstream

Branch is level with origin/main at `fb626f214`; main since step 2's design touched only
TLA frontier files. Etude pin holds. No code changed. The launch file, budget (19 runs, $3.00
machine plus $1.50), watchdogs, on-demand rows in the isolated home and speed-leg regimes all
match the protocol; nothing is rented in that home.
- Corrected: the protocol called producer learning equivalence "c5fba4's result". What exists
  is a CPU, four-stream, fixed-policy collection check, and c5fba4 is not frozen and plans no
  trained comparison. Jack Heart's "cited, not retested" stands; the citation now says what it
  rests on, so the execution review sees how thin it is.
- Checked, no change: the sampler's EXIT trap fails when `nvidia-smi` died first, but the job
  command has no `set -e`, so the job's status holds.

## concept-review b015d5 (2026-11-07): seeds x producers legs against the accepted design

Walked one reading: launch S44, read (4, 4), choose a shape. The legs, stop rules, budget and
regimes match comment ceff7d63. Two protocol changes; no code follows from either.
- **Concurrent span.** Step 1's children started together after one setup; Etude attempts each
  set up alone, so a packed leg's runs start apart and a fixed "updates 2-8" mixes K=4 with
  lighter load. Over 8 updates that is much of the leg. Shapes now read only updates inside
  the span when all K train, under 3 per run means not read. Existing rows supply this:
  `gpu-samples.csv`'s first timestamp plus `training_seconds`.
- That exposed the stagger fallback's claim. Two pairs 8 minutes apart overlap about 5
  minutes at M1, and probably not at all at S44. Stagger now covers S44 too (`--seeds`
  already allows it), and the protocol states what it costs. No new approval is implied:
  a longer S44 would be an amendment, since comment ceff7d63 set 8 updates.
- **Owners of a shape.** K is launch-time, P is regime bytes that a live continuation must
  preserve (`regime.py`), and machine selection happens after the regime is declared. So a
  per-class recommendation can conflict with "A4000 when available, L4 otherwise". The
  report now says whether one P serves both. This limits what the recommendation can
  mean; it does not need deciding before freeze, because this step adopts no P.
Checked, no change: speed-leg regimes, `threads: 1` retained. `producers.py` states no policy or
CUDA state enters a producer thread, so the device does not reach the sharded env. That is
read from code; no CUDA multi-producer run has happened here, so S14 is the first.

## experiment-freeze b015d5 (2026-11-07): step 2

Bound to main `668212567` (this branch's tree equals it). Receipt is in the protocol. L4
quoted and rentable at about 19:28 UTC; A4000 quote stays open, no stock. For execution review:
- Etude's `rent` has no quote, so the hand rental's cost is from the rate, and the per-leg
  quotes plan new rentals. Placement on a held rental is still unexercised.
- Proposed, not in the protocol: `etude rent ... --hours 5 --max-usd 3.00`, so the provider
  window itself holds the machine line. Etude's `--max-usd` ends the window early.
- The isolated controller still needs starting before the rental, under `caffeinate -is`.

## pool-validation b015d5 (2026-11-07): step 2 execution log

Approved by Jack Heart (comment e89d4889). Client: `~/.etude-b015d5-step2/e` (sets
`ETUDE_HOME`, nested Doppler, Etude `a7718f5`); controller under `caffeinate -is` from the
same home, pid in `~/.etude-b015d5-step2/controller.pid`.
- 20:42 UTC: both on-demands reported "rentable now"; `launch --quote` and the first rent
  were refused for no stock. The second rent at 20:44:43 got `b015d5-a4000x4-27c8c5c7f309`
  (RunPod `nbc3u0h4ot5rv0`, $.25/h), `--hours 5 --max-usd 3.00`, until 01:44 UTC.
- `etude launch --rent` refuses a held rental ("it is busy; etude queue would wait"),
  recording no run. Legs therefore go through `etude queue --rent b015d5-a4000x4`, same job
  bytes; placement on the held rental works.
- `etude fetch ... etude-result.json` returns StorageUnavailable 404 on every run; the
  `training/` files fetch and verify. Not needed for any reading.
- D1 `7e76419f958a`, D2 `9f8373c2ce24` (26201) / `c1a9aa5e1b72` (26202): all succeeded.
  `training_digest` D1 vs D2 26201: **digests equal**. D2 admission to finish 4.2-4.5 min
  including training, so concurrent setup is far under 7 min: M1 launched unstaggered.
- M1 `613540c9f282 2ce4ba18dedb c5610eb166a4 8814aa882b83` succeeded (20 updates, 87-105 s
  attempt start to first GPU sample with four at once). M2 26211 `673dd2a0dd9f`, 26212
  `36485589780f`, 26213 `66143fded9b1` succeeded.
- **M2 26214 `91e148fbd392` hung**: updates 1-2 normal (~40 s each), then no rows and GPU 0%
  (0.8-1.0 GB held) from ~22:12 until Etude's 1-hour limit killed it (out_of_time, exit -9,
  23:09). No error, traceback or job output after `phase run: started`. Not a dud host (the
  same host ran 7 runs clean). Counted as M2's first failure; retried once at 23:12 as
  `b3f331e7c50b`. From here a stall watcher cancels a run with no new update row for
  600 s (counts as a failure) so a second hang cannot eat the window. Cause unknown; a
  hang that reproduces goes to a bug Task with these ids.
- Pod facts (runpodctl): vcpuCount 16, memoryInGb 62. Receipts read `memory_limit` 62.0 GB
  from cgroup v1, so the memory-gate fix works on RunPod. Step 1's ~61 GB SIGKILLs match it.
- Retry `b3f331e7c50b`, M3 `7fc74afa2f41`, S14 `cb64b1093e57`, S22 `f1fc4986886b
  33f0bbe20401`, S44 `483303fadcad dbbc464c364c d265997f8111 a4dec36539bd`: all succeeded;
  watcher never fired. Machine terminated about 00:36 UTC (after killing the controller
  mid-tick, its 5-minute lease had to expire first). Bill $.807 posted to ~23:55; total
  about $.98. Report "Step 2" section published with `step-2/` evidence.
- For evidence review: packed games about 5% shorter over updates 11-20 (paired interval
  excludes zero; M3 moves the same way at 60% of the size); producers buy no speed at
  K=1 or K=4 on the A4000; L4 shape is an inference.
- Etude gaps worth Etude Tasks (not filed from this headless run): `launch --rent` and
  `terminate` refused while the controller holds the placement lease; `etude-result.json`
  fetch 404s; SDK 0.2.12 reads `a7718f5` metrics as empty.

## experiment-analysis b015d5 (2026-11-07): step 2 reconciled

- `readings.py` reproduced `readings.txt` byte for byte before any change. Two report
  figures had no producer: M1 over updates 2-8 (68.8 s, 10.88 M/h) and the phase table.
  Both now print from `readings.py` and agree with the report. Its (1, 1) phase row is M3
  (seed 26211, S14's seed); M2's four-seed mean is 7.2/8.3/15.7. The report now says so.
- Reconciled: 19 declared runs plus the hung attempt, and each run's update count and learner decisions match its own receipt.
  Exposure is 11,337,315 CUDA learner decisions in the succeeded runs plus 104,653 in the hung one.
- Added to the report: track-dir separation and GPU memory (3,620 MiB peak against 16 GB).
  Labelled the M1 table row as updates 2-8; the frozen concurrent-span reading is 60.2 s.
- An unreferenced `readings.cpython-312.pyc` object was uploaded to R2 by mistake. The
  manifest does not list it.
- Open for evidence review, unchanged: the 5% shorter late packed games, the inferred L4
  shape, the hung attempt, step 3.

## experiment-design b015d5 (2026-11-07): step 3, model size, GPU and precision

Design: protocol section "Step 3: model size, GPU and precision". Nothing built, frozen or
launched; design review returns to Jack Heart. Steers folded in: 1f4c2872 (planner
objective), 0216b052 and af047a69 (worked examples; af047a69's $100 / 336 h supersedes
5 days), 8ac4eb82 (size as input), 518e5f09 (refocus on large-model speed, measure
ourselves), b9215b8a (packing crossover ladder).
- w64 planner table is $0 arithmetic in `scratch/planner_sketch.py` (stdlib; `python3
  scratch/planner_sketch.py`). Its curve, seed-spread and selection models are my
  simplifications for review: piecewise log-linear MTG-183 rating, late slope 44 or 20 per
  doubling, Gaussian seed spread with eval-noise shrinkage, halving at 3:4:7, at most 3
  A4000s at once. The worked example matches the conversation's figures (~32k solo, ~27k
  halving, ~19-20k pack-4).
- 8ac4eb82 says to take w160l8 speed from c5fba4's calibration. That calibration has not
  run (MTG-267 at experiment-software-review, PR #445 open; its protocol says nothing has
  launched). 518e5f09 then says to measure ourselves, which the design does.
- The steer asked for 07c490's seed spread at 20/40/75/150. Per-seed scores exist only at
  u40 in `results.json`; u20/u75 per seed need the eval database (a fetch was tried and
  stopped, it held 16 GB). No 07c490 run reached u150.
- `elemental_variants.elemental_recipe` resolves to policy input v6 with 512 facts, while
  07c490's protocol describes the recipe's input as v1. Not investigated; step 3's
  parameter counts use the resolved regime. Possibly a bug Task (how a frozen regime with no
  `policy_input_version` loads), not this step's.
- Reversible choices: ladder w128l4 (4 heads, default ff) as the intermediate; a384l8 heads 8,
  ff 1536; third GPU chosen at freeze by quoted bf16 dense throughput per dollar, default RTX
  4090, ceiling $.95/hour; crossover threshold K=2 gain < 1.10; K=4 only at K=2 gain >= 1.30;
  4 updates per speed cell; learning check at w160l8, seed 26301, fp32 x2 / tf32 / bf16,
  12 updates; seeds 26301-26304; reservation $10.
- bf16 autocast covers collection inference too, so behavior numerics change as well as
  learning. Mixed precision limited to inference only would be a narrower option; not
  proposed.
