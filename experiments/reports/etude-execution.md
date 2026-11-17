# Etude execution comparison

**The RunPod RTX A6000 completed all four Elemental arms and both baseline
evaluations.** The branch averaged 954 decisions/s during collection and
learning, against 945 on main. The two repeats disagreed on which revision
was faster. Both baseline endpoints scored 11/128 on the same replay-verified
legs. CUDA forwards batched up to 14 inputs. The first three rentals and the
successful A6000 measurement are recorded below.

The repeat rents from RunPod or Vast only, taking any reasonable single GPU
on offer (L4, 4090, 3090, A5000 or A6000 class) under the remaining cap. Lium
is excluded: its rental strands the pod, which Etude Task 73e1a9 is fixing.
A host that dies or a check that finds no capacity is retried a few times
across the session before the measurement is given up; each try, its machine
id and its confirmed termination is listed below.

Exploratory, not a playing-strength claim. Does execution adoption change
Elemental update throughput on the same GPU, do managed Manabot players batch
on CUDA, and how does a stopped/resumed baseline compare with uninterrupted
learning? Compare main 3afd92d8f3 with this branch, alternating three-update
Elemental runs at seed 197, the recipe's 256 streams, 64 transitions, float32, width 64,
two attention layers and the same resolved current-world settings. Retain
collection/learning seconds and real branch save-call durations. Run a six-update
branch baseline uninterrupted and as two three-update attempts, keeping its
six-update schedule; compare curves and replayed paired-deal scores. Resume
starts fresh games and RNG streams, so bitwise equality is not expected.
Managed evaluation uses each endpoint's immutable checkpoint and 16 paired deals,
retaining actual CUDA forward batch sizes and Etude performance records.
Small samples describe execution, not strength equivalence. Total approved
rental cap USD 10, one reasonable GPU, no synthetic load. Stop and confirm
termination even if setup or measurement fails.

## Attempt — 2026-11-17

The source archives were main
`3afd92d8f3bf48b87d873693580f0d5f8ccf3612` and branch
`ee370d9757b86422db0ff0140d651c37a911de3f`, with this measurement module
overlaid on each. Python source manifests bound the archive bytes; the engine
implementation is unchanged between the revisions (the Rust diff names test
files only). Each archive kept its own locked Etude SDK. Setup used CPython
3.13, Rust 1.98.1, the locked CUDA Torch and a shared Cargo build directory,
with only one environment build at a time and two Cargo build jobs.

Etude rented Vast instance `61090179`, machine
`mtg329-speed-4090-vast-ffe7d7083bd2`, at 15:36 UTC with:

```sh
ETUDE_HOME=/tmp/mtg329-performance-etude doppler run --project etude --config prd -- doppler run --project manabot --config prd -- uv run --extra artifacts etude rent mtg329-speed-4090-vast --hours 2 --max-usd 10
```

The provider confirmed creation at 15:37:03 UTC. SSH became usable after
the image's startup package installation. `nvidia-smi` reported an NVIDIA
GeForce RTX 4090, 24,564 MiB, driver 595.71.05. Source upload, Rust installation
and CUDA dependency downloads completed; native compilation began. At
15:50 UTC the Cargo target held 181 MB, with no completed environment or
training log. At 15:52 UTC SSH disconnected and Vast reported
`actual_status=offline`, `cur_state=running`. The cause is unknown. No Torch
CUDA check, training timing, policy export or managed CUDA forward was
observed. Etude received no bootstrap/worker output; direct SSH observations
do not demonstrate launcher job delivery.

Terminated with `uv run --extra artifacts etude terminate
mtg329-speed-4090-vast-ffe7d7083bd2` in the same isolated/Doppler environment.
At 15:53 UTC Etude listed the machine deleted and Vast's instance lookup
returned absent. The launcher's final retained cost estimate was **~USD 0.12**;
this is not a reconstructed provider bill. No machine remained rented and no
remote workload survived deletion. A fallback query for RunPod NVIDIA RTX
A6000 (4 vCPUs, 24 GB, USD 1/hour ceiling) at 15:53:11 UTC returned no capacity;
no second rental was made. The shared launcher database was untouched.

## Harness checks and limits

The measurement module's actual Etude CLI path was checked locally before
remote training: a two-update CPU baseline at seed 197 stopped after update
one (exit 75) and resumed to update two (exit 0). Both attempts timed their
actual saves. The production managed callback completed two paired deals
(16 replay-checked games) on the existing three-update CPU checkpoint in
7.72 seconds, recording 645 CPU forward calls, all batch size one. This was
a harness preflight, not CUDA batching or a speed comparison. The remote
cohort was increased to 16 paired deals before any remote scored run.

Ruff check/format and `git diff --check` pass. No experiment-specific test
suite or saved schedule was added. Working setup script and source archives
remain in `/tmp/mtg329-gpu`; those paths are temporary, not archival evidence.

## Retry — 2026-11-17, 16:00–16:03 UTC

At branch `f962de12a174a17515692cd69a42a759feaa7b78`, fresh capacity checks
reported no matching RunPod L4, 4090, A6000 or A5000 and no Vast 4090 or
3090. Lowering Vast's minimum to four vCPUs and 24 GB did not find capacity.
Lium's previously unreadable inventory now offered a 4090, 23 vCPUs and
58 GB at an admitted USD 0.41/hour. Requested the same two-hour window,
with the retry cap reduced to USD 9:

```sh
ETUDE_HOME=/tmp/mtg329-performance-etude doppler run --project etude --config prd -- doppler run --project manabot --config prd -- uv run --extra artifacts etude rent mtg329-speed-4090-lium --hours 2 --max-usd 9
```

Lium created pod `55297a92-fa2b-47dd-9c73-b327672e74ca`, named
`etude-7b0c01e641d4`, at `2026-11-17T16:01:32.886331` (the provider's exact
timezone-naive value). Etude's `Pod.created_at: AwareDatetime` (pin d9a6bc8f06,
the code released as 0.4.1) rejected
that response (`timezone_aware`). Its next inventory read rejected the same
field. The launcher therefore retained machine
`mtg329-speed-4090-lium-7b0c01e641d4` as starting, with no provider id or
receipt; a subsequent `tick()` could not reconcile it. No SSH address,
bootstrap, uploaded source, CUDA check or scored workload was observed.
No Manabot or SDK workaround was added.

`etude terminate mtg329-speed-4090-lium-7b0c01e641d4` failed with
`Lium create already attempted; awaiting reconciliation`. The authenticated
Lium API accepted deletion of this exact pod. Its inventory then excluded
the pod; its individual lookup briefly reported `DELETING`, then returned
404/absent by 16:03 UTC. Termination is confirmed at the provider. The isolated
launcher still has the unresolved intent and must not be treated as a live
pod or a reconciled deletion. No launcher/server loop was left running.

The launcher's retained estimate at 16:03 UTC was **~USD 0.01**, not a final
settlement: the client cannot reconcile this deleted pod. A retry needs a
provider whose lifecycle the client can read (RunPod or Vast); Etude Task
73e1a9 owns the fix for Lium's timestamps and interrupted-creation recovery.

## RunPod/Vast capacity check — 2026-11-17, 16:09–16:10 UTC

On released Etude SDK 0.4.1, from a fresh isolated launcher
(`/tmp/mtg329-runpod-vast-etude`: storage, RunPod/Vast configuration and their
eight on-demand templates only), `uv run --no-sync etude availability TEMPLATE
--timeout 30` reported none available for every template:

| Provider | GPU | Minimum vCPUs / RAM | Hourly ceiling |
| --- | --- | --- | --- |
| RunPod | NVIDIA L4 | 6 / 48 GB | USD 0.70 |
| RunPod | NVIDIA GeForce RTX 4090 | 6 / 32 GB | USD 1 |
| RunPod | NVIDIA RTX A6000 | 4 / 24 GB | USD 1 |
| RunPod | NVIDIA RTX A5000 | 4 / 24 GB | USD 1 |
| Vast | NVIDIA L4 | 6 / 48 GB | USD 0.70 |
| Vast | RTX 4090 | 6 / 32 GB | USD 1 |
| Vast | RTX 4090 | 4 / 24 GB | USD 1 |
| Vast | RTX 3090 | 4 / 24 GB | USD 1 |

Nothing was rented or started: the launcher lists only the owned local
machine, with zero sessions, rentals and runs. The launcher holding the Lium
intent and the shared launcher were not touched.

## Retry — 2026-11-17, 18:57 UTC

Released Etude 0.4.1 found Vast RTX 4090 capacity at
USD 0.42702738444444444/hour. Rented machine
`mtg329-speed-4090-vast-711bee426992`, instance `61113127`, for two hours
with `--max-usd 9`, using the isolated RunPod/Vast launcher above. Source
archives: main `3afd92d8f3bf48b87d873693580f0d5f8ccf3612`, branch
`3670249eb38de3bf89de7dd834b8408028759f9b`, current harness overlaid on each,
verified source manifests and each archive's own lockfile.

The provider reported running before SSH was available. Its gateway refused
connections; its direct mapped SSH endpoint became usable around 19:00 UTC.
The host reported NVIDIA GeForce RTX 4090, 24,564 MiB, driver 595.71.05.
Uploaded source and started the sequential setup/measurement script at
19:01 UTC. Results and final termination follow below.

No update completed. The first branch arm exported its initial raw/EMA policies
and recorded update zero, then spent over six minutes with about 18 CPU seconds,
0% sampled GPU utilization and kernel waits `flush_workqueue` / `rq_qos_wait`.
Stopped that unscored process and retained its directory as `elemental-stalled`.
A fresh identical invocation with stdlib `faulthandler.dump_traceback_later(45)`
showed the main thread blocked at Etude 0.4.1 `run.py:412`,
`os.fsync(stream.fileno())` in `declare`, before learning. An external py-spy
snapshot had been refused by the container's ptrace permissions. The Python
wrapper changed no model, settings, SDK or persistence code. These observations
identify a durable-write stall on this host; they do not measure training speed.
RAM-backed output was not substituted for the durable save path.

Terminated the rental at 19:17 UTC. Etude lists it deleted and ended; Vast's
individual lookup confirmed instance `61113127` absent at 19:17:48 UTC. The
launcher's retained cost estimate is **~USD 0.14**. Deletion stopped all remote
work; the remaining local SSH client was stopped explicitly. No scored policy,
resume comparison, save-time sample or managed CUDA evaluation was produced.
The CUDA availability/scalar check alone passed (Torch 2.15.0+cu132).

Follow-up capacity checks, each capped at 30 seconds, were:

| UTC | Provider / template | Outcome |
| --- | --- | --- |
| 19:17:08 | RunPod 4090, 6 vCPUs / 32 GB | none available |
| 19:17:48 | Vast 3090, 4 vCPUs / 24 GB | none available |
| 19:17:58 | RunPod A6000, 4 vCPUs / 24 GB | none available |
| 19:18:27 | Vast 4090 in US, 4 vCPUs / 24 GB | none available |
| 19:18 | Vast unrestricted 4090 offer read | only offer 58865929, China/CN; no alternate host |
| 19:18:55 | RunPod A5000, 4 vCPUs / 24 GB | none available |
| 19:18:56 | RunPod L4, 6 vCPUs / 48 GB | none available |
| 19:19:12 | Vast A5000, 4 vCPUs / 24 GB | none available |
| 19:19:21 | Vast L4, 6 vCPUs / 48 GB | none available |
| 19:19:46 | Vast A6000, 4 vCPUs / 24 GB | none available |

No alternate machine was rented. GPU speed, durable save cost, managed CUDA
batching and resumed-versus-uninterrupted GPU learning remain unmeasured.
A repeat needs a rentable host with working durable storage. No server,
watcher or rented machine remains from this run.

## RunPod A6000 — 2026-11-17

At 19:36 UTC RunPod briefly offered a 4090 at USD 0.89/hour. The rental
request returned no capacity. The other RunPod and Vast templates had no
stock on that sweep. At 19:37 UTC a retry rented RunPod RTX A6000
`mtg329-speed-a6000-2f5283ae0409`, pod `k9uj7aajsbbplr`, at USD 0.69/hour
for two hours with `--max-usd 9`. Direct SSH, driver 595.112.07 and a Torch
2.15.0+cu132 CUDA tensor operation worked. The two source archives were the
same revisions as the 18:57 attempt. All 899 main and 831 branch source
manifest entries matched on the rented machine. Each environment built from
its own lockfile before the probe.

The output directory was on the root overlay mount (60 GiB, 54 GiB free),
mounted rw,relatime with lowerdir, upperdir, workdir and nouserxattr. The
container's cgroup v1 blkio read/write BPS and IOPS device-limit files were
all empty. The container cannot identify which host disk backs the overlay.
At 19:43 UTC Dirty/Writeback started at 1,132/0 kB. One 4 KiB write, flush
and fsync took 48.23 ms. System sync took 7.55 ms and Dirty/Writeback reached
0/0 kB. Two-second samples then showed 1,248/0, 1,964/0 and 4,004/8 kB
as the host continued working. The second write and fsync on the same file
took 0.86 ms, followed by 7,916/0 kB. Timed arms began after the setup
backlog drained. Both revisions ran under the same filtered strace
(`-f -yy -ttt -T --seccomp-bpf -e trace=fsync,fdatasync`), recording the
file, result and duration of each call in memory while training output stayed
on the overlay.

All four Elemental arms completed on the original archives. The first baseline
on branch 3670249eb3 then failed before update one in CUDA attention backward
with `Invalid max_seqlen_k:58`. A five-token minibatch containing sequences
of two and three tokens reproduced the same failure with 58 padded slots.
Bounding the kernel's sequence length by the smaller of the slot count and
the total packed length produced finite gradients on this GPU. The fix keeps
live-token attention enabled and reads both bounds from tensor shapes.

The baseline retry uses a fresh checkout archive at 5826f7502e, including
the attention fix and the intervening main merge. Both baseline trajectories
and their managed evaluations use that revision. The throughput comparison
retains the two earlier archives to hold engine behavior fixed between sides.

### Elemental throughput

| Arm in execution order | Decisions | Games | Optimizer steps | Collection s | Learning s | CLI wall s¹ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Branch 1 | 152,546 | 775 | 597 | 92.71 | 64.13 | 166 |
| Main 1 | 154,506 | 774 | 605 | 101.16 | 66.22 | 179 |
| Branch 2 | 152,874 | 774 | 599 | 98.73 | 64.46 | 169 |
| Main 2 | 152,139 | 774 | 595 | 92.67 | 64.49 | 163 |

¹ Elapsed between successive arm starts at one-second resolution. Every arm
completed three updates. Pooling decisions over collection plus learning
gives 954 decisions/s on the branch and 945 on main (about +1%). Across
both runs the six updates took 335 s of wall time on the branch and 342 s on
main. The ordering of the faster revision reversed on the second pair.
These two repeats do not resolve a consistent speed difference. Branch
`run.save()` durations by update were 0.17, 0.06, 50.59 ms and 0.09, 0.06,
103.01 ms. Only the final call of each arm wrote state files. Main reported
a metric cardinality warning on each run.

### Durable writes during Elemental

Each cell gives the call count and total syscall time in milliseconds. SQLite
includes the database, journal, WAL and directory descriptors. Main's first
durable write was the SQLite journal, at 1.782 and 1.240 ms. Branch's first
write was the run declaration, at 3.980 and 0.745 ms.

| File or operation | Branch 1 | Main 1 | Branch 2 | Main 2 |
| --- | ---: | ---: | ---: | ---: |
| Run declaration | 1 / 3.980 | 1 / 0.286 | 1 / 0.745 | 1 / 0.578 |
| Game receipts | 775 / 197.150 | 774 / 151.373 | 774 / 162.839 | 774 / 158.478 |
| Outcome tables | 3 / 105.682 | 3 / 2.316 | 3 / 124.623 | 3 / 1.927 |
| Metric blocks | 6 / 3.691 | 8 / 4.834 | 6 / 74.163 | 8 / 5.834 |
| SQLite | 0 | 27 / 13.064 | 0 | 27 / 8.879 |
| Save learner.pt | 1 / 1.278 | 0 | 1 / 21.521 | 0 |
| Save optimizer.pt | 1 / 0.871 | 0 | 1 / 6.030 | 0 |
| Save ema.pt | 1 / 0.949 | 0 | 1 / 2.579 | 0 |
| Save state.json | 1 / 0.112 | 0 | 1 / 0.705 | 0 |
| Save current.json.partial | 1 / 0.142 | 0 | 1 / 0.880 | 0 |
| Published policy checkpoint or checkpoint record | 0 | 0 | 0 | 0 |

The checkpoint row counts explicit fsync/fdatasync calls on those paths.
Etude records an immutable policy file already inside `out` by reference.
Recovery saves on the branch follow the separate five-file sequence above.
The slowest Elemental call was 123.708 ms on a branch outcome table. This
host did not reproduce the multi-minute Vast stall. The earlier trace
establishes where the Vast process waited; its device limit was never
measured, so disk throttling remains unconfirmed.

### Six-update baseline

After rebuilding, sync reduced Dirty from 1,716,988 to 852 kB with zero
Writeback. The CUDA regression test passed on the A6000. The uninterrupted
baseline finished, the second baseline stopped after update three (exit 75),
and the resumed process finished update six with its original target.
The table shows cumulative decisions and Adam steps, followed by the logged
per-update value loss.

| Update | Full decisions | Resumed decisions | Full Adam steps | Resumed Adam steps | Full value loss | Resumed value loss |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 962 | 962 | 141 | 141 | 0.3511 | 0.3512 |
| 2 | 1,743 | 1,743 | 243 | 243 | 2.1507 | 2.1507 |
| 3 | 2,862 | 2,862 | 377 | 377 | 0.1716 | 0.1701 |
| 4 | 3,541 | 3,450 | 469 | 450 | 0.1883 | 0.2192 |
| 5 | 4,589 | 4,032 | 599 | 529 | 0.1679 | 0.3029 |
| 6 | 5,685 | 5,070 | 739 | 649 | 0.2187 | 0.2929 |

The independent CUDA trajectories already differed slightly in loss before
the stop. Resume kept the update and optimizer counters and started fresh
game and sampling streams. At update six the resumed arm had made 615 fewer
decisions and 90 fewer optimizer steps. The final uninterrupted save took
67.8 ms. The stop and final resumed saves took 61.4 and 67.8 ms respectively.

### Managed CUDA evaluation

Both six-update EMA endpoints played the same 16 deals with eight deck/seat
legs per deal against the heuristic. All 256 games terminated, passed replay
and satisfied row integrity checks. Each endpoint won 11 and lost 117, with
the same score on every paired leg. Their EMA bytes differed. Each arena
registration's checkpoint SHA-256 matched the corresponding archived endpoint.
One seed and one paired cohort do not establish playing-strength equivalence.

| Six-update endpoint | Played deals | Wins / legs | Elapsed s | CUDA inputs | Forward calls | Calls with batch >1 | Max batch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Uninterrupted | 16 | 11 / 128 | 86.19 | 6,587 | 2,204 | 1,574 | 14 |
| Resumed | 16 | 11 / 128 | 85.99 | 6,587 | 2,209 | 1,584 | 13 |

Each evaluation loaded one checkpoint into Etude's shared CUDA inference
process, with up to 16 games active and one inference thread. Mean batch sizes
were 2.99 uninterrupted and 2.98 resumed.
Direct SSH exercised local Etude training and managed evaluation. Queued
launcher delivery was outside this measurement.

The evidence archive contains the per-call fsync/fdatasync traces for every
arm and the failed baseline, storage probe, learning curves, save durations,
endpoint policies, batch histograms, Etude performance records, arena results
and Command traces. Retrieve it with
`uv run --extra artifacts python -m manabot.data fetch experiments/data/etude-execution/a6000-results.tar.gz`.
The local copy matched the rental's SHA-256
`5b5345e2adc0b234b8d3a956a1b10839c67e3122e2c1e9064f5de1af254c4827`.

Etude terminated the A6000 at 20:07:53 UTC and reported **~USD 0.35** for
the rental. RunPod's lookup of pod `k9uj7aajsbbplr` confirmed absence.
Together with the three prior launcher estimates, the total was approximately
USD 0.62 of the approved USD 10.
