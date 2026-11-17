# Etude execution comparison

Exploratory, not a playing-strength claim. Does adopting Etude execution change
Elemental update throughput on one GPU, do managed Manabot players batch on
CUDA, and how does a stopped and resumed baseline compare with uninterrupted
learning?

**On a RunPod RTX A6000 the branch averaged 954 decisions/s during collection
and learning, against 945 on main; the two repeats disagreed on which was
faster. Durable writes did not stall. Both six-update baseline endpoints
scored 11/128 on the same replay-verified legs, and managed CUDA forwards
batched up to 14 inputs.** Total launcher estimate ~USD 0.62 of the approved
USD 10.

## Method

Main `3afd92d8f3bf48b87d873693580f0d5f8ccf3612` against branch
`3670249eb38de3bf89de7dd834b8408028759f9b`, each a source archive with this
harness (`experiments/etude_execution.py`) overlaid and its own locked Etude
SDK. All 899 main and 831 branch source-manifest entries matched on the rented
machine; the engine is unchanged between the revisions (the Rust diff names
test files only). Four alternating three-update Elemental runs at seed 197:
the recipe's 256 streams, 64 transitions, float32, width 64, two attention
layers and the same resolved current-world settings (the main arm reads the
branch's settings as JSON). CPython 3.13, Torch 2.15.0+cu132, driver
595.112.07, direct SSH; queued launcher delivery was outside the measurement.

The six-update baseline ran uninterrupted and as two three-update attempts
(stop after update three, exit 75, then resume with the original target).
Resume keeps update/optimizer counters and starts fresh games and RNG streams,
so bitwise equality is not expected. Both baseline trajectories use
`5826f7502e` (see the attention fix below). Each EMA endpoint then played 16
paired deals of eight deck/seat legs against the heuristic through Etude's
shared CUDA inference process (up to 16 active games, one inference thread).

Storage was checked before timing. The output directory was on the root
overlay (60 GiB, 54 GiB free, rw,relatime, nouserxattr); the container's
cgroup v1 blkio BPS/IOPS limit files were empty, and it cannot see which host
disk backs the overlay. With Dirty/Writeback at 1,132/0 kB, one 4 KiB
write+flush+fsync took 48.23 ms; `sync` took 7.55 ms and drained Dirty to 0;
the same probe then took 0.86 ms. Timed arms began after the setup backlog
drained. Every arm ran under
`strace -f -yy -ttt -T --seccomp-bpf -e trace=fsync,fdatasync`, with output
on the overlay.

## Elemental throughput

| Arm in execution order | Decisions | Games | Optimizer steps | Collection s | Learning s | CLI wall s¹ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Branch 1 | 152,546 | 775 | 597 | 92.71 | 64.13 | 166 |
| Main 1 | 154,506 | 774 | 605 | 101.16 | 66.22 | 179 |
| Branch 2 | 152,874 | 774 | 599 | 98.73 | 64.46 | 169 |
| Main 2 | 152,139 | 774 | 595 | 92.67 | 64.49 | 163 |

¹ Elapsed between successive arm starts, one-second resolution.

Pooled over collection plus learning: 954 decisions/s on the branch, 945 on
main (about +1%); six updates took 335 s of wall time on the branch and 342 s
on main. The faster revision reversed on the second pair, so two repeats do not
resolve a speed difference. Branch `run.save()` took 0.17, 0.06, 50.59 ms and
0.09, 0.06, 103.01 ms by update; only each arm's final call wrote state files.
Main warned about metric cardinality on each run.

## Durable writes

Call count / total syscall milliseconds. SQLite includes the database,
journal, WAL and directory descriptors.

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
| Policy checkpoint or record | 0 | 0 | 0 | 0 |

The first durable write was main's SQLite journal (1.782, 1.240 ms) and the
branch's run declaration (3.980, 0.745 ms). Etude records an immutable policy
file already inside `out` by reference, so it adds no fsync. The slowest call
was 123.708 ms, on a branch outcome table. This host did not reproduce the
multi-minute stall seen on Vast (appendix); that host's device limit was never
measured, so disk throttling there remains unconfirmed.

## Six-update baseline

The first baseline on 3670249eb3 failed before update one in CUDA attention
backward (`Invalid max_seqlen_k:58`); a five-token minibatch of sequences of
two and three tokens with 58 slots reproduced it. `5826f7502e` bounds the
kernel's `max_seqlen` by the smaller of the slot count and the packed length,
keeping live-token attention, with a CUDA gradient regression test that passed
on the A6000. Before the retry, `sync` drained Dirty from 1,716,988 to 852 kB.

Cumulative decisions and Adam steps, and per-update value loss:

| Update | Full decisions | Resumed decisions | Full Adam steps | Resumed Adam steps | Full value loss | Resumed value loss |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 962 | 962 | 141 | 141 | 0.3511 | 0.3512 |
| 2 | 1,743 | 1,743 | 243 | 243 | 2.1507 | 2.1507 |
| 3 | 2,862 | 2,862 | 377 | 377 | 0.1716 | 0.1701 |
| 4 | 3,541 | 3,450 | 469 | 450 | 0.1883 | 0.2192 |
| 5 | 4,589 | 4,032 | 599 | 529 | 0.1679 | 0.3029 |
| 6 | 5,685 | 5,070 | 739 | 649 | 0.2187 | 0.2929 |

The CUDA trajectories differed slightly in loss before the stop. At update six
the resumed run had 615 fewer decisions and 90 fewer optimizer steps. The final
uninterrupted save took 67.8 ms; the stop and final resumed saves took 61.4 and
67.8 ms.

## Managed CUDA evaluation

| Six-update endpoint | Played deals | Wins / legs | Elapsed s | CUDA inputs | Forward calls | Calls with batch >1 | Max batch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Uninterrupted | 16 | 11 / 128 | 86.19 | 6,587 | 2,204 | 1,574 | 14 |
| Resumed | 16 | 11 / 128 | 85.99 | 6,587 | 2,209 | 1,584 | 13 |

All 256 games terminated, passed replay and satisfied row integrity checks;
the two endpoints scored identically on every paired leg although their EMA
bytes differed. Each arena registration's checkpoint SHA-256 matched its
archived endpoint. Mean batch size was 2.99 uninterrupted and 2.98 resumed.

## Cost and limits

Launcher estimates: Vast 4090 ~USD 0.12, Lium ~USD 0.01 (unreconciled), Vast
4090 ~USD 0.14, RunPod A6000 ~USD 0.35; ~USD 0.62 total. Every machine was
terminated and confirmed absent at its provider.

One seed, two throughput repeats and one paired cohort describe execution,
not strength equivalence or a speed difference. Queued launch, a throttled
disk and a real native panic or hang were not exercised.

## Evidence

The archive holds per-call fsync traces for every arm and the failed baseline,
the storage probe, learning curves, save durations, endpoint policies, batch
histograms, Etude performance records, arena results and Command traces:
`uv run --extra artifacts python -m manabot.data fetch experiments/data/etude-execution/a6000-results.tar.gz`
(SHA-256 `5b5345e2adc0b234b8d3a956a1b10839c67e3122e2c1e9064f5de1af254c4827`).

Before any remote run, the harness's Etude CLI path passed locally: a two-update
CPU baseline stopped after update one (exit 75) and resumed to update two, with
both saves timed; the managed callback played two paired deals (16
replay-checked games) on CPU in 7.72 s, 645 forwards all of batch one.

<details>
<summary>Rental attempts, 2026-11-17</summary>

Rentals came from RunPod or Vast under the USD 10 cap, any reasonable single
GPU. Lium was excluded after its attempt; Etude Task 73e1a9 owns its lifecycle.

- **15:36 UTC, Vast 4090** `mtg329-speed-4090-vast-ffe7d7083bd2`, instance
  `61090179`, branch `ee370d9757`. Rented with
  `ETUDE_HOME=/tmp/mtg329-performance-etude doppler run --project etude --config prd -- doppler run --project manabot --config prd -- uv run --extra artifacts etude rent mtg329-speed-4090-vast --hours 2 --max-usd 10`.
  RTX 4090, 24,564 MiB, driver 595.71.05. During native compilation (181 MB of
  Cargo target at 15:50) SSH dropped at 15:52 and Vast reported
  `actual_status=offline`, cause unknown. Nothing was measured. Terminated with
  `etude terminate`; absent at Vast at 15:53. ~USD 0.12. A RunPod A6000
  fallback found no capacity.
- **16:00 UTC, Lium 4090** (branch `f962de12a1`), after no RunPod or Vast
  capacity: pod `55297a92-fa2b-47dd-9c73-b327672e74ca` created at the
  timezone-naive `2026-11-17T16:01:32.886331`, which Etude 0.4.1's
  `Pod.created_at: AwareDatetime` rejected, leaving launcher machine
  `mtg329-speed-4090-lium-7b0c01e641d4` unreconcilable ("Lium create already
  attempted; awaiting reconciliation"). Deleted through Lium's API; 404 by
  16:03. ~USD 0.01, unsettled. The intent remains in
  `/tmp/mtg329-performance-etude`.
- **16:09 UTC**, fresh launcher `/tmp/mtg329-runpod-vast-etude` (RunPod/Vast
  only): RunPod L4/4090/A6000/A5000 and Vast L4/4090 (two templates)/3090 all
  unavailable. Nothing rented.
- **18:57 UTC, Vast 4090** `mtg329-speed-4090-vast-711bee426992`, instance
  `61113127`, USD 0.427/h, branch 3670249eb3. After exporting initial
  policies and update zero, the first branch arm spent over six minutes with
  ~18 CPU seconds, 0% GPU and kernel waits `flush_workqueue`/`rq_qos_wait`.
  A repeat with `faulthandler.dump_traceback_later(45)` showed the main thread
  in Etude 0.4.1 `run.py:412`, `os.fsync(stream.fileno())` in `declare`;
  py-spy was refused by ptrace permissions. RAM-backed output was not
  substituted. Terminated 19:17, absent at 19:17:48. ~USD 0.14. Ten follow-up
  checks across RunPod and Vast (4090, 3090, A6000, A5000, L4) found nothing
  except one China-hosted Vast 4090 offer.
- **19:36 UTC**: a RunPod 4090 offer at USD 0.89/h returned no capacity on
  rent. **19:37 UTC, RunPod A6000** `mtg329-speed-a6000-2f5283ae0409`, pod
  `k9uj7aajsbbplr`, USD 0.69/h, produced the results above. Terminated at
  20:07:53, absent at RunPod. ~USD 0.35.

</details>
