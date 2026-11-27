# Collection closure and host/GPU overlap on an L4 (MTG-496)

Exploratory throughput run (no prediction or protocol).

**Question.** In two-seat self-play, how much collection throughput comes from
filling every stream for the whole update (`batch_closure="exact-quota"`)
instead of waiting for each stream's game to end (`complete-games`)? After that,
does overlapping one cohort's host work with the other cohort's GPU inference
add at least 15% more collection decisions per second?

**Answer.** Exact quota removes idle stream slots. On error-free updates it
collects 1.38× as many decisions per second at `w160l8` (2,597 vs 1,885) and
1.17× as many at `w384l8` (1,260 vs 1,078). Each update makes 186 full-width
policy calls, where complete-games makes about 640–740 mostly empty ones.
Overlap with two cohorts gave +25% at `w384l8` and nothing at `w160l8`. It
failed the keep rule and was removed. Exact quota stays opt-in, and the
frontier recipe keeps `complete-games`. Moving the default needs a learning
comparison.

## Setup

`experiments/collection_overlap.py` defines four arms. Each runs on one
RunPod L4 with 256 streams, 4 native workers, 2 Torch threads, the TLA trophy
corpus, seed 40902 and 5 updates. Exact-quota arms use `transitions=186`
(47,616 learner rows per update), matching complete-games' ~47.7k rows from a
late-training run (`tla_trophies_colab_500k_v2/w160l8`, about 186 decisions
per game). Both sizes ran with fresh weights. The u2000 `w160l8` checkpoint
comes from `v0.2.4+10`, and its vocabulary (`token_embedding` 4634 vs 4603,
`facts.kind` 15 vs 21) no longer loads strictly on HEAD.

Update 1 includes warm-up and is excluded. Rates are total collection
decisions over total collection time across the listed updates.

## Closure: complete-games vs exact quota

Pass 1, at `468b79e88a`:

| Arm | Updates | Collection dec/s | Idle stream-steps / update | Collection s / update | Learning s / update | Inference share |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `w160l8` complete-games | u2–u4 | 1,885 | 118,768 | 31.3 | 27.0 | 51% |
| `w160l8` exact quota | u2–u4 | **2,597** | 0 | 18.3 | 22.1 | 43% |
| `w384l8` complete-games | u2–u3 | 1,078 | 152,470 | 54.6 | 54.3 | 67% |
| `w384l8` exact quota | u2–u3 | **1,260** | 0 | 37.8 | 44.6 | 69% |

All four arms had no game failures on these updates. Counting the updates with
one recovered failure (u2–u5 for every arm), the rates are 1,972 vs 2,417
(1.23×) at `w160l8` and 1,121 vs 1,185 (1.06×) at `w384l8`. After a failure,
the failed stream stepped alone until it reached its quota. That added 46,665
idle stream-steps (`w160l8` u5) and 38,760 (`w384l8` u4).

2026-11-26: after these measurements, two-seat exact quota moved to a
pool-total row target. Every stream steps while at least `streams` rows
remain, so a failure's tail is now under one round (fewer than `streams` idle
stream-steps). Error-free updates are unchanged; the failure updates above were
not rerun.

At `w160l8`, an exact-quota update spends its collection time as follows:
inference 45.7%, environment step and encoding 23.6%, observation preparation
20.1%, bookkeeping and refill 4.4%, recording 1.9%, retention 0.7% and cut
bootstrap 0.3% (u2–u5). At `w384l8`, inference is 67.1%, and host plus
environment work is 31.1%.

Learning time also falls. Learning operates on 47,616 rows per update, where
complete-games produces ~57k. A whole update (collection plus learning) took
40.4 s instead of 58.3 s at `w160l8`, and 82.4 s instead of 108.9 s at
`w384l8`.

The waiting seat's viewer-read bootstrap at each cut differed from that seat's
next own-decision value by `mean_abs` 0.13–0.39 at `w160l8` and 0.11–0.22 at
`w384l8`, over 241–251 cut rows per update. With fresh weights this says
nothing about bias in a trained critic, and the gap also includes the opponent's
moves and the weight update between the two reads. The metric was removed
afterwards. A same-state, same-weight comparison belongs with the learning
comparison.

## Overlap: two cohorts (built, then removed)

The prototype was built at `baed2815e1` and `e9a1128dfc` and removed in
`9c51e12886`. It split the streams into two cohorts that shared one producer
pool and ran preparation and stepping on the main thread while an inference
thread ran the other cohort's forward. Each arm was paired with a fresh serial
exact-quota run on its own L4:

| Size | Updates | Serial dec/s | Pipelined dec/s | Change |
| --- | --- | ---: | ---: | ---: |
| `w160l8` | u2–u4 | 2,425 | 2,434 | +0% |
| `w384l8` | u2–u3 | 1,312 | 1,640 | +25% |

At `w160l8` the threads competed instead of overlapping. Inference time per
update grew from ~7.9 s to ~16.8 s, and observation preparation grew from ~3.8 s to
~10 s. The two cohorts' halves took as long as one serial step did. At
`w384l8`, where the forward is larger, part of the host work hid behind it.

Pass 1's split had left headroom at both sizes: the ceiling was +97% at
`w160l8` and +49% at `w384l8`. The +15% keep rule had to hold at both sizes.
`w160l8` missed it, so the cohorts, the `prepare_step`/`infer_step` split and
the `collection_pipeline` setting were deleted. A future attempt should start
by removing the forward's ~20 small CUDA host reads, which hold the GIL and
synchronize the device. The deferred finding lists them in
[collection_tail_graphs.md](collection_tail_graphs.md).

## Cost

Eight L4 rentals at about $0.10 each: four arms in pass 1 (~$0.45) and four in
the overlap check (~$0.35), $0.80 in total, under the $10 cap.

## Limits

- One run per arm, on solo L4s with whatever host CPU each rental had.
  Repeated serial exact-quota runs at `w384l8` u2–u3 differ by 4% (1,260 and
  1,312), so differences below about 5% are noise.
- Five updates with fresh weights. Game lengths come from an untrained policy
  at both sizes, not the late-training ~186-decision games the `transitions`
  setting was sized for. At these lengths a complete-games update still
  produces ~57k rows.
- At most one recovered failure per arm. A higher failure rate makes the
  refill tail matter more.
- Throughput only. The change in what the critic learns, which now bootstraps
  across update boundaries, has not been measured. A learning comparison
  stays outside this run.

<details>
<summary>Per-update readings</summary>

Collection decisions/s by update (u1 is warm-up):

| Run | u1 | u2 | u3 | u4 | u5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| pass 1 `w160l8` complete | 2,055 | 1,816 | 1,832 | 2,013 | 2,273 |
| pass 1 `w160l8` exact | 3,058 | 2,545 | 2,605 | 2,641 | 2,002* |
| pass 1 `w384l8` complete | 980 | 1,070 | 1,086 | 1,160 | 1,181 |
| pass 1 `w384l8` exact | 1,590 | 1,258 | 1,261 | 1,019* | 1,242 |
| overlap check `w160l8` serial | 3,090 | 2,189 | 2,472 | 2,661 | 2,360 |
| overlap check `w160l8` pipelined | 2,886 | 2,429 | 2,508 | 2,370 | 2,034* |
| overlap check `w384l8` serial | 1,556 | 1,284 | 1,342 | — | — |
| overlap check `w384l8` pipelined | 1,901 | 1,624 | 1,656 | 1,431* | 1,428* |

\* An update with a recovered game failure and its refill tail. The serial
`w384l8` run in the overlap check stopped after u3.

</details>
