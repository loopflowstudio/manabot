# Removing the host-bound small-call tail of collection on an L4 — MTG-318

**Committed before any scored run.** *Question:* can CUDA graphs, through
`torch.compile(mode="reduce-overhead")` or a hand capture per bucket signature,
or coalescing concurrent small requests at the collector, remove the per-call
host cost of collection's small policy calls? In the
[L4 step profile](l4_step_profile.md), calls of eight rows or fewer were 1,385
of 1,717 calls in a w160l8 update. Each took 16–27 ms for about 5 ms of GPU
work, and bucketed shapes made them 1.38× slower. *Prediction:* compilation
breaks the graph at each remaining compaction read, so it gains less than 1.5×.
A hand capture per bucket signature brings bucketed calls of eight rows or fewer
from about 25 ms to 8 ms or less (≥ 3×) at both w160l8 and w384l8. Coalescing
removes under 20% of tail calls, because the tail is mostly one game left per
producer rather than many concurrent small requests. Graphs are therefore the
better lever. *Refuted if* no graph path reaches 1.5× on calls of eight rows or
fewer at both sizes, or coalescing removes more tail time than the best graph
path. *Cost cap:* USD 5 under the launcher's `--max-usd`.

**Decision rule.** A path is accepted if, on the retained real collection
inputs replayed on the L4, it meets all of these:

1. Median per-call latency for calls of eight rows or fewer is at least 1.5×
   faster than eager with bucketed shapes, at both w160l8 and w384l8.
2. Calls of 9–32 rows are no slower than eager bucketed (within 5%).
3. Logits, values and log-probabilities match eager bucketed within 1e-5
   (maximum absolute difference) on every replayed input.
4. Projected whole-collection time per w160l8 update falls, counting capture or
   compilation cost spread over a 100-update run.

An accepted path ships as a typed `Execution` field, omitted from the
serialized regime when unset, with tests that pin its outputs to eager on
retained minibatches. If no path is accepted, this report is the result and no
code changes.

## What is measured

Arms, each at w160l8 and w384l8 on the same retained inputs as the step profile,
in interleaved order with a baseline repeat at the end:

| Arm | Forward |
|---|---|
| eager | default shapes |
| eager bucketed | `Execution(bucketed_shapes=True)` |
| compiled | eager bucketed under `torch.compile(mode="reduce-overhead")` |
| captured | eager bucketed with one CUDA graph per bucket signature (when compilation leaves breaks) |
| coalesced | eager bucketed, with the collector merging concurrent small requests |

Each arm reports:

- per-call latency and GPU busy time by bucket signature, grouped as calls of
  eight rows or fewer and calls of 32 rows or fewer;
- whole-collection time per update, projected from each signature's latency and
  the in-situ call mix, and for coalescing measured as the number of calls made;
- graph breaks and recompilations under compilation, each attributed to the
  host read that caused it;
- the remaining data-dependent host reads in the bucketed forward (nonzero legal
  rows, used columns, program lengths, CSR routes and packed GRU lengths), with
  what removing each would cost;
- one-time capture or compilation cost, and memory held by captured graphs.

*Rollout recording*, the 76–172 s of update 1 that the step profile left
unexplained, is timed by sub-phase over several updates of the same in-situ job.
It is explained if its sub-phases add up to within 10% of the phase. If not,
the report bounds it.

## Answer

**No path is accepted.** Neither graph path reaches 1.5× on calls of eight rows
or fewer, at either size:
- `torch.compile(mode="reduce-overhead")` is 1.32× faster at w160l8 and 1.19×
  at w384l8. It leaves 38 graph partitions per forward and costs about 6
  minutes of compilation per size.
- A hand capture of the whole forward is impossible while the forward still
  reads the host 23 times. Capturing its one read-free stretch, the attention
  trunk, gives 1.17× and 1.06×.

Coalescing removes no calls: collection already makes one policy call per
environment step, covering every active stream. The prediction is refuted on
the graph side: no path reached 1.5×, and the predicted ≥ 3× hand capture could
not be built at all.

The tail is also smaller than MTG-301 suggested:
- Calls of eight rows or fewer took 4.5–10.4 s of a 75–100 s collection phase.
- Rollout recording took 2.7–4.6 s per update, not 76–172 s.

Even a perfect graph would save at most a few seconds per update. No
`Execution` switch ships.

## Results

### Small calls stay host-bound under every arm

![Median per-call latency by rows per call, four arms, two sizes](data/collection_tail_graphs/latency-by-band.svg)

Each value is the median over the retained inputs in the band of 15 timed
repetitions, measuring collection's preparation, forward and action selection:

| Arm | w160l8 ≤ 8 rows | w160l8 9–32 | w160l8 129+ | w384l8 ≤ 8 rows | w384l8 9–32 | w384l8 129+ |
|---|---:|---:|---:|---:|---:|---:|
| eager, default shapes | 19.1 ms | 53.3 ms | 525 ms | 23.0 ms | 96.4 ms | 1,156 ms |
| eager, bucketed (reference) | 24.5 ms | 37.4 ms | 211 ms | 25.0 ms | 54.6 ms | 379 ms |
| compiled | 18.6 ms (1.32×) | 35.0 ms | 196 ms | 20.9 ms (1.19×) | 51.1 ms | 345 ms |
| trunk CUDA graph | 20.9 ms (1.17×) | 37.0 ms | 212 ms | 23.7 ms (1.06×) | 54.3 ms | 377 ms |

The profiled pass shows why. A call of eight rows or fewer keeps the GPU busy
for only about 1.9 ms (w160l8) or 3.1 ms (w384l8) of a 24–28 ms profiled call:
- **Compiled:** cuts launches from 681 to 145 and syncs from 33 to 19. Dynamo
  guards, the eager stretches between graph breaks and preparation still run
  on the host.
- **Trunk graph:** replaces only about 165 of the 681 launches.
- **Bucketed against default shapes:** bucketed small calls are 1.28× slower,
  as MTG-301 found. The extra cost is host-side.

Decision rule, as pre-registered:

| Rule | compiled | trunk graph |
|---|---|---|
| 1. ≤ 8 rows ≥ 1.5× faster at both sizes | **fails** (1.32×, 1.19×) | **fails** (1.17×, 1.06×) |
| 2. 9–32 rows within 5% | passes (0.94×, 0.94× time) | passes (0.99×, 0.99×) |
| 3. outputs within 1e-5 | passes (≤ 4.8e-7) | passes (≤ 2.4e-7) |
| 4. projected w160l8 collection falls | passes narrowly (58.8 → 57.8 s, see limits) | passes narrowly (58.8 → 58.5 s) |

The largest differences from the reference were 9.8e-9 on legal logits, 2.4e-7
on log-probabilities and 4.8e-7 on values. The bucketed repeat at the end of the
job read within 1% of its first run (24.5 / 24.5 ms; 37.4 / 37.5 ms).

### What still reads the host

A one-row bucketed forward on CUDA makes **23 synchronizing calls**. Each is
listed below with what removing it would take:

| Site | Syncs | What it is | Cost to remove |
|---|---:|---|---|
| `buckets.py:306–307` | 12 | object-reference columns indexed with a Python list (a host→device index copy per key) | small: a device index buffer built once |
| `facts.py:84` | 5 | `torch.isin` against `new_tensor(values)` | small: registered buffers |
| `actor_critic.py:706`, `:764` | 3 | list column indexing | small: device index buffers |
| `facts.py:156–157` | 1 | boolean-mask gather of program rows and a `len(rows) == 0` branch | moderate: run the program GRU over the fact bucket with a mask |
| `shared_effects.py:87` | 1 | effect headers copied to the host to build CSR record rows | large: device-side CSR from offsets and counts |
| `actor_critic.py:394` | 1 | `nonzero` over legal actions to gather only legal slots | moderate: score every slot in the action bucket, then mask |

Whole-forward capture fails at the first of these (`buckets.py:307`, a
CPU→CUDA copy during capture).

Under compilation, one pass over the 24 inputs also broke the graph at:
- `nonzero` (29);
- the program GRUs, which Dynamo does not trace (16);
- `sort` with SymInt keys (30);
- the per-call admission binding (`model_dump_json` and `named_modules`, 23);
- two scalar reads.

The admission binding is pure Python per call. The GRUs need either
`allow_rnn` or a masked unpacked recurrence.

Removing the 22 cheap and moderate reads is a prerequisite for any capture.
The headroom is bounded, though: the GPU work is about 2–3 ms, and preparation
(admission, host bucket measurement and transfer) took a median 7.3 ms per
small call in situ, outside the forward.

### The tail in situ, and rollout recording

Three real updates of the frozen w160l8 regime on the current world, with
bucketed shapes:

| Update | Collection | Calls (= steps) | Calls ≤ 8 rows (time) | Calls 9–32 (time) | Preparation | Inference | Recording |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 73.7 s | 532 | 166 (4.5 s) | 75 (2.9 s) | 24.6 s | 30.6 s | 4.6 s |
| 2 | 99.7 s | 635 | 254 (6.8 s) | 119 (4.5 s) | 23.0 s | 31.6 s | 2.7 s |
| 3 | 74.8 s | 780 | 400 (10.4 s) | 104 (4.0 s) | 23.8 s | 34.1 s | 2.7 s |

Calls equal environment steps in every update, so there were never two
requests to merge. A shorter tail needs a different batch closure, which
changes the learning data and is outside this question.

**Rollout recording is explained on this host.** Each update copied 10.1–10.6
GB of observation rows (about 210 KB each) into a store allocated fresh per
update. Recording split into three parts:

| Update | Append | Packing | Pending-row loop | Garbage collection |
|---|---:|---:|---:|---:|
| 1 | 3.8 s | 0.3 s | 0.5 s | 0 s |
| 2 | 1.9 s | 0.3 s | 0.5 s | 0 s |
| 3 | 1.8 s | 0.3 s | 0.5 s | 0 s |

- **Append** includes update 1's two store doublings (20,480 → 40,960 → 81,920
  rows).
- **Packing** is the per-call `PackedActions.pack`.
- **The pending-row loop** is the remainder of the phase, not timed on its own.

Append and packing together account for 81–89% of the phase. The 76–172 s
reading in MTG-301 did not recur. Its cause lies in that host, not in the
recording code.

## Cost

Etude run `1a2e6cb86345`: about $0.27 (launcher), under the USD 5 cap.

## Limits

- **One host and one GPU type.** This L4 pod had a 128-CPU AMD EPYC 7542 host;
  MTG-301's had 48 CPUs, so host-bound absolute times differ. Every ratio here
  is within this job.
- **24 retained inputs.** They are spread over row bands from update 1 and
  weighted to the tail; band medians rest on 1–5 inputs each.
- **The rule-4 projection undercounts compilation.** It multiplies band
  medians by update 1's call mix and spreads the replay's first-call cost
  (385 s for 24 inputs) over 100 updates. In situ there were 367–396 distinct
  bucket signatures per update, each a new CUDA graph recording under
  `reduce-overhead`.
- **w384l8 is a random initialization;** w160l8 uses the in-situ initial
  weights.
- **Rollout recording was read on one host only.** The host that produced
  76–172 s was not available, so that reading is bounded here, not reproduced.
- No learning evidence; no arm changes what is learned.

<details>
<summary>Appendix: method, run and evidence</summary>

### Method

`experiments/collection_tail_graphs.py` (revision `8b753ade3`) runs four stages
in one job.

1. **`insitu`.** `tla_frontier.frozen()` w160l8 runs with `updates=3` and
   `Execution(bucketed_shapes=True)`, from its own initialization on the current
   world. The MTG-301 weights no longer load, because the card pool grew. The
   regime is otherwise unchanged: 256 streams × 64 transitions with complete
   games, both seats, four producers, float32 on CUDA.
   - Hooks record each policy call's rows, bucket signature, preparation,
     inference and recording time.
   - Recording is split into the observation-store append (with minor page
     faults and growth), action packing and the rest.
   - `gc.callbacks` attribute garbage-collection pauses to phases.
   - Inputs are kept per row band, taking every k-th call of the band in
     update 1.
2. **`probe`.** On the smallest kept input (one row):
   - synchronizing calls per site, through `torch.cuda.set_sync_debug_mode`;
   - `torch._dynamo.explain`, with scalar and dynamic-shape capture on, to list
     breaks past the first;
   - one attempt to capture the whole bucketed forward with `torch.cuda.graph`.
3. **`replay`.** One process per arm and size, interleaved, with a bucketed
   repeat at the end:
   - Per input: one first call (which includes compilation or capture), two
     warm calls, then 15 timed calls synchronized on the device.
   - Outputs are compared with bucketed eager.
   - One `torch.profiler` pass gives GPU busy time, kernels, launches and syncs
     per call.
   - The compiled arm is `torch.compile(model.forward_distribution,
     mode="reduce-overhead")` with default flags and
     `cudagraph_mark_step_begin` per call.
   - The trunk arm routes `_attend_objects` through one `torch.cuda.CUDAGraph`
     per input shape, all sharing one memory pool.
4. **`analyze`.** Writes `summary.json`, including `decision`: the rule applied
   to each candidate arm.

Stage times on the L4: in situ 369 s, probe 17 s, eager 47/89 s, bucketed
30/44 s, compiled 421/394 s, trunk 31/46 s (w160l8/w384l8).

### Evidence

Everything is under `experiments/data/collection_tail_graphs/` (manifest entries
in `experiments/data.manifest.json`):
- `summary.json` and `job.json`;
- `insitu/receipt.json`, with every call's record;
- `probe/receipt.json`;
- `replay/<arm>-<size>/`: receipt, outputs and gzipped trace;
- `logs/`: compile logs carry `TORCH_LOGS=perf_hints,recompiles,graph_breaks`;
- `inputs/`: initial weights, regime and the 24 inputs;
- `gpu-samples.csv`.

Reproduce with

```sh
etude launch experiments/collection_tail_graphs.py --rent l4-etude \
  --cohort NAME --max-usd 3 --revision COMMIT --hours 2.0
```

and test locally with `ETUDE_OUTPUT_DIR=… uv run python -m
experiments.collection_tail_graphs job --smoke`.

</details>
