# Does MLX train the policy faster than 8-thread CPU on an Apple M2? — MTG-321

**Committed before any scored run.** Question: on one Apple M2 (24 GiB), does
an MLX port of `ActorCritic` and its PPO learner step, numerically equal to the
PyTorch model, beat Torch 2.15 on 8 CPU threads? Prediction: MLX is 1.3× faster
on the w160l8 whole update (collection plus learning on real self-play) and
1.8× faster on the w384l8 learner step. The pilot is **accepted** only if MLX is
at least 1.5× faster on both. Anything less closes it, and a parity failure
closes it before any timing. Cost cap: the already-rented host, plus at most
one further day (about USD 21) if the soak needs it.

**Parity bar, fixed before timing.** Against PyTorch float32 on CPU, on the same
real checkpoint and captured w160l8 inputs: logits and values within 1e-4
absolute, PPO loss within 0.1% relative, every parameter's gradient cosine
at least 0.999. Checked with
[`manabot.model.mlx.parity`](../manabot/model/mlx.md#parity-check).

## Answer

**Accepted.** The MLX port matches the PyTorch policy to rounding, and on the
M2 it runs a whole w160l8 update 1.67× faster than PyTorch on 8 CPU threads
(452 s against 756 s) and the w384l8 learner step 1.57× faster (5.15 s against
7.84 s, median). Both clear the 1.5× bar. The prediction was low on the update
(1.3×) and high on the learner (1.8×): collection policy calls gain most
(2.2–2.7×), learner steps about 1.6×. PyTorch's own MPS backend gains nothing
at w384l8 (0.93×). The port needed two fixes to get there; as first built it
lost to CPU8 everywhere except one-row calls.

| Decision input | CPU8 | MLX | MLX speedup | Bar |
|---|---:|---:|---:|---:|
| w160l8 whole update (replayed) | 755.7 s | 452.1 s | 1.67× | 1.5× |
| w384l8 learner step (median) | 7.84 s | 5.15 s | 1.57× | 1.5× |

Parity passed far inside its bar, before any timing: logits within 7.7e-9,
values within 3.6e-7, loss within 3.3e-7 relative, minimum gradient cosine
0.99994 over the 168 tensors with nonzero gradients.

## Results

All numbers come from one Apple M2 (8 cores, 24 GiB), all arms on the same host
in the same hour, from one fresh capture of a real w160l8 update (method in the
appendix). Speedups are medians of per-call or per-step ratios from arms that
alternate call by call.

### 1. Collection policy calls: MLX is 2.2–2.7× faster at every batch size

![Collection latency](data/mlx_pilot/collection_latency.png)

| Rows per call | CPU8 | MLX | MLX speedup |
|---:|---:|---:|---:|
| 1 | 10.1 ms | 4.5 ms | 2.22× |
| 8 | 35.3 ms | 16.0 ms | 2.16× |
| 64 | 253 ms | 92.7 ms | 2.69× |
| 256 | 1,026 ms | 387 ms | 2.62× |

MLX wins even at one row (4.5 ms against 10.1 ms); its host-side index
preparation takes 0.2 ms of that at B = 1 and 6 ms at B = 256. Both arms drop the object and fact columns no row uses, as PyTorch
CPU inference already does; without that compaction MLX lost at B ≥ 8
(see *Where MLX stalled*).

### 2. Learner step: MLX is about 1.6× faster at both widths; MPS is not faster

![Learner step](data/mlx_pilot/learner_step.png)

| Width | CPU8 | MLX | MLX speedup | torch MPS | MPS speedup | MLX peak | MPS driver peak |
|---|---:|---:|---:|---:|---:|---:|---:|
| w160l8 | 2.79 s | 1.74 s | 1.60× | 2.28 s | 1.21× | 13.5 GiB | 16.0 GiB |
| w384l8 | 7.84 s | 5.15 s | 1.57× | 8.83 s | 0.93× | 23.3 GiB | 23.0 GiB |

Steps are the first 24 captured minibatches of the update (12 for the MPS
pair), 12–108 rows each, weights and Adam state evolving in every arm. The
speedup holds across minibatch sizes except the smallest: a 12-row step is
about as fast on either side. MPS speedups are against the CPU8 steps of their
own alternating pair (medians 2.86 s and 8.96 s). The MPS arm answers whether
"CPU rollouts, MPS updates" is worth revisiting at the width we intend to
train: it is not. Step by step MPS runs at 0.68–1.27× CPU8's speed and is
slower on the largest w384l8 minibatches; its median at w384l8 is 0.93×, far
below 1.5×.

### 3. Whole update: MLX is 1.67× faster

![Whole update](data/mlx_pilot/whole_update.png)

Every captured collection call and all 197 learner steps were replayed from
the initial checkpoint in each arm. CPU8 took 755.7 s: 27.3 s engine, 187.5 s
collection policy calls, 540.9 s learning. MLX took 452.1 s: the same engine
share, 77.4 s of policy calls (2.4× less) and 347.4 s of learning (1.56× less,
1.76 s against 2.75 s per step). MLX peaked at 15.9 GiB over the update.

### 4. Memory: w384l8 is the widest that fits in 24 GiB

![MLX memory by width](data/mlx_pilot/memory.png)

On the largest captured minibatch (108 rows), the MLX learner step peaks at
13.5 GiB at width 160, 17.7 GiB at 256 and 20.5 GiB at 320. Growth is about
47 MB per unit of width at this batch size, mostly activations of eight layers
over 450 padded objects; weights and Adam state are a small share. The w384l8
learner phase peaked at 23.3 GiB, 0.7 GiB under the machine's memory. The
sweep stopped before 384 on its own 95% guard, so that point comes from the
learner phase. Extrapolated at the same slope, w448 needs about 26 GiB, w640
about 34 GiB and w1024 about 51 GiB at 108 rows. Those are extrapolations,
not measurements. On a 24 GiB Mac, MLX trains w384l8 at these minibatch sizes
and nothing wider. What a larger Mac buys is width, roughly 3 GiB per 64
units.

## Where MLX stalled, and what changed during the pilot

The port as first built lost: MLX collection calls were 1.7–1.8× slower than
CPU8 at B ≥ 8 and learner steps 1.5–2.5× slower, with a floor near 5 s. Two
fixes, each a change that keeps the question, turned that around (2026-11-12):

- **The GRU backward was quadratic.** Profiling one 108-row step put 3.39 s of
  its 5.9 s gradient in the catalog GRU (713 recurrent steps), whose forward
  costs 0.04 s. A synthetic recurrence on the M2 showed why: MLX materializes
  a zero cotangent the size of the whole input for every per-step slice or
  split, so the backward grows with the square of the step count (713 steps:
  0.07 s forward, 15.5 s gradient). Gathering each step's rows from the
  fixed-size token table instead is linear (0.39 s). The catalog GRU's
  forward and backward fell from 3.39 s to 0.22 s.
- **Inference compaction.** PyTorch CPU inference attends only over object
  columns some row uses (about 30 of 450 early in a game) and drops unused fact
  columns. The port does the same now (median B = 256 call: 1.89 s → 0.39 s).

What remains is arithmetic. Attention over the fixed 450-object table is 0.74
of a 0.81 s forward at 108 rows, mostly the 1,280-wide MLP of eight layers over
padding. Learning keeps fixed shapes in both arms, as production does. Kernel
dispatch is not the limit at these sizes, and compiling the GRU cell made no
measurable difference. Shape variety rules out compiling the whole step: legal
counts and packed schedules change with every batch, so the step runs eagerly.

The bench run before these fixes is kept on the host and summarized in the
appendix; it is not the result.

## Implications

- **MLX is the faster Apple path for this policy.** It beats CPU8, the recipe
  the MPS study settled on, by about 1.6–1.7× at both widths and in the whole
  update, and it holds parity. Training through it is the next step. That
  means MLX collection and learning inside the executor, plus writing PyTorch
  checkpoints back from MLX (the reverse of `convert`). It is a design change
  for its own Task, not part of this pilot.
- **"CPU rollouts, MPS updates" is not worth revisiting.** The bar for
  revisiting it was 1.5× over CPU8 at w384l8; PyTorch MPS reaches 0.93× there
  and 1.21× at w160l8.
- **The remaining cost is padded attention.** Eight layers over all 450 object
  slots dominate the learner forward on the GPU, as on the L4 (MTG-301).
  Bucketed shapes, which cut that padded work by about 2.4× on CUDA, are
  untested on MLX and would compound with it.
- **Memory caps the Apple width, not speed.** A 24 GiB M2 fits w384l8 with
  0.7 GiB to spare.
- **Context, not comparison:** an L4 runs the padded float32 w384l8 learner
  step in about 0.92 s (MTG-301, a different machine), about 5.6× faster than
  MLX on the M2. At about USD 0.59 an hour against the M2's USD 0.88 (USD 21
  a day), rented CUDA stays far cheaper per learner step. MLX makes Macs we
  already own faster; it does not make renting Macs worthwhile.
- **An MLX lesson worth keeping:** in MLX, never slice or split one array
  per recurrent step under autograd. Each piece's gradient materializes the
  whole input, so the backward turns quadratic. Gather each step's rows from
  a fixed table instead (`PackedGRU.run`).

## Cost

No new rental. The M2 was already paid through 2026-11-12 23:39 UTC; the pilot
finished inside that window, so the extra day the cap allowed was not used.

## Limits

- One machine, one capture, one update. The ratios are medians over 23 learner
  steps per width and 96 collection calls per batch size, not a population
  estimate. The host's idle load is near 1.8 from system daemons.
- During the first collection block, a package install on the host overlapped
  the B = 64 and B = 256 calls (load average 27.7). The collection phase was
  re-run alone afterwards. The table uses that re-run; the overlapped run read
  within 10% of it at every size.
- The whole update is a replay (the MPS study's X3 shape), not a training run
  with MLX in the loop. Engine time comes from the capture and is at most
  27.3 s, since that figure also absorbs writing the captured inputs. The
  bootstrap forward for value targets, a single no-gradient pass, is not
  replayed.
- w384l8 starts from fresh seeded weights, not trained ones.
- Learning trajectories diverge slowly. At the checkpoint the two frameworks
  agree to 1e-7, and over the first eight steps losses stay within 4e-4.
  After about 20 steps, single-minibatch losses differ by up to 0.064 at
  w160l8 and 0.007 at w384l8, against typical losses of 0.05–0.7. Adam
  divides each gradient by the root of its second moment, so rounding noise
  in tensors whose gradient is almost zero becomes full-size updates that
  differ between frameworks. This is
  the usual divergence between float32 implementations, not a parity failure;
  a training-quality comparison would need repeated seeds in both.
- The surrogate's likelihood ratio is float32 in MLX (float64 in PyTorch);
  the two differ only for |log ratio| > 88.
- The L4 figures from [l4_step_profile.md](l4_step_profile.md) and
  [cuda_precision.md](cuda_precision.md) (padded float32 learner step about
  400 ms at w160l8 and 920 ms at w384l8) come from a different machine, as
  context only.

<details>
<summary>Method, process and evidence</summary>

**Host.** One AWS mac2-m2 (Apple M2, 8 cores, 24 GiB unified memory, macOS
26.7) with nothing else scheduled; its idle one-minute load average is near
1.8 from system daemons, and every measured block records the load before and
after. Python 3.13, torch 2.15.0, MLX 0.32.4, source at this report's commit.
Every run, smoke test and install happened on the M2 over ssh.

**Inputs.** The MPS study's captured inputs are bound to an earlier card world
and no longer load, so a fresh capture (`experiments/mlx_pilot.py capture`)
ran one update of the frozen TLA frontier w160l8 regime
(`experiments.tla_frontier.frozen`) on CPU, one worker, four threads, from its
seeded initial checkpoint. Hooks copied every collection policy input (1,526
calls) and every learner minibatch with its targets and settings (197
minibatches) and read clocks; they changed nothing the run computed. The update
took 826 s; collection 193.8 s, of which 166.5 s were policy calls, leaving at
most 27.3 s for the engine.

**Parity** (`manabot.model.mlx.parity`) ran first, on 8 learner minibatches and
16 collection calls, against PyTorch float32 on CPU with the training code's
own loss. One clarification was fixed before any timing: the policy-score bias
and every attention key bias have analytically zero gradients, so a tensor
below 1e-6 of the global gradient norm in both frameworks counts as zero (33 of
201 tensors; their norms were about 1e-9 against a global norm of 3.1) instead
of having its noise cosine scored. A later bench phase refuses to run without a
passing parity record in the same ledger.

**Arms.** CPU8 is the production PyTorch path on 8 threads: one
`PolicyInference` scope per collection for inference; for learning the damped
loss, `OptimizerHealth` backward, clipping and Adam. MPS is that learner on
torch's MPS device, same venv. MLX is `manabot.model.mlx`: host-side index
preparation in numpy, then the MLX graph, timed to evaluated outputs (logits
copied back to numpy for collection). Each bench phase is its own process: MLX
and MPS share unified memory, and an early run that held both in one process
ran MPS out of memory.

**Collection latency.** For B in 1, 8, 64 and 256: eight batches of exactly B
real rows cut from consecutive captured calls spread over the update, three
repetitions, CPU8 and MLX alternating per batch after one warm call each.

**Learner step.** The first 24 captured minibatches in order (12 for the MPS
pair), weights and Adam state evolving in each arm; CPU8 alternates per
minibatch with MLX or with MPS. The first step of each phase is warm-up and is
excluded. w160l8 starts from the capture's initial checkpoint, w384l8 from a
fresh model of the same architecture at width 384 (seed 21501), identical in
every arm.

**Memory.** MLX peak (`mx.get_peak_memory`) over each learner phase, the MPS
driver allocation peak, and an MLX width sweep: two learner steps on the
largest of the first 24 minibatches, stopping before a predicted overflow of
24 GiB.

**Whole update.** The MPS study's X3 shape: every captured collection call,
then all 197 minibatches with weights evolving, replayed in each arm from the
initial checkpoint (MLX first, then CPU8). The update time is that plus the
capture's engine share, which both arms run identically.

**Bench run before the two fixes** (same host, same capture; ledger kept as
`bench-v1`): collection medians CPU8/MLX 10.2/8.8 ms (B = 1), 35.8/59.6 ms (8),
254/470 ms (64), 1,031/1,891 ms (256); the first eight w160l8 learner steps
took 2.0–4.9 s on CPU8 and 4.7–7.5 s on MLX. Its MPS arm ran out of memory at
step 8 beside MLX in the same process.

**Profile of one 108-row learner step** (eager MLX, before the GRU fix):
catalog GRU forward 0.04 s, forward and backward 3.39 s; whole gradient 5.90 s,
2.42 s with the catalog held fixed. After the fix: 0.22 s and 2.67 s. Forward
by section: attention 0.74 s, catalog 0.03 s, facts 0.02 s, permanent links
0.01 s, of 0.81 s.

**Evidence** in `experiments/data/mlx_pilot/` (artifact store): `bench.jsonl`
(every timed call and step, parity record, loads, memory), `soak.jsonl`,
`bench-v1.jsonl`, `profile.jsonl`, `summary.json`, the four charts, the
capture's `capture.json`, `calls.json`, `minibatches.json` and `regime.json`,
and a parity subset (`capture/initial.pt`, the first eight minibatches, the
sixteen collection calls) so `manabot.model.mlx.parity` can be re-run. The
full 12 GB capture is not retained. `experiments/mlx_pilot.py report`
regenerates `summary.json` and the charts from the ledgers.

</details>
