# The forward layout in served evaluation and on CPU — MTG-328

Exploratory. Bucketed padding with live-token attention makes a w384l8 learner
step 4.4× faster on an L4 ([MTG-319](live_token_attention.md)) and is the CUDA
training default. Two questions decide where else it applies.

## Does the layout make a served evaluation pass faster?

**No.** One evaluation pass of a w384l8 policy on an L4 took 118 s with the
padded forward and 118 s with both switches on. The served forward itself was
slower with the layout: 27.3 ms against 23.7 ms per call, 15% more. Evaluation
therefore keeps the padded forward by default and accepts the switches
explicitly (`EvalAllocation.bucketed_shapes`, `live_tokens`).

![Mean served forward per pass: padded 24 ms, bucketed with live tokens 27 ms,
both repeats](data/eval_forward_layout/served_forward.svg)

| Pass (128 games, 24,653 decisions) | Wall | Forward calls | Rows per forward | Forward (mean) | Input admission and transfer (mean) |
|---|---:|---:|---:|---:|---:|
| Padded, 1 | 119.1 s | 2,718 | 9.1 | 23.7 ms | 3.8 ms |
| Bucketed, live tokens, 1 | 119.7 s | 2,741 | 9.0 | 27.3 ms | 5.7 ms |
| Padded, 2 | 117.7 s | 2,701 | 9.1 | 23.7 ms | 3.9 ms |
| Bucketed, live tokens, 2 | 117.0 s | 2,620 | 9.4 | 27.4 ms | 5.9 ms |

The two repeats of each arm agree within 0.3% on the forward, so the 15%
difference is not host drift. The wall time does not move because the forward
is a minority of the pass: about 64 s of forward across the pass with the
padded path and 73 s with the layout, served while 32 games run at once, against
about 79 s of trace replay after the games in every pass.

The difference from training is batch size. A collection forward in MTG-319
held 256 rows and a served forward here holds about nine, with a tenth to a fifth of
the calls carrying one row. At that size the L4 is not saturated by the padded
tables (GPU utilization averaged 21% over the job), so attending over fewer
tokens saves little, while trimming and packing add host work to every call.

All four passes played the same 24,653 decisions and scored the same 0.492, so
the layout changed no sampled action in 128 games.

## What does live-token attention cost on CPU?

**About twice the collection forward; the learner minibatches measured here
were not slower.** On CPU the jagged attention runs one call per observation.
Replaying MTG-319's retained w160l8 inputs on a laptop CPU (4 threads):

![Median learner pass and collection forward on CPU per
layout](data/eval_forward_layout/cpu_cost.svg)

| Layout, w160l8 on CPU | Collection forward, 244–256 rows (median) | Learner forward and backward, 23–84 rows (median) |
|---|---:|---:|
| Padded | 459 ms | 2,252 ms |
| Live tokens, full tables | 855 ms | 954 ms |
| Live tokens over buckets | 793 ms | 1,249 ms |

A collection forward is 1.7–1.9× slower, and single calls on later, fuller
boards reached 3.5 s. Collection is most of a CPU stage's forwards, so
`live_tokens=True` stays refused off CUDA (`live tokens require CUDA`). The
learner result says the per-observation fallback is not uniformly slow: on
minibatches of a few dozen rows, skipping padded slots more than pays for it.

## Cost

USD 0.11 (launcher) for one L4, 8 minutes of job time, under the USD 3 cap. The
CPU measurement ran locally.

## Limits

- The policy is a seeded random initialization playing itself, so boards and
  game lengths are those of random play, not of a trained policy. A trained
  policy's fuller boards raise live-token counts, which favours padding less,
  but does not change the batch size.
- One allocation shape: 32 active games and 32 batch rows, which gave about nine
  rows per forward. Larger served batches were not measured; at collection's
  256 rows the layout was 4.4× faster in MTG-319, so an allocation that fills
  large batches may gain from writing the switches on.
- Search players and range likelihoods were not measured. They keep the padded
  forward.
- The CPU timings come from a laptop with other work on it: timed passes of one
  arm differ by up to 2×. Read the collection result as "roughly double", not as
  a precise ratio. Four learner minibatches and eight collection observations.

<details><summary>Appendix: run and evidence</summary>

- Etude run `2e3990779ba4` (`mtg328-eval-forward-layout`), source `4fe550046`
  on this branch, machine `mtg328-l4-8c03d33b42b9`: NVIDIA L4, AMD EPYC 9254
  (48 CPUs), torch 2.15.0+cu132, CUDA 13.2. Launched 21:22 UTC on 2026-11-12,
  succeeded, job 499 s; the machine was terminated by hand after the run.
- Dated line, 2026-11-12: at the run's source an omitted allocation layout
  resolved on for CUDA, and the `layout` arm relied on that. After this result
  the allocation default is off and the script writes both switches on for that
  arm. The arms compared are unchanged: both off against both on.
- Each pass is `manabot.eval.evaluation.evaluate` over 32 deals in four legs
  (128 games) of the selected matchup, in its own process, in the order padded,
  layout, padded, layout. "Forward" is the serving child's `forward_host` plus
  `output_transfer_host` (the wait for CUDA outputs) over `forward_calls`.
- The CPU measurement: `uv run python -m experiments.eval_forward_layout cpu
  experiments/data/live_token_attention/inputs experiments/data/eval_forward_layout`,
  one warm-up pass and three timed passes per layout, torch 2.15.0, 16 CPUs.
- Figures: `uv run --extra reports python -m experiments.eval_forward_layout
  charts experiments/data/eval_forward_layout`.
- Evidence under `experiments/data/eval_forward_layout/` (manifest entries in
  `experiments/data.manifest.json`): `summary.json` (pass receipts and serving
  measurements), `job.json`, `gpu-samples.csv`, `cpu.json` and the two figures.

</details>
