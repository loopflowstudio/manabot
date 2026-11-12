# Live-token attention against bucketed shapes on an L4 — MTG-319

**Committed before any scored run (2026-11-12).** Question: does attending over
valid objects only (`Execution.live_tokens`, packed jagged attention) beat
bucketed shapes enough to build out as the layout for larger models? Prediction:
at w384l8, with bucketed shapes on in both arms, live tokens take the learner
step to about 1/1.3 of bucketed's time (bucket padding is roughly half the
object slots, and attention blocks are most but not all of a step), so the
pre-registered bar is **not** met. The result that refutes the prediction and
passes the bar: `bucketed-live` learner step ≤ 1/1.5 of `bucketed` at w384l8,
with logits, values, losses and gradients equal to the padded path within float
tolerance. Cost cap: USD 3 for the L4 job (Task budget USD 10).

## Method

`experiments/live_token_attention.py` launches MTG-301's profile
(`experiments/l4_step_profile.py`, candidate `live`) on one rented L4: one
in-situ w160l8 update of the frozen MTG-267 regime keeps collection observations
and learner minibatches; they replay through `baseline`, `bucketed`, `live` and
`bucketed-live` in separate processes at w160l8 and w384l8, interleaved with
baseline repeats. Each arm writes fixed-weight numerics compared with baseline.

The layout is described in
[live-token attention](../docs/observation-and-model.md#live-token-attention-profiling).
On CPU, `tests/model/test_live_tokens.py` checks logits, values and gradients
against the padded path within 1e-5, alone and over buckets, with masked-mean
and value-token critics, and that collection packs with its host-measured count.

Attention runs as `scaled_dot_product_attention` over jagged nested tensors,
which dispatches to fused variable-length kernels on CUDA; the tensors are built
with host `min_seqlen`/`max_seqlen` bounds so dispatch reads nothing back from
the device. `flex_attention` (block masks, compiled) was not measured. Learning
reads the packed-token count back once per forward (`nonzero`); collection
measures it on the host while preparing the batch and reads nothing.

## Results

**Answer.** Yes. At w384l8 on an L4, live tokens over bucketed shapes take the
median learner step from 386 ms to 202 ms, 1.91× faster, which clears the 1.5×
bar. The prediction (about 1.3×) is refuted. Outputs match the padded path as
closely as bucketed shapes already do: losses within 9e-8, learner logits
within 4e-8, values within 6e-7 and gradients within 2.5e-5 (cosine ≥ 0.999999999). Live tokens should be built
out as the layout for larger models.

### Learner step and collection forward

The same retained minibatches and collection observations were run through each
layout, two passes each. A baseline repeat at w160l8 lands within 0.5% of the
first baseline run, so differences of a few percent are noise.

![Median learner step per layout at w160l8 and w384l8, with the bucketed ÷ 1.5
bar](data/live_token_attention/learner_step.svg)

| Layout, w384l8 (31.5M parameters) | Learner step (median) | Learner pass | Collection pass | Learner peak memory |
|---|---:|---:|---:|---:|
| Padded (default) | 882 ms | 18.0 s | 20.4 s | 12.8 GB |
| Bucketed shapes | 386 ms | 7.7 s | 8.0 s | 5.4 GB |
| Live tokens, full tables | 254 ms | 5.2 s | 4.8 s | 3.1 GB |
| **Live tokens over buckets** | **202 ms** | **4.1 s** | **4.6 s** | **3.0 GB** |

| Layout, w160l8 (5.8M parameters) | Learner step (median) | Learner pass | Collection pass | Learner peak memory |
|---|---:|---:|---:|---:|
| Padded (default) | 387 ms | 7.8 s | 9.2 s | 5.3 GB |
| Bucketed shapes | 165 ms | 3.3 s | 4.0 s | 2.2 GB |
| Live tokens, full tables | 143 ms | 2.9 s | 2.4 s | 1.2 GB |
| **Live tokens over buckets** | **114 ms** | **2.3 s** | **2.9 s** | **1.1 GB** |

Against bucketed shapes, live tokens over buckets are 1.91× faster per learner
step at w384l8 and 1.45× at w160l8, so the gain grows with model width, as it
should when attention dominates. Collection passes are 1.75× and 1.37× faster.
Against the padded default the combined layout is 4.4× (w384l8) and 3.4×
(w160l8) per learner step, with a quarter of the learner memory. Kernel launches
per learner step barely change (7,241 against 7,363 for bucketed shapes at
w384l8), so the saving is less work per kernel, not fewer kernels.

At w160l8, live tokens over the full tables give a faster collection pass than
over buckets (2.4 s against 2.9 s), although the learner step is slower. Bucket
trimming costs something on small collection calls that live packing already
covers. This pilot did not separate that cost.

### Numerics

At fixed weights (w160l8 in-situ initialization, CUDA, four learner minibatches
and eight collection observations), each layout was compared with the padded
path:

![Largest absolute deviation from the padded path per metric and layout, log
scale](data/live_token_attention/equivalence.svg)

| Layout | Collection logits | Values | Losses | Gradients (max abs) | Gradient cosine |
|---|---:|---:|---:|---:|---:|
| Bucketed shapes | 8e-9 | 3e-7 | 7e-8 | 2.5e-5 | ≥ 0.9999999989 |
| Live tokens, full tables | 7e-9 | 5e-7 | 2e-8 | 8.9e-6 | ≥ 0.9999999994 |
| Live tokens over buckets | 7e-9 | 6e-7 | 9e-8 | 2.5e-5 | ≥ 0.9999999989 |

The gradient differences are atomic-accumulation order on CUDA, the same size
bucketed shapes already show, and live tokens add none of their own. As in
MTG-301, `compare`'s CPU-tuned gradient tolerance reports `ok: false` for every
CUDA layout, bucketed shapes included. Numerics were taken at w160l8 only.

### Cost

USD 0.24 (launcher), one L4 for 17 minutes of job time, under the USD 3 cap.

### Limits

- One rented L4, one in-situ update, two replay passes per arm. Timings are of
  retained inputs replayed in separate processes, not of a whole training run.
- Packing reads the live-token count on the host once per learning forward.
  It is included in the learner step above. Collection takes the count from
  host preparation.
- The CPU path is correct but slow (per-observation attention), so the layout
  is for CUDA.

## Decision

Build out live tokens. The pilot passes its pre-registered rule: the
bucketed-live learner step is 0.52 of bucketed's at w384l8, below the 0.67 bar,
with outputs equal within float tolerance.

## What a default for larger models would take

Recommended as its own Task, not done here. Today live tokens run only where a
training stage's collection and learner policies set `Execution.live_tokens`;
evaluation and search build their policies without it and still attend over
padded tables. Making it the default for larger models needs:

- **A device rule.** On CPU, jagged attention falls back to per-observation math
  attention: correct, but slow. Either the default applies to CUDA only, or a
  CPU stage that asks for it is refused rather than silently slowed.
- **Adoption by evaluation and search**, so the layout a model trains with is
  the one it is evaluated and searched with, and their cost falls with it.
- **The w160l8 collection question.** At w160l8, collection over full tables was
  faster than over buckets (2.4 s against 2.9 s per pass), while the learner
  step favoured buckets. Whether a default should skip bucket trimming for
  collection at small widths needs a measurement that separates the trimming
  cost; this pilot did not.

It composes with bucketed shapes, so the default can be both together.

## What MTG-306's pooled stream becomes

Recommendation: strip `token_stream="pooled"`, in its own Task. Its only benefit
was attending over fewer tokens; live tokens deliver that per observation
without padding every observation to a common width, and the two are mutually
exclusive (a configuration that asks for both is rejected). Keeping it would
mean maintaining a second reduced-attention layout that the faster one
supersedes.

<details><summary>Appendix: run and evidence</summary>

- Etude run `308602b435fe` (`mtg319-live-token-attention-live`), source
  `53cf1357d` on this branch (main `4c9ba1957` plus this change), machine
  `mtg319-l4-161dd1d181d2`: NVIDIA L4, AMD EPYC 9254 (48 CPUs), torch
  2.15.0+cu132, CUDA 13.2. Launched 16:45 UTC on 2026-11-12, succeeded, job
  1,037 s. The machine was terminated by hand after the run ended, since its
  window ran to 19:01 UTC.
- Dated line, 2026-11-12: the planned `l4-etude` on-demand was held by another
  Task's run from 15:51 UTC, and RunPod had no L4 stock until 16:45. The job ran
  on `mtg319-l4`, an on-demand with `l4-etude`'s template (L4, 6 vCPUs, 48 GB,
  50 GB disk, same image, ≤ USD 0.60/hour). The question is unchanged.
- Medians are over the 40 timed learner steps (20 minibatches × 2 passes) in
  each arm's `receipt.json`. Passes are `summary.json`'s
  `learner_total_s_per_pass` / `inference_total_s_per_pass`, averaged over two
  passes.
- Figures: `uv run --extra reports python -m experiments.live_token_attention
  experiments/data/live_token_attention` writes `learner_step.svg` and
  `equivalence.svg` from `summary.json` and `replay/<arm>-<size>/receipt.json`
  (2.4 MB of the folder's 1.2 GB; fetch those paths with
  `uv run --extra artifacts python -m manabot.data fetch`). The SVGs are
  byte-stable, so redrawing them leaves the manifest unchanged.
- Evidence is under `experiments/data/live_token_attention/` (manifest entries
  in `experiments/data.manifest.json`): `summary.json`, `job.json`,
  `gpu-samples.csv`, `insitu/`, `replay/<arm>-<size>/`, `numerics/`, `logs/`,
  `inputs/`.

</details>
