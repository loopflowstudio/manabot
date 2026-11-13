# PPO CUDA update throughput — MTG-338

## Decision pass 1: iterate on the unmeasured lookup candidate

The preceding pass intended to establish CPU equivalence, bound CUDA rounding
and measure a 20% reduction in update time. It prepared the candidate and paired
harness at `02c202956`, but authenticated admission `947aca1850c9` failed before
execution. There are no paired samples or CUDA numerical results. The measure
keep rule restored runtime and candidate tests/harness to `734f2cd5c`.

Both rows of `scratch/measure.html` are inconclusive about improvement: the first
is a trusted baseline, the second an unavailable comparison. No optimization
has been rejected on performance or behavior. The evidence gap does not justify
changing mechanisms or declaring the target met. Operational progress is the
recovered comparison and the resolved credential injection; speedup is unproven.

## Hypothesis and boundary

Repeated fact-source and continuation IDs accumulate large gradient tensors
into a small definition table through advanced indexing. Replace only
`SemanticCardEncoder.embed_definitions` with functional embedding during CUDA
autograd, so the specialized repeated-ID reduction can make that work cheaper.
Predict **at least 20% less median update time**, from the prior **60.521 s**
to **48.417 s or less**, equivalent to **at least 0.33046 optimizer steps/s**
at the same 16 steps/update. The paired baseline, not the historical rate,
determines the relative gain.

The trusted baseline is 0.26437 steps/s (five repeats, 0.26366–0.26474).
Backward/clipping takes 45.975 s/update. The separate first-step profile
attributes 1.126 s to fact-source lookup and 0.542 s to continuation definitions:
56.3% of 2.964 s summed GPU kernel time. Source inspection confirms both still
call this method (`facts.py` via `RecordEmbeddings.sources`, and
`backends/torch/policy.py` via continuation encoding). This is a credible target,
not a whole-update speedup estimate.

Compute `self._rows(ids)` once. Use `F.embedding(rows, definitions)` only when
the table is CUDA, autograd is enabled and the table requires gradients. Keep
indexing otherwise. Preserve the differentiable table, row-zero gradients,
repeated-ID sums, dtype, shape and upstream program gradients. Use embedding's
defaults: no padding suppression, frequency scaling, norm constraint or sparse
gradients. No other gather sites, optimizer settings, schemas or dependencies
change in this candidate.

## Delete — do not maintain

Replace unconditional indexing inside `SemanticCardEncoder.embed_definitions`
for CUDA autograd only. Preserve indexing for CPU, MPS and inference, the shared
row mapping, differentiable program table and all checkpoint/data contracts.
No files, schemas or exclusive tests are deletion targets. The kept lookup in
`scratch/speed/baseline_lookup.py` is solely the frozen experimental comparator;
it is not a second production implementation.

## Implementation handoff

Recover the existing candidate rather than design a second optimization.
`02c202956` holds `manabot/model/semantic_cards.py`, its focused tests in
`tests/model/test_semantic_cards.py`, the paired `experiments/ppo_cuda.py`,
`scratch/speed/{baseline_lookup,paired}.py` and
`scratch/equivalence/{run,updates,compare}.py`. Restore those specific paths;
retain the current report, measurement table and failed-admission evidence.
The full earlier design is also recoverable from that commit. Do not infer
CUDA correctness from the prior eight CPU passes and eight CUDA skips.

Implementation pass 2 recovered all eight named files verbatim from `02c202956`.
The report and failed-admission evidence are retained. Remaining acceptance is
the rental checks and paired measurement below; recovery alone admits no gain.
Focused check: `uv run --locked --extra dev pytest tests/model/test_semantic_cards.py -k definition_lookup -q`
→ 8 passed, 8 CUDA skips, 2 deselected. Ruff, compilation and generated shell
syntax checks passed; full CPU/CUDA equivalence and performance remain unrun.

Measurement pass 2 verified all eight recovered files against `02c202956` and
requested that unchanged freeze after a positive availability read. Admission
`db4502e2006a` nevertheless failed for no capacity at 2026-11-13 11:10:12 UTC;
requested run `4bacd2187ed1` was never recorded and no command started. The
measure keep rule restored the candidate/harness to `734f2cd5c` again. Current
tree state is the kept implementation; the recovery instructions above remain
a future handoff conditional on successful L4 admission, not an active change.

Use the frozen numerical bounds in the [report](../experiments/ppo_cuda.md):
CPU outputs/gradients/trained weights bitwise equal; identical initial CUDA
outputs and first loss; raw gradient element error at most
`1e-6 + 1e-4 * max(abs(reference parameter gradient))`; first/final weight and
final probability/value absolute error at most `1e-4`. Check gradient support,
counts, float64 repeated sums and same-arm drift. These are one-update bounds,
not long-run learning equivalence; reject unexplained differences.

Run the recovered CPU harness and relevant model/training checks on the rental.
Then use one captured Elemental w64l2 batch, one locked build, equal warmups and
five alternating samples per arm with fresh Adam and initial weights each time.
Preserve 53,341 learner rows, 26,671 retained rows, 106,684 exposures and 16
steps/update, production threads/cuDNN settings, TF32, bucketed shapes and live
tokens. Retain raw numerical audits outside timing and profile separately.
Keep only a gain above measured spread with behavior accounted for; distinguish
a useful smaller gain from meeting the 20% prediction.

## Execution prerequisite and alternatives

A read-only availability check at **2026-11-13 11:02:39 UTC** again returned
“none available”, “0 of 1 in use”, “not rentable now”. This planning pass launched
nothing. Implementation recovery is reversible; measurement still needs an L4.
Recheck availability before a launch under **`--max-usd 4`**, injecting Doppler
`manabot/prd` (R2) and `etude/prd` (RunPod) together. Do not substitute another
GPU, queue work, poll indefinitely or repeat failed launches automatically.
If capacity remains unavailable at measurement, retain the evidence gap and
report the external prerequisite; no additional runtime candidate is justified
by that failure alone.

Caching across updates would use stale trainable definitions; smaller
minibatches change the learning schedule. Adam is only 0.029 s/update and
attention backward only 0.044 s in the profile. Padding compaction needs an
occupancy measurement and a wider shape/packing change. These are deferred
priorities, not measured rejections. The cheapest remaining test is this
prepared lookup substitution. Extend it to other measured definition joins
only if it wins; if sorting costs erase the gain, measure live/zero-ID occupancy
before trying compaction or a custom reduction.
