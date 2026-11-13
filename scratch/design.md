# PPO CUDA update throughput — MTG-338

Implementation pass 11 recovered all eight candidate/harness paths byte for byte
from `02c202956`, preserving the report, measurement table and capacity evidence.
No availability read, launch or measurement occurred.

Focused check: `uv run --locked --extra dev pytest tests/model/test_semantic_cards.py -k definition_lookup -q`
→ 8 passed, 8 CUDA skips, 2 deselected (2.44 s). Runtime/tests/launcher and speed
harness Ruff checks passed; all eight files match and compile, and both generated
shell commands pass `bash -n`. The frozen equivalence scripts retain their
previously recorded import-style diagnostics to preserve exact bytes.

Remaining work: whole-update CPU equivalence, CUDA numerical audits and paired
L4 timing in measurement; broader acceptance at gate. Recovery establishes no
speedup or CUDA equivalence. Latest capacity evidence remains measurement pass
10 at 2026-11-13 11:43:03 UTC; its keep-rule restoration has now been reversed
for this implementation handoff.

## Decision pass 10: iterate on the unmeasured lookup candidate

The preceding pass intended to recover the frozen candidate and obtain CPU/CUDA
numerical checks and paired L4 timings. Implementation recovered all eight files
exactly, with eight CPU lookup passes and eight CUDA skips. Measurement pass 10
added one availability observation at 2026-11-13 11:43:03 UTC: “none available”,
“0 of 1 in use”, “not rentable now”. No launch occurred; the keep rule restored
the eight paths to the baseline.

All eleven rows of `scratch/measure.html` contain one trusted baseline and ten
unavailable comparisons: three capacity refusals and seven negative availability
reads. This is new operational evidence, not a performance rejection or a failed
workload. Whole-update CPU equivalence, CUDA audits and paired timing remain
missing. Repeated recovery establishes neither speedup nor CUDA equivalence.

Choose **iterate**: recover the same eight paths from `02c202956` in implement.
Rereading the current source confirms unconditional indexing in
`SemanticCardEncoder.embed_definitions`, reached by fact sources through
`RecordEmbeddings.sources` and by continuation relation definitions in
`manabot/backends/torch/policy.py` (also used by name-candidate definitions).
The frozen CUDA-autograd embedding substitution remains the cheapest prepared
change reaching the 56.3% profile attribution. Its prediction remains at least
20% lower median update time, with all numerical bounds below unchanged.

No capacity read or launch occurs in planning. Measurement gets one fresh
availability read and, only if positive, at most one named-cohort L4 launch
under $4. No queue, polling, alternative GPU or within-step retry. The baseline
remains trustworthy, a credible optimization remains untested, and no human
design decision is missing. Neither target attainment nor a blocked verdict
is supported by this pass.

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

Measurement pass 2 verified all eight recovered files against `02c202956` and
requested that unchanged freeze after a positive availability read. Admission
`db4502e2006a` nevertheless failed for no capacity at 2026-11-13 11:10:12 UTC;
requested run `4bacd2187ed1` was never recorded and no command started. The
measure keep rule restored the candidate/harness to `734f2cd5c` again.

Implementation pass 3 recovered all eight named files verbatim from `02c202956`.
That implementation tree contained the candidate and paired harness; measurement
pass 3 subsequently restored the kept runtime. All admission records remain retained. No availability read or launch occurred.
Focused check: `uv run --locked --extra dev pytest tests/model/test_semantic_cards.py -k definition_lookup -q`
→ 8 passed, 8 CUDA skips, 2 deselected (5.98 s). Ruff, compilation and both
generated commands' shell syntax checks passed. Remaining acceptance is the
rental checks and paired measurement below; recovery alone admits no gain.

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

The latest availability read, **2026-11-13 11:43:03 UTC**, returned “none
available”, “0 of 1 in use”, “not rentable now”; no launch followed.
Implementation recovery is reversible; measurement still needs successful L4
admission. In the next measurement step, make one fresh
availability read before at most one launch under **`--max-usd 4`** with an
explicit named cohort, injecting Doppler `manabot/prd` (R2) and `etude/prd`
(RunPod) together. A positive read permits that single attempt; it does not
establish capacity or authorize retries. Do not substitute another GPU, queue
work, poll indefinitely or repeat failed launches automatically.
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
