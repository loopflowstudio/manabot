# Disambiguate structural encoder optimization, capacity, and CPU cost

## Problem

W2-214 established a valid static semantic instrument and an exact negative
control, but it did not produce an admissible structural encoder. On the
content-addressed `structural-semantic-katas-v1` suite, `bag_v1` remained at
exactly 50% for every family and seed while
`relational_semantic_encoder_v1` reached only 82.1% mean test accuracy. The
structural arm solved order and hierarchy consistently, but seed-dependent
families locked at either chance or near-perfect accuracy. Four of five runs
selected their final 800-step checkpoint. Its worst end-to-end p95 latency was
9.657x the bag control and its worst batch-128 throughput ratio was 0.026x.

Those observations leave three materially different explanations:

1. the fixed architecture can represent the signal but the 800-step training
   recipe stopped too early;
2. the one-block encoder lacks a usable structural path or sufficient
   admissible capacity; or
3. accuracy is recoverable, but Python tensor materialization, dense model
   execution, or both make the encoder too expensive.

The Semantic Representation Katas Project needs this distinction before it
can spend W2-213 on held-out recombination. A false optimization diagnosis
would scale the wrong model; a false capacity diagnosis would add parameters
without evidence; a false projector diagnosis would optimize Python around a
model whose dense attention remains intrinsically too slow.

This design is the pre-result contract. No new training or performance result
may be generated until the machine-readable form of this contract, the runner,
and their fail-closed tests are committed together. The result commit must be
later and mechanically trace its preregistration revision.

## The demo

A developer runs:

```bash
uv run experiments/runners/run_structural_encoder_discriminator.py \
  --contract experiments/workloads/structural-semantic-katas-v1-discriminator.json \
  --suite experiments/katas/structural-semantic-katas-v1.json \
  --preregistration-revision <contract-commit> \
  --out experiments/data/structural-semantic-katas-v1-discriminator.json \
  --report experiments/w2-266-structural-encoder-discriminator.md
```

The command verifies the preserved suite and authority hashes, executes only
the preregistered sequential arms, and prints exactly one terminal outcome:
`NOMINATE_FOR_W2_213 <encoder>` or `KILL_REDESIGN <surviving-hypothesis>`.
The report shows every seed and family, the step at which training fit was
first achieved, and an online-projector/cached-projector/model-only CPU cost
decomposition. The command does not start W2-213.

## Approach

### Preserve the W2-214 instrument

The discriminator consumes the checked-in suite; it never rebuilds or edits
it. The contract pins these existing authority receipts verbatim:

| Artifact | Required SHA-256 |
|---|---|
| suite | `5595ce579017c4ec84b8746cb30a3f4bb09a69e4801ccaf7748467f7bec2f948` |
| source | `1a61c10d15a0ed5aa2746f4bb4fb9773db841c0c6affffe8cad21fac32ef5f2d` |
| oracle | `87a3e918f513c4bd8bfc4d29634440d0c1c6526542078b88b3f1e50b4e165c11` |
| compiler | `f10f34d7b2e458a0ea01b124261a0c0fbbda4648e4452f277254aa6cc3250367` |
| learning schema | `a156c592414d0d4838c5423e2cb471fc49a4450f21d62e5e5c198ba948ae7034` |

The 800 authoritative typed-program labels, 400 pair assignments, 48/16/16
pair split per family, content hashes, nuisance audits, exact `bag_v1`
symmetry, and model seeds `21401` through `21405` are immutable. Definition
references and opaque identity features remain forbidden. Any mismatch ends
the run as `KILL_REDESIGN instrument_invalid`; it is not repaired in place.

`bag_v1` is replayed with its original initialization and 800-step recipe for
all five seeds. It must retain exact paired-prediction symmetry and exactly
50% train, validation, and test accuracy per family and seed. Its same-process
CPU measurements are the denominator for every cost ratio.

### Execute a sequential, not factorial, discriminator

The runner is a fail-closed state machine. It may not skip ahead, add a retry,
or inspect test results to choose a later arm.

#### Stage O: optimization only

Arm name: `relational_semantic_encoder_v1_opt4000`.

Keep `RelationalSemanticEncoder` byte-for-byte identical: one 24-wide,
two-head block, `d_ff=28`, the same token/kind embeddings, sinusoidal preorder
and depth features, eight relation biases, 24-wide context, probe head,
initialization offsets, and 8,838 trainable parameters. Keep AdamW at learning
rate 0.003, zero weight decay, deterministic family-balanced batches of 64,
single-thread CPU execution, validation every 20 steps, and lowest validation
NLL with earliest-step tie-break for the candidate checkpoint. Change exactly
one factor: extend the maximum from 800 to 4,000 optimizer steps.

At each validation interval, record full-training-split metrics in addition to
validation metrics. The optimization/capacity trigger uses the maximum full
training accuracy seen anywhere on the trajectory, not the validation-selected
checkpoint. This prevents checkpoint selection from hiding a valid
memorization witness. The original trainability and all nomination gates are
still evaluated on the validation-selected checkpoint.

If every seed reaches at least 99% full-training accuracy at some checkpoint,
the known training signal is fit and Stage C is forbidden. The selected
checkpoints continue to semantic, calibration, and cost adjudication. If any
seed never reaches 99%, Stage C is required.

#### Stage C: conditional minimal architecture/capacity repair

Arm name: `relational_message_encoder_v1`. This stage runs only after Stage O
fails the training-fit trigger.

The current one-block dataflow has a specific blind spot: relation biases
alter token-row attention, but the summary row attends to pre-block token
values. Because the runner reads only the summary row after that same block,
the output is mathematically independent of the relation tensor. Order, depth,
and field/role token identities can still solve some katas, explaining the
W2-214 pattern, but no amount of optimization can make the summary consume an
explicit argument link in that architecture.

Repair only that path. Before the existing attention block, compute one
directed relation message for each token from the same eight checked relation
matrices:

```text
base_i = token_i + token_kind_i + preorder_i + depth_i
message_i = mean over relation types present at i of
            (mean over linked j of base_j + learned_relation_embedding_r)
input_i = base_i + message_i
```

Empty relation rows contribute zero. Direction remains encoded by the existing
directed relation names. The only new parameters are eight 24-wide relation
embeddings: 192 parameters. The predicted total is 9,030, a 1.1% difference
from the 9,128-parameter bag control and therefore inside the original 5%
gate. No second block, wider hidden state, new relation, new token, new label,
or family-specific encoder path is allowed.

Train Stage C with the exact Stage O 4,000-step recipe and the same five
initialization identities. Thus the only changed factor between O and C is the
pre-attention one-hop relation message. If Stage C also cannot reach 99%
training accuracy in every seed, the bounded admissible architecture/capacity
hypothesis survives and the task ends in redesign rather than scaling again.

#### Stage P: semantics-identical projection and CPU attribution

Stage P always benchmarks `bag_v1` and Stage O; it also benchmarks Stage C if
that conditional arm ran. Model weights do not affect this stage's attribution,
so it remains informative even when no accuracy-eligible candidate exists.

Add a content-addressed `KataTensorCatalog` that parses and tensorizes the
static suite once, keyed by the preserved suite SHA-256. It caches token IDs,
token kinds, masks, depths, the same eight relation matrices, families, labels,
and stable record rows. Seed-dependent candidate ordering is still derived by
the existing pair-ID hash at selection time. The cache changes execution only;
it does not change a semantic input.

Before timing, assert tensor equality between online `records_to_batch` and
cached selection for every program and all five seeds, then assert exact logits
and metrics from the same model state. Any discrepancy is
`KILL_REDESIGN projection_semantics_changed`.

Measure three hot paths separately:

1. `online_projector_e2e`: current Python record traversal and dense tensor
   construction plus model;
2. `cached_projector_e2e`: catalog selection/collation plus model; and
3. `model_only`: an already materialized batch passed to the model.

Report the one-time cold cache-build time separately and never amortize it into
the hot gate. Use `model.eval()`, disabled gradients, one PyTorch thread, the
original 200 warmups/2,000 batch-one samples and 20 warmups/100 batch-128
samples, and the same record order for every arm. Record raw call durations so
p50 and p95 are computed directly. A secondary
[`torch.utils.benchmark.Timer`](https://docs.pytorch.org/docs/stable/benchmark_utils.html)
median/IQR audit may detect timer overhead, but it does not replace or alter the
primary gate. The official benchmark utility's warmup, replicate, and fixed
thread-pool behavior is the reason to use it only as an audit of the preserved
measurement.

The original numeric CPU gate applies to the semantics-identical cached hot
path relative to the cached `bag_v1` hot path: worst per-seed batch-one p95
ratio at most 2.5x and worst per-seed batch-128 throughput ratio at least 0.4x.
The legacy online ratios and model-only ratios are reported beside it. This
keeps the admission threshold unchanged while testing the deployment-relevant
fact that a static card catalog can be projected once rather than rebuilt at
every decision.

### Preserve every original admission gate

No W2-214 threshold may be weakened or reinterpreted after results:

| Gate | Required result |
|---|---:|
| bag symmetry and ceiling | exact pair symmetry and 50% per family/seed |
| selected train accuracy | at least 99% in every seed |
| selected validation accuracy | at least 95% in every seed |
| aggregate test accuracy | at least 95% mean over five seeds |
| per-family test accuracy | at least 90% mean for every family |
| per-family uplift over bag | at least 40 points mean for every family |
| aggregate uplift uncertainty | two-sided t95 lower bound strictly above 35 points |
| Brier score | at most 0.10 mean |
| NLL | at most 0.35 mean |
| parameter difference from bag | at most 5% per seed |
| cached end-to-end batch-one p95 | at most 2.5x bag, worst seed |
| cached end-to-end batch-128 throughput | at least 0.4x bag, worst seed |

ECE with five bins remains required reporting but is not promoted into a new
post-hoc gate. Training seed remains the experimental unit; t intervals remain
two-sided over exactly the five preserved seeds.

### Terminal decision table

| Observed path | Surviving diagnosis | Terminal decision |
|---|---|---|
| bag symmetry, authority hash, or cached equivalence fails | instrument invalid | `KILL_REDESIGN instrument_invalid` |
| O fits training and selected checkpoints pass semantic/calibration gates | original optimization horizon was insufficient | cost-adjudicate O; nominate only if CPU gate passes |
| O fits training but selected validation/test gates fail | representational generalization or checkpoint-selection defect, not insufficient raw capacity | `KILL_REDESIGN structural_generalization` |
| O cannot fit; C fits and passes semantic/calibration gates | missing one-hop structural capacity/path | cost-adjudicate C; nominate only if CPU gate passes |
| O cannot fit and C cannot fit | admissible structural capacity remains insufficient | `KILL_REDESIGN structural_capacity` |
| accuracy candidate passes model-only cost but fails cached end-to-end cost | Python/cache collation remains the cost cause | `KILL_REDESIGN projector_execution_cost` |
| accuracy candidate fails model-only cost | dense encoder execution is the cost cause, whether or not caching also helps | `KILL_REDESIGN model_execution_cost` |
| one accuracy candidate passes all preserved gates | no surviving admission failure | `NOMINATE_FOR_W2_213 <arm>` |

These reason codes refine rather than replace the original result branches.
`instrument_invalid` and `teacher_or_label_error` remain fail-closed authority
branches; Stage O resolves `optimization_or_capacity_unresolved`; a Stage C
recovery records `missing_structural_relation`; unrecovered accuracy or
generalization failures map to `encoder_redesign`; and projector/model CPU
failures map to `cost_redesign`. The JSON receipt records both the original
parent branch and the refined W2-266 reason.

A nomination names exactly one static encoder contract and only makes it
eligible for W2-213. It does not start W2-213, claim recombination or dynamic
binding, broaden semantics, alter the gameplay ABI, or authorize gameplay
integration.

## Pre-registered predictions

1. `bag_v1` remains exactly symmetric and at 50% for all splits, families,
   and seeds.
2. Stage O reaches 99% full-training accuracy in all five seeds by 4,000 steps
   and passes the original semantic and calibration gates. The W2-214
   end-of-budget checkpoint pattern makes optimization horizon the leading
   accuracy hypothesis.
3. If Stage C is triggered, its one-hop relation message reaches 99% training
   accuracy and passes the semantic gates without exceeding the parameter
   gate. A failure would be evidence that the allowed single-block boundary is
   too small, not license for another scale-up in this task.
4. Cached projection removes roughly 250-350 microseconds from structural
   batch-one end-to-end latency, but it does not recover the CPU gate. W2-214's
   model-only structural p95 was already about 5x the bag model and its
   model-only batch throughput was only about 3% of bag. The preregistered
   modal outcome is therefore `KILL_REDESIGN model_execution_cost`, even if
   optimization recovers accuracy.

Predictions do not affect branching or thresholds.

## Budget and stop rules

- Runtime: CPython 3.12 through `uv`, CPU only, one PyTorch thread, deterministic
  algorithms enabled.
- Seeds: exactly `21401`, `21402`, `21403`, `21404`, `21405`.
- Maximum trained arms: three (`bag_v1`, O, and conditional C).
- Maximum optimizer steps: 44,000 total: 4,000 bag steps, 20,000 O steps,
  and at most 20,000 C steps.
- Maximum presented examples: 2,816,000.
- Maximum total wall clock: 1,800 seconds, including performance measurement.
- Maximum retries or substituted seeds: zero. A crashed or non-deterministic
  run is discarded and the task returns to redesign; it is not silently rerun.
- Test metrics are computed only after an arm's five selected checkpoints are
  fixed. They never trigger Stage C or change a hyperparameter.
- Results that exceed any cap are reported as `KILL_REDESIGN budget_exceeded`;
  no partial nomination is possible.

## Required receipt

The machine-readable result and rendered report include, for every arm, seed,
split, and kata family:

- accuracy, Brier, NLL, ECE-5, example count, and candidate-order balance;
- maximum trajectory train accuracy, first 99%-fit step, selected checkpoint
  step, and selected train/validation/test metrics;
- optimizer steps, presented examples, training wall clock, and total wall
  clock;
- parameter count, parameter bytes, and component breakdown;
- p50/p95 batch-one latency for model-only, online end-to-end, and cached
  end-to-end execution;
- batch-128 examples/s and tokens/s for all three paths;
- cold cache-build latency and the online-minus-cached and cached-minus-model
  cost attribution;
- original gate booleans, all worst-seed ratios, suite/authority/contract/code
  hashes, environment versions, and the one surviving diagnosis.

The report repeats the static claim boundary and explicitly states whether
Stage C was triggered or skipped and why.

## De-risking

| Question | Finding | Impact on design |
|---|---|---|
| Is the W2-214 instrument itself suspect? | No checked split overlaps exist for normalized programs, nuisance signatures, or pair templates; authoritative oracles, labels, hashes, exact bag symmetry, and five seed receipts are already landed. | Reuse the suite byte-for-byte and fail closed on every authority hash rather than regenerate data. |
| Did the original run actually exhaust optimization? | Probably not. Four of five structural seeds selected step 800, the cap; the fifth selected step 780. Training and validation moved together rather than showing a generalization gap. | First change only the optimizer horizon from 800 to 4,000 steps. |
| Can validation checkpointing make a representable model look unable to fit? | Yes. Lowest validation NLL can select an earlier state even if a later state memorizes training. | Track the maximum full-training accuracy and first fit step separately; keep the original validation-selected checkpoint for admission. |
| Does the one-block summary actually receive explicit relation information? | No. `RelationalAttentionBlock` computes every value from pre-block inputs; relation bias changes token rows, while the summary row has zero relation edges. The model returns the summary immediately after that block, so its output is independent of `batch.relations`. | If O cannot fit, add one pre-attention relation message rather than a wider or deeper transformer. |
| Can the architecture repair remain matched capacity? | Yes. Eight 24-wide relation embeddings add 192 parameters, predicting 9,030 total versus bag's 9,128, a 1.1% difference. | Preserve the original 5% parameter gate without shrinking an unrelated component. |
| Is Python projection plausibly the whole CPU problem? | No. W2-214 structural end-to-end p50 was about 500 microseconds versus about 205 microseconds model-only, so projection is material; however model-only p95 and batch throughput already miss the gate by large margins. | Measure cached, online, and model-only paths independently. Expect caching to help but not absolve the model. |
| Is precomputation semantically legitimate? | Yes. These relations are a deterministic function of a versioned static typed-program catalog, not runtime state. W2-215 already treats semantic catalog binding as cold work and hot viewer projection separately. | Cache only suite/catalog structure, key it by content hash, retain seed-dependent candidate ordering at selection, and require tensor/logit equality. |
| How should microsecond CPU timings be made comparable? | PyTorch's benchmark guidance emphasizes warmups, replicates, and a fixed thread pool; inference-only execution also avoids autograd overhead. | Preserve the original gate sampler and single-thread setting, record raw replicates, and use `torch.utils.benchmark` only as a non-gating audit. |
| Could this result be mistaken for transfer or gameplay evidence? | Yes; the suite has shared family skeletons by design and no runtime objects or legal offers. | Repeat the W2-214 claim boundary and prohibit W2-213 startup or gameplay integration in every terminal branch. |

## Alternatives considered

| Approach | Tradeoff | Why not |
|---|---|---|
| Grid-search optimizer, learning rate, schedule, and restarts | Might find a strong seed quickly, but creates researcher degrees of freedom and no longer isolates one factor. | One fixed horizon extension gives a falsifiable optimization diagnosis across the same five seeds. |
| Add a second transformer block or widen `d_model` immediately | Makes relation-biased token updates reachable by the summary, but adds substantial parameters and dense quadratic CPU work. | It violates the spirit or letter of the original capacity/cost match and hides the smaller one-hop dataflow defect. |
| Recur the same block twice with tied weights | Preserves parameter count and enables two-hop flow, but nearly doubles the already inadmissible model execution. | The pre-attention relation message tests the same missing path with much lower expected cost. |
| Rewrite projection in Rust or vectorize all record traversal before training | Could improve an end-to-end number while the candidate still fails accuracy and model-only CPU gates. | Precompute static tensors now for attribution; defer a language rewrite unless projector cost alone survives. |
| Use `torch.compile` as the cost arm | May reduce Python dispatch but adds compile/cold-start and version-specific confounds, and it cannot explain projector versus model cost cleanly. | Keep eager execution and decompose the existing hot path first. |
| Relax the 2.5x/0.4x CPU gates for a more accurate encoder | Would guarantee an answer without establishing practical inference cost. | The task explicitly preserves the original gates; failure ends in redesign. |

## Wild success

The same one-block model simply needed a longer horizon, cached structural
catalog tensors remove most Python overhead, and the selected encoder clears
all five families in every seed while staying under the original CPU ratios.
The surprising win is not just a nomination: the receipt makes the next
experiment cheap because W2-213 inherits a content-addressed encoder contract,
a deterministic cache, and a family-by-family optimization trajectory rather
than another opaque architecture guess.

## Wild failure

The longer run memorizes order and hierarchy but still flips which binding
families it learns by seed; the conditional relation message fixes accuracy
only by making dense relation execution even slower. Six months later, the
team would regret treating tiny-kata accuracy as permission to put a quadratic
Python/PyTorch path into gameplay. This design prevents that outcome by making
model-only CPU cost a first-class diagnosis and by ending with an explicit kill
instead of weakening the gate or starting W2-213 anyway.

## Key decisions

- Optimization horizon is tested before architecture because W2-214 stopped at
  the boundary and the directive requires that ordering.
- Training fit and validation-selected admission are separate facts. Capacity
  is triggered by whether the architecture can ever fit training, while
  nomination still uses the original selected checkpoint.
- The conditional architecture repairs relation reachability with 192
  parameters rather than buying a second block.
- Static relation tensors are cached once per suite/content hash; runtime
  candidate ordering stays seed-derived and uncached.
- The candidate cached path may satisfy the unchanged CPU thresholds, but all
  online and model-only numbers remain visible so a cache cannot hide dense
  encoder cost.
- One surviving failure is terminal. No additional arm, seed, semantic family,
  gate amendment, W2-213 run, or gameplay integration is authorized here.

## Scope

- In scope: a committed machine-readable sequential contract; fail-closed
  provenance and branching; same-seed `bag_v1` replay; one same-architecture
  optimization arm; one conditional minimal relation-message arm; a
  semantics-identical cached projection path; full accuracy, calibration,
  parameter, CPU, throughput, and wall-clock receipts; one nomination or kill.
- Out of scope: changing any kata, label, split, oracle, compiler, learning
  schema, semantic opcode, runtime binding, legal command representation,
  gameplay ABI, policy model, teacher experiment, W2-213, gameplay integration,
  GPU measurement, Rust projection, or post-result gate change.

## Done when

The preregistration commit passes:

```bash
uv run pytest tests/semantic/test_structural_katas.py \
  tests/semantic/test_structural_encoder.py \
  tests/semantic/test_structural_discriminator.py
```

The result commit then reproduces from its pinned preregistration revision with
the demo command, emits the JSON receipt and timeless Markdown report, stays
inside the 1,800-second/44,000-step/2,816,000-example caps, and terminates in
exactly one nomination or kill branch. The existing semantic test suite remains
green:

```bash
uv run pytest tests/semantic
```

This advances the Wave measure that a matched-capacity structural encoder be
reported with multi-seed per-kata accuracy, calibration, parameters, latency,
throughput, and an explicit continuation decision. It does not claim the
separate Wave measures for recombination, dynamic binding, gameplay strength,
or semantic-conditioned integration.

## Measure

The before state is the checked W2-214 receipt: 82.1% mean structural test
accuracy; 70.0-90.0% selected train accuracy by seed; 8,838 parameters; about
200-249 microseconds model-only p95; about 509-643 microseconds online
end-to-end p95; about 1,187-1,254 online end-to-end batch-128 examples/s;
9.657x worst p95 and 0.026x worst throughput versus `bag_v1`.

The after state reports the same quantities, plus first training-fit step and
cached cost attribution, under the unchanged gates. “Better” is not a relative
improvement: it is all original semantic, calibration, parameter, latency, and
throughput gates passing in all required places. Anything else is a named
redesign result.
