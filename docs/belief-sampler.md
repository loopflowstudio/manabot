# Frozen-policy belief sampling

The sampler learns a joint distribution over opponent hand counts without
listing every compatible hand. It uses managym's viewer-safe pool, public known
hand minima and hand size, shared with the exact possible-world provider. The
complement is the opponent library multiset; library order and other hidden
state are outside this domain.

```bash
uv run manabot train --regime experiments/regimes/belief-sampler.json --seed 421 --out .runs/belief-pilot
```

This bounded three-game, small-deck pilot demonstrates the dependency chain.
It does not score the Allies/Lessons challenger or modify the ETU-91 campaign.
Use a fresh output directory for every attempt. No additional paid compute or
scientific allocation follows from this recipe.

## Stages and artifacts

`collect_belief` references an earlier policy stage and explicitly chooses `raw`
or `ema`. It reloads the admitted checkpoint and freezes those bytes for both
seats throughout collection. `train_belief` references its immutable dataset;
only the sampler receives gradients. A policy can learn useful behavior before
this stage, with no circular policy/belief dependency.

Supported behavior exports are observation-only flat policies (PPO or Ataraxos,
raw or explicitly exported EMA) and compound policies (`train_compound`, raw
only). Compound recipes may append `collect_belief` and `train_belief`; they may
not mix flat policy training stages into the run. Belief-conditioned behavior
policies remain rejected. The collector owns game boundaries and resets queued
compound Commands after each reset; native lowering owns their legality.

Collection uses the checkpoint's saved observation capacity for both policy
kinds and fails explicitly if a game exceeds it. The compound player's wider
native offer support does not remove this collector limit. Pin sufficient
capacity before training and collection; do not reinterpret checkpoint bindings.
The collector owns private labels and never passes them to the behavior player.

The dataset retains complete games with disjoint train, validation and test
membership, alternating deck assignments, policy and world identities, public
input records and private supervision labels. Capture occurs at the current
revision before acting. Later reveals never rewrite earlier inputs. These
artifacts contain actual hidden hands and belong to private training/audit
storage, not viewer-facing Study or play responses.

The existing VerifyStore/TrainingRun records retain dependencies, hashes,
configuration, seed streams, failures and measured costs. Sampler checkpoints
have their own schema/world/dataset/policy admission; they are not ordinary
playable policy checkpoints. Last-complete-raw selection continues to select
the policy. Neither checkpoint type promises process resume.

## Estimator

The autoregressive decoder samples one definition count at a time in the bound
vocabulary order. Each conditional mask enforces public minima, remaining
slots and remaining pool capacity. Teacher-forced training minimizes the sum
of conditional negative log probabilities of the actual joint hand. Hidden
truth is a target only; it never enters inference. This retains count
correlations that independent marginals cannot represent.

The safe reference is the physical compatible-deal prior: known cards are
removed, unknown slots are dealt without replacement, then known cards are
restored. It is not uniform over distinct count vectors. The learned decoder
adds trainable corrections to conditional physical-deal log odds. Public history
features currently summarize typed commitments; this does not model arbitrary
ordered history or establish sufficient statistics for a true posterior.
History dropout is a configurable training treatment, not immunity to baiting.

## Evidence and scientific boundary

The instrument reports joint log loss, sampled marginal/query calibration,
support violations and sampling cost. Exact reference comparison consumes the
existing tracker output on tractable supports; no second Bayesian updater is
introduced. Self-play holdouts test the frozen training population. Foreign or
adversarial histories must be labeled separately, retain the generating policy
assumption and preserve the physical baseline.

Before expensive scoring, freeze a separate protocol with independent sampler
seeds, exact frozen policy/world identities, whole-game partitions, held-out
foreign/adversarial histories, dropout arms, total cost cap and untouched
complete-game evaluation deals. Compare exact, physical-prior and learned
search at equal elapsed inference time, including sampling. Calibration alone
cannot establish search strength. The post-RL search consumer is now connected through
`collect_local_update.sampler` and direct native count materialization; see
[local-search contracts](local-policy-search.md#learned-joint-hand-search).
It preserves separate sampler/policy admission and labels learned beliefs as
approximate. This does not replace the exact reference or establish calibration.

Larger pools need a newly pinned world and measured complete-loop cost. The
INT-17 failure motivates avoiding enumeration, but neither a large synthetic
vocabulary nor a fast sampler is evidence of wider-world strategic strength.
The proposed scientific allocations in
[ETU-99 transfer protocols](../experiments/ataraxos-transfer.md) remain
proposals. Full evaluation, opponent transfer and challenger strength remain
unmeasured until those separately authorized cohorts complete.

## Bounded implementation proof — 2026-10-04

The documented pilot completed locally in 17.41 seconds: three complete games,
303 environment decisions, both deck assignments, and four belief optimizer
steps (32 training-row exposures). The private dataset, policy and sampler
remain in `.runs/etu96-belief-pilot`, Run
`83cc540334c2426597cb36343836e7c0`, with source digest and exact artifact hashes
in `run.json`/VerifyStore. Its 840 held-out learned samples had zero support
violations. This tiny run is pipeline evidence only; it does not establish
calibration improvement, opponent transfer or playing strength.

Focused checks exercise joint normalization and learning, physical-prior
agreement, held-out label isolation, synthetic adversarial history, artifact
rejection, real frozen-policy collection and regime execution. Native debug
checks cover the shared constraint projection and existing exact/prepared
possible-world contracts. No scientific cohort ran in this implementation pass.

## Offline saved-game quality report

No training or game generation is needed. Use the immutable dataset and sampler
from a completed run, and the sampler SHA-256 recorded in its artifact receipt:

```bash
uv run python -m manabot.belief.sampling_report \
  --dataset /path/to/evaluation-dataset.json \
  --training-dataset /path/to/sampler-training-dataset.json \
  --checkpoint /path/to/sampler.pt \
  --checkpoint-sha256 "$SAMPLER_SHA256" \
  --samples 16 64 256 --seed 421 \
  --out .runs/sampler-report.json
```

For a physical-prior-only report, omit all three learned-artifact arguments.
Evaluation and training dataset paths may be identical. A different evaluation
dataset must retain its own generating-policy identity and match the schema and
world; overlap with sampler fitting game IDs is rejected. Files retain original
whole-game splits, and reports list their game IDs without copying private hands.
Train-split scores are descriptive in-sample evidence. Foreign-policy identity
means a distribution shift, not successful transfer; adversarial selection cannot
be inferred from identity alone and must be documented in the cohort protocol.
Malformed labels/constraints, altered digests and incompatible bindings fail
explicitly instead of being silently excluded. Output paths must be new.

`report_saved_sampler` returns a typed `SamplerQualityReport`. For each split and
sample count it reports observed-label joint NLL (nats), card-presence Brier and
ten-bin ECE, adjacent vocabulary-pair conjunction Brier, violating sampled hands,
sample totals, sampling seconds, peak Python allocation bytes and largest output
tensor bytes. Support violations count draws violating any pool capacity, known
minimum or total hand-size constraint. These constraints are the saved managym
projection; the report does not reconstruct a hidden-world authority. Calibration
pools decision/card forecasts, so games with more decisions carry more weight;
these aggregate fields are descriptive means. The whole-game evidence and
separate cohort intervals below retain their different weighting.
The conjunction panel is O(vocabulary size), not all possible queries. Arbitrary
typed queries, exact-posterior error, native peak memory and strength acceptance
are explicitly unavailable. Joint NLL is an observed-label proper score, not KL
to an enumerated posterior. No hands are enumerated at any vocabulary size.

Every sample count restarts the declared RNG seed. Counts expose Monte Carlo
sensitivity; they are not independent cohorts or necessarily nested draws.
Statistics reproduce under the same artifacts, evaluator-source digest and
Python/Torch runtime. Timing and allocation peaks are fresh measurements and
will differ between reports. Execution is serial, deterministic CPU with one
Torch thread; sampling time excludes artifact loading and label scoring. Python
allocation tracing includes metric work and excludes native tensor workspace;
parameter bytes and returned count-tensor bytes are separate. This is not process
peak RSS or an uncontended hardware benchmark.

Deterministic non-optimized fixture checkpoints exercise ordinary reload, foreign
cohorts, CLI regeneration and a 40-definition, 80-copy count domain in
`tests/belief/test_sampling_report.py`. They prove software functionality only.
ETU-99 owns empirical calibration, dropout/foreign-policy comparisons and broader
acceptance. ETU-91's frozen campaign and allocation remain unchanged.

## Whole-game evidence and independent-fit uncertainty

The v2 saved report retains each game's ID, deal seed, assignment, decision
count, learned/prior metrics and ten calibration-bin counts, probability sums
and truth sums. Both viewers stay in that game. Repeated `(viewer, revision)`
rows fail rather than overweight a duplicated view. Different revisions remain
correlated observations in one cluster. Aggregate fields remain decision-weighted;
pooled ECE is reconstructed from bin totals, not averaged from game ECE.
Sampling streams now derive from `(sampling seed, game ID)`, so reordering other
games cannot perturb a game's draws. This is a versioned change from v1's
split-wide stream; historical outputs are not rewritten.

Generate one saved report per admitted sampler using the command above. Then
create a `CohortSpec` JSON manifest and run the purely offline analysis:

```bash
uv run python -m manabot.belief.sampling_cohort \
  --manifest /path/to/cohort/manifest.json \
  --out .runs/sampler-cohort.json
```

The command writes JSON and a Markdown companion, refuses overwrites and resolves
input report paths relative to the manifest. The [synthetic example generator](../experiments/runners/sampler_uncertainty_example.py)
writes a complete manifest and hashed input reports; its [report](../experiments/data/sampler-uncertainty-synthetic/analysis.md)
is **fabricated software evidence**, not measured calibration. Regenerate it in a
fresh directory without training, games or hand sampling:

```bash
uv run python experiments/runners/sampler_uncertainty_example.py \
  --out .runs/sampler-uncertainty-synthetic
```

Each attempt declares its treatment, independent training seed, fit receipt and
configuration identities, immutable training dataset, status, and either a hashed
saved report or a failure/missing reason. Copy provenance from the original
TrainingRun/artifact receipts; names or seed integers alone cannot establish
independence. The analyzer verifies the training seed retained in admitted
checkpoint metadata, report bytes, producer/evaluation policy, datasets,
world/schema, evaluator source, runtime, sampling seed and sample count. Historical
checkpoints without fit-seed metadata still support descriptive saved reporting,
but cannot enter independent-fit analysis. Receipt/configuration identities are
explicit declarations, not automatic validation of an external TrainingRun store.
Do not label a different architecture or optimizer setting as another seed of one
treatment.

The implemented estimands and limits are explicit:

- NLL is the mean decision log loss within a game (nats), then the mean over
  games, then the mean over fits. Presence/conjunction Brier follow the same
  hierarchy. ECE here is **mean game ECE**, distinct from pooled forecast ECE
  retained in the original report. Legality reports game-mean violation rate;
  original game evidence retains violating counts and total draws.
- Learned, physical-prior and learned-minus-prior panels use equal game weights,
  regardless of game length. Paired treatment contrasts subtract at the same
  fit seed and exact same games before aggregation. Unequal decision counts are
  allowed; unequal game membership/counts across reports are rejected. There
  is no silent intersection, imputation or length weighting.
- A 95% percentile fit bootstrap resamples independent fits with replacement,
  keeping all saved games fixed. A separate 95% game bootstrap resamples whole
  games with replacement using the same indexes across fits and contrast arms,
  keeping fitted models fixed. All decisions and both viewers stay together.
  These are separate conditional intervals, **not a combined uncertainty
  interval**. Very small fit cohorts give coarse, exploratory intervals.
- One fit has no fit-seed interval; one game has no game interval. Repeated
  sampling seeds, sample-count variants and repeated checkpoint bytes cannot
  become independent fits. Fixed-panel sampling randomness is conditioned on,
  not integrated out. All-zero observed violations yield a zero empirical
  bootstrap interval; that is not an upper bound on unseen violations.
- The cohort is conditional on **one frozen producer and training dataset**,
  and one evaluation dataset/population. Cross-producer variance, unequal-data
  comparisons, checkpoint selection and a combined producer/fit analysis remain
  unsupported. Foreign evaluation retains a separate generating-policy identity;
  it is not an independent producer fit. Run separate panels for other datasets,
  sample counts or sampling seeds and do not pool them as independent models.
- Every declared failed/missing fit remains in the output. An incomplete cohort
  retains available per-fit descriptive means but suppresses all cohort means
  and intervals, including contrasts. A declared complete report that is absent,
  malformed or mismatched raises an error; amend its status explicitly, never
  silently drop it. Unsupported metrics stay unavailable, never zero.
- Train-split analysis remains in-sample. No arbitrary queries, posterior KL,
  game generation or hidden-truth inference is added. NLL uses retrospective
  labels only. No calibration, transfer or strength acceptance follows.

The output binds the manifest, all saved-report hashes, evaluator and analysis
source identities, bootstrap seed/count, analysis Python/NumPy and saved evaluation runtimes.
Statistical regeneration reads these frozen reports and does no sampling; measured
sampling time and Python allocations remain in the hashed inputs. New sampling
reports measure fresh costs and must not be substituted as if only their intervals
had changed. The synthetic and fixed-weight saved-artifact checks in
`tests/belief/test_sampling_cohort.py` and `test_sampling_report.py` exercise this
software contract. ETU-99 retains empirical cohorts and budget authorization.
