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
[training-regime follow-ups](../experiments/training-regime-followups.md) remain
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
these are descriptive means, without game- or training-seed confidence intervals.
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
