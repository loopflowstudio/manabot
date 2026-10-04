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
cannot establish search strength. Coordinate this consumer with the post-RL
search Task; the sampler does not replace its planner or admission protocol.

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
