# Omitted Ataraxos controls (ETU-93)

2026-10-04. Jack Heart authorized reusable implementation, bounded real-game
proof and landing. Scientific allocation is **zero**. This protocol is separate
from ETU-91: its live checkout, frozen cohort, recipes and results are unchanged.
All techniques below remain **unresolved** scientifically. Existing ETU-91
results cannot be silently reused as independent training replicates.

## Executable contrasts

Generate a fully resolved, digest-bound workflow plan without executing it:

```bash
uv run -m experiments.runners.omitted_controls --contrast behavior-ema --out .runs/etu93-behavior-plan.json
uv run --extra notebook -m experiments.runners.run_training_regimes --study omitted-controls --plan .runs/etu93-behavior-plan.json --out .runs/etu93-behavior
uv run --extra notebook -m experiments.runners.run_training_regimes --report-only .runs/etu93-behavior
```

Every generated plan has one seed (693), two checkpoints, four streams,
64 learner transitions/stream/update, two updates/checkpoint, one CPU thread,
and a 600-second total deadline. Each training arm has a 150-second cap; the
remainder covers arena/replay/reporting. Failures consume the same allowance.
These numbers prove wiring, not learning. Do not scale this profile into a
scientific experiment without freezing a separately budgeted `ResolvedStudy`.
The generator refuses to overwrite a plan; the runner retains attempts in
VerifyStore and binds native runtime, source, world, setup, seed and artifact
identities. Arena uses the selected Allies/Lessons runtime, not an Interactive
mirror fingerprint. No second identity or session store is introduced.

| Contrast name | Control → treatment; other settings held fixed |
| --- | --- |
| `discount` | PPO gamma 1 → .99 |
| `policy-trace` | Policy lambda .95 → .99, value lambda stays .95 |
| `paper-estimators` | Scalar PPO policy/value lambdas .95/.95 → .5/.8; explicitly a scalar estimator adaptation |
| `reference` | Offer uniform → action-type uniform |
| `collection-kl` | Reverse KL coefficient .1 → 0 |
| `paper-filter` | Stable top-half → inclusive .75 magnitude quantile AND .01 floor |
| `filter-ties` | Stable top-half → inclusive median, with the same floor |
| `filter-scope` | Filter actor and critic → filter actor only |
| `evaluation-ema` | One training run per seed, both raw and EMA evaluated; no duplicate training arm |
| `behavior-ema` | Raw self-play → EMA self-play; both arms maintain/evaluate EMA |
| `lr-decay` | Constant LR → decaying LR, constant regularization |
| `regularization-decay` | Constant regularization → decaying regularization, constant LR |
| `combined-no-filter` | Combined → remove filtering |
| `combined-no-decay` | Combined → remove coordinated decay |
| `combined-no-value-trace` | Combined → restore value lambda .95 |
| `paper-value-head` | Paper move recipe scalar MSE → categorical outcome CE; explicit optional representation hypothesis |

The final [Nature paper](https://www.nature.com/articles/s41586-026-11036-y)
and [supplement S3.4 / Table S7](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41586-026-11036-y/MediaObjects/41586_2026_11036_MOESM1_ESM.pdf)
were checked directly on 2026-10-04. Move learning uses policy/value lambda
.5/.8, inclusive .75/.01 filtering, clipped action ratios, both reverse KLs,
and evaluation EMA .999. EMA behavior is an additional hypothesis, not a
paper-fidelity claim. The complete move recipe and explicit adaptation table
remain in [docs/ataraxos.md](../docs/ataraxos.md). Auxiliary predictions and
setup heads are unsupported architecture hypotheses, never bundled into a
control. The historical five-arm document remains evidence of that campaign.

## Mechanism contracts

`Learning` and `AtaraxosMoveLearning` own selection kind and scope. The existing
PPO and paper defaults are preserved. Quantiles use Torch linear interpolation
and include all ties; top-count selection breaks ties by flattened row order.
Both operate on detached raw advantages before any PPO normalization. Policy
normalization uses retained rows only; the paper recipe never normalizes.
Actor-only filtering applies to the policy surrogate and both regularizers;
critic loss uses all rows. Each loss averages over its own support. Empty actor
support still trains the critic; shared encoder updates may change policy
outputs even when there is no policy gradient. Actor and critic exposures are
recorded separately from total optimizer row exposures, including repeated PPO
epochs. An empty actor+critic filter skips Adam but advances the EMA clock.

`TrainSelfPlay.behavior="ema-self"` requires an EMA rate and preserves the choice
across same-process continuation. The collector uses averaged weights for both
seats and saves their actual action probabilities and outcome predictions.
Paused next-observation bootstrap uses the same pre-update averaged model.
Learner gradients still update raw weights; EMA advances afterward, once per
collection/update iteration. Likelihood ratios therefore refer to the actual
behavior, not reconstructed raw probabilities. Games can span iterations:
this is not full-episode frozen-policy sampling or trajectory importance
correction. Run/recipe identity, behavior kind and iteration identify collection;
raw/EMA checkpoint digests identify exported models. Recovery retains the existing
raw optimizer, averaged model, RNG and exact collector journal contract.

PPO schedules retain the declared elapsed-budget or iteration-fraction clock.
The paper recipe always retains its one-based collection/update-iteration
clock. No generic executor label overwrites that distinction.

## Analysis and scientific launch boundary

Diagnostics retain signed raw/selected advantage min/quartiles/max and absolute
lambda-target residual means, crossed by chosen action type and observed distance
to termination. Distances count learner transitions: terminal=0, 1–4, 5+;
tails without a terminal in the collected batch are censored. They are never
classified as distant. These bins are descriptive, not a frozen scientific
short/long-horizon split. `selection-diagnostics.json` regenerates offline from
run exports. Entropy, reference/collection KL and optimizer exposure accompany
selection; the last minibatch loss/KL measurements are not epoch averages.

Compare selection rates and residuals **within** each action/distance stratum.
A retained residual excess is an association with the estimator's own target,
not proof that filtering selects true critic error. Independent terminal-return
residuals for censored full episodes remain unavailable in these batch summaries.
Freeze a held-out whole-game analysis population before making that causal claim;
no positions from a game may cross partitions. Pass/combat choices remain in
the population. No action type is assumed irrelevant.

Before costly scoring, freeze at least three independent training seeds per
contrast, fresh disjoint development/final paired-deal families, both deck/seat
assignments, all attempts/exclusions, measured CPU calibration, numeric predictions,
kill criteria, inference envelope, checkpoint cutoffs and an inclusive approved
training/evaluation/recovery budget. The current generator does not allocate or
extrapolate that budget. A scientific ResolvedStudy must retain all three fixed
anchors, untouched endpoint deals and paired seed schedules. Compare only common
observed cost support using the last checkpoint available at each cutoff; no
interpolation or latest-checkpoint substitution. Report per-seed effects and
seed/common-deal bootstrap uncertainty. Raw/EMA are correlated within seed;
variant grouping never doubles sample size. No-overlap and incomplete cohorts
remain unavailable. Report training, export, diagnostics and arena costs.

Leave-one-out interactions require a predeclared follow-up when the combined
screen contradicts isolated effects. LR-only and regularization-only mechanisms
are executable now, but are not automatically selected by the first screen.
Retain/reject requires a separately frozen criterion and complete cohort;
workflow smoke always reports unresolved. No result here establishes S1–S5,
checkpoint-to-demo admission, completed human games or chapter strength.

## Bounded implementation evidence

At implementation commit `a8895883`, one behavior-EMA workflow attempt completed
in 116.83 seconds at `.runs/etu93-behavior`. Raw-self Run
`c5bb16b267494d1ab6ce044d7bc90b60` and EMA-self Run
`9c9a696e725345b48b73999a813aa747` each completed seven training games and 1,024
learner transitions/exposures, in 7.05 and 6.66 seconds respectively. All eight
raw/EMA checkpoint exports were admitted. Twelve arena cells produced 48 full
games with exact replay. Observed cost ranges overlapped. These timings were
collected alongside tests and are not CPU calibration or learning comparisons.
No standalone training/arena attempt failed. The first local test invocation
failed during import because this checkout lacked its native extension; it ran
no training. The extension was built locally before verification.

Offline regeneration preserved byte-identical cost comparisons, uncertainty,
selection diagnostics and report. The saved workflow protocol's old generic
`selection` label said raw, while its explicit variants and receipts included
both raw and EMA. Final validation now requires that label to agree with the
variants; the original proof bytes were retained unchanged. The affected suite
passed 98 tests; 20 focused checks then passed after that metadata correction,
including variant replicate counting and actual collection values matching the
previous EMA checkpoint rather than raw weights. No Rust source changed.

Integration with main preserved the compound-study and frozen-opponent APIs.
Compound training explicitly rejects quantile or actor-only filtering because
its joint-action optimizer does not implement those controls; the contrasts
above target ordinary self-play. Post-sync verification passed 39 focused
omitted-control, frozen-opponent and study tests. CI owns the full merged matrix.
