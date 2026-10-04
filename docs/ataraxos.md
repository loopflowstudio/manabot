# Ataraxos move learning in manabot

ETU-92 implements a selectable move-learning recipe, with the existing PPO
recipe retained as a control. This is an MTG adaptation, not an exact Stratego
reproduction or a strength result. The final source corrects a premise in the
Task: the damped move update **does use clipped importance ratios**. Scheduled
entropy alone is insufficient, but the presence of PPO clipping does not imply
that the method differs from Ataraxos.

Sources checked on 2026-10-04:
[final Nature paper, Methods](https://www.nature.com/articles/s41586-026-11036-y)
and [actual supplement, S3.2–S3.4, equations (5)–(6), Table S7](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41586-026-11036-y/MediaObjects/41586_2026_11036_MOESM1_ESM.pdf).
The retrieved supplement SHA-256 is
`74f217e942cd824140dd5221c02d8c36821a715664f47aee6f489f0ca6932a08`.

## Run the bounded implementation proof

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run manabot train \
  --regime experiments/regimes/ataraxos-move.json \
  --seed 601 --out .runs/etu92-move
```

Use `ataraxos-move-scalar.json` in a fresh output directory for the representation
ablation. These are two-stage, one-thread, 180-second bounded recipes, not
scientific allocations. The normal executor retains the resolved recipe, seed
streams, source/runtime/setup identities, costs and all failed attempts in
VerifyStore; JSON is its export. Both raw and evaluation-EMA weights pass the
ordinary world-bound loader. Categorical checkpoints serve the signed expected
value through the existing interface; their saved AgentHypers bind the three
logits. An old scalar checkpoint is not converted to a categorical checkpoint.

The selected artifact remains last-complete-raw; EMA selection is explicit in
arena registration. Checkpoints do not resume collection across processes.
Same-process continuation preserves Adam, the collector and the iteration
schedule. Neither a reload nor this smoke establishes demo admission, completed
human games, strength, or equilibrium.

## Objective and data ownership

`manabot/training/ataraxos.py` owns the move objective and targets. For each
retained observation the policy loss is the negative clipped-ratio advantage
surrogate plus `collection_kl * KL(current || behavior)` and
`tau * KL(current || magnet)`. All legal offers enter both KL terms. Actual
collection probabilities are retained per row; saved selected-action likelihoods
must agree with them. Invalid support or nonfinite values fail explicitly.

Advantages use signed expected outcomes and a separate lambda from value
supervision. Outcome targets are three-way lambda mixtures, stored in
**loss, draw, win** order; terminal labels are one-hot. The critic receives
cross-entropy with these soft targets, equally weighted with policy loss.
Only current logits receive gradients: behavior distributions, references,
estimates and targets are detached. Shared encoder weights receive both losses.
The scalar ablation uses squared-error lambda-return supervision instead; equal
means do not imply equal categorical targets.

Collection freezes learner weights within each batch. Games may span updates,
so this is close-to-on-policy continuation, not immutable-policy full episodes.
One epoch still changes parameters between timestep minibatches: its per-action
ratios remain necessary. No replay buffer, trajectory product of ratios, or
importance correction for changing historical state occupancy is implemented
or claimed. Filtering itself changes the training-state measure, as in the
source. No normalization rescales advantages after filtering.

The collector records only acting-viewer observation tensors, chosen actions,
behavior probabilities and outcome estimates. Terminal truth supplies reward
labels, never policy inputs. Each stream pauses at the exact next learner
observation; end markers stop propagation across resets. Truncation fails
rather than becoming a draw. MTG opponent actions are traversed between learner
decisions, not assigned learner policy gradients.

## Fidelity table

| Technique / source | Implementation and proof | Boundary |
| --- | --- | --- |
| S3.4 eq. (6), clipped surrogate | `damped_policy_loss`; independently enumerated mixed-policy gradients include both clipping branches | Same mathematical loss; not a new gradient formula |
| Both reverse KLs, S3.4 | Exact legal-support sums against saved behavior and chosen magnet; zero-advantage gradient tests | Positive legal support required; no smoothed fallback |
| Magnet, S3.4 | `reference_distribution`, action-type uniform by default; uniform-offer control available | MTG types replace Stratego piece-then-move groups; not semantic equivalence |
| Scalar advantage, S3.4 | Transition-end GAE, gamma=1, lambda=.5; detached collection values | Learner decisions replace Stratego same-player positions |
| Outcome estimator and eq. (5) | Vector lambda=.8, terminal one-hot, paused-tail bootstrap; terminal/reset/mixture tests | Signed loss/draw/win storage order is a permutation only |
| Value representation, S3.4 | Categorical head, CE; scalar MSE recipe tested separately | Scalar recipe is explicitly an ablation |
| Filtering, S3.4/Table S7 | Inclusive .75 quantile and .01 minimum; ties included; no advantage normalization | Torch linear quantile convention stated; source interpolation unspecified |
| Optimizer damping, Table S7 | Adam, clip=.2, collection KL=.1, max norm=.267, value weight=1 | Existing Adam betas .9/.999 and epsilon 1e-5; source unspecified |
| Schedules, Table S7 | LR clip(.5 / iteration^1.1, 5e-6, 1e-4); tau=.05 / iteration^.3 | One-based iterations; no elapsed-budget substitution |
| Batch reuse, S3.4 | One epoch, increasing learner timestep batches; no shuffle | CPU learner streams replace 202 simulator-step batches of 1536/GPU |
| EMA, S3.4 | .999 after collection/update iteration, including empty filter; parameters averaged, buffers copied | Initialized from learner; used for evaluation only, not fictitious-play policy averaging |
| S3.3 setup learning, eqs. (1)–(4) | Not implemented by move recipe | MTG has no learned piece setup; entropy-to-go predictor not silently substituted |
| S3.5 belief and S3.7 search | Outside this trainer | ETU-96 and separate search work; no claim of the full Ataraxos system |
| S3.6 network/system scale | Existing semantic Agent and viewer transport | CPU float32, small model and MTG observations are explicit adaptations |

Unresolved supplement details: Adam epsilon/betas, iteration origin, EMA
initialization, quantile interpolation, and exact bootstrap pseudocode at a
collection boundary. The choices above are explicit implementations, not claims
that unpublished details were recovered. Boundary tests verify the chosen
lambda recursion. Supplement S5–S7 contain different game-specific recipes;
this implementation selects S3.4, not a blend of their constants. DeepNash's
R-NaD is not implemented here.

## Scientific comparison remains separate

Before expensive scoring, freeze a new protocol, independent seed cohort,
training and evaluation budgets, immutable world/setup and code identities,
held-out paired deals and both seat/deck assignments. Compare current PPO,
Ataraxos categorical and its scalar ablation using existing TrainingRun exports
and `manabot.arena` registration/play/replay. Keep the full planned cohort and
all failures, and compare only overlapping observed cost intervals, including
collection, fitting and exports. Use one inference envelope and declare raw
versus EMA selection before evaluating. Whole-game partitions apply to any
retained data analysis; game noise is not cross-seed method uncertainty.

No extra scientific budget is allocated here. The active ETU-91 campaign,
recipes, frozen fingerprints and retained checkout are unchanged. Historical
small PPO experiments do not decide whether this method is viable.

## Retained implementation evidence (2026-10-04)

One standalone categorical attempt completed: TrainingRun
`c419f9aca3814983ac1e17a1492a0bb6`, seed 601, retained at
`.runs/etu92-move/run.json`. It completed five games, 512 learner transitions
and 46 retained optimizer exposures in 2.1517 seconds total. Collection took
0.9599 seconds, learning 0.5959 seconds and export 0.0374 seconds; total cost
also includes setup and persistence. All four raw/EMA exports passed ordinary
admission. Final raw SHA-256:
`a3141fe4250a5cd4cf7eea4cfb3519ae8894f38397e6c288e2b0b6bf1def68b0`.
No standalone run failed; initial test setup attempts could not find pytest or
the native extension, and ran no training. The scalar path is exercised by
bounded integration tests, not by a scored comparison. These receipts are local
workflow evidence, not a claim of repeatable strength or human completion.

Verification: 77 relevant Python tests passed across the focused objective,
regime, categorical model, existing collector and PPO regression suites. Ruff
and diff whitespace checks passed. No Rust source changed.

## Independent follow-up controls

[ETU-93's separate protocol](../experiments/ataraxos-omitted-controls.md) adds
explicit selection scope and EMA behavior contrasts to the existing trainers.
The paper recipe defaults above remain unchanged. `filter_scope="actor"` keeps
all critic rows; `behavior="ema-self"` collects and bootstraps with averaged
weights before updating raw weights. These are named hypotheses. Their bounded
proofs do not accept a technique or alter the frozen ETU-91 campaign.
