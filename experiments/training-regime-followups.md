# Follow-up mechanism protocols

2026-10-04. These proposals belong to ETU-91's experiment design; no execution
or expensive allocation is authorized by their presence. Each requires frozen
inputs and predictions, a separately approved cost cap, fresh evaluation deals,
and retained failed attempts before scoring. The frozen-policy collection and
belief fitting stages in item 4 now have a bounded implementation described in
[the sampler guide](../docs/belief-sampler.md); its scientific allocations and
acceptance comparisons below remain unexecuted proposals.

## Following the learned policy into search and belief experiments

Search remains a product capability, including belief-conditioned advice.
Policy-only evaluation isolates the source of learned strength; it is not a
decision to remove search from Etude. Deliver the following concrete follow-up
experiment specifications with prerequisites, artifact contracts and proposed
analysis, while keeping their unimplemented stages out of runnable recipes:

1. **Compound combat decisions:** integrate a trainable autoregressive decoder
   over complete legal declarations. Sum conditional log probabilities for
   the joint action; define reward/discount/value boundaries explicitly and
   forbid grouping across intervening information or opponent decisions.
   Compare sequential and grouped policies on the same game outcomes and
   wall budget; report equivalent underlying choices so collapsed prompts
   cannot manufacture throughput gains. Adapter parity alone is insufficient.
2. **Search after RL:** freeze each selected raw/EMA policy identity. On a
   tractable pool, obtain exact posterior weights from that same frozen
   behavior policy, sample compatible worlds, roll out with viewer-safe
   policy observations, evaluate leaves and take one local regularized update.
   Compare policy-only, uniform-belief search and exact-belief search at matched
   elapsed decision budgets. Never feed full sampled worlds into the policy.
   Exact ranges fitted to another population do not establish update equivalence.
3. **Distill the local update:** retain base policy, action-value estimates,
   belief/rollout policy identities, reference, step size and computed target
   distribution. Compare hard argmax targets and regularized soft targets from
   the same roots/cost. Old per-action scores without those identities cannot
   reconstruct this target. Measure student strength, policy mixing and
   adversarial response; soft targets alone do not guarantee sound bluffing.
4. **Amortized beliefs:** collect frozen-policy self-play with hidden truth as
   labels only. Train a constrained autoregressive sampler; keep whole games
   separate across train/validation/test. Compare calibration, legal support,
   sampling cost and resulting search strength against exact beliefs on a
   tractable pool before widening the pool. All widening is a new world-bound
   comparison, not an unverified ten-million-hand extrapolation.

Add a frozen-opponent exploiter protocol for both main arms: independent
attacker seeds, terminal rewards, equal attacker budgets increasing at declared
checkpoints, and fresh final deals. Plot attack success versus attacker compute.
An unsuccessful bounded attacker is not an exploitability certificate.
Charge attacks and their evaluations explicitly; they are outside the proposed
168-hour comparison unless its allocation is amended. Reuse the repaired
`NetOpponentTrainer` frozen-opponent mode, not a second training implementation.
Run S1-S5 at policy checkpoints after verifying current-world legality and
intended strategic premises; report unsupported scenarios rather than silently
substituting old scores. Behavioral failures qualify arena gains.

The document freezes seeds, budgets, numeric prediction (proposed B=.55),
kill criteria and strongest confound before costly runs via `lf commit`.
Training, validation, development and final deal families are disjoint.
The strongest confound is recipe maturity: a negative result may reflect
untuned self-play treatments, not a limit of direct RL.

## Proposed run matrix and stopping rules

These are draft allocations to review after the policy-only studies, not an
extension of either smoke. For each study, freeze the exact world, raw or EMA
checkpoint digests, runtime digest, recipe, reference distributions and cost
receipts before generating its first label. Use whole-episode partitions;
positions from one trajectory never cross partitions. All training attempts,
including invalid runs, count against the cap. A missing prerequisite stops
that study rather than substituting a different mechanism.

| Study | Required executable prerequisite | Planned comparison | Proposed cap |
| --- | --- | --- | --- |
| Compound combat | Trainable joint decoder with semantic Command parity and joint log probabilities | Sequential versus grouped, seeds 401–403; two hours per arm/seed | 12 training hours + 4 evaluation hours |
| Post-RL search | Same-policy exact posterior, viewer-safe rollouts and regularized local update | Policy alone, uniform-belief search, exact-belief search; decision budgets 50/200 ms | 8 evaluation hours; no new policy training |
| Update distillation | Receipts with the exact local-update target and its generating policy | Hard versus soft targets on the same roots, seeds 411–413; one hour per fit | 4 label hours + 6 fitting hours + 4 evaluation hours |
| Belief sampler | Constrained autoregressive sampler and exact tractable-pool reference | Exact versus learned joint samples; three sampler seeds 421–423 | 4 label hours + 6 fitting hours + 4 evaluation hours |
| Frozen exploiters | Ordinary frozen-opponent trainer with verified learner-only credit | Both main arms attacked by seeds 431–433 at 15/30/60 minutes | 6 training hours + 4 evaluation hours per attacked checkpoint pair |

Reserve deal families 940000–949999 for follow-up development and
950000–959999 for final scoring, with distinct nonoverlapping subranges per
study fixed in its launch manifest. These families must also be checked against
actual producer seeds before launch. Use 32 untouched four-leg blocks per
matched comparison as the initial proposed final cohort, then calibrate the
complete schedule before freezing it. If the proposed time allowance cannot
fit the full cohort with a 25% margin, revise the protocol before any scoring.
Do not stop on a favorable result or trim the cohort after observing outcomes.

All studies stop on illegal Commands, hidden-truth input leakage, world/setup
mismatch, or unreplayable evidence. Compound-action timing counts underlying
choices as well as grouped decisions. Search reports deadline misses and
realized simulations, with equal elapsed inference envelopes. Distillation
reports target entropy, held-out KL and full-game scores separately. Belief
sampling reports support violations, query calibration, joint log loss where
computable and search quality; a low marginal error cannot certify the joint.
Exploiters report every seed's attack curve and failed attempts; their maximum
observed win rate is a bounded attack result, never an exact exploitability
number. No mechanism is retained solely because a single seed improved.
