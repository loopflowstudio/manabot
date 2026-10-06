# Follow-up mechanism protocols

2026-10-05 ownership reconciliation: ETU-105 owns learning-rule, compound-credit
and systems screens through the [source-checked inventory](ataraxos-technique-screen.md).
Larger belief, search and exploiter studies remain ETU-99. Its
[current coverage map and prospective protocols](ataraxos-transfer.md) supersede
the search/belief launch ordering and allocations below. Production uses direct
physical or learned sampling; exact enumeration is a small-pool reference, not
a prerequisite. The proposals below retain their historical evidence and budget
boundaries; none grants a new allocation.

Historical proposal, 2026-10-04, from ETU-91's experiment design; no execution
or expensive allocation is authorized by their presence. Each requires frozen
inputs and predictions, a separately approved cost cap, fresh evaluation deals,
and retained failed attempts before scoring. Compound training (ETU-94, below)
and the frozen-policy collection and belief fitting stages in item 4 now have
bounded implementations. See [the sampler guide](../docs/belief-sampler.md) for
the latter; scientific allocations and acceptance comparisons remain unexecuted
proposals.

## Original proposal: following the learned policy into search and belief experiments

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

## Original proposed run matrix and stopping rules

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

## ETU-95 implementation boundary (2026-10-04)

The [local-search guide](../docs/local-policy-search.md) and executable
[`frozen-policy-local-update.json`](regimes/frozen-policy-local-update.json)
now cover frozen raw/EMA collection, viewer-safe rollouts, the two-KL local
update, complete target receipts, hard/soft/allocation controls, and cumulative
multi-round fitting. The recipe is a bounded mechanism proof, not authorization
for the proposed 8-hour search or 14-hour distillation allocations above.
The allocation target is not the requested genuine PUCT-visit comparison.

Full-game exact-posterior scoring is blocked: combat/target choices lack public
likelihood identities and ordinary discard cannot refresh counterfactual offers.
The local exact player stops explicitly there. The compatible-prior recipe can
complete; substituting it does not satisfy the exact-belief comparison.

Before any scientific launch, freeze three independent producer checkpoints and
all failed producer attempts; checkpoint native/setup/ABI hashes; alpha/beta,
rollout depth and 50/200-ms envelopes; source policy identities; held-out paired
deal subranges; training/validation/test whole-game membership; endpoint cohort;
and calibrated storage/time estimates. Exact-history capability must pass before
the exact-reference comparison; direct sampling comparisons do not depend on it.
The proposed B=.55 criterion uses a training-seed-level uncertainty interval
whose lower bound exceeds .50; game-level paired uncertainty remains separate.
Keep the original proposed caps and stop on deadline/support/provider failures,
with incomplete cells retained. An unavailable comparison is not a negative
strength result. No paid compute or extra ETU-91 allocation is authorized.

Evaluate additional teacher lessons separately:

| Question | Frozen comparison | Required evidence / stop |
| --- | --- | --- |
| Does full/cheap allocation pay? | All-full versus configured random full/cheap worlds at equal total label cost | Realized counts, entropy, held-out KL and complete-game cost; no inference from nominal world counts |
| Does replay help? | Latest-round-only versus cumulative immutable shards, identical fits/seeds and endpoint budgets | Immutable root membership, per-teacher target age/weight and charged historical generation cost; no split reshuffle |
| Does relabeling beat fresh games? | Stronger teacher on archived roots versus newly generated roots at equal total cost | Canonical archived root/trajectory replay and new immutable receipts; unavailable until relabeling admission exists |
| Do improvements compound? | At least two frozen collect/fit rounds versus first-round student, fixed controls and endpoint deals | Full-game cross-seed uncertainty, unchanged inference envelopes and every attempted round; pipeline iteration alone is insufficient |
| Which target preserves strategy? | Hard Q argmax, actual same-root PUCT visits and regularized soft update | Same roots, separately identified teachers, entropy, held-out KL, arena and adversarial response; allocation frequencies do not substitute for visits |

Source basis: final Ataraxos S3.7 uses a regularized local update after fixed-policy
rollouts; the implementation guide records its exact equation and adaptations.
KataGo's allocation idea is a separate ablation, not evidence about hidden-information
MTG. No new scientific result is reported by this addition.

## Compound implementation and separate comparison — 2026-10-04

ETU-94 implements a recurrent joint decoder, native lowering to canonical
Commands, complete-game `train_compound`, ordinary checkpoint serving, and
four `compound-decisions` study arms. Search/distillation/belief software now
exists; their scientific comparisons remain future work. See
[the execution contract](../docs/training-regimes.md#compound-decisions).

The primary Ataraxos construction is now available in the
[Nature methods and Extended Data Fig. 1](https://www.nature.com/articles/s41586-026-11036-y):
its decoder-only setup transformer generates placements in row-major order,
with prefix outcome and entropy predictions; setup credit uses final game
outcomes, whereas move learning uses lambda estimators. ETU-94 uses a GRU and
scalar prefix values over MTG's native offers. This is a mechanism analogy,
not a reproduction or evidence that outcome credit is superior in MTG.

The runnable 2×2 is sequential/grouped decoder credit × bootstrapped/outcome
credit, with identical initialization, model, world, setup and conditional
reference. Both sequential and grouped arms execute complete sampled native
submissions. It isolates credit boundaries without changing the policy family;
comparison against the historical flat policy would be a separate architecture
ablation. Grouped attacker declarations and single-target casts are supported.
The original implementation kept blockers and payments separate. ETU-94/100
subsequently delivered independent blockers and fixed-economics payment subsets,
including support-preserving search. Dependent choices retain sequential native
fallback; see the current [compound contract](../docs/training-regimes.md#compound-decisions).

Keep gamma=1 and all non-estimator settings fixed. Predeclare the measured
number of complete games/updates that fits each arm's same wall cap, report
unused allocation and failed attempts, and compare checkpoints only over
observed overlapping cost ranges. Native optionless-step collapse is a separate
`skip_trivial` audit, not part of the grouping contrast. Report games, native
Commands, groups, decoder factors (including forced factors), optimizer units,
wall cost and decision latency together. Full-game paired arena blocks cover
both deck and seat assignments; every Command is replayed. Whole training games
stay intact, and evaluation deal families remain disjoint from training.

The prior 12 training + 4 evaluation hour proposal covers only the original
six-run, two-arm study. It does **not** allocate twelve runs for the 2×2. Before
scientific scoring, freeze a separately authorized total cap, three or more
independent training seeds per arm, all failure/recovery accounting, development
and untouched endpoint deal schedules, matched inference limits and a numeric
criterion. Keep seeds 401–403 and reserved follow-up deal families provisional
until collision checks against actual producer receipts. No ETU-91 allocation
or retained running checkout is changed by this implementation.

Bounded checks establish normalized conditionals/joints, prefix dependence,
score-function and finite-difference gradients, terminal versus bootstrapped
credit, 65-attacker/35-target legality and Command parity, interruption rejection,
hidden-world invariance and ordinary reload. They do not establish method-level
improvement, competence, or a human challenger. Outcome versus bootstrapped
superiority remains an open scientific result even when both paths execute.
