# Frozen-policy local search and distillation

ETU-95 supplies a frozen-policy rollout teacher, retained local-update targets,
TrainingRegime collection/distillation, arena lifecycle and an advice projection.
Learned joint-hand sampling now feeds the teacher without enumerating support.
The complete-history exact-posterior comparison remains blocked by Rules
capabilities; no strength, bluffing or human-play result is established.

Run the bounded mechanism proof in a fresh output directory:

```bash
uv run manabot train --regime experiments/regimes/frozen-policy-local-update.json \
  --seed 715 --out .runs/etu95-local-proof
```

The 180-second one-thread recipe trains a tiny policy, freezes its raw bytes,
labels two complete games under the compatible-deal prior, fits three targets
on the same roots, freezes the soft-target student, labels two more games and
fits cumulatively. Both deck assignments occur. This uses a tractable two-name
pool, not Allies/Lessons or the ETU-91 campaign. The initial retained execution
completed in 9.94 seconds with 64 learner transitions and 318 labeled decisions
over four games. It predates the final diagnostic/receipt-validation edits;
the automated current-tree tests are the correctness authority. `.runs` holds
local proof, not a portable scientific artifact.

## Contracts

`manabot.sim.local_update.LocalUpdateTeacher` admits one checkpoint by exact
SHA-256 through the ordinary world/setup loader. Categorical heads return signed
expected outcomes; scalar artifacts must explicitly declare
`bc.value_semantic = signed_outcome`. TrainingRegime writes this declaration for
self-play and local distillation. Untagged historical scalar critics are rejected
because previous supervised teachers used win logits. No checkpoint port or
renaming of frozen evidence occurs.

For legal actions a, the update maximizes
`sum p(a) Q(a) - alpha KL(p || reference) - beta KL(p || base)`.
Its normalized logits are `(Q + alpha log(reference) + beta log(base))/(alpha+beta)`.
The reference is uniform over semantic offers; beta is the inverse step scale.
This follows the final Ataraxos supplement S3.7 equations (7)–(8), checked on
2026-10-04 ([supplement](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41586-026-11036-y/MediaObjects/41586_2026_11036_MOESM1_ESM.pdf)).
The MTG budget, horizon, reference and value representation are explicit
adaptations; this is not an assertion that arbitrary PPO training inherits the
paper's improvement properties.

Each sampled canonical world gets every root action, followed by frozen-policy
moves to the declared depth (including the forced first action). At each step,
only the acting player's ordinary Observation is encoded. Terminal values are
-1/0/+1; nonterminal values are that acting player's signed estimate, reversed
when the actor differs from the root viewer. Scalar estimates remain unbounded;
there is no silent sigmoid or clipping. There are no gradients in the teacher.
The student gets fixed distributions and policy cross-entropy; gradients reach
its policy head and shared encoder, not teacher weights. The critic's head is
not fitted by these policy-only stages; shared-encoder drift remains measurable.

Collection follows the **base frozen policy**, not its search update, so the
likelihood model matches the generating population where its history model is
supported. Arena play samples the updated distribution and explicitly records
that the actual opponent need not match the frozen belief model. Policy, value,
rollout and likelihood share one checkpoint identity. `sampling=compatible_prior`
uses the physical compatible-deal measure, not equal mass over count vectors.
`sampling=belief` uses zero likelihood smoothing and fails on provider gaps.
Queries restrict this distribution through managym; they never ask whether the
actual hidden hand satisfies the query. A zero-mass query fails.

## Evidence and budgets

`LocalUpdateReceipt` records semantic offer IDs, base/Q/reference/target,
coefficients, exact sampling weights, query/mass, world/belief/policy/native/source
identities, seeds, per-action counts, signed rollout values and exact branch
Command/witness tapes. `verify_replay` re-executes from an exact source root and
tracked belief, comparing all retained evidence except elapsed times. Hidden-world
swap tests verify root information invariance. Receipts alone are not a source
trajectory or a policy-free historical reconstruction system.

Collection journals each complete target and played base action before stepping;
failed runs and elapsed costs remain in TrainingRun/VerifyStore. Whole-game shard
IDs determine immutable validation membership (`game_index % 10 == 0`). Cumulative
fitting retains both older teacher provenance and Adam state. Old score-only
shards cannot claim to reconstruct a local target. Local receipt admission checks
its target formula, full action coverage and rollout means.

Decision deadlines charge prior construction/history filtering since the previous
decision as well as search. Stage/run budgets also charge setup, export and failed
work. Native enumeration and individual forwards are cooperative operations:
a call can overrun, after which the target is unavailable, never silently accepted
or replaced with policy-only play. In `belief` and `compatible_prior` modes the support cap is checked **after
enumeration**; those paths remain restricted to tractable pools. Learned mode
uses direct count materialization and never constructs that support; its cost
scales with sampled hands and vocabulary, not all compatible hands. Arena's existing subprocess game deadline supplies
an outer execution bound. Equal nominal milliseconds do not prove equal realized
cost; retain overruns and unavailable cells.

`full_probability`, `worlds`, and `cheap_worlds` permit reproducible per-root
full/cheap allocation. All actions still receive coverage; all roots are retained.
This is an instrument inspired by [KataGo's playout-cap randomization](https://arxiv.org/abs/1902.10565),
not its target-selection recipe or a measured efficiency improvement. Every
receipt carries the resolved counts. Mixing diagnostics report entropy, KL to
base and L1 distance; none establishes bluff quality.

Targets on the same retained roots:

- `local_soft`: the exact regularized distribution.
- `local_argmax`: one-hot maximal Q, with first-offer tie breaking.
- `local_allocation`: realized root allocation frequencies. Balanced flat
  allocation makes this uniform; it is **not a PUCT visit-distribution arm**.

A same-root genuine PUCT-visit comparison remains separate scientific work.
The existing `visit_distribution` trainer remains available for its own teacher.

## Consumers and unsupported paths

`make_player(kind=local_update)` supports the existing matchup lifecycle. Arena
registrations additionally pin the teacher source and explicit search semantics;
private game traces retain local receipts and replay the actual Commands. Failed
history updates retain the command that preceded the failure. The ordinary
selected arena still owns its existing suites; tiny-pool tests do not register a
new production arena or establish selected-matchup admission.

`etude.local_advice.local_update_scenario` projects into `AdvisorScenarioEvidence`;
`regularized-local-update/v1` identifies the policy semantic. Q quantities name
signed-value methods, and unmeasured uncertainty/robustness remain unavailable.
No sampled worlds, sampling weights, rollout seeds or branch tapes enter that
projection. The live `/api/advice` provider is not registered to this teacher;
its historical-root, artifact and compute admission remain required.

The exact arm stops at opponent choices without a public likelihood identity
(including combat/target choices), and at ordinary discard prompts that the
materializer cannot refresh. The existing tracker only supplies supported
hand-multiset/chance transport; it is not an exact posterior over every history
or hidden library order. This discovered Rules dependency prevents the requested
full-game exact-versus-prior acceptance today. Reference tracker behavior elsewhere
is unchanged. Compound-policy checkpoints, belief-enabled policy rollout memory,
arbitrary archived-root relabeling and new arena suites remain unsupported.
A sampler trained on a compound checkpoint cannot enter sequential local search;
this fails at policy admission even if sampler collection supports it. No automatic fallback or unmeasured replacement is provided.

The staged recipe makes repeated teacher/student rounds executable; whether they
compound, whether immutable replay helps, and whether stronger-teacher relabeling
beats new games require the separately frozen protocol in
[training follow-ups](../experiments/training-regime-followups.md). ETU-95 remains
open for these scientific outcomes. ETU-91 and its allocations remain unchanged.

Validation on 2026-10-04: 73 affected Python tests and 21 native debug world/branch
tests passed before upstream sync. After rebuilding the merged native extension,
eight local-search/compound integration tests and nine native debug world tests
passed. Ruff passed. Evidence includes complete tiny-pool arena Command replay
and retained failure on unsupported exact history.

## Learned joint-hand search

```bash
uv run manabot train --regime experiments/regimes/learned-belief-local-search.json \
  --seed 718 --out .runs/etu95-learned-search
```

The separate 180-second, one-thread proof trains a policy, collects three complete
belief-supervision games, fits a sampler, labels eight complete frozen-policy
games and fits a soft-target student. Eight label games ensure the existing
immutable `game_index % 10 == 0` validation partition is populated after belief
collection advances the run's game counter. A retained initial two-label-game
attempt failed explicitly at that split check after successful search collection;
no failed attempt or partition was replaced.

`collect_local_update.sampler` references an earlier `train_belief` stage and
requires `search.sampling = learned`. Its source dataset must use exactly the
same policy stage and raw/EMA choice as the rollout teacher. Admission checks
sampler bytes, dataset, schema, world binding and generating-policy digest.
The policy, rollout and signed-value identities remain the frozen policy's;
only the separate sampler was fitted on hidden labels.

managym's `HiddenHandConstraints.materialize_hand` and Python
`Env.materialize_sampled_hand` accept counts directly against a complete native
constraint snapshot. They validate source viewer/revision/observation identity,
public pool, known minima and hand size, preserve root viewer identity and
semantic offers, and use the same seeded hidden-card placement as indexed
materialization. They do not enumerate, map counts to support indexes, or
refresh an opponent's acting prompt. Reproducible seeds determine hidden library
order; sampled hand counts do not model a posterior over library order.

`LocalUpdatePlayer` retains each seat's canonical viewer history from game start.
Collection and search share `public_sampler_input`, including the same lossy,
order-invariant public-commitment features. Missing combat/target commitment
semantics remain missing features, not inferred meanings or an exact posterior.
Private labels and sampled hands never enter the acting policy's observation.
Foreign arena opponents are explicitly model-mismatched populations.

Learned receipts identify `approximate-learned-hand`, with public input/history
identity, exact sampler bindings, native constraint snapshot, sampling seed,
counts and joint log probabilities per draw. Rollouts use `world_index = null`
and retain the sampled counts and seeded branch audit. Empty
`sampling_probabilities` means full support was not enumerated; it does not mean
uniform mass. Values average sampled rollouts without importance weighting.
Cheap/full allocations share a deterministic sample prefix. Normalized learned
sampling supports only `True`; other queries fail because calibrated conditional
mass and conditional sampling are not implemented. `max_support` applies only
to the enumerated modes. Sampling, feature projection and materialization count
toward the cooperative decision deadline; artifact loading counts at stage setup.

Native debug checks cover direct/indexed parity, impossible and stale snapshots,
public known minima, and hidden-hand swap invariance. Python proofs cover zero
native support constructions, full pipeline execution with enumeration disabled,
receipt replay and two-seat complete arena games with exact actual-Command replay.
These are mechanism proofs, not posterior calibration, stronger play, exact-history
completion or authorization for scientific scoring.

Integrated proof after PR 212 on 2026-10-04: the recipe completed in 31.43 s,
with three sampler-data games, eight label games (753 search decisions), 32
sampler optimizer exposures and 661 student exposures. The exported student
passed ordinary checkpoint admission. The exact trained policy/sampler pair
then played both arena seats against a random fixture: two complete games,
183 actual Commands, zero replay mismatches. Private evidence is retained under
`.runs/etu95-learned-search-integrated`. The first arena export failed after
successful replay because the proof script treated a dataclass as a Pydantic
model; its failure marker is retained beside the successful second export.
The earlier split failure remains in `.runs/etu95-learned-search-proof`.
The custom arena registrations are workflow fixtures, not production admission.

Checks: 57 affected Python tests passed before final upstream integration;
11 native debug world/materializer tests and Clippy passed. The integrated CLI
proof and 12 passing PR 212 sampler collection/regime checks cover the subsequent
merge.
