# Frozen-policy local search and distillation

ETU-95 supplies a frozen-policy rollout teacher, retained local-update targets,
TrainingRegime collection/distillation, arena lifecycle and an advice projection.
Physical-prior and learned joint-hand sampling feed the teacher through one typed
interface without enumerating support.
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
SHA-256 through the ordinary world/setup loader. Compound checkpoints use their
decoder's signed-outcome prefix critic, never the unrelated flat head. Ordinary
categorical heads return signed expected outcomes; ordinary scalar artifacts must declare
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

Each sampled canonical world gets every positive-support root action, followed by frozen-policy
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
that the actual opponent need not match the frozen belief model. Where exact likelihood is supported, policy, value,
rollout and likelihood share one checkpoint identity. `sampling=compatible_prior`
uses the physical compatible-deal measure, not equal mass over count vectors.
`compatible_prior` is the default. `sampling=belief` explicitly selects the tiny
exact-reference path, uses zero likelihood smoothing and fails on provider gaps.
Only that reference supports conditional queries through managym; production
samplers accept `True` and reject other queries before rollouts. No path asks
whether the actual hidden hand satisfies a query. A zero-mass reference query fails.

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
work. Native operations and individual forwards are cooperative: a call can overrun,
after which the target is unavailable, never silently accepted or replaced with
policy-only play. Only the explicit `belief` reference enumerates; its support
cap is checked **after enumeration**, so it remains restricted to tractable pools.
Both production samplers materialize counts directly; their cost scales with
sampled hands and vocabulary, not all compatible hands. Arena's subprocess deadline supplies
an outer execution bound. Equal nominal milliseconds do not prove equal realized
cost; retain overruns and unavailable cells.

`full_probability`, `worlds`, and `cheap_worlds` permit reproducible per-root
full/cheap allocation. All positive-support actions receive coverage; all roots are retained.
This is an instrument inspired by [KataGo's playout-cap randomization](https://arxiv.org/abs/1902.10565),
not its target-selection recipe or a measured efficiency improvement. Every
receipt carries the resolved counts. Mixing diagnostics report entropy, KL to
base and L1 distance; none establishes bluff quality.

Targets on the same retained roots:

- `local_soft`: the exact regularized distribution.
- `local_argmax`: one-hot maximal measured Q on policy support, with first-offer tie breaking.
- `local_allocation`: realized root allocation frequencies. Balanced flat
  allocation makes this uniform over policy support; it is **not a PUCT visit-distribution arm**.

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
is unchanged. Belief-enabled policy rollout memory, arbitrary archived-root
relabeling and new arena suites remain unsupported. Compound exact-history
likelihood is unavailable; compound search requires a direct sampler. No automatic
fallback or unmeasured replacement is provided.

The staged recipe makes repeated teacher/student rounds executable; whether they
compound, whether immutable replay helps, and whether stronger-teacher relabeling
beats new games require the separately frozen protocol in
[training follow-ups](../experiments/training-regime-followups.md). ETU-99 owns
these scientific outcomes; ETU-95 owns the software described here. ETU-91 and
its allocations remain unchanged.

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

## Search-facing sampler interface

`manabot.sim.local_sampling.SearchHandSampler` separates native public-constraint
preparation from seeded sampling. `PreparedHands` binds distribution, viewer
history, vocabulary and native snapshot; `HandBatch` retains ordered counts and
joint log probabilities. `LocalUpdateTeacher(hand_sampler=...)` accepts an
implementation matching the admitted distribution identity, independently checks
its preparation against the ordinary sampler, and validates the entire batch
before materialization. Learned injection still requires the exact artifact and
its policy/dataset/schema/world admission. Physical sampling removes public
minima before drawing unknown slots and handles the unique empty-pool deal.
Sequential draws preserve cheap/full sample prefixes. Deadlines include both
preparation and sampling, with checks between draws and after injected calls.

New receipts use `regularized-local-update/v2`, retaining direct physical samples
without fabricated support indexes or learned artifact identities. Learned
receipts retain their existing artifact-bound evidence. The reader still admits
historical v1 receipts without rewriting their bytes. Arena source admission now
uses `local_search_source_sha256()` to bind search plus sampling implementation
owners; historical single-file fingerprints fail current admission.

## Compound conditional search

ETU-94 PR #216's landed native API is integrated. A compound checkpoint uses its
original viewer Observation, ragged offers and recurrent token prefix across
canonical microsteps. `CompoundDecoder` and `Agent.compound` accept `prefix=` to
complete a forced prefix; existing `tokens=` still requires a complete tape.
No parameters or serialized checkpoint fields were added. The decoder exposes an
end-prefix value for optionless roles as well as the existing per-factor values.

`manabot.sim.local_compound` projects a complete next-Command distribution and
checks every route through native lowering. It supports attacker include/exclude
factors, native single-target casts, independent blocker roles (one optional
attacker per blocker), and ordinary action-aligned fallback offers. Small joint
probability tests check the product of canonical conditionals against the original
compound decoder; production does not enumerate attacker subsets. Snapshots and
prefix replay use the selected branch runtime, never raw branch cloning.

Each counterfactual fixes its root action and samples the remaining declaration
once. A rollout retains one declaration at a time and drains its native Commands
before admitting the next actor's root. Leaf values inside declarations use the
retained prefix. Base-policy collection
also samples a whole declaration once, whereas improved search play computes a
local update at each retained prefix. Actor/revision/state changes and incomplete
canonical coverage fail explicitly. Saved observation capacity remains binding.

Receipts retain the original viewer root, offer projection, fixed tokens and
preceding Commands. Replay at an intermediate prefix requires that original-root
cursor alongside the source trajectory/history; a receipt alone cannot recover
its tensors or authoritative source state. `canonical_commands` counts rollout
steps, while `decoder_factors` counts executed decoder factors (including forced
ones); neither counts speculative native lowering calls. All preparation,
lowering, sampling, rollouts and value inference contribute to elapsed cost.
Individual model/native calls are cooperative; an overrun yields no target.

Ordered **waterbend subsets** preserve the original viewer root and token prefix.
The next canonical tap is represented by excluding each intervening candidate and
including that candidate; mana completion excludes the remaining candidates.
These disjoint prefixes integrate the unchosen suffix exactly. Production never
enumerates subsets or compatible hands. A legal tap outside the retained ordered
prefix has exactly zero conditional mass; it is not a numerical approximation.
Native optionless completion ends the declaration without another cached action.
Menace, effectful payments, kicker and ward still use the native sequential fallback.
Ordinary sequential checkpoints retain their payment behavior.

`regularized-local-update/v3` makes policy support explicit through the aligned
base probabilities. Because beta is strictly positive, reverse KL restricts the
updated policy to that support. The uniform reference still covers all legal
offers; the optimizer normalizes its closed-form logits only on positive base
support. Zero-base actions receive zero target and allocation, and JSON `null`
for their unmeasured Q. No rollout is allocated to an inaccessible action. There
is no epsilon smoothing, flat-head fallback or declaration resampling. Supported
probability underflow in the update still fails rather than silently losing mass.

The receipt reader checks support, null availability, per-hand supported-action
coverage and rollout means. Historical v1/v2 receipts retain their complete
positive-support contract and serialized bytes. `local_soft` preserves zeros;
`local_argmax` chooses only measured Q; `local_allocation` normalizes realized
supported counts. Numeric legacy score columns use NaN for unavailable Q and
cannot be admitted as score-softmax targets. Advice marks unvisited Q unavailable
and computes the root value over the supported target. Receipts still require
the original-root cursor for intermediate-prefix replay.

An admitted priority root can now reach a payment subset during rollout with
either direct sampler. This removes ETU-95's payment-specific full-game blocker;
other documented admission, history, capacity and deadline limits still apply.
This is software compatibility, not an arbitrary-game completion or strength
claim. ETU-99 retains scientific comparison ownership.

`collect_local_update` accepts raw compound policy stages and compatible admitted
samplers. Compound EMA remains unsupported by its training owner. Its canonical
conditional targets load through the existing `local_soft`, `local_argmax` and
`local_allocation` readers; flat students can consume these targets, but that does
not promise preservation of a joint declaration policy. Mixing flat supervised
training into a compound TrainingRegime remains rejected. No training run or
scientific comparison is required for this software proof, and the live advice
provider is still unregistered.

Fixture-only validation on 2026-10-04 passed 60 affected Python checks, six native
debug compound tests and focused Ruff/format checks. It includes physical and
learned saved-artifact search, complete-game target reader admission, compound
arena Command replay, small joint-probability parity and wide attacker/cast roots.
No optimizer ran, no paid compute was used, and ETU-91 was not modified. CI runs
the new search fixtures on Linux and macOS; these results are software evidence,
not calibration, strength or human-play acceptance.

ETU-100 fixture gate on 2026-10-04: 42 affected Python tests and six native debug
compound tests passed after integrating ETU-95's final parent. Fixed untrained
weights exercise exact payment joint parity (tap-only, mixed and mana-only),
65-candidate support projection, both saved samplers, sequential compatibility,
retained zero-support receipts/readers/advice, replay and deadlines. No optimizer
or scientific study ran locally; these checks make no strength claim.
