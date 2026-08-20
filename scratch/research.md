# Research: belief-forming agent versus the evolving plan of record

## Scope and conclusion

This review compared the current branch, the current `origin/main` tree, Wave
goals and memory, archived architecture notes, accepted INT-10 design records,
implementation-task designs, frozen experiment reports, tests, and relevant
Git history from 2026-07-09 through 2026-07-18.

The central conclusion is:

> The belief-forming agent was the original belief vision and an explicitly
> accepted part of the 2026-07-17 architecture. It was not rejected. It was
> first deferred by experimental ordering, then split across three independent
> implementation paths, and finally described as landed substrate before those
> paths were recomposed into one ordinary manabot.

The current branch design changes the **keystone and integration order** more
than the destination. It makes the missing composition—temporal belief
formation plus an explicit belief-conditioned policy/value/search core—the
next indivisible agent change.

## System understanding

### Current architecture on main

The current main tree has substantial belief machinery, but it is divided:

```text
viewer history
    -> BeliefTracker -> exact BeliefState
                         -> ExactRangePlayer -> weighted flat search
                         -> conditional PUCT adapter -> advice/search

conditional shard
    -> condition_index + condition_weight
                         -> general neural Agent policy/value

BeliefState -------------------X-------------------> general neural Agent
                          missing model-facing seam
```

The pieces are:

- `managym` owns the viewer-relative `PossibleWorldSpace`, exact physical-deal
  weights, typed `WorldQuery`, semantic history, and materialization.
- `manabot/belief/range.py` defines a real normalized `BeliefState` as one log
  probability per canonical world. It preserves correlations, samples worlds,
  reports inclusion marginals, and fails closed on normalization errors.
- `manabot/belief/tracker.py` implements the temporal update the new design
  calls `b(t-1) -> b(t)`: public-action likelihood conditioning, hidden draws,
  known exits/returns, transport to the next canonical world space, and exact
  transition receipts.
- `manabot/belief/player.py` gives that tracker an autonomous consumer, but
  only as the specialized `ExactRangePlayer`; it samples the belief into flat
  search and emits a semantic Command. The ordinary neural `Agent` does not
  participate.
- `manabot/sim/conditional_search.py` adapts `BeliefState` into the older
  `ConditionalWorldPrior` search seam and searches explicit beliefs. This is a
  root evaluator, not an agent belief lifecycle.
- `manabot/model/agent.py` has no `BeliefState`, belief model, history update,
  or semantic belief projection. Its optional condition channel embeds a
  positional `condition_index` plus one scalar `condition_weight`. If those
  fields are absent during arena inference, it silently supplies condition 0.
- `manabot/sim/conditional_distill.py` freezes the five positional roles
  `true/has/lacks/q/not_q`; human-readable query identities live only in
  provenance. Its frozen toy deliberately repeats identical labels across all
  five roles.

There is no `BeliefTensorView`, `BeliefModel` protocol, belief override API, or
test proving that the general neural agent's generated and supplied beliefs
converge on one decision core.

### Tests and evidence

The quality is asymmetric in a useful way:

- `tests/belief/` thoroughly checks exact support, normalization, transport,
  semantic-history updates, viewer equivalence, replay receipts, and matched
  belief/prior search behavior.
- `tests/sim/test_conditional_search.py` checks conditional search alignment
  and that viewer-safe results do not leak hidden worlds.
- `tests/sim/test_conditional_snapshot.py` thoroughly checks the positional
  condition schema, immutable bytes, neutral fallback, and tamper failures.
- No test connects an exact or learned `BeliefState` to the general neural
  `Agent`, checks semantic equivalence of two beliefs reached by different
  queries, or proves that an intervention replaces all belief-carrying paths.

The INT-17 calibration attempt also exposed a systems boundary: likelihood
evaluation materialized every world through a provider that re-enumerated the
complete support for every row, making an update quadratic in support size.
The frozen run stopped without calibration curves after proving it could not
meet its six-hour cap. The semantic belief representation is sound; its first
production update path is not yet practical at the largest retained supports.

## Chronology

### Before beliefs: observation-only policy and generic memory ideas

The archived model plan proposed an optional LSTM for imperfect information,
but it treated memory as an opaque neural state. The implemented `Agent`
remained a current-observation attention model. Nothing gave beliefs explicit
world semantics or made them replaceable.

### 2026-07-09: the original PBS design already contains the agent idea

Commit `dd66516` captured `wave/intelligence/02-beliefs-design.md` (then a
separate beliefs Wave). It proposed:

- a public belief state as public state plus player ranges;
- a temporal Bayesian update from public action likelihoods, draws, and
  reveals;
- canonical semantic action identity for cross-world likelihood updates;
- ranges as inputs to policy and value;
- supervised calibration from logged hidden truth; and
- the policy net as an amortized solver table.

This is the clearest historical ancestor of the current branch design. It was
also more ambitious: both players' ranges, genuine mixed strategies, per-hand
counterfactual value vectors, and eventual CFR/public-belief solving.

The belief idea was initially marked dormant until a search/exploitability
tripwire fired. That was a priority decision, not a different model of what an
eventual belief-aware agent was.

### 2026-07-10: belief and search were joined, but belief stayed gated

Commit `ede07b3` folded the beliefs Wave into Search and added important
game-theoretic cautions. The search plan treated uniform determinization as a
useful initial player and deferred belief-aware or information-set-consistent
machinery until diagnostics showed belief error or strategy fusion was
binding.

This established the first long-lived tension: belief was architecturally
fundamental to correct hidden-information value, but operationally treated as
a later remedy for a measured search failure.

### 2026-07-15: the measured-research recharter separated the work

Commit `c562fe3` chartered two parallel programs:

1. a KataGo-shaped search teacher and distillation loop on the existing ABI;
2. semantic representation katas.

Belief estimation remained behind the information-by-continuation diagnostic.
The directive to make teacher progress “without waiting” was sensible for
experimental throughput, but it made the search teacher—not the complete
agent—the organizing abstraction.

### 2026-07-17 morning: the research map restored a complete belief player

Commit `c3319a0` added `manabot/RESEARCH.md`. It explicitly says the first
belief build should be a complete player, not a standalone posterior demo:
track an exact range from public history, sample weighted determinizations,
play the selected matchup, and explain the result with calibration and a
uniform control.

That led to the later `ExactRangePlayer`. It recovered the temporal belief
half of the intended agent but coupled it directly to search rather than to the
general policy/value model.

### 2026-07-17 evening: INT-10 accepted the intended target architecture

Commits `1f79603` and `04af5a1` made the interactive INT-10 design the plan of
record. The accepted record states:

```text
Observation history -> BeliefHead -> BeliefState b

visible state + semantic offers + b -> PolicyHead
                                     -> ValueHead
```

It also states that query text is provenance rather than a model feature,
equivalent induced beliefs must produce the same model result, and actual
hidden truth supervises belief calibration rather than policy.

The important sequencing decision was different from the current branch:

1. canonical world/query authority;
2. conditional search teacher;
3. conditional shards and a student trained on canonical query-restricted
   priors;
4. learned belief head and INT-9 adaptation.

The concrete belief-head representation and whether policy had another history
path were explicitly left open. The design said those questions did not block
the first conditional teacher.

### 2026-07-17 night: the first semantic/code divergence

INT-13 (`522b2fb`) implemented honest conditional determinized PUCT. It was a
reasonable teacher slice: explicit world weights in, aligned strategy results
out, no belief inference claim.

INT-14 (`8ca40e2`) is where the general model diverged from the accepted
semantic boundary. Its task design explicitly chose the smallest plumbing
change:

```text
condition_index + condition_weight -> one Agent object row
```

It repeatedly stated “no learned belief head” and “condition is a tag.” The
experiment was honest about being a no-strength plumbing receipt, but the code
could not represent which card, zone, distribution, or correlation a belief
described. Two equivalent beliefs reached through different query roles would
not necessarily encode identically; two different card queries in the same
role would encode identically.

There was also a coordination contradiction in the retained INT-14 design. It
justified excluding belief work by citing the old beliefs document as
“dormant, trigger-armed,” even though the INT-10 architecture merged earlier
that evening and explicitly changed that header to say the dormant gate was
superseded. The task updated its assumptions after rebasing onto INT-13, but it
did not refresh this larger architectural premise.

### 2026-07-18: real beliefs landed on a separate player path

INT-9 (`3efd25b`) implemented a substantial part of the original design:

- canonical exact `BeliefState` values;
- temporal `BeliefTracker` updates from viewer-safe semantic history;
- a matched compatible-prior control;
- likelihood-weighted world sampling; and
- an autonomous `ExactRangePlayer` that acts through semantic Commands.

It intentionally rejected per-card marginals as the **canonical** belief
because they cannot preserve correlated worlds or materialize hands. The
current branch design agrees: marginals are only a derived policy view, while
`BeliefState` remains the full source.

INT-9 was not connected to the neural `Agent`. The tracker fed search-world
sampling, while INT-14's unrelated condition tag fed policy/value.

### 2026-07-18: results-first reprioritization froze the split

Commit `823b0a3` observed that many instruments existed without production
results and adopted an R1-R4 results ladder:

- freeze a recommendation flip;
- put tracked belief on live advice;
- measure belief calibration; and
- run arena and production-teacher results.

It explicitly deferred the supervised belief head and factorized marginal
representation. This was evidence-driven and correctly resisted more idle
substrate, but it also described the belief-conditioned substrate as already
landed. In reality, three narrower instruments had landed:

1. exact temporal beliefs for a specialized search player;
2. explicit root beliefs for conditional search/advice; and
3. positional condition tags for the neural student.

The missing composition was not named as an unfinished instrument.

R1 later succeeded via INT-15 (`23a4c8b`), a post-hoc curated deterministic
search fixture whose own report disclaims strategy-strength or cross-seed
stability. It was not a trained policy changing action from its own or supplied
belief. R3's INT-17 run retained a systems failure rather than calibration
curves, and the exact-range player remained absent from the first retained
arena rating.

## How the current branch design differs

### It does not change these accepted laws

The current design preserves the durable plan of record:

- managym defines authoritative worlds and typed query meaning;
- manabot owns probability, beliefs, memory, policy/value, and search;
- `BeliefState` is a normalized distribution over the managym domain;
- actual hidden truth is calibration evidence only;
- queries induce beliefs rather than becoming strategy features;
- policy and value are belief-conditioned; and
- search planners remain honest about determinization versus
  information-set consistency.

### It changes the keystone

The recorded plan built conditional teacher and student plumbing before the
agent's belief model. The current design makes this indivisible path first:

```text
b(t-1) + viewer history -> generated b(t)
                         -> generated-or-supplied belief seam
                         -> policy/value
                         -> optional search
```

That is the main difference. Belief formation becomes ordinary agent behavior,
not a specialized player or a later head attached after teacher infrastructure.

### It chooses a model-facing representation without redefining belief

Earlier INT-10 left the belief-head representation open. INT-9 then rejected
marginals as a canonical world model. The current design resolves both by
introducing two levels:

- `BeliefState`: exact semantic distribution/scorer, used for queries,
  conditioning, sampling, identity, and evidence;
- `BeliefTensorView`: deterministic per-owner/zone/card count marginals and
  global diagnostics used only by policy/value inference.

This is new specificity, not a change to world or query semantics.

### It formalizes intervention as an agent capability

The accepted architecture could evaluate `b_Q`, but it did not define an
ordinary-agent API that generated `b(t)` and an explicit API that replaced it
for one decision. The current design requires both paths to converge on the
same decision core and requires a supplied generated belief to reproduce the
autonomous result exactly.

### It permits shared parameters while forbidding a hidden bypass

Historical docs named a `BeliefHead`, `PolicyHead`, and optional separate
history memory, but did not decide parameter sharing or causal override
semantics. The current design shares semantic embeddings/current-state
encoding while prohibiting private recurrent belief activations from bypassing
the explicit belief seam. Joint policy gradients into the belief updater are a
later recorded experiment.

### It is narrower than the original PBS solver vision

The 2026-07-09 design ultimately wanted both players' public ranges, per-hand
counterfactual value vectors, calibrated mixed strategies, and CFR/safe
continual resolving. The current branch does not commit to those mechanisms.
It establishes the belief-forming agent substrate and retains ordinary scalar
policy/value plus optional determinized search initially. Solver-specific PBS
work remains a later hypothesis.

## Interpretation: how the drift happened

The evidence fits a sequence of locally reasonable decisions rather than one
explicit architectural reversal:

1. **Diagnostic gating:** beliefs began as a later treatment for search
   failure, even though the belief design considered them fundamental to
   correct value.
2. **Parallelization:** the teacher/student Project was authorized to progress
   on the existing ABI without waiting for the complete agent architecture.
3. **Stale task premise:** INT-14 retained the obsolete “beliefs are dormant”
   premise after INT-10 had superseded it.
4. **Smallest plumbing shortcut:** a positional condition tag was much cheaper
   than choosing and integrating a real belief representation.
5. **Semantic overloading:** “belief-conditioned” came to describe exact
   posteriors, query-restricted root distributions, and positional condition
   tags, even though those are materially different things.
6. **Results-first closure:** once each narrow path had an instrument, the plan
   prioritized artifacts and deferred the learned head. It did not recognize
   their missing composition as a product-sized gap.

The plan therefore drifted in implementation and sequencing while its top-level
architecture continued to state the intended belief-forming policy/value
model.

## Tensions

- **Build versus proof:** the results-first correction was justified by too
  much substrate without results, but the missing agent composition is not
  merely more infrastructure; it is the behavior the substrate was supposed
  to enable.
- **Exact semantics versus practical inference:** exact `BeliefState` support
  is excellent for authority, queries, and search, but production likelihood
  updates already hit a severe materialization cost. Policy needs a cheap view
  without pretending marginals preserve the world distribution.
- **Counterfactual product versus autonomous agent:** advice made explicit
  belief comparisons visible quickly, while ordinary policy play remained
  unaware of the tracked belief.
- **Shared representation versus causal intervention:** sharing semantics is
  efficient, but an opaque history path can make a belief override
  uninterpretable unless the activation boundary is explicit.
- **Frozen compatibility versus cleanup:** INT-14 artifacts must remain
  reproducible, but their positional condition encoding should not remain the
  forward architecture.

## Complexity and quality observations

### Complexity

- Belief meaning is distributed across `managym` world/query types,
  `manabot/belief`, conditional search adapters, advice resolvers, shard
  adapters, and the neural Agent. The hard part is composition and identity,
  not creating another probability container.
- Conditional search retains a legacy fixture world/query protocol beside the
  canonical `BeliefState` path. This is justified for frozen evidence but
  makes the current conceptual surface look more duplicated than the target.
- Exact likelihood evaluation is the dominant known systems risk: the INT-17
  path proved accidental `O(S^2)` support enumeration at real support sizes.

### Quality

- World/belief identity, normalization, viewer safety, semantic Commands, and
  immutable evidence receive strong fail-closed coverage.
- Experiment reports are unusually honest about claim boundaries: INT-14 says
  plumbing only, INT-15 says curated fixture only, and INT-17 retains a systems
  failure rather than manufacturing calibration evidence.
- Naming is the largest quality failure. “Belief-conditioned student” names a
  positional tag path that cannot express a belief, while the actual
  `BeliefState` never reaches that student.

### Potential

Most of the difficult authority work already exists. A belief-forming neural
agent can reuse canonical world support, exact priors, temporal tracker
receipts, query support, conditional search, and semantic action alignment.
The genuinely new seam is the deterministic model projection plus autonomous
and supplied-belief decision APIs.

## Open questions

- Should policy/value receive viewer history only through `BeliefState`, or
  also through a separate explicit non-belief memory? The new design chooses
  no opaque belief-carrying bypass for the keystone, but a future public-memory
  channel may still be useful.
- Should the first autonomous model recompute the compatible prior from full
  history or carry `BeliefTracker` state between decisions? Both can implement
  the same `BeliefModel` contract; cost and replay behavior differ.
- At what content size should the policy view move beyond per-card marginals?
  The canonical source must remain correlation-aware regardless.
- Which frozen INT-14 loaders/checkpoints need compatibility support, and can
  the production `Agent` remove the silent neutral fallback immediately?

## Recommendations

### Reclassify the plan-of-record gap

**Observation:** main has belief tracking, conditional search, and conditioned
student plumbing, but not a belief-forming general agent.

**Cost:** documentation and roadmap correction only.

**Benefit:** prevents “substrate landed” from hiding the missing composition.

**Verdict:** worth doing. Name the current state as “belief/query/search
instruments landed; fundamental policy integration missing.”

### Use the current branch keystone as the convergence slice

**Observation:** the new design composes existing authoritative pieces instead
of creating another ontology.

**Cost:** non-trivial Agent/data-path work plus latency measurement; lower than
rebuilding beliefs or queries.

**Benefit:** produces the agent architecture accepted on 2026-07-17 and makes
both autonomous play and interventions real.

**Verdict:** worth doing before another teacher-only or advice-only result.

### Freeze `condition_index` as historical compatibility

**Observation:** its evidence contracts are valuable, but its semantics cannot
generalize across cards, zones, query syntax, or learned beliefs.

**Cost:** keep legacy readers/tests while moving new datasets and checkpoints
to belief identities and tensor views.

**Benefit:** preserves frozen evidence without allowing the shortcut to define
new model architecture.

**Verdict:** retain for replay only; do not emit it from new training paths.

### Keep teacher results downstream of the agent boundary

**Observation:** teacher policy/value targets and belief overrides are
currently conflated in language even though the code treats them separately.

**Cost:** schema and naming cleanup in the next conditional dataset design.

**Benefit:** one decision problem `(Observation, BeliefState, legal actions)`
can be served by policy-only, search, teacher generation, and Study without
role-specific belief semantics.

**Verdict:** adopt as the next teacher/distillation contract.
