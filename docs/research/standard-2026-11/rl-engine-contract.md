# Why managym's engine contract suits reinforcement learning

Research date: 2026-11-04. Jack Heart asked which engine-level choices make
manabot more suitable for RL than Phase. This is an architectural comparison,
not a measured training, throughput or playing-strength comparison.

The strongest advantage is that a mechanic is expected to be executable,
observable and learnable in the same change. A fast game engine cannot train a
policy to distinguish states or actions its input represents identically.

## The strongest concrete differences

**1. Learnability is checked at the rules boundary.** The exhaustive guards in
[`observability.rs`](../../../managym/cards/tests/cards/observability.rs) require
new rule/state variants to name their tensor and learning-token representation.
Directed tests distinguish otherwise-equal states and protect viewer privacy.
Native and Python encoders have parity checks. These mechanisms reduce the risk
of training for hours against an observation that omitted the relevant fact.
They are not a theorem that the input is sufficient for every strategy: the
retrospective includes past action collisions and capacity failures.

**2. Card programs are model inputs derived from the executable definition.**
[`cards/tokens.rs`](../../../managym/src/cards/tokens.rs) traverses the same
`CardDefinition` the engine runs. Effect structure, values, predicates and
references to created-token/back-face definitions reach learning without a
second authored rules vocabulary. Printed card names and Oracle text do not
stand in for mechanics. This provides a basis for sharing learned behavior
across similar cards; actual generalization still requires empirical evidence.

**3. Collection is a native batched workload.**
[`VectorEnv`](../../../managym/src/agent/vector_env.rs) owns batched game roots,
reset seeds and stepping. Its [Python bindings](../../../managym/src/python/vector_env_bindings.rs)
write encoded observations, action masks, rewards and termination flags into
caller-owned arrays with the GIL released. The interface avoids making a
Python/UI round trip for every internal rules operation. Native trivial-decision
skipping means the policy acts at actual alternatives rather than being asked
to select the sole legal action. Measure both samples per second and actual
decision/game work; skipping does not make every trajectory equally informative.

**4. Reproducibility includes the game's meaning.** World/content/input versions
bind checkpoints and retained evidence; old omitted configuration values retain
their historical interpretation. The [semantic decision contract](../../../managym/src/decision.rs)
binds commands to their exact legal offers and revision. The
[search witness](../../../managym/src/search_state.rs) checks mutable rules
authority, legal surface, fixed-viewer projections and RNG diagnostics
separately. These boundaries help distinguish a learning change from a changed
game, input or replay setup. They also make failure prefixes actionable.

## Valuable foundations that are not unique to us

Phase also has typed actions, explicit pending choices, exact-incarnation work,
hidden-information handling, deterministic execution and search. We should not
claim it lacks those capabilities. Their presence alone does not establish our
comparative RL advantage.

Managym's explicit [Observation](../../../managym/src/agent/observation.rs),
shared Command surface and [possible-world authority](../../../managym/src/possible_worlds.rs)
are nevertheless important to preserve. Policies, humans and search should not
get conflicting legal choices or privileged information. A possible world must
respect the viewer's known cards and configured disclosure; it is not just a
random shuffle of everything unseen. That authority currently has a bounded
supported information/query domain, not a complete learned belief model for
every future mechanic.

Exact forks include pending payments, choices, effects, history and RNG, while
sharing immutable definitions and appropriate caches. Forking is useful for
search-based targets, counterfactual evaluation and debugging. Phase's persistent
containers pursue a related goal through a different storage strategy. No
controlled comparison here establishes either strategy as universally faster.

## What the Standard work must preserve

New layers, permissions, counters and history must reach the actual policy
input. Wider legal choices must remain representable without shortlists or
silent fallback. Every newly suspendable operation must be cloneable at the
choice boundary; the [counter-replacement example](phase-architecture.md#concrete-example-what-suspend-means)
shows why that affects execution architecture.

Catalog growth also needs separate accounting from per-game work. The existing
model encodes referenced definitions during learning and can cache the full
definition table within an inference scope; preserve that separation. A larger
card file should not force unrelated card programs through every learner step.
Engine benchmarks remain separate from end-to-end model/learner throughput.

The comparison's strongest conclusion is that our current engine is more
directly integrated with RL's input, action and transition requirements. Phase's
broader rules coverage exposes semantics we still need. Both matter: complete
rules that the model cannot observe and excellent tensors over incomplete rules
are each insufficient for the intended Standard milestone.

Validation: current local interfaces and the pinned Phase architecture review;
no new execution, training, benchmarks or performance ranking.
