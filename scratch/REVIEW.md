# Belief-forming manabot — branch review

The full design is
[`search-learning-architecture.md`](search-learning-architecture.md). This is
the short review path.

## Core decision

Belief generation is fundamental to a manabot:

```text
previous belief + new viewer-safe history
                    -> current belief
                    -> policy/value
                    -> optional search
```

The teacher does not own a special conditioning channel. Search becomes a
teacher only when its policy/value results are recorded as training labels.

The generated belief is sometimes called the search prior. It already includes
all viewer-safe history through the current decision; “prior” means before a
query intervention or rollout, not before observing the game.

Training, evaluation, and Study can explicitly replace the generated belief
for one decision. The override and autonomous paths then use the same decision
core. Policy/value teacher labels remain separate from the belief override.

## Semantic boundary

The architecture distinguishes:

```text
world language   managym defines possible hidden worlds
belief           manabot weights those worlds
query language   managym defines typed predicates over worlds
```

Queries measure and condition the full belief. The belief model may expose
per-card count probabilities to policy/value, but those are derived marginals.
They cannot replace the full distribution because they lose correlations.

Thus `Has(Bolt)` is not an input tag. It selects worlds and produces a new
normalized belief. Equivalent beliefs yield identical policy inputs regardless
of which query produced them.

## Shared parameters

The belief updater and policy/value share semantic embeddings and visible-state
encoding. They retain distinct heads and an explicit `BeliefState` boundary.
The policy receives no private recurrent belief activation that could bypass an
override.

Belief calibration and strategy losses remain separate initially. Joint
end-to-end gradients are a later measured choice, not an invisible default.

## Delivery

This is an additive series:

1. Build the fundamental belief-forming agent and intervention seam.
2. Use it to measure paired conditional search targets.
3. Distill those targets and prove a held-out policy-only action change.
4. Learn calibrated belief updates from viewer history and evaluate the full
   autonomous loop.

The keystone lands only with a real autonomous decision plus an intervention
demo. It cannot stop at data structures.

## Real proof

The keystone command must show that the ordinary agent generates a belief,
that explicitly supplying the same belief reproduces its policy/value exactly,
and that `Has(Bolt)`/`Lacks(Bolt)` changes only the supplied belief while
Observation and legal actions remain fixed.

The later research result remains the stronger claim: a preregistered,
multi-seed student reproduces stable teacher action changes on held-out roots
without search at inference time or hidden-truth leakage.
