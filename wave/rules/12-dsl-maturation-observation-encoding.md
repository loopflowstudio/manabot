# 12: DSL Maturation + Observation Encoding

## Finish line

The declarative effect DSL feeds the observation space. The agent can see
what cards do structurally and generalize across cards with shared mechanics.

## Changes

### Effect Tree → Observation Features

Flatten declarative effect trees into fixed-width feature vectors for the
neural network. Each card/permanent in the observation includes effect
features alongside existing features (P/T, mana cost, card type, zone).

Design dimensions:
- Feature granularity: "has an activated ability" (1 bit) through full
  effect decomposition (trigger condition type, effect type, target filter,
  value)
- Fixed-width encoding: effect trees vary in depth/breadth, but observation
  features must be fixed-size per object for the neural network
- Coverage: encode effects for cards on battlefield, stack, and hand (agent
  needs to plan sequences)

### Transfer Learning Validation

Test the hypothesis that structured effect features enable cross-card
generalization:
- Train agent on card set A (e.g., Bolt + Ogre + Elves)
- Evaluate on card set B with unseen cards sharing mechanics (e.g., Shock
  instead of Bolt, different creatures with same keywords)
- Compare vs baseline agent without effect features

### DSL Expressiveness Audit

Review all implemented cards against the DSL. Document:
- Cards fully representable declaratively
- Cards requiring special handling or DSL extensions
- Patterns that appear frequently and deserve first-class DSL support

## Done when

- Observation space includes structured effect features for all cards.
- Transfer learning experiment shows positive generalization signal.
- DSL expressiveness audit complete with documented gaps.
