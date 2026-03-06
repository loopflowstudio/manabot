# Structure

## Vision

Improve the engine's internal architecture independent of feature or card
expansion. Every sprint here makes the existing engine cleaner, more
observable, and harder to break — without adding new rules, cards, or
mechanics.

This wave was motivated by a comparative analysis against XMage, Forge,
Magarena, and MageZero. The key finding: our RL interface and state model
are ahead of the field, but the agent can't see what spells do before
casting them.

### Not here

- New cards or card types
- Migrating spell resolution from string-matching to Effect enum dispatch
- Triggered ability improvements
- Continuous effects / layer system
- New keywords or mechanics
- Changes to training, PPO, or model architecture

## Strategy

Sprints 01-03 are shipped: `game.rs` decomposed to 66 lines, stack/zone
unification with debug invariants, and event observations (dual-drain
accumulator, `EventData` in observations, matching Rust/Python encoders).

The remaining sprint (04) adds static spell metadata — what spells do,
not what happened. This is orthogonal to the event system and touches
only card definitions, observation structs, and encoders.

## Goals

1. The agent can distinguish spells by what they do, not just their stats.
2. All existing tests pass unchanged after every sprint.

## Risks

- **Card feature dimension change** affects training. Any trained models
  need retraining after `CARD_DIM` increases. This is expected and
  acceptable for the current stage.

## Metrics

- Event observation coverage (events appear for all major game actions)
- No regression in SPS or test pass rate
