# Structure

## Vision

Improve the engine's internal architecture independent of feature or card
expansion. Every sprint here makes the existing engine cleaner, more
observable, and harder to break — without adding new rules, cards, or
mechanics.

This wave was motivated by a comparative analysis against XMage, Forge,
Magarena, and MageZero. The key finding: our RL interface and state model
are ahead of the field, but `game.rs` is a god object, the stack has a
dual-tracking desync risk, and the agent can't see what happened between
its decisions or what spells do.

### Not here

- New cards or card types
- Migrating spell resolution from string-matching to Effect enum dispatch
- Triggered ability improvements
- Continuous effects / layer system
- New keywords or mechanics
- Changes to training, PPO, or model architecture

## Goals

1. No method in `game.rs` knows a card name.
2. Stack membership is tracked in exactly one place.
3. The agent sees recent game events in its observation.
4. The agent can distinguish spells by what they do, not just their stats.
5. All existing tests pass unchanged after every sprint.

## Risks

- **Decomposing game.rs** is mechanical but tedious. Risk of introducing
  bugs through mis-scoped `pub` visibility or broken imports.
- **Stack unification** touches zone management, which is load-bearing
  for observations. Desync bugs during the refactor would be ironic.
- **Event observations** add to the observation tensor size, which could
  affect training throughput or destabilize existing trained models.

## Metrics

- `game.rs` line count (target: <400)
- Stack/zone consistency assertions (zero failures across full test suite)
- Event observation coverage (events appear for all major game actions)
- No regression in SPS or test pass rate
