# Rules

## Vision

Grow managym into a materially fuller implementation of Magic's rules while
keeping every expansion testable, attributed to CR references, and shippable in
small diffs.

The path starts with visibility (what is implemented vs not), then builds rule
systems in dependency order: event system, priority/stack, targeting, triggers,
keywords, SBA depth, layers, replacement, and card-driven validation.

Two innovations distinguish this from other MTG engine efforts:

1. **Declarative effect DSL** that is both executable by the engine and
   encodable into the observation space — the agent can *see* what a card does
   structurally and generalize across cards with shared mechanics.

2. **Trace-based test harness** where rule tests are scenario data (JSON), not
   bespoke code. Scales to hundreds of rules without proportional test code
   growth.

### Not here

- Multiplayer/casual variants (8xx/9xx)
- Automated upstream CR sync process (defer until core work is stable)
- Compatibility mode split runtime
- Multi-target spells (single-target first, extend later)
- Cancel/rollback of in-progress casting (engine guarantees legal actions)

## Goals

1. Every implemented rule family has focused CR-cited trace tests (plus negative paths).
2. Lightning Bolt and Man-o'-War land early to force stack/target/trigger behavior.
3. Keyword abilities batch expands strategic depth cheaply after structural work.
4. Rule expansion stages remain independently shippable (~500-1000 LOC per diff target).
5. Training remains stable as branching factor and interaction depth grow — smoke
   tested every stage, not bolted on at the end.
6. Declarative DSL enables cross-card generalization in the observation space.

## Risks

- Rule-family coupling causes oversized refactors.
- Card additions outpace engine semantics, creating false confidence.
- RL instability from larger action spaces and longer horizons.
- Ambiguous rule ownership without explicit CR citations.
- DSL design locks in too early before enough cards exercise it.

## Metrics

- `rules_coverage` entries with status `implemented_tested` (count, target +N per stage)
- CR-cited trace tests added per stage (count)
- Negative-path rule tests per stage (count >= 1 per family)
- Invalid-action rate during training after each rules milestone (%)
- Mean episode length and truncation rate before/after milestones
- Average branching factor per decision point (tracked from stage 01)
- Action space size distribution (tracked from stage 01)
