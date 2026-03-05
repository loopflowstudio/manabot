# Rules

## Vision

Grow managym into a materially fuller implementation of Magic’s rules while
keeping every expansion testable, attributed to CR references, and shippable in
small diffs.

The path starts with visibility (what is implemented vs not), then builds rule
systems in dependency order: priority/stack, triggers, SBA depth, layers,
replacement, and card-driven validation.

### Not here

- Multiplayer/casual variants (8xx/9xx)
- Automated upstream CR sync process (defer until core work is stable)
- Compatibility mode split runtime

## Goals

1. Every implemented rule family has focused CR-cited Rust tests (plus negative paths).
2. Lightning Bolt and Man-o'-War land early to force stack/target/trigger behavior.
3. Rule expansion stages remain independently shippable (~500–1000 LOC per diff target).
4. Training remains stable as branching factor and interaction depth grow.

## Risks

- Rule-family coupling causes oversized refactors.
- Card additions outpace engine semantics, creating false confidence.
- RL instability from larger action spaces and longer horizons.
- Ambiguous rule ownership without explicit CR citations.

## Metrics

- `rules_coverage` entries with status `implemented_tested` (count, target +N per stage)
- CR-cited Rust tests added per stage (count)
- Negative-path rule tests per stage (count >= 1 per family)
- Invalid-action rate during training after each rules milestone (%)
- Mean episode length and truncation rate before/after milestones
