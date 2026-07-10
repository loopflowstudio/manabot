# Rules

> **Status: parked** (2026-07-10, owner call — parallelizable but not now;
> resumes when an experiment pulls a capability or the owner unparks it).
> Destination unchanged (updated 2026-07-09): The destination
> pool is the owner's cube — https://cubecobra.com/cube/list/elemental — and
> the path is capability-ordered:
>
> 1. **Milestone 1 — the two-deck slice** (UR Lessons vs GW Allies, lists in
>    review): implement the union of the two decklists, trace-tested. Sits
>    almost entirely inside the audit's evergreen substrate (targets, EOT
>    modifiers, tokens, +1/+1 counters, the trigger family, activated costs)
>    plus three pulled extras (waterbend, multi-target, exile-until-leaves).
>    Unlocks games in the cube's real texture AND the first non-mirror matchup.
> 2. **Milestone 2 — TLA commons complete** (audit verdict, `00-pool-audit.md`:
>    TLA dominates FIN on every cube-relevant axis — 34 TLA commons are
>    literally cube cards, closure 58 vs 65, and the bendings are the cube's
>    identity mechanic on 59 unique cube cards). Completing it puts the cube
>    at ~80% weighted expressible. Earthbend arrives here regardless of any
>    two-deck-slice decision.
> 3. **Milestone 3 — full cube closure:** the tail is MULTIFACE (12% of cube,
>    in neither commons set), sagas, replacement effects, vehicles, X-costs —
>    "done" is defined by ten named cards (The Legend of Kuruk et al., see
>    audit §Hardest).
>
> Still no open-ended rules grinding: every rung is pulled by a named pool
> with a named payoff. Completeness of the CR remains a non-goal;
> completeness of *this cube* is a finish line.

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
