# wave/

Manabot's work is organized into **waves**: focused portfolios with a charter
(`GOAL.md`), running context (`MEMORY.md`), and numbered working documents.
A wave owns a surface and its contracts; tasks that change another wave's
contract live in that wave and appear here as dependencies. Charters bind to
Linear via their frontmatter.

Cross-wave authority and convergence are defined in the top-down
[Manabot architecture](../docs/ARCHITECTURE.md). Wave roadmaps schedule
provider and consumer work without redefining those contracts.

| Wave | Owns |
|---|---|
| [intelligence/](intelligence/GOAL.md) | manabot training: self-play, policy/value learning, beliefs, search, and arena evaluation |
| [rules/](rules/GOAL.md) | managym authority: curated semantic worlds, match/Command/replay, canonical Observations, possible-world queries, and safe high-throughput forks |
| [game/](game/GOAL.md) | Academy as a lab for researchers: load a trained manabot, watch it play or play against it, and inspect its decisions |

Retired portfolios are preserved under [archive/](archive/README.md) with
their dispositions.
