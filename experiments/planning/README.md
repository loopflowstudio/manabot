# Training planning records

Retained records of the September 2026 training plan, which ranked width-64
search distillation first. Manabot no longer trains policies from search
labels; self-play from terminal reward is the training path. Nothing here is a
current plan, a strength measurement or an execution authorization.

| File | What |
|---|---|
| [training-evidence.md](training-evidence.md) | Primary-source research behind the method ranking |
| [grid.md](grid.md), [grid.json](grid.json) | The eight time × cash cells the budget calculator produced |
| [prices.json](prices.json) | The dated price snapshot those cells used |

The calculator (`scripts/budget_plan.py`) and the supervised consumers named in
`grid.json` were removed; regenerating these files requires the source pinned
at commit `07a091bca`. The [accepted plan](../../docs/plans/strongest-manabot-training.md)
records the decision they informed.
