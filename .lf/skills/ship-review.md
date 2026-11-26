---
extends: builtin
---
## In this repository

**Diagnose engine performance impact.** When the diff touches `managym/`,
report the `Engine Budget` numbers (steps throughput, search total, worst reply,
digest changes) alongside hot-path or allocation changes. When any metric
regresses, inspect the CI artifact's per-seed/per-sim breakdown and the diff's
hot paths before asking what to do—separate trajectory shifts from per-step
overhead instead of punting investigation to `deliver`/`gate`.

**Ask in concrete behavior, not code symbols.** Write each numbered question (in
chat and on the page) as what the player or user sees under each option rather
than internal field or function names, and give each distinct choice its own
number so a bare reply like `4b` needs no follow-up.
