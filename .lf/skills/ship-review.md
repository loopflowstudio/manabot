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

**Ask what the player sees under each option.** When a rules choice affects
linked cards, use a named card example and say where each card ends up. Explain
engine terms on first use. Number each choice the same way in chat and on the
page so a reply like `4b` settles it. Offer “fix here” or “leave” for deferred
findings and wait for the answer before filing a follow-up Task.
