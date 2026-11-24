---
extends: builtin
---
## In this repository

For engine, RL and training code (`managym` and `manabot` outside Academy and
CLI surfaces), the consumer is the algorithm or caller, not a human operator.
Judge concepts, APIs and data layouts by throughput, memory and cache
ownership, thread safety, statistical validity and composability rather than
human UX. Still challenge opaque wrapper names that hide what a type is and
arbitrary numeric caps or tested-grid `Literal` sets lacking a structural
invariant.

**Skip a pass that adds no concept.** In a headless run, when the work since
the last review only authored cards or moved callers onto types, records and
APIs that `scratch/concept-review.html` already covers, report "nothing to
change" in one line and leave the page alone. Review, and rewrite the whole
page to match `HEAD`, when the work added or changed a rules type, an
observation or token kind, a Command or an offer shape.
