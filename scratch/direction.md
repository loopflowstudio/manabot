# c5fba4 direction

Task comment `694f4190`, 2026-11-05 (Jack Heart). It supplements the brief and
was posted after the Task's launch, so the first `implement` step's context
did not include it.

- One seed per model size.
- Prioritize regular evaluations during training, not only at the end. At each
  milestone, play the two sizes against each other (at matched updates/compute)
  and each against the heuristic, so progress is visible as it goes.
- Evaluation stays within the USD100 and a fraction of training cost, on rented
  CPU.
