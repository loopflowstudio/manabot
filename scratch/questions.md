# MTG-241 decisions — 2026-11-04

Jack Heart authorized one composable-rules PR, stopping at interactive review;
no merge, auto-merge or experiments. Directive snapshot:
`.lf/prompts/20261104-121409-da34bb73-ee11-4664-ae5a-7799bda7c435-aryo5a-summary.md`.
Inherited T7 history remains at `4463e57f5:scratch/questions.md`.

The [plan](mtg-241-composable-rules-execution.md) owns the implementation decisions,
coordination, evidence and remaining deletions. No product judgment or new
permission is needed. Actual linked-entry replacement suspension remains unbuilt.

Reversible implementation choice: top and bottom ordering share `OrderLibrary`;
top keeps its inspected prefix visible while bottom retains remaining-only
visibility. Existing bottom code 23 stays fixed; top code 27 is appended.
No real current definition prints Surveil above 1, so a constructed rules program
proves arbitrary kept order while Joo Dee verifies existing real-card behavior.
This does not claim atomic Surveil movement or its full keyword-event contract.

Full event/result and policy-input version choices remain implementation work.
