# MTG-241 decisions — 2026-11-04

Jack Heart authorized one composable-rules PR, stopping at interactive review;
no merge, auto-merge or experiments. Directive snapshot:
`.lf/prompts/20261104-121409-da34bb73-ee11-4664-ae5a-7799bda7c435-aryo5a-summary.md`.
Inherited T7 history remains at `4463e57f5:scratch/questions.md`.

The [plan](mtg-241-composable-rules-execution.md) owns the implementation decisions,
coordination, evidence and remaining deletions. No product judgment or new
permission is needed. Actual linked-entry replacement suspension is now implemented with a passing
Command witness; complete hidden selection and versioned projection remain open.

Reversible implementation choice: top and bottom ordering share `OrderLibrary`;
top keeps its inspected prefix visible while bottom retains remaining-only
visibility. Existing bottom code 23 stays fixed; top code 27 is appended.
No real current definition prints Surveil above 1, so a constructed rules program
proves arbitrary kept order while Joo Dee verifies existing real-card behavior.
Surveil now retains the graveyard partition until selection finishes, moves it
through one retained event batch, and emits its keyword event after
kept ordering, including empty-library completion. Planetarium
matches both Scry and Surveil.

Full event/result and policy-input version choices remain implementation work.
Shared public-result selection appends obligation code 28 (retiring legacy 5);
MTG-243 received that reservation through its Task comments. MTG-244 received
the concrete schema-coordination question; the latest shared MTG-244 risk directive names v8, but its concrete schema
is not yet present in the shared Task comments. This Task has not assigned a
competing v8. OrderReplacement appends obligation code 29.

## Sync with main — 2026-11-04

Resolution assumption: this checkout keeps MTG-241's active decisions above.
Incoming MTG-213 pilot decisions, concept review and CI evidence remain at
`8e8d702f807d61675fc1bca46417eee69316daa0:scratch/questions.md`.
That Task's experiment and delivery authorization does not extend this Task's
interactive-review boundary. Main's implementation changes are retained.

Shared selection currently migrates public mill results. Hidden hand/search
migration must update retained commitments during sampling, not only the active
action space; the plan records that concrete ownership constraint. No new human
judgment is needed.
