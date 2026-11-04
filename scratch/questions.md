# MTG-241 decisions — 2026-11-04

Jack Heart authorized one composable-rules PR, stopping at interactive review;
no merge, auto-merge or experiments. The directive snapshot is in
`.lf/prompts/20261104-121409-da34bb73-ee11-4664-ae5a-7799bda7c435-aryo5a-summary.md`.
The inherited T7 note remains historical at `4463e57f5:scratch/questions.md`.

Nested iteration owns a fresh result scope; branches/sequences within it share
that scope. Parent results and finalization stay suspended until every child
completes. Native Effect remains the only authored program. This internal cut
and both contextual-read fixes are implemented and tested, not the full Task.

The proposed frame-local replacement continuation proved insufficient: a mana
cost's sacrifice can return an exiled card whose entry asks a replacement
question; synchronous payment and zone postambles would overwrite that question.
A retained completion protocol across effect, event-batch and payment owners is
the unresolved architecture boundary. The [plan](mtg-241-composable-rules-execution.md)
records exact callers, the revised proposal, deletions and remaining acceptance.
Replacement and selection executors remain unchanged pending that design review.
MTG-242 received the concrete payment-interface finding; MTG-243 the frame shape.
MTG-244's layers remain untouched. No competing Tasks or worktrees were created.
