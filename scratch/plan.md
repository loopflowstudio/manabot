# MTG-289 — abandoned; draft removed

The filed-in-error steer is quoted in `scratch/questions.md`. Existing
`Env.hand_snapshot` and `VectorEnv.hand_snapshots` remain the supported path
(`managym/src/agent/env.rs`, `managym/src/agent/vector_env.rs`). Library counts,
both seats in one call and definition-indexed buffers await a consumer request.

The unimplemented
`hidden_zone_counts_into_buffers` declarations and their exclusive import from
`manabot/managym/__init__.pyi`, plus the guide section, example, navigation link
and API table row from `docs/engine.md` have been removed, along with the draft
implementation/tests. No runtime behavior changed.

Reconciliation on 2026-11-11: `lf task status MTG-289 --json` reports
`state: abandoned`, no PR and no Wave, with the same cancellation steer and no
later direction. Nothing remains to implement or publish for this Task; the
library/buffer extensions above are not delivered by the existing hand API.
No product decision is open.

Validation: `git diff --exit-code 668555c09 -- . ':(exclude)scratch'` — passed;
all tracked product, test and documentation files match the checkout base.
No build or runtime tests were rerun for this context-only reconciliation.

Concept review on 2026-11-11: the fresh Task read still reports abandoned,
with no later direction. No design change or open product choice remains.
The cancelled brief's library/buffer requirements are not a reason to resume
implementation. Loop-or-next outcome: nothing to change and no open choice;
preserve the cancellation, with no implementation or publication work.
