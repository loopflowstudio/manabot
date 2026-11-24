---
requires: a decomposition plan, the changed consumers and their behavior checks
produces: composability findings in `scratch/plan.md`
action_style: procedural
---
Judge whether real cards compose through one rules authority without card-specific engine logic.

1. Follow at least two distinct representative consumers through the shared
   mechanism. Confirm the planned deletions removed their exclusive
   predecessors, tests and data paths while keeping required behavior on the
   surviving owner.
2. Flag card-specific dispatch, duplicated rules state, adapters and unused
   abstractions. A smaller diff or a renamed type doesn't prove cards
   compose.
3. Reuse the current head's focused test results; leave full suites to CI.
   Write concrete findings into `scratch/plan.md` without editing code.

When deciding whether to continue:

- **Iterate** with the remaining duplication, unbuilt slices or any rules
  defect (a rules defect voids dependent numbers and fixes the contract
  first).
- **Advance** when the criterion holds.
