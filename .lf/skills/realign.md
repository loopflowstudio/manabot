---
extends: builtin
---
## In this repository

**The next slice comes from the plan.** Name the plan's next unbuilt slice,
whole; don't shrink it to one card. Iterate while the phase has unbuilt
slices and advance when none remain: the phase's review then reads the whole
batch at once.

**Keep the plan short.** Every later step reads all of `scratch/plan.md`.
Collapse a finished slice to one line with its commit; its detail is in git.
Keep the remaining slices, open defects with their reproductions, and what
people said.

**Reuse what implement ran.** Read the slice's commits and the results
implement recorded; don't rerun its checks or reread finished slices.

When a slice depends on another Task's release, record the version or artifact
that unblocks it and the first command to run once available. Finish the step
without waiting inside it or returning to implementation while the release is pending.
