---
extends: builtin
---
## In this repository

**The next slice comes from the plan.** Name the plan's next unbuilt slice,
whole; don't shrink it to one card. Iterate while the phase has unbuilt
slices and advance when none remain: the phase's review then reads the whole
batch at once.

When a slice depends on another Task's release, record the version or artifact
that unblocks it and the first command to run once available. Finish the step
without waiting inside it or returning to implementation while the release is pending.
