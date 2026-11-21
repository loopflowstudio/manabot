---
extends: builtin
---

When a slice depends on another Task's release, record the version or artifact
that unblocks it and the first command to run once available. Finish the step
without waiting inside it or returning to implementation while the release is pending.
