# Decisions and assumptions

No missing input blocks this gate. Reuse the previous gate's checks because
`git diff 349d643e52 HEAD -- . ':!scratch'` is empty. Prepare the corrected PR
copy locally for delivery; publication is outside this gate step.

The accepted scope is recoverable at
`fe30e33d62:scratch/mtg-320-plan.md`. Library and hand retain one ordered
checked-card/transient-object sequence. Graveyard/exile use mixed ObjectId
lists narrowed by card-only readers. The physical-card store, new objects per
zone change and carry-over records remain separate follow-up work.

Preserve the exact direction: "Stop at review. Do not enable auto-merge."
