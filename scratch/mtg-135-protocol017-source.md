# Exact protocol017 source retained in this checkout

The complete source used for MTG-135's protocol017 is retained here, independently
of another worktree or a branch retaining its original commit:

- Archive: [source.pack](../.runs/mtg135-handoff/protocol017-source/source.pack),
  105,942,147 bytes, SHA256
  `5c41c3af06513e8c3bfc3569e38bd6298295d29ca3811282fdbe77a19153468b`.
- Commit: `3c3681d2b26a7fc8167928fa63fe24a6cd3d058a`.
- Tree: `bd880479f77bf1d506af1db1f3cee813bfb2be4f`.
- [Verification receipt](../.runs/mtg135-handoff/protocol017-source/receipt.json):
  isolated Git import and strict object verification passed; 2,208 tracked files
  and 2,378 objects. No source checkout or scientific execution was performed.

This is a shallow source snapshot; its parent history is intentionally omitted.
The archive includes repository source and dependency locks, not the separately
verified native binary, installed environment, private run checkpoints or full
evaluation traces. Those remain separately identified inputs. The operational
logging patch is retained in the [Mini evidence package](mtg-135-mini-operations-20261018/README.md).

For source inspection, from this Task checkout use a new temporary repository:

```sh
mtg135_source_checkout=$(mktemp -d /tmp/mtg135-protocol017-source.XXXXXX)
git init "$mtg135_source_checkout"
git -C "$mtg135_source_checkout" index-pack --stdin < .runs/mtg135-handoff/protocol017-source/source.pack
printf '%s\n' '3c3681d2b26a7fc8167928fa63fe24a6cd3d058a' > "$mtg135_source_checkout/.git/shallow"
git -C "$mtg135_source_checkout" checkout --detach 3c3681d2b26a7fc8167928fa63fe24a6cd3d058a
git -C "$mtg135_source_checkout" fsck --strict --no-reflogs
```

This does not integrate code into MTG-122 or admit its next experiment. The exact
recipe transfer still needs the [documented schedule/plan integration](mtg-135-longer-learning-017-preparation/README.md),
full-panel result review, research-catalog identity checks and fresh cost admission.
No model is selected by retaining source.
