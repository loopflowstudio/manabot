# Implementation notes

- `lf rebase --plan` now resolves a direct rebase onto `origin/main`, but
  `lf rebase` is blocked by live writer
  `writer_ba61f2a7-6ce8-4157-89b7-70e7cfa673eb` (PID 64829), an active
  human design Ask in this worktree. The branch's compiled extension exposes
  none of the current possible-world or semantic-step methods. The Python
  implementation now targets the native APIs on current main and the demo no
  longer has a fixture fallback; native proof must remain red until that writer
  releases the worktree, the rebase completes, and the extension is rebuilt.
- The current visible-card encoder has no `CardDefId`/name identity to share
  with a belief row. Reusing its dense projection would erase the distinction
  between cards with the same public numeric features. The keystone therefore
  shares the current-state attention/decision core while using an explicit
  schema-bound card-definition embedding for belief rows. A later content
  vocabulary migration can make the card embedding itself shared without
  weakening the intervention boundary.
