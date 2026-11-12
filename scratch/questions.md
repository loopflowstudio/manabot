# MTG-289 cancellation — 2026-11-11

Inherited MTG-285 release history is at `4c041d6ce:scratch/questions.md`;
that completed work does not select this Task.

Comment `733e8a22-2256-4e71-9471-393d0055c913`, delivered as a steer attributed
to Jack Heart:

> Filed in error 2026-11-11: MTG-272 (PR #458, merged 2026-11-11 23:46 UTC) already added `Env.hand_snapshot(player)` and `VectorEnv.hand_snapshots(players)`. The brief was written from jace at its older pin (b5e8f404a) and a checkout that predated the merge. Not covered by #458, if a consumer ever asks: library contents, both seats in one call, and counts written into caller buffers by definition id.

Implementation stopped on this steer; its native methods and draft tests were
removed. The stopped debug test failed its fixture assertion (expected 23
library cards; actual 30). No passing runtime validation, extension replacement
or publication occurred.

The abandoned docs step passed stub/example AST parsing, Markdown fence and
whitespace checks. Its `lf open` attempt returned "No Desktop Files view
attached; nothing opened." Compression removed the unimplemented API's guide
and stub declarations; existing hand snapshots remain unchanged.
