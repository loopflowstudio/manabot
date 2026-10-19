# Storage visibility and efficiency — 2026-10-18

Jack asked for future request visibility, then clarified that SDK accounting and
storage-efficiency improvements belong in one Task. The boundary is Manabot's
usage versus framework capability. He directed immediate work that does not need
SDK changes, and asked to start the SDK Task and open its session in this cmux
window. This adds to MTG-135's ongoing recommendation work; it does not authorize
new GPU rentals or laptop evaluations.

The existing private cost Task **78b6b1** retains billing/account evidence and
Manabot adoption in `/Users/jack/src/etude-studio.reduce-experiment-transfer-costs`.
The combined public-client Task **88f96c**, "Explain and reduce experiment storage
transfers", was filed in `/Users/jack/src/etude` and placed at
`/Users/jack/src/etude.storage-transfers`, branch `jack-heart/88f96c-storage-transfers`.
Its copied design SHA256 is
`1187a160e7edfc183f455d9b604179e1f0f4552c47ac73cecfc05b793bb912f9`.
Checkout started from released SDK 0.1.2, commit
`df6ec0523c0a3c0ead9109ca81e3d199ae32bc1f`, including merged logging PR 10.

Pursue Flow `7b560d68-42b5-4150-ba99-1640430f4b2a` was launched and observed live
(PID 88923); implement Process `d14a34a3-01dc-4d71-a3a6-dbc35ca76a42`,
Session `session_21fe4aeb5a4d45769a66f187e675a40f`. It owns the sole SDK implementation.
The primary conversation is `b403fdf5e70c485a9a5cc63cef6bea76`. Its cmux client
initially failed to reconnect to Loopflow's local Codex relay; do not confuse that
display failure with the independent Flow. Reread current liveness before acting.

SDK scope: actual byte/request counts, partial and failed transfers, request-ID
correlation, conserved shared-download attribution, durable verified reuse and
resumption, offline totals with coverage, cost estimates and exact Manabot
integration proof. Private lower-egress bulk storage remains an option within the
same Task, not an authorized migration. Cold acquisition and warm reuse must be
measured separately. No private billing/account data belongs in public code.

Direct account work completed: both experiment buckets were observed with
server access logging disabled, then enabled in the AWS console to a separate
private, same-account/us-west-2, SSE-S3 access-log bucket. Block All Public Access
is on, ACLs are disabled, and AWS log delivery has write permission; no user
credentials/permissions or research artifacts changed. Records partition by
source bucket and event date. Configuration success and saved properties were
verified. Delivery was still pending at the check; AWS logs are delayed and
best-effort, so this is prospective visibility, not retroactive attribution.
Private configuration observations and the screenshot are in the cost Task's
`scratch/s3-logging-20261018/`. A CLI one-byte GET canary was denied by the existing
issuer and is not a successful transfer or log-delivery test.

The main protocol017 evaluation continues on the Mini, not the laptop. At
20:13:32 UTC, evaluator PID 68448 and audit/report PID 69462 were live; 4,184 games
were recorded and all 4,184 passed replay, with 38 completed and four running
panels and no panel errors. No model ranking is inferred from partial results.
The $5 ancillary reserve is still not verified against attributed S3 costs.

No cost savings, SDK implementation completion, PR publication or final MTG-122
training recommendation is claimed by this operational setup.

Final connection check: the same primary Session was opened in cmux workspace
`workspace:1000070048` in the existing window
`A13D14F4-A590-48DC-B734-943411818060`. The native conversation remained unusable:
Loopflow's local Codex relay reset its WebSocket connection before a provider
thread was established. Reconnecting and bypassing the optional cmux PATH
wrapper did not repair it. No provider/account upgrade, duplicate Session or
second implementation Flow was launched. The obsolete failed terminal was
closed; the current workspace is selected with the connection error visible.
This is a conversation transport blocker, not an implementation blocker.

Fresh `lf ps --json`, `lf flow show --processes`, Task status, and OS process
checks observed the sole pursue Flow and implement Process alive. The worker
reported synthetic HTTP tests covering cache reuse, partial-byte accounting,
resumed ranges and invalid responses; these are progress reports, not a final
review or shipped result. Task automation is disabled and no CI watcher runs.

At 20:22:51 UTC the Mini had 4,600 completed game records, all replay-passing,
43 completed panels and four running panels. Both evaluator PID 68448 and
audit/report PID 69462 remained live. No laptop evaluation was launched.

## SDK recovery ownership — 20:37 UTC

The first SDK pursue Flow `7b560d68-42b5-4150-ba99-1640430f4b2a` stopped after
implementation and compression succeeded, when synchronization conflicted with
newer upstream timeout/runtime changes. The existing task-run runner started its
automatic recovery Flow `702a0532-4e5e-4a59-8bbe-4b528af4c6e0`. Fresh Task records
identify implement Process `d3e2fff5-d103-4fc8-b784-b51edcd001bf`, Session
`session_e142ffcd653b40308dd80dc93c9e1901`. It is resolving both changes together
and checking SQLite usage retention across background HTTP I/O deadlines. No
additional implementation was launched by this operator. The initial Flow is
history; read all current Flows and process identity before any next action.

The worker reported 47 focused tests passing before sync, and a 24 MiB fixture
whose two reads transferred 24 MiB total, with zero payload GETs for the warm
read. Those are synthetic implementation results. Integration checks, publication,
landing, Manabot adoption and measured cloud savings remain unverified.

## Existing conversation recovered through a text client — 20:53 UTC

Exiting the failed native terminal exposed a concrete transport failure:
`Space limit exceeded: Message too long: 16791848 > 16777216`. A later resumed
attempt reported `16792112 > 16777216`. The oversized message's contents were
not identified; do not treat earlier generic socket-reset messages as its cause.
No tool upgrade or account change was made.

The supported `lf -b session connect <existing-id> --message ...` path completed
a read-only turn in the same primary Session
`b403fdf5e70c485a9a5cc63cef6bea76`. Native reconnect then displayed that history
but still failed on the size limit. Its client was closed normally.

The same cmux workspace/surface now hosts `/tmp/etude-storage-conversation.py`,
a small text client forwarding each entered line as an argument to that supported
continuation command. It was launched with the Task venv interpreter and `-I`:
an unrelated `/tmp/selectors.py` otherwise shadowed the standard library and
failed before the client started. That unrelated file was neither changed nor
removed. `/status` and a no-tool connection-check message both returned, and the
client returned to `You>` after the agent confirmed the existing Task identity.
The workspace is selected and its tab is named `88f96c conversation`. No duplicate
primary Session or implementation was created. This restores a usable text
conversation; it does not fix the native relay.

At 20:50:04 UTC the Mini evaluator and audit owner remained live, with 5,456
game records, all replay-passing, 51 completed panels and four running panels.
The SDK worker subsequently reported the combined synthetic workload proof;
its measured scoped savings are not AWS billing savings or a released SDK.

## SDK landed and completed; conversation retained — 21:30 UTC

Public Etude PR 16 merged at 21:22:38 UTC:
https://github.com/loopflowstudio/etude/pull/16 . The merge commit is
`2b3fe66e36f6638257d64af82be878ae3c0fe97a`; reviewed head was
`77a19d1b360eae80c75adefaf9eea696ed2d2a92`. Both have tree
`a569d0e25bc464709f35f20d67bc5963c2099cd8`. The exact pinned adoption command
is now in the merged PR description. The earlier `6d5aa468` implementation
does not contain the final review fixes and is superseded for adoption.

The two reproduced issues are fixed: local cache verification/copying checks the
shared deadline before publication and preserves an existing destination on
expiry; range downloads retry transient HTTP statuses with one shared retry
allowance across ranges. The real HTTP retry proof retains both request IDs and
all 16 response bytes for a 12-byte artifact after a 503. The revised full
network-denied suite passed 358 tests, with 6 opt-in measurement skips and 3
scale deselections. Mypy, Ruff lint/format, PR checks and merge-queue checks passed.
The source tree has verified cache reuse/resumption, retained usage and offline
reporting; synthetic savings remain distinct from observed AWS savings.

The gate agent completed its checks, but its native relay again failed the
16 MiB frame limit (`16792491 > 16777216`). The task runner automatically
retried ship twice; the operator stopped that exact runner group rather than
letting native retries repeat. Fresh records and OS identities confirmed no live
owner; the unchanged tested four-file repair then went through only the authored
remaining headless `lf -b --task 88f96c land --wait-and-fix` operation. It exited
successfully after merge. Historical stopped gate records are not unfinished
implementation and must not be replayed.

`lf task complete 88f96c` completed at 21:24:47 UTC. Fresh status confirms
revision 3, completed true, state completed, status done, no completion pending
and PM writeback current. Completion removed its worktree, which made the same
Session's continuation fail with a missing-directory error. To fulfill the
requested open conversation without reopening the Task, its exact merged files
were restored as a detached checkout at the recorded path. No branch, Flow or
implementation was restarted. A headless connection check succeeded in the same
primary Session `b403fdf5e70c485a9a5cc63cef6bea76`.

The existing cmux workspace `workspace:1000070048`, surface
`surface:1000070125`, remains in window
`A13D14F4-A590-48DC-B734-943411818060`. Its text client was restarted with the
stable Manabot venv interpreter and isolated mode; it shows a compact fresh
completed-Task status and a `You>` prompt. The native UI is still unavailable.
The Task's detached checkout contains `scratch/storage-delivery.md`; do not
mistake it for newly opened work. Task automation and CI watcher remain disabled;
no landing watcher remains after the waited landing completed.

The private cost Task 78b6b1 received actual copied API reference contents and
the exact commit, proof, adoption boundary and remaining financial work in
`scratch/sdk-storage-88f96c-handoff.md` and `scratch/sdk-storage-88f96c-launcher.md`
inside `/Users/jack/src/etude-studio.reduce-experiment-transfer-costs`.
Its plan now distinguishes the shipped SDK from the earlier implementation-gap
baseline. No live Manabot environment, database, scientific source or receipt
was upgraded; account-log delivery and cloud bill reductions remain unverified.

At 21:30:00 UTC the Mini had 6,576 recorded games, all replay-passing, 61 completed
panels and four running panels. Launchd and OS command identities confirmed
evaluator PID 68448 and analysis owner PID 69462 live; analysis still waits for
the full panel. The MTG-122 recommendation goal is still active. No partial score
ranking, new rental, laptop evaluation or claimed S3 budget attribution follows
from this SDK delivery.
