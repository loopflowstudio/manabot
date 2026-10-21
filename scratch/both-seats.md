# Two-seat learning (MTG-165)

Initial direction: Jack Heart, 2026-10-20, implement and publish for review without paid work or a ladder. Later direction in this conversation authorizes one $10 two-seat pilot and a two-seat default; the no-ladder boundary remains. Existing pursue Flow owns later compression, synchronization, reconciliation and publication.

Jack Heart revised direction on 2026-10-20: make two-seat self-play the default and launch one run with a $10 ceiling. New schema-2 regimes and direct collectors default to both learning seats for self-play, and one for off-policy opponents. Saved schema-1 regimes preserve their one-seat meaning and bytes. The ordinary baseline becomes baseline-v2. Current-self records both policies' decisions with separate seat trajectories. Random/frozen modes reject two-seat learning. Distinct learning models use a paired collector/update API, with independent optimizers and behavior records.

Collection boundary: both-seat collection takes at least `transitions` total decisions per native stream, then finishes that stream's current game. Batch size can exceed the floor. Pack seat trajectories into one column, separated by terminal flags, so GAE never crosses seats or episodes. An explicit `games_per_stream` overrides the decision floor on either control, enabling equal-game comparisons with the same terminal estimator. All actions are fresh and no pending samples cross an update. This trades exact row quotas and bootstrapped partial games for complete terminal targets. Retain actual learner/environment decision counts; this is not a claim of 2x speed or strength. A matched comparison remains required for a gain claim; the default itself is now explicitly selected by Jack Heart.

Delete — do not maintain: no separate predecessor is removed; retain one-seat collection as the requested experimental control and for off-policy opponents. Share inference, recording and terminal accounting across collection modes.

Missing context: supplied `.lf/wave/intelligence/GOAL.md` and `MEMORY.md` are absent in this checkout. Task directive is available from `lf task status MTG-165 --json`.

Implemented: shared inference/recording, per-seat rewards and distributions, regime switch and actual-row accounting, matched-game controls, independent paired learners, checkpoint setup binding, continuation guards and documentation. Frozen evidence and Academy runtime paths are unchanged. The paired API trains both models but does not introduce a remote dual-model executor. Ataraxos packed rows each form one timestep minibatch, an explicit cost/grouping difference from the rectangular control.

Verification: focused collector and new two-seat checks pass, including native w6 games/recovery, raw/EMA regime execution, independent optimizer updates and checkpoint reload; Ruff and diff checks pass. Matching native w6 module rebuilt locally; no Rust source changed. Affected suite gate remains deferred to the delivery procedure; no strength/cost comparison was launched.

Remaining authorized work belongs to the active pursue Flow: compress, sync, realign, routing and PR publication, then stop at the review node for Jack Heart. Do not merge or complete the Task. Jack Heart subsequently authorized one two-seat run within $10; that supersedes the original no-paid-run steer, without authorizing a ladder.


## Authorized pilot

Jack Heart authorized a $10 two-seat learning run in this conversation on 2026-10-20. Prepared `.runs/mtg165-two-seat-001/`: one RTX 5090, eight-hour absolute allocation, $1.20/hour compute ceiling plus $0.02/hour storage allowance ($9.76), plus $0.10 guardian allowance ($9.86 inclusive admission). No ladder or replacement rental is authorized. Seed 12261; current w6 Lessons/Allies world and width-64/depth-2 semantic policy; 64 cross-balanced streams, 4,096-update ceiling, both seats, PPO with constant learning rate, checkpoint every 128 updates and heuristic monitoring (eight paired development deals). This is a single training pilot, not an equal-cost or equal-game superiority comparison.

Admission: local CLI completed one same-model PPO update in 26.98 seconds, four games, 968 learner and environment decisions, checkpoint export. Default/frozen identity and experiment checks: 46 passed; baseline/regime checks: 18 passed. Existing saved schema-1 regimes and preset digests remain unchanged.

Submission status: source commit `352edf6be070993605756308630785a2f0b88a45` is pushed. Plan compiled and saved at `.runs/mtg165-two-seat-001/plan.json`; job ID `mtg165-two-seat-001`. Submission failed in the initial S3 intent write, before submit_job/provider-create, because AWS SSO profile `default` expired and could not refresh. No new rental was created and no pilot cost was incurred. Jack Heart was asked to renew with `aws sso login --profile default`; resume with the SAME job ID after renewal. Redacted failure evidence: `.runs/mtg165-two-seat-001/submit.log`.

Additional validation: `uv run pytest tests/model/test_inference.py tests/training/test_producers.py tests/model/test_categorical_value.py -q` passed 42 tests in 266.71 seconds. `manabot-issuer` is an existing AWS profile. After Jack Heart renews the default SSO login, use that profile for worker issuance (`MANABOT_REMOTE_ISSUER_PROFILE=manabot-issuer`) and the existing `manabot-remote-jobs` role in the same SSO account (`MANABOT_REMOTE_ROLE_ARN`). SSO alone cannot issue an eight-hour worker session because of AWS's one-hour role-chaining limit. No auth policy or profile was modified.

AWS renewal: Jack Heart confirmed the default SSO login was refreshed. Original job ID persisted successfully; scoped eight-hour worker credentials admitted through the existing manabot-issuer profile. Submission is in progress with deadline 2026-10-21T09:59:54+00:00; no replacement job or policy change.

Pilot outcome after AWS renewal: `mtg165-two-seat-001` failed to publish any supervisor record. The guardian proof succeeded, and the training rental was created, but no accepted/running TrainingRun, learner updates, or checkpoints were confirmed. A local DNS outage interrupted submitter status observation; connectivity recovered, and authoritative reads still showed no runtime record well beyond the 900-second setup allowance (training age approximately 4,416 seconds when checked). The exact bootstrap failure was not captured; the pre-supervisor bootstrap log was not published to S3.

Cancellation was persisted first. Because no supervisor existed to consume it, both recorded resource identities were checked against their owned job claims and the remaining training pod was deleted through RunPod. Authoritative job_status confirms provider_state=absent and cleanup. Estimated total guardian plus training cost: $1.5312598409473894; approximately $8.47 remains of Jack Heart's $10 ceiling. No model training result exists. Local receipts: `.runs/mtg165-two-seat-001/cancelled-status.json`, `latest-status.json`, and `resumed-submit.log`. Original deadline 2026-10-21 09:59:54 UTC is superseded operationally by confirmed deletion. Diagnose bootstrap/startup observability before any additional attempt; never reuse the cancelled job ID for training.

## Ladder-matched pilot requested 2026-10-21

Jack Heart authorized one new rented GPU experiment within $10, matching the
currently running MTG-159 ladder but training both seats. The control was read
from its owning checkout: seed15902, w6 Allies/Lessons, width64/depth2, 256
streams, Ataraxos (not the prior PPO pilot), L4 at $0.59/hour. Preserve absolute
iteration schedules for this matching pilot. Protocol and recipe are frozen in
experiments/protocols/two-seat-ladder-002.md and its linked regime.

Before admission, corrected Ataraxos batching: credit still uses terminal seat
trajectories; packed rows now group by native stream count instead of taking one
optimizer step per row. Both-seat checks:15 passed; Ataraxos regression checks:
21 passed. The new attempt keeps provider-owned supervision and uses a public
SSH key to observe bootstrap before a supervisor receipt exists. The standard
launcher admits11.9 hours ($7.259 plus $0.10 guardian allowance), keeping the
prior $1.53126 failure and this rental below $10 inclusive. No further rental
or paid evaluator is authorized. Job ID mtg165-two-seat-002.

Pilot 002 is running. Source fabb96db2 (full identity retained in plan.json),
job mtg165-two-seat-002, TrainingRun b855d41288c542ffb8af8c337db81186. Guardian
proof passed; new L4 resource dsist8j859mj94 at $0.59/hour, independently owned
from the still-running v1 ladder. Bootstrap completed, including the native
release build in20.25 seconds. A job-specific public SSH diagnostic key made
pre-supervisor logs readable; standard S3 supervision and scoped worker
credentials remain the lifecycle owners. The account also has CPU evaluation
pods, which the local diagnostic provider excludes from its GPU inventory.

Confirmed2 completed updates on cuda:0:525 complete games,101,463 environment
decisions and101,463 learner decisions,25,367 retained optimizer exposures and
397 optimizer steps. Both updates report zero nonfinite values and rejected
optimizer steps. Collection24.75 seconds; learning50.51 seconds (early observations,
not a cost/strength comparison). Generation000004 was fetched and verified at
.runs/mtg165-two-seat-002/fetched/generation-000004; it includes the initial raw
checkpoint and current training evidence. Learned milestone exports are every250
updates and were still pending at this observation.

Learning cutoff2026-10-22 07:37:48 UTC; independent deletion deadline07:47:48 UTC
(00:47:48 Pacific). Rental admission$7.259 plus$0.10 guardian allowance and
prior attempt$1.53125984 totals$8.89025984, below$10 inclusive. No extension,
replacement or additional evaluator was launched. The supervisor continues
training, exports, S3 snapshots and cleanup independently of this conversation.
Reconnect with uv run --extra artifacts manabot deploy status --job-id
mtg165-two-seat-002. Existing PR330 is updated with the batching correction and
frozen recipe/protocol; no merge or Task completion.

## $50 expansion and Mac mini controller (2026-10-21)

Jack Heart raised the two-seat experiment to $50 total, selected the scale ladder
in Manabot PR #333, authorized necessary Etude changes, and selected the Mac mini
as controller. This supersedes the earlier no-continuation/$10 restriction.
The original 002 job remains running on its original lease; a later observation
confirmed 297 updates and a fresh supervisor heartbeat.

Prepared `.runs/mtg165-two-seat-50/cohort.json`: adopt exact job 002, then chain
003–007 at 11.9 hours/$7.36 each and 008 at 4.9 hours/$3.10. Including the existing
$8.35 reservation, prior $1.5312598409473894 and $0.20 controller allowance,
reservations total $49.98125984094739. Absolute cohort deadline is seven days
from 002 admission. Keep seed 15902, recipe and source fabb96db2, and 250-update
exports. Stop on failure/incomplete export/unknown cleanup; no failed-job replay.

Controller source 4ff408601a3ecebabb845e5abce429eaa6775027 adds explicit
`last-complete` continuation after a clean budget pause and `cohort start
--adopt-job` for the exact running job. Model/Adam/EMA/absolute counters survive;
collector/game/RNG streams restart. The first cohort entry cannot change the
live job's intent, cadence or original deadline. Regression tests for adoption,
continuation and controller services: 39 passed, 1 skipped. Broader job checks
passed 42 tests; a pre-existing learning-state assertion assumed fixed row
quotas. It now asserts actual cumulative learner rows across the three segments;
all 10 learning-state tests pass with complete-game two-seat batches.

The mini is reachable as jack@mini-heart.local, configured awake on AC, with uv,
Rust and Doppler installed. Controller deployment paths are
`~/src/manabot.mtg165-controller` and the retained pinned worker source
`~/src/manabot.mtg165-worker`. Etude's existing SSH controller was configured
using isolated local ETUDE_HOME `.runs/mtg165-two-seat-50/etude-client`; its package
installed successfully on the mini. No Etude source change was needed for SSH
controller selection. Existing launcher databases and other jobs were untouched.

Activation requires credentials on the mini. Doppler initially reported no
configured token, and Jack Heart was asked to run `doppler login` there. Added
nonsecret AWS configuration for `manabot-controller` (the existing Manabot SSO
account/session settings) and `manabot-issuer` (existing Doppler-backed worker
issuer selector), preserving all pre-existing profiles. No credential values
were copied. Jack Heart was also asked to run
`aws sso login --profile manabot-controller`. Both sign-ins must be verified
before installing/starting the budget-owning service. The $50 queue is prepared,
not yet active; the current GPU run continues independently meanwhile.
