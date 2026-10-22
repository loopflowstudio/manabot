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

### $50 controller active after sign-in

Jack Heart completed Doppler and AWS SSO login on the mini. Installed missing
GnuPG 2.5.24 there, per Doppler's CLI prerequisite. AWS account readback matched
the existing Manabot account. SSH could not retrieve the Doppler token from the
macOS keychain (exit 36), so activation ran through a temporary LaunchAgent in
the mini's logged-in GUI domain. This succeeded without copying the token.

Service `manabot.cohort.mtg165-two-seat-50` is running under launchd on
mini-heart.local, PID 76098 at verification, state directory
`/Users/jack/.manabot/mtg165-two-seat-50`. Authoritative S3 cohort readback showed
phase running, heartbeat age 12.94 seconds, no error, exactly original job
mtg165-two-seat-002 adopted at its unchanged deadline, and 441 completed updates.
Continuations 003–008 are queued. Reserved total is $49.98125984094739 against
$50; overall deadline is 2026-10-28 19:53:48.795631 UTC. The existing run was not
restarted, and no new training rental was created during activation.

A separate GUI-domain preflight successfully issued and discarded job-scoped
credentials for a full 11.9-hour planned worker lease using the existing issuer
and role. Only `{ok: true, hours_admitted: 11.9, rental_created: false}` was
retained; no credential values were stored or printed. Local receipts are
`.runs/mtg165-two-seat-50/active-state.json` and `credentials-result.json`.
Both temporary activation/preflight LaunchAgents were removed after success;
the budget-owning cohort service remains loaded. The controller now owns later
admission, exact exported-state binding, cost accounting, and cleanup without
this laptop or conversation remaining connected.

## Report publication goal — active

Jack Heart requested publication of a report on the $50 two-seat experiment.
At 2026-10-21 23:57 UTC the experiment was still running in its first segment;
the final $50 endpoint was not reached. Published an explicitly interim report
at `experiments/two-seat-ladder.md`, machine-readable observation
`experiments/data/two-seat-ladder/2026-10-21-interim.json`, and an experiment-index
entry. `lf pr publish` pushed these and marked PR #330 ready, using refreshed
copy that corrects the stale earlier description of authorized rentals. GitHub
contents API verified the report exists on the published branch. PR head at
verification: 26d0c68edeeddc902fc815c482e96c5895d1e5fb. Publication warned about a
missing Linear Initiative/Wave binding; GitHub publication succeeded. Do not
repair that unrelated planning binding or merge/complete MTG-165.

Verified/fetched immutable generation 414 of job 002, TrainingRun
b855d41288c542ffb8af8c337db81186: 704 updates, 211389 games, 20909111 learner
and environment decisions, 5227546 filtered optimizer exposures and 81988
optimizer steps. Counts equal sums of diagnostic rows/steps; zero numerical
nonfinite or rejected optimizer steps. Both raw milestone checkpoints load with
ordinary readers. Run JSON and checkpoint hashes are pinned in the report data.
The later fresh provider/S3 read found job 002 running at 712 updates. Estimated
incurred cost to observation was $4.010065 including prior attempt 001 and the
002 guardian; $49.98126 is reserved admission, not spend. Controller remains live
on the mini with a fresh heartbeat; all six continuations are still queued.

A bounded CPU evaluation of update 250 is now running on the existing mini,
without another rental. Uses the predeclared 25 deals 1590100000–1590100024,
heuristic, mirrors/eight legs = 200 games, bootstrap seed17/4000 repetitions,
600-second per-game allowance. CheckpointQueue owns the evaluator with a
3600-second attempt cap (3660 total queue budget), isolated subprocess group
and persistent attempt/result records. LaunchAgent `manabot.mtg165-eval250` was
confirmed live, PID85134; its child processes were confirmed live too. Inputs
and all queue evidence are under
`/Users/jack/src/manabot.mtg165-controller/.runs/mtg165-two-seat-50/evaluation-250`
on mini-heart.local. The local counterpart is `.runs/mtg165-two-seat-50/evaluation-250`.
Do not retry failed/incomplete evaluations silently. Fetch `queue/**/evaluation/monitor.json`
and its arena evidence, verify completed rows/replay, then update the published
report with any valid result. Native evaluator source4ff4086 matches the w6
training engine; the ordinary admission checks exact source/world/content ABI.

Goal remains active for the final report: inspect authoritative cohort state and
exact current job/provider before assuming terminal state; no restart solely on
observation timeout. After terminal evidence, settle all attempted costs and
cleanup, fetch/bind final raw/EMA/Adam exports, score the selected tier checkpoints
with the declared anchor on owned CPU, retain failures, and publish the final
report. Do not present this interim publication as a completed $50 experiment.

### Early heuristic evaluation verified (2026-10-22)

The previous goal turn made progress by publishing the interim report and
starting a bounded evaluation. Fresh authoritative reads in this turn confirmed
cohort/controller running and evaluator PID85134 live; evaluation advanced from
72 through 184 to200 games, then completed with exit0. The GPU job remained
running (727 updates at the first fresh mini read), no continuation yet.

Copied all evaluation queue evidence to local
`.runs/mtg165-two-seat-50/evaluation-250/retained/`. MonitorResult is completed,
200/200 unique expected seed/leg rows, every row valid/terminal/replayed, no
integrity failures. Verified the SHA-256 of every compressed trace shard for
both this cohort and the completed one-seat #333 cohort at
`~/manabot-runs/mtg159/eval/results/monitor/out/heuristic-0/monitor.json`.
Engine/content/input ABI, heuristic registration and monitor-source identities
match. One-seat checkpoint a9fe2d23…, two-seat checkpoint2bd07fa2….

Both-seat250 score .515 [.485,.545] versus one-seat250 .475 [.445,.505]. Shared
whole-deal bootstrap (4000 draws, seed17) gives paired mean difference .04
[-.01,.085]. LvA .24 [.12,.38] versus .22 [.12,.32], paired .02[-.10,.14];
AvL .74 vs.82, paired-.08[-.18,0]; LvL .60 vs.38, paired .22[.08,.36]; AvA
.48 vs.48, paired0[-.08,.08]. One training seed; descriptive same-update,
post-hoc and unmatched cost/decision budgets, not method-level superiority.
Two-seat250 used7,595,063 learner rows and5222.94 seconds of training wall;
one-seat250 used4,096,000 rows and1530.62 seconds (1.85×rows,3.41×wall).
Evaluation on the mini took425.84 seconds, with no paid evaluation rental.
Compact scored rows/intervals/receipts are in
`experiments/data/two-seat-ladder/250-update-heuristic.json` and the report now
includes the full matchup table and the null/unequal-cost interpretation.

Installed standard bounded projection companion using the same frozen cohort
and existing controller allowance: LaunchAgent
`manabot.cohort.mtg165-two-seat-50.projection`, PID85625 at verification, saves
report.json/report.html and original dashboard exports under
`/Users/jack/.manabot/mtg165-two-seat-50/projection`. First verified saved report
showed job002 running at731 updates, generation427, no report error. It polls
at300seconds with120-second child allowance and3600seconds total. It cannot
rent machines; scheduler ownership/lease stays unchanged. Removed the temporary
report-service bootstrap LaunchAgent and the finished evaluation LaunchAgent;
main scheduler and projection remain running. Goal remains active for the final
budget-limited endpoint/report. Do not restart a stopped or failed job or add
paid evaluation work merely to obtain a more favorable result.

## Live Etude evaluations requested 2026-10-21/22

Jack Heart explicitly requested live reports with regular evaluations through
Etude Eval and authorized necessary Manabot/Etude PRs. Added the `evaluation`
extra, pinned public Etude source 1ee96ffb0a077cc83953b2806e99d8c8fb92b4e3 plus
DuckDB; separate/conflicting with legacy `reports` (0.1.1), whose imports were
verified in an isolated environment. No Etude repository changes were made:
its local `jack/live-eval-reports` branch already has other uncommitted report
work, preserved untouched.

Manabot `manabot.training.etude_evaluation` holds a real Etude Benchmark: each
match is one fixed development deal, eight arena legs, heuristic, legal/terminal
replay admission. Etude writes match records and Parquet and resamples whole
deals (its default 2000 draws/seed0). Original 250-update monitor evidence remains
separate (4000 draws/seed17). `manabot.remote.live_evaluation` discovers digest-
verified 250-update raw exports from pinned S3 manifests, serializes CPU evals
with a one-hour child allowance, saves per-digest attempts, and never rents or
mutates training. Each checkpoint gets an admission JSON and arena evidence.
Interrupted/failed attempts are not automatically replayed; explicit
`--retry-checkpoint SHA` retires that observer attempt and resumes completed
Etude matches. Live old PID fences replacement. Shared absolute deadline is
cohort.deadline+1h. Reporting exceptions cannot stop a valid evaluator.

Separate deployment `/Users/jack/src/manabot.mtg165-evaluation` on Mini, pinned
33054b347bed7f486d8de36214f9bc7baafe5e7f, separate uv environment with artifacts+
evaluation and copied matching controller native .so. Running controller and
worker source checkouts were untouched. LaunchAgents:
- manabot.mtg165.etude-evaluation (no KeepAlive; explicit restart/reconcile)
- manabot.mtg165.eval-report (KeepAlive HTTP server, serves ONLY public/)
Runtime/evidence: `/Users/jack/.manabot/mtg165-two-seat-50/evaluation`.
Live local-network URL: http://mini-heart.local:8766/ ; index, eval.html,
curves.json, status.json. Public page refreshes30seconds during evaluation;
last observed training coordinates and timestamp distinguish stale observations.
AWS_PROFILE manabot-controller; no Doppler token/key copying required for reads.
Persistent service configurations retained locally under
`.runs/mtg165-two-seat-50/live-evaluation/` and on Mini ~/Library/LaunchAgents.

Activation history: first callback had an Etude declaration but no plays table;
report curves query raised and interrupted it. Empty table handling and report
exception isolation fixed, with a real failed-match/no-score regression. First
attempt retained in original digest directory; ledger retired key suffix has
started_at1792628818.586645, PID88075 now absent. Explicit second attempt started
1792628989.676289 PID88471. It retained two complete Etude matches (16 games),
all terminal/replayed; first deal1590100010 scores/decision counts match the
original monitor. Background launchd ProcessType made each deal~118seconds versus
~16original. Explicitly suspended observer88410, SIGINTed evaluator88471 (arena
finally kills/reaps its native child), booted out ONLY this evaluation observer.
Confirmed88471 exited. Changed saved evaluator plist to normal ProcessType default
and OMP_NUM_THREADS/MKL_NUM_THREADS=1; resumed explicitly again, retaining old
attempt and existing Etude matches. Remove-once retry flag was copied for the
bootstrap then immediately replaced on disk with normal no-retry plist. Do not
boot out the training controller or projection service. All paid work unaffected.

Latest fresh training check: cohort running, original002 at831updates, heartbeat
9.29seconds old. Final $50 endpoint still pending. Live Etude callback run name:
checkpoint-2bd07fa24a524536471c3e4d0a03e9b5a42fc19201f007db6d1bb6b490126897.
Full250 Etude result still pending; 500 and750 exports should queue next. Fetch
curves.json/status.json and exact retained result records before reporting a
completed evaluation. No need to rerun original200-game monitor. Unit and
focused changed suites passed23checks (4long checkpoint checks excluded),
isolated evaluation+continuation19passed; package boundaries2passed. Added
explicit CI step for eval extra (legacy suites still use reports extra), optional
module collection skips cleanly without eval extra. New tampered-byte admission
check passed (4 Etude adapter checks total). PR330 has source commits adc10393,
a40f801e and33054b34; final PR copy/report notes still need publication after
this activation verification. Keep goal active for final experiment report.


### Etude activation fully verified and published

Normal-scheduling resume fixed throughput to16–18seconds/deal. Third observer
attempt PID89126 started1792629324.551751, completed1792629710.4152782, exit0.
Etude result played23/skipped2;25unique done MatchRecords,200unique valid terminal
replayed native rows. Every (deal,leg) score equals the original monitor's200
rows. Etude curve score.515 [.485,.545],200rows/25matches. CPU-only next500
checkpoint launched automatically PID90603, at least3matches already complete.
Source33054b34 and saved no-retry plist remain stable; do not restart.

Copied all250evidence, both interrupted attempts and Etude track to
`.runs/mtg165-two-seat-50/live-evaluation/retained`. Verified all trace receipt
hashes and complete originalscore equivalence. Archive4,518,522bytes published
using ordinary S3ArtifactStore with full byte-verified readback:
s3://etudefantasia/manabot/cohorts/mtg165-two-seat-50/evaluations/sha256/c18e4cefe8440182d3b88313634dda91b2f93d36c857ddf40d71517deb3874ca
Version7BT325KA4NoxTcGnsXrNJ42qUAzVuBhd. Machine-readablepublicreceipt
`experiments/data/two-seat-ladder/250-update-etude.json` includes deckbreakdowns,
Etude result, attempts and per-fileinventory. No credentialvalues included.

PR330 was published withliveevalcode/instructions atHEAD43ac2028; final full250
receipt/report publication follows. Latestfreshtraining860updates, heartbeat4.87s,
cohortrunning. Keepgoalactiveforfinal$50report; live250cadence+CPUserver now work
withoutthisconversation/laptop. Futurecheckpoint evidence remains onMini; publish
verifiedreceipt/report snapshots at meaningfulmilestones, retainingfailures.

## Goal continuation: updates500/750 and corrected Etude deck labels

Previous goal turn was progress: live Etude activation, verified250 and automatic
500 handoff published. Fresh current-turn checks confirmed original002 live at864
updates, providerpresent, controllerPID76098, evaluation500PID90603live. No restart
on observation timeouts. Subsequent500and750evaluations completed200games each,
zero native integrity/replay failures; preserved all evidence locally and S3.
Aggregates250=.515[.485,.545],500=.52[.475,.565],750=.55[.51,.59]. Paired750minus250
+0.035[-.02,.095],2000whole-deal bootstrapseed0, descriptive within-run only.

Found and fixed a real scientific labeling bug before publishing500/750 details:
Etude adapter used leg//2 to name candidate crossmatchups. ActualcandidateLessons
legs0/3, Allies1/2. Aggregates/nativegames/oldmonitor250comparisonwerecorrect;
Etude's cross-deck subgrouplabelswerewrong. New publicpurearena_tables function
validatesauthoredseat/decks and labels eachparticipant'sownactualdeck. Replaced
faultyprojection, addedregressioncheckingall8legs/opponents andrejectingcorrupted
seats. Benchmarkname now two-seat-ladder-heuristic-deck-seats tokeepcorrected
records distinct; old implementation isdeleted, frozenrecords retained.

Alsoaddedfinalstage raw checkpointselection toliveobserver foractualpaused/completed
budgetendpoints between250milestones, digestdedup andterminalmanifestcursor. Tests
coverpaused/completed vsrunning/failed/interrupted andidenticaldigestdedup. Small
supervisorobservations nowrefreshduringeval; sourceheartbeatage displayed separately
fromreporttime. Existingtraining/controller code andworkersourceunchanged.

Codecommit9467dd9d0889622309cf43e0d6f433a6dc69ec68. FocusedEtudetests10passed;
earlierendpoint+batch14passedbeforenewdeckregression, Ruff/diffpass. Native750
completedbeforecontrolledobserverstop; ledgerhadno runningeval, thenONLYevaluation
LaunchAgentbootedout. No live evaluator ortraining interrupted fordeployment.
PushedcodeandupdatedMini~/src/manabot.mtg165-evaluationtothat exactcommit.

Used `.runs/mtg165-two-seat-50/live-evaluation/correct-labels.py` toverifyalloriginal
native rowregistrationbindings andtracehashes, thenholdcorrectedEtudebenchmark
usingarena_tablesoverretainedrows:25matches/checkpoint, no newgames orfreshreplay.
RunIDs deck-seat-projection-SHA. Config explicitlyrecords nativeproducer33054b34,
projection9467dd9, source-rowandtracehashes. Corrected-trackincludes250/500/750.
OnMini movedoriginaltracktoevaluation/retired-cross-deck-labels, preservedoriginal
ledgerasledger-before-deck-correction.json, installedcorrectedtrackasactive track.
Completedledgerentries nowprojection=retained-arena; interruptionhistoryretained.
Restartednormal no-retry evalservice, PID92528live atverification, noerror. Original
cohortcontrollerPID76098unchanged. Header nowshowsfreshtraining908updates, source
heartbeat~12secondsold; owncontrollerobservation907slightlyearlier, consistent.
1000checkpointpending; observeridleandpollingnormally. Next1000 usesnewnative
correctlabeladapterandshouldjoincorrectedbenchmark withoutprojection.

Evidence receipts, all fullbyte-readbackverified:
- 250originalnativec18e4cef… remainsretained, unchangedarchive.
- 500native1d12faa7c924508fc71ab8e0f78bda6493b49129420ceae173386f66bcf668c0,
  4,413,111bytes, S3versionCQxJlQ1jOGg9_b3xbDo9ezIhj5I..BPP.
- 750nativec33af3bc1368cef2dab05f4646e6038a3548863774b191ecb7241442789743d2,
  4,549,072bytes, versionBdeRis.6p9XvmIn4s_QT4LYzqFVcQ4IM.
- CorrectedEtudetableprojectionc4b36f41df77885534f04847b12b163758635777d254be4cbca272213b9aefe7,
  31,950bytes, version33UHlzlwGpJHOY.EaXFa.I3kqdLCTgI5.
Allunder s3://etudefantasia/manabot/cohorts/mtg165-two-seat-50/evaluations/sha256/.

Public250receiptupdatedwithexplicit table_projection+supersession provenance;
originalnativearchive/fileinventoryremains. New500/750JSONreceiptsincludecorrected
queries, intervals, pairedchanges, coordinates, nativeattempts andper-fileevidence.
Correctedmatchups250/500/750:LvA.24/.30/.34;AvL.74/.80/.86;LvL.60/.46/.54;
AvA.48/.52/.46. Do notuseunadjustedoriginalEtude subgroupParquet labels.
Reportappendsexplicitcorrection andthree-pointcurve, neverclaimsfinal$50complete.
Finalendpoint/cleanup/coststillpending, goalactive. Publishcurrentreport/data next.

Genericpublisherignoredscriptnowrequires --updates N --source EXACT_EVALUATOR_COMMIT;
future1000nativeon9467dd9usescurrentBenchmarkfromresult.experiment, no projection
needed. Inputlayout retained-N/{track,evaluations,ledger.json}; datareceiptarchive
hashreadbackusualS3ArtifactStore. `publish-correction.py` isone-offforfirstthree
points; donotrerunfuturevalidnativepointsthroughlegacybenchmark. `correct-labels.py`
isretainedreproducibleprojectionproof, notnewtrainingorinference.


Publication verified: PR330head901c7752df0e047f5bb680736a6f45141ae20c62 includes
allthreepointsandexplicitcorrection. Follow-upreviewcaught final-endpoint explicit
retryneededclearing settled_jobs cursor; fixed in901c7752. Idle queueconfirmed
(all600gamesfinished, noactiveeval) beforeONLYevaluationobserverwasbootedoutand
updatedto901c7752. Policy/tablecode remains9467dd9, so correctedEtudebenchmark
identitiesstaystable. Sourceworkerfabb/controller4ffandcohortPID76098 untouched.
Goalstillactive; final$50 endpoint/cleanup/costreportnotachieved.
