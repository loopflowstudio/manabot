# Remote jobs that survive a disconnected laptop

A submitted manabot job owns its training, checkpoint evaluations, S3 snapshots
and shutdown on RunPod. After `deploy submit` reports remote acceptance, the
submitting process can exit. Reconnect from another machine using the job ID and
ordinary AWS/RunPod authentication. `run` observes this same lifecycle.

The implementation builds on TrainingRun, CheckpointQueue, Bundle and reporting
APIs. Experiment callers can submit one resolved case with
`prepare_experiment_job` and `submit_job`, or run the frozen `Experiment.jobs`
order through the cohort service below. Experiment declarations retain case/seed
and evaluation authority; the service sequences the existing job lifecycle. Arbitrary SSH execution callbacks cannot survive a
client disconnect and are rejected before rental. Historical calibration scripts
requiring them must run at their original pinned source.

## Cohorts need an independent owner

`Experiment.cohort(...)` freezes the existing `jobs` order and compiled plans,
including original case receipts, seeds, monitoring and checkpoint cadence. It
requires an absolute cohort deadline, inclusive dollar ceiling, prior charges
and an allowance for the controller host. Construction does not submit anything.
For an existing Experiment declaration and pinned `Source`:

```python
cohort = experiment.cohort(
    source,
    "comparison-attempt-002",
    deadline=admitted_deadline_unix,
    spending_limit=approved_total_dollars,
    prior_dollars=all_prior_attempt_charges,
    controller_dollars=controller_host_allowance,
)
Path("cohort.json").write_text(cohort.model_dump_json(indent=2))
```

The full **JobSpec spending limits**, prior charges and controller allowance must
fit the ceiling. Every newly admitted job must also fit within the remaining
cohort deadline. The service admits live GPU prices through ordinary `submit_job`.
It retains original job deadlines across restart and charges confirmed cleanup
receipts once. An unresolved attempt retains its full reservation. Late provider
absence observations can conservatively exceed that reservation; further admission
then stops. Provider outages can prevent timely deletion, so this is an admission
bound and cost ledger, not a provider billing guarantee.

Run `start` **on the chosen always-on controller host**, in the exact clean source
checkout used to compile the plans, with the artifacts dependencies installed:

```bash
uv run --extra artifacts manabot deploy cohort start \
  --plan cohort.json --state-dir /absolute/private/cohort-state --doppler

# Reconnect from another authenticated machine; no original local directory needed:
uv run --extra artifacts manabot deploy cohort status --cohort-id comparison-attempt-002
uv run --extra artifacts manabot deploy cohort cancel --cohort-id comparison-attempt-002
```

`--doppler` resolves the repository's etude/prd secrets when the service starts;
without it, use the service account's configured provider credential chain. The
service copies only AWS profile/region and issuer role/profile selectors, never
credential values. The host must have renewable credentials for S3, RunPod and the
worker issuer. An expired SSO login cannot be repaired by restarting a service.
Issuer credentials stay on this controller; workers receive ordinary job-scoped
STS sessions with their original permission deadline.

On macOS this installs a launchd LaunchAgent; the account must remain logged in.
On Linux it installs a systemd user service and requires preconfigured lingering.
Both restart a crashed worker and survive the invoking terminal/agent exiting.
Neither keeps a sleeping/offline laptop available. This command does not provision
an always-on host, change login/lingering policy or claim a chat is a service.
Local `driver.json`, `status.json` and redacted `error.json` in the state directory
retain process identity, observations and storage/network failures.

S3 retains immutable cohort intent, attempts and sticky cancellation. Each job's
identity and deadline are recorded before submission. A stable OS machine identity/directory binding
plus a process lock excludes concurrent supervisors. Hostnames alone are not identities;
Linux controllers need `/etc/machine-id`, and macOS uses the platform UUID. Those
identifiers are hashed before storage. Another directory or host
cannot take over automatically; loss of that host requires explicit ownership
recovery after proving the old owner cannot act. There is no expiring lease that
could let a disconnected old owner resume alongside a replacement.

Restart reconciles existing job, provider and artifact identities. A lost create
response uses the same ID and permanent creation claim. Empty inventory cannot
prove an uncertain request never created a rental. Transient observation failures
retry; failed training, paused targets, cancellation, missing evidence and
uncertain creation never create replacement jobs. Failed attempts and their costs
remain visible. Paused CUDA exports do not provide exact process continuation.
A numerical change or replacement scientific attempt needs its own frozen protocol
and allocation, including the earlier dollars.

Only completed jobs with confirmed cleanup and a verified final manifest admit
the next entry. The service reads small control records and final manifests; it
never downloads checkpoint bundles or runs notebooks/W&B inside the scheduling
loop. `deploy fetch/report` remain independent, reconnectable consumers of durable
worker snapshots. An optional independent report service is described below; scheduling never
waits for its downloads or W&B acknowledgements.
Cancellation records the cohort request first and forwards it to the current job
even if the supervisor is offline. A returning supervisor also forwards it before
provider reads. Worker cancellation acknowledgement, final artifacts and provider
absence remain separate observations.

The offline suite drives the actual deploy entry point with persistent fake
provider/storage adapters: launcher exit, SIGKILL after provider creation, restart,
no duplicate submission, failure/pause stops, cancellation, competing owners and
inclusive cost accounting. On 2026-10-07 the explicit native launchd check also
completed two fake jobs after a killed supervisor was automatically restarted:

```bash
MANABOT_TEST_LAUNCHD=1 uv run --extra dev --extra artifacts pytest \
  tests/remote/test_cohort_service.py::test_launchd_restarts_killed_cohort_without_launcher -q
```

This proof rents nothing and establishes no live cloud cohort, systemd host
acceptance, CUDA recovery or scientific result. ETU-103's failed frozen attempt
remains unchanged; its recovery additionally requires the numerical-health fix,
a versioned attempt, live-price admission and all earlier charges within $100.

## Reports while clients are offline

Add `--reports` when installing the cohort to create a separate OS-managed
companion. Add both W&B selectors for native graphs in the declared workspace:

```bash
uv run --extra artifacts manabot deploy cohort start \
  --plan cohort.json --state-dir /absolute/private/cohort-state --doppler \
  --reports --wandb-project etude --wandb-entity loopflow-studio
```

The companion reads the current job's exact published generation and retrieves
only existing training/monitoring Dashboard JSON exports. It saves those bytes and
writes `projection/report.json` and a compact `projection/report.html` before
attempting W&B. The report shows the intended versus admitted job count, phases,
updates, evaluations, generations, heartbeat timestamps and charged/reserved
cost. Its links expose the saved Dashboard exports; W&B owns the metric graphs.
Full editable notebook analysis still uses the ordinary `deploy fetch/report` path.
Unpublished data is unavailable, not a zero, and a saved timestamp is not a live
heartbeat. A network outage retains the earlier local evidence.

The queue index at `monitoring/dashboard.json` is not a Dashboard export. The
projector reads the per-run files, including protocol-specific dashboard names.
In-progress evaluation counts remain visible in local exports and W&B summaries;
only settled evaluation rows enter W&B's append-only history. A later completed
or failed attempt appends once, while rewriting a published result still fails.

The report service has its own process/host lock and cannot submit training.
It polls every five minutes by default, with 120-second child attempts and a
cumulative 3,600-second allowance inside the declared controller allocation.
Reservations are persisted before starting children; interruption cannot reset
consumed allowance. Each child owns an absolute process-group deadline even if
its parent disappears. A restart can conservatively charge the full reservation.
Failed attempts and telemetry error types are retained; credential-bearing error
responses are omitted. W&B retries use its existing acknowledged history cursor;
a confirmed publication of unchanged Dashboard bytes is skipped.

Both services stop after settled outcomes; reporting stops only after a successful
final projection, or on allowance exhaustion. Unresolved observation/reporting is
bounded to 30 minutes beyond the cohort deadline, inside the controller allowance.
The training rentals retain their own earlier absolute deadlines. Service and
projection configuration is frozen on installation; a different owner/configuration
fails rather than creating competing publishers. Renewing a reporting allowance or
moving a lost host requires explicit recovery, not deleting its ledger.

Local fake-provider/telemetry checks cover retained reports during W&B failure,
idempotent publication, interrupted allowance accounting and the child's deadline.
No live W&B service or cloud cohort was exercised by those checks.

## Submit, disconnect and reconnect

### Bounded CUDA numerical admission

Add `--validate-numerics` to `deploy submit` (or direct `deploy --plan ...`) for
an explicitly allocated validation job. The flag is immutable job intent; jobs
without it retain their historical identity. Bootstrap installs the development
extra, then the supervised process runs the frozen numerical contract tests with
`MANABOT_NUMERICS_DEVICE=cuda` and the CUDA optimizer-overhead probe before
starting the declared TrainingRegime. A missing GPU or failed test blocks learning;
there is no CPU fallback or automatic replacement.

Experiment authors can set `PlannedRun(validate_numerics=True)` for the same
gate through `prepare_job` or a frozen cohort. It stays attached to each admitted
job across supervisor restarts; omitted/false flags preserve historical identities.

```bash
uv run --extra artifacts manabot deploy submit --plan validation-plan.json \
  --job-id capacity-validation-v2 --validate-numerics
```

The gate has a five-minute ceiling within the existing work deadline, plus one
second for timeout receipt cleanup. Its private process group remains owned by
the job supervisor and guardian. Cancellation stops validation and descendants.
`numerical-validation.json` and `numerical-validation.log` enter the ordinary
artifact snapshots, including failure before TrainingRun creation. The receipt
binds exact source/native bytes, Torch/CUDA/device identity, exit codes and elapsed
time. An interrupted receipt stays incomplete. The small declared training and
replayed evaluation that follow establish lifecycle feasibility separately.

This flag grants no allocation or CUDA recovery. Include validation, bootstrap,
evaluation, artifact delivery and deletion in the admitted JobSpec and shared
cohort ceiling. ETU-103's current overall ceiling is **$100**, superseding the
earlier $30 ceiling and including all historical attempts. Original proof-era
budgets below remain historical records.

Use the [existing compiler](remote-training.md#compile-and-deploy) to create a
source-pinned deployment plan. The exact commit must be publicly fetchable.
Private, versioned `s3://etudefantasia/manabot/jobs/` stores control records and
verified artifact generations. No bucket policy, public access or lifecycle
settings are changed.

For AWS SSO, configure the worker role once through the current AWS principal:

```bash
uv run --extra artifacts manabot deploy setup-worker
```

This creates `manabot-remote-jobs`, trusting only that principal and granting
control reads plus runtime-evidence writes under the job prefix. It never changes
another role; an existing conflicting trust or policy fails. The rental gets an
expiring, job-scoped STS session, never the account's AWS or RunPod keys. A supplied
`MANABOT_REMOTE_ROLE_ARN` selects an already configured alternative. IAM-user
credentials can instead use STS federation without role setup. SSO/role-chained
jobs must fit below 3,540 seconds including all reserves; admission fails before
rental if credentials cannot cover the absolute deadline. This is a current
runtime limit, not portable credential renewal or long-job recovery. AWS documents
the [one-hour role-chaining limit](https://docs.aws.amazon.com/STS/latest/APIReference/API_AssumeRole.html).

For a long job, an explicitly configured dedicated IAM issuer can assume the
worker role directly. Set `MANABOT_REMOTE_ISSUER_PROFILE` to that local AWS profile
and `MANABOT_REMOTE_ROLE_ARN` to the worker role. The profile may load its key
from the approved secret manager through `credential_process`; never store a key
in source or pass the issuer environment to the pod. Client S3 operations keep
the ordinary AWS profile. Issuance requests the full remaining rental deadline
plus a 60-second reserve, applies the same job-prefix session policy, and verifies
returned expiration before provider creation. Role duration/trust/permissions
remain enforced by AWS. A temporary-role source still fails above one hour.

Issuer account setup remains an explicitly authorized operation. ETU-103's
restricted issuer was configured on 2026-10-07, with its sole AssumeRole key in
Doppler and the existing SSO trust preserved; no key enters source or workers.
The runtime only consumes that configured profile and verifies delegated expiry.

`MonitoringBudget.require_initial_admission` optionally holds a fresh single
self-play stage after initial export. The supervisor releases the exact artifact
only after its initialization cohort completes and the evidence generation is
published. Failure prevents learning; upload failure keeps the gate closed.
Waiting uses the stage watchdog/rental allowance, not active training time.
This evaluates initialization on the training rental without another smoke job.

A monitoring budget may also supply `terminal_protocols`. The last completed
stage's raw export alone gets these disjoint cohorts; initialization and live
exports use `protocol`. This permits untouched held-out final deals and a separate
random diagnostic without putting an experiment callback inside the supervisor.

Author an optional `MonitoringBudget` JSON using the existing contract. For a tiny
workflow check, this declares two minutes per evaluation and a four-minute total:

```json
{"seconds":240,"attempt_seconds":120,"include_initial":true,
 "protocol":{"deal_seeds":[1910123000],"game_seconds":20}}
```

One CPU is reserved for evaluation; the learner's declared threads must leave it
free. Deals here are inspected monitoring deals, not scientific held-out evidence.

```bash
doppler run --project etude --config prd -- uv run --extra artifacts manabot deploy \
  --plan .runs/remote-plan.json --job-id example-001 --monitoring .runs/monitoring.json

# Any later client, with no original output directory:
doppler run --project etude --config prd -- uv run --extra artifacts manabot deploy status --job-id example-001
uv run --extra artifacts manabot deploy logs --job-id example-001 --follow
uv run --extra artifacts manabot deploy fetch --job-id example-001 --out .runs/example-001
uv run --extra artifacts --extra notebook manabot deploy report \
  --evidence .runs/example-001/generation-000003 --out .runs/example-001/report
```

`deploy --plan ... --job-id ...` is the direct submission entry point;
`deploy submit` is the explicit equivalent. `deploy --regime ... --spec ... --job-id ...`
compiles and submits in one invocation. No previous CLI namespace is registered.

The generation number comes from `fetch`. Repeating a fetch verifies and reuses
completed files; interruption can be resumed into the same output directory.
Files are installed atomically and the complete bundle marker is written last.
Corrupt or conflicting retained bytes fail explicitly and are not overwritten.
`logs`/`attach` retrieve only the pinned manifest and training log, never model
weights or the database; a newer remote generation cannot change the observed
log's identity mid-fetch. Report generation reuses the existing
create-once editable notebook and offline HTML dashboard. It needs neither the
rental nor SSH. Existing notebook edits survive refresh.

Within one supervisor, intermediate snapshots reuse immutable S3 version receipts
that this process has already verified by full readback. Local snapshot bytes still
receive digest/size checks. New content and unversioned (`null`) objects always
receive full readback; failed publications never populate this cache. Losing the
process loses the cache. Final publication re-verifies every artifact remotely
before declaring complete, including unchanged checkpoints and closed evaluations.

`attach` follows published log prefixes like `logs --follow`; Ctrl-C stops
observation only. Log freshness is limited by the declared publication interval.
The ID prints before provisioning. A timeout before acceptance means acceptance
is **unknown**, not that training was cancelled. Retry the same plan and ID;
never use a new ID to retry an uncertain rental.

## Identity, failures and cancellation

`Job` binds exact source, resolved regime, optional Experiment receipt,
hardware, seed, evaluation allocation, artifact location and original absolute
cost/deadline allowance. An existing ID with different content fails. S3
conditional creation records intent before provider creation. Concurrent clients
share that fence. A lost create response is reconciled by the unique name; an
empty inventory alone cannot prove that an uncertain create never happened.
A permanent claim left before the request is intentionally not retried.

`JobRecord` keeps execution phase, heartbeat, training coordinates,
evaluation count, cancellation acknowledgement and committed artifact generation.
Stale heartbeat means unknown/unreachable. Completed execution, complete uploaded
artifacts, provider absence and confirmed cleanup are separate facts. Provider
reads that fail never imply deletion. Cost estimates conservatively include
intent through observed absence, including setup, upload and teardown; reconnecting
late can overestimate the rental interval. They are not invoices.

```bash
uv run --extra artifacts manabot deploy cancel --job-id example-001
# For uncertain provisioning, failed setup or unconfirmed deletion:
doppler run --project etude --config prd -- uv run --extra artifacts manabot deploy reconcile --job-id example-001
# Explicit forced deletion can lose unpublished evidence:
doppler run --project etude --config prd -- uv run --extra artifacts manabot deploy reconcile --job-id example-001 --delete
```

Normal cancellation is a durable request; the supervisor acknowledges it, stops
its process groups and attempts final publication. Before supervisor acceptance,
that request is pending. Forced reconciliation deletes only the exact claimed
resource. Unknown/ambiguous creation remains unresolved even after an empty read.

A short separate rental proves pod-side guardian deletion before the training
rental. Both share the original time and dollar cap. Provider startup launches
setup without SSH; acceptance follows source and price/resource admission. The
independent shell guardian enforces the original billing deadline if setup,
supervision or storage fails. Provider outages or a container that never starts
can still leave deletion unconfirmed; inspect and reconcile that same ID.

## Durable evidence and recovery limits

The supervisor takes SQLite's online backup and derives the TrainingRun JSON from
that same database cut. Committed raw/EMA/optimizer artifacts retain their bytes,
paths and hashes. Closed evaluator attempts, replay evidence and log prefixes
enter the generation. All blobs pass S3 readback before an immutable manifest is
published; only then does the current job record point to it. Failed upload keeps
the prior generation, and missing final evidence stays explicit. Fetch verifies
hashes and Bundle paths; ordinary checkpoint loading retains world/ABI admission.

Each pod can acquire its execution claim only once. Container restart does not
restart learning at initialization. Reconnecting is observation and retrieval,
not CUDA process recovery, pause/resume, a portable collector snapshot, or a
scientific rerun. TrainingRun and VerifyStore remain the learning authority.

## ETU-123 proof allocation

Jack Heart reserved at most **$3 total** from ETU-103's existing **$15**, leaving
ETU-103 at most $12. ETU-103's five deleted rentals retain $0.5359766177 estimated
cost before this proof. The proof ledger records both this inherited expenditure
and every ETU-123 attempt; they are never fresh additive allowances. Per-deployment
below-$5 admission remains intact. Mini ETU-105 and laptop ETU-118 are untouched.
The [compact proof](../experiments/data/etu123-remote-jobs/proof.json) and
[shared spend ledger](../experiments/data/etu123-remote-jobs/shared-spend.json)
retain both attempts. Full originals remain in `.runs/etu123-disconnect`; the
private S3 job records and artifacts remain available by ID.

The first live attempt at `70f76e0f` reached remote acceptance, then failed before
learner launch: a job-scoped S3 reader without ListBucket receives HTTP 403 for a
missing cancellation key. Scoped readback reproduced that response. The job
published its setup evidence and deleted; those original records remain retained.
Submission now creates the readable empty cancellation mailbox before renting,
and concurrent cancellation requests preserve the first timestamp. No bucket-list
permission was added. A subsequent proof uses a new ID and retains this charge.

At `def37708334d8153688a4bdb2be4386607e609c7`, job
`etu123-disconnect-001` was accepted at zero updates and the submitting process
exited. A separate client observed 19 updates/one completed evaluation, then
100 updates/three evaluations. The final authoritative TrainingRun contains
**160 CUDA updates, 81,920 optimizer exposures and 259.445 training seconds**
across two stages. The initial checkpoint and two later checkpoints completed
12 arena games; both later evaluations began after client exit. Their coordinates
0, 73 and 36 are stage-local, not a monotonic run-wide sequence.

After provider deletion, retrieval verified all 48 bundle files and ordinary
loading admitted four raw/EMA policies. Fresh `deploy status`, `logs`, `fetch`
and offline `report` clients worked without the submitting process; notebook and
HTML were regenerated from the returned generation. The proof source and its
original commands remain immutable through the later CLI naming correction.
No compatibility namespace is registered.

The final proof-source status counter retained its last poll (139 updates),
while the database and returned export agree on 160. The current supervisor reads
VerifyStore for progress and completion; a focused regression retains a stale
export and checks the authoritative result. The frozen job record is unchanged.

The failed attempt cost $0.0339593 and the completed attempt $0.0818858,
**$0.1158450 total estimated**, including guardian probes and observed deletion.
Together with ETU-103's prior rentals, the shared estimate is **$0.6518216** of
$15. These conservative observations are not invoices. Final inventory was empty.
This is one bounded execution/evidence proof, not learning improvement, portable
CUDA recovery or the Trained Challengers chapter's scientific acceptance.

## Step targets and machine allocations

New jobs declare `JobSpec` separately from `TrainingRegime`. The regime owns
updates and learning settings; the JobSpec owns the machine, `lifetime_hours`,
`spending_limit`, artifact access scope and setup/checkpoint/upload/cleanup
reserves. Hours include all those phases. No throughput estimate is translated
into the learning target. Keep the same target for equal-step comparisons, or
compare available checkpoints at declared common allocation/cost cutoffs for
equal-time comparisons. A paused endpoint is not a completed target.

A concrete authoring example (configuration only; it does not rent):

```python
from pathlib import Path

from manabot.remote.plan import AccessScope, JobSpec, Machine, Source
from manabot.training.checkpoint_queue import MonitoringBudget
from manabot.training.experiment_execution import PlannedRun
from manabot.training.experiments import Baseline, Experiment
from manabot.training.models import TrainingRegime

regime = TrainingRegime.model_validate_json(Path("ops/examples/step-target.json").read_text())
# One ordinary self-play stage; its authored updates remain the learning target.
# Select iteration schedules explicitly for new recipes, not frozen evidence.
regime.schedule_clock = "iteration_fraction"
spec = JobSpec(
    machine=Machine.model_validate_json(Path("machine.json").read_text()),
    lifetime_hours=4,              # 0.5 and 24 * 30 use the same model
    spending_limit=3,
    access=AccessScope(destination="s3://etudefantasia/manabot/jobs"),
    setup_seconds=300,
    checkpoint_seconds=120,        # export reserve, not monitoring cadence
    upload_seconds=300,
    cleanup_seconds=120,
)
experiment = Experiment(
    name="step-comparison",
    baseline=Baseline.capture("fixed-target", regime),
    jobs=(PlannedRun(
        case="step-comparison", seed=197, spec=spec,
        monitoring=MonitoringBudget(seconds=600, attempt_seconds=120),
        checkpoint_seconds=3600,   # monitoring cadence
    ),),
)
source = Source.model_validate_json(Path("source.json").read_text())
plan, = experiment.compile_jobs(source)
Path("plan.json").write_text(plan.model_dump_json(indent=2))
```

`machine.json` contains the existing RunPod shape and price fields: GPU types,
CPU threads/vCPUs, memory/storage, digest-pinned image, hourly compute ceiling and
storage allowance. It has no lifetime, credential duration or secrets.
`experiment.prepare_job(index, source, job_id)` persists the selected run's
monitoring and original authoring receipt through the existing job client;
`submit_job(job)` performs the explicit submission. Neither method schedules the rest of the experiment. Use `experiment.cohort(...)`
and the independently managed deploy service for that order.

The same compiler is available directly:

```bash
uv run manabot deploy compile --regime regime.json --spec job.json --out plan.json
# Execution requires its separately authorized compute allocation:
uv run manabot deploy submit --plan plan.json --job-id fixed-target-001
```

`deploy --regime ... --spec ... --job-id ...` compiles and submits directly.
New schema-2 DeploymentPlans contain one `spec: JobSpec`; machine configuration
lives only in `spec.machine`. `Job` binds that plan to its admitted ID and deadline.
The private schema-1 reader retains historical plan/job JSON, digests, active-time
watchdogs and fractional deadline semantics. Current compilation accepts only
JobSpec; there is no `--mix`, public hardware-mix type or derived mix projection.
Historical
`active_seconds` recipes retain their exact meaning and serialized identities; they are refused under a new JobSpec.
The duration-specific capacity preparation API/CLI is removed. ETU-103's running
four-hour cohort continues at its pinned source
`68e0fbe9265a689891615274123587d653b5e66b`, without rewritten configurations or evidence.

Admission derives one absolute deadline, rounding down to the guardian's Unix
second precision. STS session policy explicitly denies access at that deadline;
the guardian terminates the machine at the same deadline. STS's minimum token
lifetime or issuance margin cannot extend artifact permissions. Issuer credentials
stay on the launcher; workers receive only the job-scoped artifact session and
the provider's existing pod termination capability. Access scope has no separate
duration input. Historical recipe watchdog fields remain serialized, but
JobSpec execution uses the admitted allocation for timing and iteration-based
learning schedules; changing machine hours does not change the update schedule.

Collection stops at `pause_at = deadline - checkpoint - upload - cleanup`.
An unfinished batch is discarded and its collection time remains charged. The
executor exports the last updated raw/EMA policy and optimizer, reports `paused`
when updates remain, and reports `completed` only on reaching the target. Exports
must finish before `deadline - upload - cleanup`; exhausting that reserve is an
interrupted/failed attempt, not a successful pause. Monitoring may inspect a
paused checkpoint, while final completed-target cohorts remain excluded.

Supported boundaries and refusals:

- Bounded single-worker RunPod jobs work through the existing disconnected
  supervisor. With a directly configured issuer, the maximum is twelve hours
  minus the 60-second STS issuance margin; temporary-role sources remain limited
  to 3,540 seconds. No renewed session is issued after the allocation deadline.
- A 720-hour JobSpec constructs and compiles with its full projected cost,
  but admission refuses it before persisting job intent or renting: in-worker
  credential renewal, portable complete-state CUDA recovery and exclusive
  replacement-worker ownership are not implemented. There is no capability
  boolean that bypasses this refusal, nor a month-long bearer credential.
- CPU recipes with existing recovery enabled can explicitly resume a paused run
  on the same host under a new allocation, retaining the original update target,
  optimizer, EMA, RNGs and collector. A collection interrupted mid-batch retains
  the preceding complete recovery boundary. CUDA exports do **not** provide this
  continuation: attempting recovery fails instead of restarting from initialization.
- Cancellation requests stop the learner and finalize available evidence before
  pod deletion. They **do not immediately revoke** an already issued STS session;
  copied credentials retain their restricted scope until the deadline. IAM
  permission changes or a separate revocation service would be needed. See
  [AWS's session permission controls](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_credentials_temp_control-access_disable-perms.html).
- In-place extension is refused without changing the old deadline. Updating only
  a local plan cannot update persisted job intent, the running guardian and an
  issued immutable session policy coherently. A new allocation is explicit;
  remote CUDA replacement is still unsupported.

The offline tests exercise actual CLI compilation and fake-provider submission,
absolute permission/shutdown bounds, paused artifact publication and native CPU
pause/recovery equivalence. They do not establish live IAM enforcement, CUDA
recovery, or month-long operation. No rental or scientific scoring is needed to
run them.

### Final checkpoint discovery

A learner may finish while its supervisor uploads an intermediate snapshot.
After observing learner exit, supervision scans the closed TrainingRun again
before deciding evaluation is finished. Pending checkpoints with exhausted
allowance make the job failed; they cannot become a completed job merely because
no evaluator process is running. The CUDA recovery pilot exposed this race:
its original record says completed despite omitting its endpoint evaluation.
Keep that record as evidence; a separately recovered evaluation does not rewrite
it or establish remote lifecycle acceptance. See the
[capacity recovery record](../experiments/model-capacity.md#live-cuda-recovery--2026-10-08).
