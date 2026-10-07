# Remote jobs that survive a disconnected laptop

A submitted manabot job owns its training, checkpoint evaluations, S3 snapshots
and shutdown on RunPod. After `deploy submit` reports remote acceptance, the
submitting process can exit. Reconnect from another machine using the job ID and
ordinary AWS/RunPod authentication. `run` observes this same lifecycle.

The implementation builds on PR #253's initial/raw checkpoints, CheckpointQueue,
Bundle and reporting APIs. It does not add a learner or experiment scheduler.
Experiment callers select an existing resolved case with `prepare_experiment_job`,
then call the same `submit_job`; case/seed selection and campaign budgets remain
with the experiment owner. Arbitrary SSH execution callbacks cannot survive a
client disconnect and are rejected before rental. Historical calibration scripts
requiring them must run at their original pinned source.

## Submit, disconnect and reconnect

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
`deploy submit` is the explicit equivalent. `deploy --regime ... --mix ... --job-id ...`
compiles and submits in one invocation. No previous CLI namespace is registered.

The generation number comes from `fetch`. Report generation reuses the existing
create-once editable notebook and offline HTML dashboard. It needs neither the
rental nor SSH. Existing notebook edits survive refresh.

`attach` follows published log prefixes like `logs --follow`; Ctrl-C stops
observation only. Log freshness is limited by the declared publication interval.
The ID prints before provisioning. A timeout before acceptance means acceptance
is **unknown**, not that training was cancelled. Retry the same plan and ID;
never use a new ID to retry an uncertain rental.

## Identity, failures and cancellation

`RemoteJobSpec` binds exact source, resolved regime, optional Experiment receipt,
hardware, seed, evaluation allocation, artifact location and original absolute
cost/deadline allowance. An existing ID with different content fails. S3
conditional creation records intent before provider creation. Concurrent clients
share that fence. A lost create response is reconciled by the unique name; an
empty inventory alone cannot prove that an uncertain create never happened.
A permanent claim left before the request is intentionally not retried.

`RemoteJobRecord` keeps execution phase, heartbeat, training coordinates,
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
