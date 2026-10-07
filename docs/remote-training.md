# Remote training on RunPod

The current submit/reconnect/cancel interface and remote-owned lifecycle are in
[Remote jobs](remote-jobs.md). `deploy run` now observes that same durable job.
The compiler, hardware admission, source pinning and Bundle verification below
remain shared contracts. The ETU-114 measurements and client-owned execution
instructions below are historical evidence, reproducible at their named source
commits; they do not describe the current shutdown owner. PR253's custom SSH
calibration callbacks also require their pinned source until their caller adopts
a serializable remote workload.

ETU-114's eight retained attempts and shakedown total **$1.6591 estimated/reported**
from its separate historic allowance. Those dollars are not available to ETU-123.
Its disconnected CUDA proof uses the separately recorded reservation in the
[current remote job contract](remote-jobs.md#etu-123-proof-allocation).

## Live attempts, 2026-10-06

Jack Heart authorized **$50 total across every Task attempt**, including the
previous $0.29 shakedown, and required tens of minutes of CUDA self-play before
review. Normal non-force pushes of this Task branch are authorized to make exact
source commits fetchable; opening/readying a PR, landing and completing review
remain prohibited. Each command retains its own smaller cost/deadline limits.

Evidence stays under `.runs/remote-acceptance`; failed directories are retained.
Dollar estimates use observed rental intervals and the declared storage allowance,
not a provider invoice. The prior shakedown's $0.29 is reported expenditure.

| Attempt | Source | Result | Estimated rental cost |
| --- | --- | --- | --- |
| 000 | `d8307606` | No create: unrelated zero-price GPU quotes rejected admission | $0.0000 |
| 001 | `46c80fd2` | Pending-pod port list rejected; exact owned probe manually deleted | $0.0193 |
| 002 | `5f1c34d6` | Guardian passed; bootstrap exceeded the eight-minute setup deadline; deleted | $0.0686 |
| 003 | `5f1c34d6` | Guardian and CUDA smoke passed; first stage interrupted; all 21 bundle files verified; deleted | $0.3260 |
| 004 | `62fc7e15` | 350 updates/stage completed; records/checkpoints returned; four replayed arena games; deleted | $0.3661 |
| 005 | `62fc7e15` | 600 updates/stage completed; four replayed arena games; deleted | $0.5010 |
| 006 | `88375ad0` | Local-disk setup; 16 CUDA updates; four replayed arena games; deleted | $0.0450 |
| 007 | `f4189656` | Two experiments on one pod; setup once; both returned and loaded; deleted | $0.0430 |

Attempt 001 retains the original unresolved receipt plus explicit manual
reconciliation. An authenticated query observed the unique receipt-owned pod at
$0.49/hour, 6 vCPUs and 62 GB; the corrected client deleted that exact name and
confirmed empty inventory. Its cost conservatively ends at the later reconciliation
confirmation. Pending `ports=["22/tcp"]` declarations now mean no public mapping;
only published `portMappings` supplies an SSH endpoint. The regression covers both.

Attempt 002 demonstrated real pod-side deletion without laptop DELETE, then hit
its setup deadline during dependency installation. Attempt 003 preserves the same
1,200-update/two-stage recipe and 45-minute training allowance, increasing setup
to 20 minutes and the total command allowance to 75 minutes ($0.775 projected
ceiling). Setup completed in roughly ten minutes. Network-backed `/workspace`
and uv's cross-filesystem copy warning were observed; no isolated throughput
explanation is claimed.

The rented-hardware command `uv run --locked --extra dev pytest
tests/remote/test_device.py -q` passed **2 tests in 189.74 seconds**, including
actual CUDA optimizer work, continuation and raw/EMA CPU reload. It overlapped
main training and is functionality evidence, not a performance benchmark.
Its logs, exit status and fixture artifacts are retained inside the returned
bundle. Pytest's two convenience symlinks were recorded separately and removed
before bundling; all underlying artifact bytes remain intact.

The first main stage stopped after 567 updates / 290,304 optimizer exposures
in 1,354 recorded seconds. It is an interrupted run, not a completed deployment.
Attempt 004 reduced each stage to 350 updates without editing an active run.
It completed in **1,663.91 seconds (27.73 minutes)**: 2,213 self-play games,
369,748 native decisions, 179,200 learner transitions and 358,400 optimizer
exposures. Both stages used `cuda:0` and exported raw, EMA and Adam artifacts.
The selected raw checkpoint SHA-256 is
`d380bc42de41402411a608656d06a21762c9e794d0b65ee3e73a48f417d0ce1a`.
The original run/SQLite and all 12 declared bundle files verified. Four ordinary
arena games covered both seats/deck assignments and completed with exact replay
in 11.75 seconds. Final authenticated inventory was empty. Prior Task expenditure,
including the shakedown and all failures through attempt 004, was **$1.0701 estimated/reported**.

Jack Heart then requested an immediate longer run after confirmed deletion.
Attempt 005 restores 600 updates per stage, with one-hour stage watchdogs,
a two-hour run allowance and a 150-minute rental allowance ($1.55 projected
ceiling). The elapsed-budget schedule uses the larger denominator too; this is
a feasibility run, not a controlled timing-only or strength comparison.
It completed 1,200 updates, 3,131 games, 307,200 learner transitions and
614,400 optimizer exposures in 2,577.73 seconds. All returned artifacts verified;
four terminal arena games replayed exactly in 10.72 seconds. Inventory was empty.
Total through attempt 005 is **$1.5710 estimated/reported**, including shakedown.
ETU-119 owns the separate persistence fix; no such change entered these attempts.

Saved update coordinates provide wall intervals without changing training.
Attempt 003's median interval rose from 1.69 seconds over early updates to
2.24 seconds over updates 500–567. In attempt 004, stage-one medians over
successive blocks were 1.58, 1.64, 1.77 and 1.91 seconds; stage two was 1.78,
2.03, 3.10 and 2.36 seconds. Blocks use diagnostics [1,100), [100,200),
[200,300), [300,350), differencing adjacent cumulative coordinates.
These intervals include ordinary work and preceding persistence; they do not
isolate the save cost. On failure, the executor assigns unaccounted time to its
last active phase, so failed-run `learning_seconds` is not pure optimizer time.
Jack Heart's separately reported laptop persist timings identify a plausible
contributor, not a measured complete explanation of pod slowdown.

Local checks: **405 passed, 3 skipped** across remote, training and net-opponent
tests. These checks do not substitute for the live deployment. No strength,
scientific comparison or chapter acceptance follows.

## Compile and deploy

Commit source first. Compilation reads source identity but makes no provider
request. Deployment additionally checks that the exact commit/tree are publicly
fetchable from this repository before renting. Push the exact commit to the Task branch before the live gate. A source push
does not require opening or readying a PR for review.

```bash
uv run manabot deploy compile --regime experiments/regimes/direct-self-play.json \
  --mix ops/mixes/runpod-small.json --seed 197 --out .runs/remote-plan.json

doppler run --project etude --config prd -- uv run manabot deploy run \
  --plan .runs/remote-plan.json --out .runs/remote-example
```

The single-command form takes `--regime`, `--mix` and `--seed` in place of
`--plan`. It prints the complete plan before provisioning and saves it with the
private deployment receipt. There is no interactive prompt. Use a fresh output
directory for every attempt; never replace failed receipts.

The hardware mix declares GPU alternatives in priority order, one worker, vCPUs,
RAM, thread limit, image digest, disks, hourly ceiling, dollar cap, wall time and
setup/transfer/cleanup reserves. Threads resolve to the smallest of requested
threads, rental vCPUs, mix thread limit and the executor's four-thread maximum.
The pod's reported host CPU count is telemetry only. GPU allocation and returned
CPU/RAM/rate are admitted before bootstrap. The first implementation deliberately
caps each deployment below $5; the larger weekly allocation is not a scheduler.

Bootstrap uses uv 0.8.22, Rust 1.98.1, locked dependencies and Python 3.12. The
exact source tree binds bootstrap/guardian code. CUDA is passed to the model,
collector and optimizer owner; unavailable CUDA fails without CPU fallback.
Phase boundaries synchronize the GPU. CUDA receipts add actual device, memory,
capability and runtime; the returned toolchain record includes driver version.
CPU defaults and receipt fields retain their prior meaning.

## Local-disk setup measurement

Source and its `.venv` live at `/opt/manabot/repo`, with uv's cache alongside at
`/opt/manabot/uv-cache`. Only returned evidence and input recipes use `/workspace`.
The 20 GB pod volume is deleted with the pod; no independent persistent volume is
created. Container storage remains 30 GB. This is a fresh install, not a prebuilt
image or a warm-environment claim.

On 2026-10-06, attempt 006 measured **53 seconds for uv sync, 20 seconds for the
native build, and 90 seconds for the complete bootstrap**. The comparable interval
from `.venv/pyvenv.cfg` to final `toolchain.txt` was **71 seconds**, versus **486
seconds** on attempt 005's network-volume environment. That interval includes
install and build; the older observation is not a direct uv timer. Both are
single observations on different rentals, and the newer source includes PR #250;
this supports the placement choice, not a controlled hardware benchmark.

The short run completed 16 CUDA updates in 27.76 seconds, returned verified
raw/EMA records, and played four exact-replayed games in 6.51 seconds. The pod
was deleted and inventory was empty. Task cost through 006 is **$1.6160** including
the reported shakedown. The measured bootstrap plus guardian/provisioning fits the
existing five-minute example setup reserve; longer earlier attempts used twenty
minutes. Keep extra reserve for larger dependencies or slow availability.
`bootstrap-timing.json` travels with the immutable evidence bundle.

## Worked example: two experiments on one rental

[ops/examples/two_experiments.py](../ops/examples/two_experiments.py) keeps one
machine for two seeds of a small regime at the **same exact committed source**.
It uses the existing compiler, price/resource admission, startup guardian,
transport, bundle verifier and cleanup. It adds no public command or reuse mode.
The normal deployment command continues to rent a fresh pod and delete it.
Changing dependency/native inputs between experiments is outside this example;
do not silently reuse an incompatible environment.

Prepare the small recipe, commit and non-force push any source edits, then run:

```bash
uv run python - <<'PYCODE'
import json
from pathlib import Path
recipe = json.loads(Path('experiments/regimes/direct-self-play.json').read_text())
recipe['wall_seconds'] = 300
for stage in recipe['stages']:
    stage['updates'] = 8
    stage['execution']['wall_seconds'] = 150
    stage['learning']['ema'] = 0.9
Path('.runs').mkdir(exist_ok=True)
Path('.runs/two-experiments.json').write_text(json.dumps(recipe, indent=2))
PYCODE
doppler run --project etude --config prd -- uv run python ops/examples/two_experiments.py \
  --regime .runs/two-experiments.json --mix ops/mixes/runpod-small.json
```

Setup happens once. Both training commands use `uv run --no-sync`, assert the
pinned commit/tree and create separate TrainingRuns/databases for seeds 197 and
198. Every artifact returns and ordinary raw/EMA loading must pass. The original
30-minute allowance never extends: the pod guardian deletes at minute 28, with
two minutes reserved for laptop cleanup. The laptop also bounds SSH and deletes
in `finally`. The default path's guardian probes established scoped deletion on
this image; the example uses the same startup guardian without another probe.

An idle L4 still costs **$0.49/hour compute**, plus storage (the plan reserves
$0.02/hour). Keeping it live saves setup, not rental charges. `experiments.json`
separates first-run setup, idle gaps, training command and transfer time;
TrainingRun retains training phase costs. `deployment.json` includes the entire
rental, including setup and teardown. Records stay under the same ignored
`.runs/remote-acceptance` ledger; unresolved prior costs/deletions prevent another
rental and the example retains its conservative $4.90 ledger ceiling.
Use `deploy status` to inspect live count/hourly compute and `deploy cleanup`
with the recorded deployment receipt if cleanup is unconfirmed.

The example ran on 2026-10-06 as attempt 007 at `f4189656`. Both experiments
completed 16 CUDA updates / 8,192 optimizer exposures and returned hash-verified
TrainingRun/SQLite plus raw/EMA/Adam exports. Ordinary policy admission passed.
The second command performed no dependency sync or native build.

| Seed | Setup including provisioning | TrainingRun time | Command time | Transfer/admission |
| --- | ---: | ---: | ---: | ---: |
| 197 | 114.57 s | 30.06 s | 40.64 s | 54.25 s |
| 198 | **0 s** | 25.58 s | 34.14 s | 55.64 s |

Bootstrap itself was 77 s (uv sync 36 s, native build 22 s). Recorded idle gaps
were below 1 ms; this back-to-back run is not an idle-throughput benchmark.
Command time includes CLI startup/bundling; TrainingRun time is a subset, not an
additional cost. Laptop cleanup confirmed deletion and a fresh provider query
found **zero pods**. The rental estimate was **$0.0430**. Including every retained
failure and the earlier $0.29 shakedown, Task expenditure is **$1.6591**; no
provider invoice has been obtained. No stopped pod or persistent volume remains.

The final remote suite passed 47 checks (one local CUDA-host skip); focused merged
persistence/continuation checks passed three. The two-run example also has local
success/second-run-failure cleanup coverage. These are workflow proofs, not
scientific comparisons. PR review and landing remain outside this execution.

## Deadlines and cleanup

Every command first rents a short guardian probe, then waits for pod-side deletion.
Training cannot start unless the probe disappears at its deadline without laptop
DELETE. Both rentals share the same command budget and absolute outer deadline;
there is never more than one active rental. The startup guardian is independent
of Python/bootstrap/SSH. It uses the image's provider-supplied pod-scoped
runpodctl configuration, never the laptop account key. That credential's self-delete capability was observed in the retained live probes.

The laptop bounds API/SSH operations and attempts deletion in `finally`, including
interruptions. Provider DELETE must be followed by confirmed absence. A failed
create is reconciled by its unique name; it is never retried speculatively.
Unobserved ambiguous creates stay unresolved even if a brief inventory is empty.
Provider-ordered GPU alternatives use one create request. Unrelated rentals are
never deleted.

```bash
doppler run --project etude --config prd -- uv run manabot deploy status
doppler run --project etude --config prd -- uv run manabot deploy cleanup \
  --deployment .runs/remote-example/deployment.json
```

`CLEANUP UNCONFIRMED` means possible continued billing, not success. Keep the
receipt and retry cleanup. Estimated cost includes observed rental time and the
declared storage allowance; it is not a provider invoice. Price admission is not
an atomic spending cap. Provider outage, a container that never starts, or laptop
loss before startup can defeat the two guards. No independent persistent volume is created.
The provider's [delete operation](https://docs.runpod.io/api-reference/pods/DELETE/pods/podId)
terminates the rental; stopping alone is insufficient because storage remains.

## Returned evidence

Evidence lives beneath the ignored, permission-restricted deployment directory.
The closed SQLite database, original run.json, optimizer files, raw/EMA policies,
toolchain record and training log return unchanged. `bundle.json` binds every
file's producer path, relative destination, size and SHA-256. `Bundle.resolve`
locates a verified local artifact without rewriting producer paths or costs.
Traversal, symlinks, missing files and changed bytes fail admission. TrainingRun
export and authoritative database must agree; only raw/EMA files enter the normal
policy loader. Optimizer files are verified opaque artifacts, not policies.

Incomplete transfers retain whatever arrived, then deletion proceeds. Training
errors retain their closed evidence when retrieval is still possible. Timeout
may prevent retrieval; partial evidence never becomes a completed training run.
Portable process recovery and transparent relocation for all historical report
consumers are not supported.

## Explicit live gate

The helper is not collected by pytest. Prepare a bounded version of the existing
recipe with EMA enabled, then invoke it after the source push:

```bash
uv run python - <<'PY'
import json
from pathlib import Path
recipe = json.loads(Path('experiments/regimes/direct-self-play.json').read_text())
for stage in recipe['stages']:
    stage['learning']['ema'] = 0.9
Path('.runs').mkdir(exist_ok=True)
Path('.runs/remote-proof-regime.json').write_text(json.dumps(recipe, indent=2))
PY
doppler run --project etude --config prd -- uv run python -m tests.remote.live_acceptance \
  --regime .runs/remote-proof-regime.json --mix ops/mixes/runpod-small.json
```

It conservatively limits retained deployments under `.runs/remote-acceptance` to
$4.90 and refuses unresolved prior cost/deletion. The Task totals above separately
include the earlier $0.29 shakedown; the authorized $50 total is not a spending
target. Success requires CUDA
optimizer exposures, raw/EMA reload, complete returned records, and one four-leg
paired-deal arena block through the existing evaluator: both decks/seats,
terminal completion and exact Command replay. Arena cost is separate from rental
cost. This proves the workflow, not strength or chapter acceptance. Do not delete
that directory between retries or infer live acceptance from local fixture tests.

Cleanup recovers estimates from retained or newly observed rental rates, the
original hash-bound plan's storage allowance and intent-to-confirmed-absence
elapsed time. Missing rates, unobserved creates, conflicting rental evidence or
missing/changed plans leave billing unresolved. Repeated cleanup preserves the
first confirmed deletion time; deletion alone never supplies a zero cost.
The helper records initial inventory and requires final inventory to be empty
when it started empty. Otherwise it requires no owned pods; unrelated rentals
are counted and never deleted. Its arena input uses checkpoint training coordinates.
The paid proofs above are complete. A full mocked default-deployment success and
training/transfer timeout matrix is not claimed by the existing tests.
