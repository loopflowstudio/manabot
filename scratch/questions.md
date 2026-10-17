# MTG-122 open decisions and assumptions

2026-10-14 — Jack Heart authorized running and analyzing MTG-122 through the
public Etude SDK. Jack Heart permits adding vanilla creatures without abilities.
Scientific compute may use Mini for up to 48 hours OR GPU for up to $25;
local experiment execution is forbidden. These alternatives are not additive.
Calibration, setup, evaluation, failures and cleanup count inside the selected
cap. Existing cohorts and evidence must not be displaced. Mini is the preferred
route, subject to capacity/admission; no campaign has launched.

The earlier design-only hold and proposed local 48-wall/64-process-hour allocation
are superseded. Exact content, source freeze, calibrated counts and remaining
SDK deployment/reporting software must be admitted before scored execution.
Catalog-size and curriculum comparisons remain separately scoped.

## Vanilla-only basic pool (2026-10-13)

Jack Heart requested removing haste and first strike from L1. The design now
requires creatures with no abilities and carries that base into L2. Replacement
definitions and resulting colors/curve remain unresolved; the old keyword
creature choices are withdrawn, not edited into fake vanilla versions.
This changes design only and authorizes no implementation or experiment.

## Catalog-size comparison requested (2026-10-13)

Jack Heart selected a separate minimal-versus-full catalog comparison on identical
simple games. This is design/capture direction, not a training budget or launch.
Keep MTG-122's three-world comparison fixed-catalog. Proposed separate Intelligence
Task: measure resource and learning costs of unused semantic catalog entries.

Hold native world/decks, active canonical definitions, embedding width, shared
encoder, learner and evaluation fixed. Match common-token embeddings and shared
parameters by semantic identity, not seed alone. Measure definition count, unique
semantic-token count, program lengths/padding, parameter/optimizer bytes, peak
memory, encoding and full-update latency, samples/time and learning at matched
updates and matched elapsed cost. Use independent seeds, frozen controls and
held-out paired deals. Resolve exact roster and bounded calibration budget before
execution; preserve product bindings and other frozen cohorts.

Static source: SemanticCardEncoder embeds token-kind/value pairs, not one learned
row per card; definitions() encodes every catalog definition before visible-row
selection. Extra unused definitions can cost compute without task gradient. More
genuinely encountered cards is a different exposure/interference question.
Prediction: unused catalog expansion primarily hurts resources and progress per
wall time, not learning per matched update after identity-matched initialization.
No runtime measurement establishes magnitude.

Task filing pending: Intelligence Wave status did not return during this pass,
so overlap/ownership could not be reconciled. No duplicate Task or worker launched.

## Project naming and SDK migration (2026-10-14)

Jack Heart confirmed Manabot names the whole project, manabot.manabot the training
system, and manabot.academy the play web app. Etude is the external experiment
library at loopflowstudio/etude, distributed as etude-sdk. Jack Heart requested
running this experiment through that SDK and replacing corresponding local
experiment infrastructure, not the engine or learning algorithms. Broader removal
scope and machine/budget remain to be fixed before scientific execution.
Public etude-sdk 0.1.0 is pinned and installed. The integrated academy/ rename
removes the old etude import collision. Documentation distinguishes current
components from historical identifiers.

## Current execution direction — 2026-10-17

Jack Heart reactivated the three-pool study on GPU with 64 streams and a separate
$35 inclusive cap. This supersedes the Mini preference and older allocations
above. Calibration/setup/evaluation/failures/storage/cleanup count inside $35;
initial calibration has at most $2, settlement reserve $3, science at most $30.
No local scientific execution. The same learner trains one fixed cross-role
matchup per pool; future diverse-deck curriculum is not bundled into this run.
The goal is full learning/cost/decision-horizon understanding that can guide that
future curriculum, not a cross-world raw-score leaderboard.

Baseline execution and monitoring transferred to MTG-137; its existing jobs and
allocations are separate. Historical baseline launch/cadence notes and receipts
remain in this file at `fc170aed` and in `experiments/corrected-world-baseline.md`.
Do not interpret those old PID/job observations as current liveness or restart
those jobs here. Original private `.runs/mtg122` evidence must remain intact.

GPU preparation preserves the old CPU declaration and uses explicit CUDA/64-stream
calibration. The fixed 64-transition horizon means 4,096 learner rows/update;
MTG-135's 64-stream/8-transition measurements cannot supply its timing. Full
GPU plan admission and source/transport/dependency preflight still precede rental.

Check (2026-10-17): fourteen declaration/campaign/memory-guard tests and focused
Ruff pass. GPU plan v2 binds CUDA/64 streams and the inclusive cost ceiling while
retaining all 36,000 evaluation games. The existing `c8946070` child-RSS race fix
is reused without baseline-runner changes. Mocked executor checks establish
recipe selection/failure retention, not GPU runtime or scientific evidence.

Read-only RunPod quote at 20:52 UTC: L4 available, $0.59/hour, retained in
`.runs/mtg122/gpu-complexity/quote.json`. The repository-documented `etude/prd`
secret scope supplied it; `manabot/prd` lacked the provider variable and made no
provider request. No credentials changed. Remote source-upload, publication and
setup preflight remain before the uniquely owned calibration rental. No rental
or scientific execution occurred in this pass.


## GPU calibration submitted — 2026-10-17

Exact source `5ed397ae`, 1,977 files, compressed archive 80,335,492 bytes passed
multipart upload/full readback and scoped worker publication. Private receipts
and the single-owner controller are in `.runs/mtg122/gpu-complexity/`.
Run `b689bba03c66`, session `4f58be96e16f`, attempt `79d938953765` is admitted
for seed12250, three pools ×100 updates, 64 streams ×64 transitions, and48
complete replayed calibration games. Cap $2, job2.5h, pool max rental2.8h;
Etude owns guardian/deletion. No scientific cohort is admitted by submission.
SDK history is isolated from MTG-137 and local pool disabled. The first detached
shell exited before intent/queue; foreground controller owns the one retained
intent. Do not launch a replacement: reconcile this exact run and preserve any
failure. Runtime completion, costs and scientific admission remain unverified.


## Laptop interruption and corrected-world admission repair — 2026-10-17

Jack Heart reported closing the laptop and authorized retry, deferring an Etude
bug report. Original run `b689bba03c66` has a cancellation request; provider lookup
found no matching pod, while SDK kept a conservative `renting` estimate. Bootstrap
was absent; source upload was incomplete. A later source-fetch probe succeeded.
No original training is established; its uncertain ledger stays separate.

Retry `cdad6c0a67fd` used exact-source server-side S3 copy and reached provider
RUNNING (`lmdo5yr6046fbg`) and native build. It failed before learning: the GPU
recipe inherited w4 while the runtime was w6. SDK reports failed and rental ended,
$0.01595487 estimated. This is a runner preparation defect, not a learning result.
The new CUDA derivative now explicitly pins w6 (rejecting other native worlds);
historical CPU recipes and evidence remain unchanged. Fifteen focused tests now
include ordinary native-world admission for each CUDA pool, not merely recipe
construction. New source freeze and verified cleanup precede further submission.
Private controller/attempt evidence: `.runs/mtg122/gpu-complexity-retry/`.


Corrected-source calibration `970a26b94c0f` pins `0cd95fa8` and is setting up on
GPU session `d4416e8e387d`; private owner/receipts live in
`.runs/mtg122/gpu-complexity-w6/`. Two pre-rental source uploads hit SDK transport
timeouts; 5MiB parts/180-second preflight transport passed full readback. Launch
uses exact-digest S3 server-side copy, avoiding laptop upload during rental setup.
No SDK installation or frozen source was patched. Reserve the original uncertain
$2 cap, $0.02 for the settled world-tag failure, $2 for this calibration, $3
settlement and at most $27.98 for science. All remain inside $35. These conservative
reservations are not invoiced costs. Training has not yet been observed.
