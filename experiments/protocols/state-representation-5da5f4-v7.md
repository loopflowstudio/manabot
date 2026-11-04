# Same-input v7 state reduction — Task 5da5f4

Prospective protocol, 2026-11-04. Jack Heart authorized this replicated learning
study within $25 total. The bounded shared-effect repair already shipped in
PR419; this experiment does not change its ABI, defaults or frozen evidence.

## Question and treatments

Does a GRU over typed records improve full-game learning relative to sum pooling
on the same complete viewer-safe v7 observations? Both arms use explicit shared
payloads and membership routing into objects before attention. `typed_pool`
sums tokens, applies tanh and a learned projection; `typed_gru` applies a GRU
then tanh/projection. This changes BOTH shared-effect payload reduction AND typed
continuation reduction over each frame's header and ordered edge tokens. Local
facts use the same FactEncoder. Neither arm retains hidden state across decisions.
The comparison does not isolate effect reduction from continuation reduction,
match parameter count, or prove injective neural encoding. `rows` is omitted:
v7 already shares its effect reducer with typed_pool and it would spend a third
of the budget on a secondary continuation contrast.

## Frozen learning and evaluation

The executable authority is `experiments/state_representation.py`, committed
before scientific runs. Training source, native world/content/ABI, architecture,
recipe digest and checkpoint digest are retained by TrainingRun. Source revisions
are explicit; old canaries remain v5 evidence at their original sources.

- Four independent training seeds: 73201, 73202, 73203, 73204. Each appears in
  both arms. Alternate treatment order within seeds. No seed/checkpoint selection.
- Existing complete-game self-play, both seats, 32 streams, one game per stream
  per update, 128 updates: 4,096 games per arm/seed, with actual decisions and
  optimizer steps reported. Same Allies/Lessons authored match and cross-balanced
  curriculum, width64/depth2, policy FiLM, no recent events or persistent memory,
  v7 with128local facts and the delivered effect capacities. Shared schedule and
  learning rule derive from batched_baseline, including its128-update schedule.
- CPU float32, one thread, sequential runs on the same local host;16GiB per-run
  memory watchdog and4200seconds per training run. Compare measured elapsed time.
  Before scientific execution, RunPod refused the available4090 offer (launch
  8dd337797e0a; no rental acquired); L40/L40S reported unavailable. The entire
  cohort therefore uses the supported local CPU fallback. No device-selected
  survivors are combined. Earlier CUDA declarations remain in git atb73d97eeb.
- Only final raw update128 is scored. Initial exports are retained but not used
  for selection. No-update controls are absent, so this is an endpoint architecture
  comparison, not proof of improvement over initialization.
- Held-out deal seeds84001–84032 against the source-pinned heuristic; four legs
  exchange both authored deck assignments and both seats:128games/endpoint,
  1,024scheduled games total. CPU float32 inference, two active games, batch16,
  catalog cache,1GiB residency. Each endpoint has a1800second outer deadline;
  each game120seconds and10,000commands. Retain ordinary replay validation.
- Calibration seed73200 and deals83901–83902 are excluded. A one-update v7
  calibration may establish runtime feasibility before the scientific cohort;
  no calibration score chooses treatment, seeds or horizon.

## Prediction, uncertainty and failure policy

Primary difference: typed_gru score minus typed_pool score within each training
seed, then mean of four differences. Predict at least+0.03 score at no more than
1.25× measured training time. Report the95% Student-t interval (df3, critical
3.182446305284263) across paired training-seed means, explicitly conditional on
small-sample approximately normal differences. Four replicates give weak power.
Per-checkpoint paired-deal bootstrap intervals (2,000replicates, seed84000) describe
within-seed evaluation noise separately; games are never training replicates.
A favorable result requires primary lower bound>0, mean>=0.03, and time ratio
<=1.25. This is a review criterion, not automatic promotion or a proof of
strategic mechanism. No broad equivalence claim follows from a null estimate.

All eight trained endpoints and every scheduled evaluation leg must complete.
Capacity overflow, nonfinite training, failure, missing/rejected checkpoint or
replay discrepancy remains in the denominator and prevents a qualified cohort
claim. Do not retry failed scientific seeds, substitute survivors, extend based
on scores or alter the world mid-cohort. Stop dependent work on a systemic error.
Report partial receipts and the concrete blocker. Never exclude playable cards
or erase state to make training finish.

## Cost and cleanup

Separate from MTG-213. Prior deleted L4 provider bill=$0.07258259388618171;
reserve$0.073. Reserve$1 for excluded admission, at most$16 for the scientific
cohort INCLUDING rental setup/idle/delivery, and$6 for storage/API/transfer and
other ancillary charges: total$23.073, leaving$1.927 unallocated. Quote the
complete cohort before purchase; reject a quote exceeding this allocation.
Retain launch failures, per-rental start/end/bills and transfer usage. Source and
artifact uploads are counted; no network volumes are requested. Deadline and
spend guardians plus launcher cleanup bound each rental. Verify deleted pods
and no unresolved rentals after execution, and publish result bytes to the
artifact store. Billing estimates and bounded reserves are not settled invoices.

Strongest confounds: the joint treatment and extra GRU parameters, a short fixed
training horizon, one authored matchup, four seeds and any GPU/environment
variation. A follow-up isolating each reducer and extending the horizon would
separate those explanations; this study authorizes neither that follow-up nor
automatic model promotion.
