# MTG-174: the single Elemental baseline stopped on native legality

**Actual result, 2026-10-22: failed; MTG-174 remains incomplete.** The one
paid run stopped with `PyAgentError: illegal mana activation` during collection
following update 4. No updated checkpoint survived. Both task rentals are
confirmed absent. Inclusive conservative lifetime cost was **$0.051831584**
of the authorized $10; no replacement run, top-up or paid evaluation followed.

All new numbers below trace to [summary.json](data/mtg174/paid-run-v1/summary.json)
and its retained [TrainingRun](data/mtg174/paid-run-v1/remote/run/run.json),
[training log](data/mtg174/paid-run-v1/remote/training.log),
[control receipts](data/mtg174/paid-run-v1/control/), and
[provider absence](data/mtg174/paid-run-v1/provider-absence.json).
Fetch bytes with `uv run --extra artifacts python -m manabot.data fetch experiments/data/mtg174/paid-run-v1`.

## Frozen experiment and actual execution

The [prospective protocol](protocols/elemental-174.md) was committed before
allocation at `6006a360addbe2a3dcf093750070b45a3c659223`, retained publicly at
`mtg174-frozen-uniform-20261022` and as a [source archive](data/mtg174/paid-run-v1/source.tar).
The superseded, unexecuted short plan remains beside it. Fresh existing
`manabot-issuer` admission supported 43,140 seconds without account changes;
allocation expiry and all shutdown/export reserves remained enforced.

Job `mtg174-elemental-uniform-001`, TrainingRun
`58c2a9c8aea0464eb45281fbc1756075`, used seed 17402, w10, the exact authored
ten-deck/Lesson roster, independent uniform ordered pairs at each game root,
256 streams and one learner seat per stream. Historical 64×2, four-head,
FiLM=false Ataraxos settings were retained. Linux source/native/toolchain and
world/ABI identities are in the TrainingRun and bundle; the failed bundle
contains no Linux extension bytes or updated weights. CUDA numerical checks
passed; that does not establish native action legality.

| Recorded result | Actual |
|---|---:|
| Complete collection/update iterations | 4 |
| Learner / environment decisions | 65,536 / 131,593 |
| Retained optimizer exposures / optimizer steps | 16,384 / 256 |
| Game roots / terminal games / truncations | 773 / 517 / 0 |
| Decks / ordered pairs / mirror pairs observed | 10 / 100 / 10 |
| Pair/learner-seat cells observed | 197 / 200 |
| TrainingRun elapsed time | 34.927 seconds |
| Recorded learner decisions / TrainingRun second | 1,876.36 |

[Exposure counters](data/mtg174/paid-run-v1/exposure.json) cover the last
successful update; the failing collection can consume additional unrecorded
actions and time. Root sampling does not promise equal transition exposure.
This short startup sample is not a steady-state throughput estimate.
The minimum-25-update and no-illegal-step predictions failed; root coverage
predictions held. Outcome-score predictions and usefulness remain untested.

## Checkpoints, reporting and evaluation limits

Only initialization, SHA256
`57b5ee3d64156e41bd88934767e1b4a60371a80cc787dbd72195c88921858b50`,
was retained and [successfully reloaded](data/mtg174/paid-run-v1/reload.json).
The final manifest has no trained raw/EMA endpoint. It cannot support a gain,
confidence interval, final replay proof, strength estimate or balance claim.

The durable scientific owner was attached before rental. It registered the
frozen initialization request `25314209a04f`; source publication was still
in progress when the native failure became known. After stopping the owner,
the exact stable ID was absent from Etude's launcher: **zero accepted
scientific eval jobs and zero evaluated games**. No replacement was launched.
The native [Etude training report](data/mtg174/paid-run-v1/training-report.html)
retains the missing request and failed training projection. Its actual local
location is `.runs/mtg174-execution-v2/owner/reports/mtg174-elemental.html`.
There is no first successful scientific evaluated checkpoint or learning curve.
The earlier successful engineering evaluation remains separate evidence under
`data/mtg174/training-report-v1/`.

The frozen 500-update cadence, initialization/endpoint monitoring, disjoint
final scoring, both-seat/full-pair evaluation and mirrors were **not completed**.
The owner was disabled and its installed service removed; configuration,
ledger, stop and missing-job receipts are retained. Initialization was not
relabeled an updated endpoint. No bounded strength scoring can repair the
missing trained artifact.

## Cost, comparison and next step

[Deletion receipts](data/mtg174/paid-run-v1/control/) sum guardian
**$0.015915816** plus training/startup/CUDA checks/storage **$0.035915768**.
These use intent through observed provider absence, not a provider invoice.
No known prior task rental charges or paid evaluations are omitted.
A later broad inventory read rejected an unrelated pod shape; exact GETs
independently confirmed both owned resources absent. No other job was changed.

The retained [MTG-159 baseline](scale-ladder.md) reached update 9,250 and
151,552,000 learner decisions at $9.88 pod time, with heuristic score
0.11 initially and 0.69 [0.64, 0.735] at that checkpoint over 200 games.
Its cost excluded separately charged calibration/evaluation and its w6 world
and heuristic differ. Those weights were not imported or retrained here.
This failed w10 run supplies no comparable strength or learning-per-dollar
result. Game intervals for one checkpoint would not establish method effects
across training seeds even if evaluation existed; no cross-world Elo is valid.

The immediate next question is why the frozen native environment rejected this
collection action. Preserve this source, failure and consumed cost for engine
owner diagnosis; the log alone does not identify a scalar action-prefix
reproduction. Repair and admission evidence must precede any separately
authorized further paid execution. Adaptive sampling and curriculum research
remain separate from this failed uniform baseline.
