# MTG-122 open decisions and assumptions

2026-10-10 — Jack Heart authorized design only. Proposed cards, fixed catalog,
four-stream cross-role assignments, three seeds and the 48-wall/64-process-hour
CPU ceiling are reversible design choices, not an approved allocation. No resource
is reserved and no waiting/automatic launch is authorized. Calibration and science
require later explicit budgets; transfer remains separately selectable.

Source inspection establishes that semantic subset/mirror admission and deck-order
balancing are missing in this base. The design chooses shared owner extensions,
not loader bypasses or a competing collector. Native proofs remain unavailable
until the later software gate; no trained candidate or human-play proof exists.

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

## Active execution goal and integration (2026-10-14)

Jack Heart set the goal to run and analyze the experiment using public Etude.
The work is no longer design-only: implement and prove the SDK-based path now.
Scientific machine/budget authorization is awaiting a response in the ongoing
conversation; no rental or scientific cohort has launched. Main 9576dc40 was
merged at fcbea2f8 via lf sync, supplying Academy namespace separation,
cross-balanced collection and explicit mirror admission. Its naming resolution
retained older Fantasia prose; Jack Heart's latest Manabot naming still governs.
The native extension has been built; bounded game-level learner/export tests pass.
Etude 0.1.0 imports successfully in an isolated Python 3.12 environment.

## SDK learner integration (2026-10-14)

Live Experiment/Trainer W&B logging is replaced by Etude; frozen config fields
and historical readers remain. Standalone and launcher-owned identity/directory
paths pass. Self-play logs cumulative coordinates and retains admitted raw stage
exports; failed attempts remain failed. Scientific TrainingRun/VerifyStore and
old scheduling/evaluation still exist and must be migrated before claiming the
requested Etude experiment path. No science launched.

Check: 19 SDK/Trainer/Ataraxos/recovery tests passed, followed by four SDK tests
including launcher environment admission. Initial strict-clock failures exposed
missing clock values and stage-local counts; fixed without clock suppression.
Open user decisions are compute allocation and whether to add standard vanilla
card definitions or restrict the simple pools to Gray Ogre and Craw Wurm.
