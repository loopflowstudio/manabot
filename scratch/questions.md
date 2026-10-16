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
paths pass. Self-play logs cumulative coordinates and retains admitted initialization,
monitoring milestones and raw stage exports; failed attempts remain failed. Scientific TrainingRun/VerifyStore and
old scheduling/evaluation still exist and must be migrated before claiming the
requested Etude experiment path. No science launched.

Check: 19 SDK/Trainer/Ataraxos/recovery tests passed, followed by four SDK tests
including launcher environment admission. Initial strict-clock failures exposed
missing clock values and stage-local counts; fixed without clock suppression.
Open user decisions are compute allocation and whether to add standard vanilla
card definitions or restrict the simple pools to Gray Ogre and Craw Wurm.

Check (2026-10-14): SDK initialization/midpoint/end hashes and coordinates passed
real scalar/categorical learner tests; 15 cadence checks passed. The pause test
missed a learner finishing between five-second polls (retained run was completed,
not failed); requesting pause at worker startup removed that test race and its
recovery check passed. Scientific launch and actual Etude placement remain open.


## Direct Etude worker (2026-10-14)

TrainingJob now freezes single-stage fresh self-play into the public SDK Job,
with explicit source/target/lifetime and one attempt. Its real subprocess command
runs the ordinary learner, records initialization/milestones/end in SDK tracking,
and writes etude-result.json only after target and policy admission. No legacy
ExperimentSchedule runner is used. TrainingRun/VerifyStore remain scientific
receipts. Late deadlines, target mismatch, occupied outputs and policy-only
continuation fail explicitly. Cancellation retains failure rather than a partial
success. This does not prove actual SDK placement or portable source setup;
native extension provisioning, simple catalog admission, matrix evaluation and
campaign execution/analysis remain. Do not mistake a constructed Job for a launch.

Check: 15 direct-worker/SDK/real-learner tests passed (including a real subprocess
and retained cancellation receipt); focused ruff and diff checks passed. No
scientific cohort, rental or source-frozen experiment was launched.


## Catalog admission prerequisite (2026-10-14)

The encoder now obtains native/IR bindings through an explicit policy catalog
registry, requiring matching pack/IR/source receipts and complete native coverage.
CatalogShape separates definitions, unique tokens, expanded sequence length and
padding. Selected product counts are 31 definitions, 163 unique tokens, 2200
expanded positions, maximum 148, and 4588 padded positions. These are static
representation counts, not timing or learning results. Fixed-seed original state
and output bytes were unchanged by extraction; nine model/learner tests passed.

Trying the already-native Jeong pack exposed an unrepresented Rebel subtype.
It was not admitted or silently added to the product schema. The existing schema
also lacks Ogre/Wurm: research content needs a separately bound schema and
checkpoint identity, not a global schema rewrite. Only the product policy pack
remains registered. Simple-world admission is still open pending exact vanilla
roster. The budget and roster questions were presented again in this conversation;
no scientific run is authorized by this software proof.


## Exact compiled-pack reload (2026-10-14)

Ordinary checkpoint loading now reconstructs the saved compiled pack selection
alongside decks/sideboards, then checks the complete runtime binding. Previously
both explicit product mirror checkpoints fell back to general content and failed
reload. Both now reload without relaxing roster, manifest or sideboard checks;
native setup rejection is translated to the existing binding error. This is a
prerequisite for a research pack with more than two registered deck roles, not
proof that that pack exists. Check: 25 world/catalog/Etude-worker checks passed;
focused lint and diff checks passed. No scientific games were scored.


## Shared authored evaluation matrix (2026-10-14)

AuthoredMatchup now names an exact native pack and two role keys. The existing
arena/MonitorProtocol accepts that roster, binds a separate suite identity,
executes all four role cells in both seats, and replays the actual retained
Commands. The training matchup must equal the explicit roster's cross setup;
source/world/ABI checks and checkpoint repetition admission remain. Omitted
rosters preserve existing selected-world protocol serialization. This is not
permission to load Jeong through the unsupported neural schema.

Check: 24 monitoring/matrix/legacy mirror-plan tests passed in 46.81s, including
8 complete native Jeong anchor games and 8 complete ordinary selected-checkpoint
games through the explicit matrix with replay. Bounded fixtures establish the
shared path, not learning or the proposed simple-world cohort. Etude evaluation
records, research dashboard role labels, simple native/schema content and the
actual campaign remain unfinished; compute/vanilla decisions remain requested.


## Etude evaluation publication (2026-10-14)

The public SDK publisher consumes retained monitor.json, verifies checkpoint
bytes, recomputes cohort admission/whole-deal intervals, and records whitelisted
outcomes with coordinates. Incomplete cohorts become failed evaluations without
strength estimates; n counts deals, never training seeds. Original arena/replay
receipts remain authority. This does not certify untrusted historical replay,
archive local artifacts remotely, or import pre-manifest admission failures.
Each publication key includes the source result digest and a unique publication
suffix so SDK episode files are never appended twice under the same key. Reports
must deduplicate publications by evidence digest, not count them as new attempts.

Check: 23 publisher/monitoring checks passed, including actual checkpoint arena
results and deliberate incomplete games; three focused publisher checks passed
after the final key change. Source-byte changes and fabricated summary estimates
are rejected. No scientific cohort launched. Remaining work includes SDK campaign
orchestration, evidence-deduplicated research reporting, native/simple schema
admission and calibration after the pending vanilla/compute decisions.


## Paired learning analysis (2026-10-14)

The offline MTG-122 analysis now admits matched candidate/frozen-initialization
results, verifies the producer initialization receipt, and subtracts scores before
whole-deal resampling. Four cells follow the candidate's actual deck/seat; the
cross-role A/B cell uses legs 0 and 3. It preserves coordinates and source hashes.
The separate three-seed summary requires the entire declared seed cohort and
matching protocol/recipe/world/milestone, reports the df=2 t interval without
clipping, and suppresses incomplete cohorts. No game-level CI substitutes for
training-seed uncertainty. The experiment metric guide owns formulas/limits.

Check: nine paired-analysis/publisher tests passed; lint/diff checks passed.
These are synthetic arithmetic/admission proofs, not learned improvement. The
campaign still needs time-to-progress/cost accounting, notebook assembly,
simple content/schema, declaration/calibration and actual execution. Pending
vanilla-card and machine/budget choices are unchanged.


## Observed progress accounting (2026-10-14)

MTG-122 analysis now distinguishes interval-censored threshold crossing confirmed
at endpoint, right-censored non-attainment, an unconfirmed earlier crossing,
and unavailable/missing cohorts. It uses planned checkpoints only, never
interpolates, and rejects mixed seeds/initializations/experiment identities.
Update, transition, exposure, active-learning and executor-elapsed clocks stay
separate. None reconstructs inclusive campaign cost. Check: ten synthetic paired
analysis/censoring tests passed; these do not measure learning. Full cost ledger,
notebook, simple native/schema admission and the authorized campaign remain open.

## Research content admission (2026-10-14)

The research pack adds Savannah Lions (W2/1), Glory Seeker (1W2/2) and Hill Giant
(3R3/3), all ability-free. L2 adds Bolt and unchanged Yip Yip!; none of the vanilla
creatures is an Ally. Six exact roles share the full research catalog, including
unchanged product decks/programs. Its explicit native pack and extended learning
schema are hash-bound by ordinary checkpoint admission. Original product JSON
bytes stay unchanged; the general native catalog gains three definitions and a
new digest. No frozen artifacts are adapted. The generator is reproducible.

Check: native debug roster test, release extension build, two static content tests
and six catalog/reload checks passed; the 20-test product/catalog/world suite
passed before adding the three research reload cases. These are software tests,
not local calibration/training. Remote full-game admission and the actual
calibrated campaign/report remain unfinished.

## Mini calibration admitted (2026-10-14)

Etude run `ff9f4cd90d03`, session `372ed2d82ced`, source `5f007a18` owns the
first remote calibration attempt: one hour including native build, seed 12250,
100 updates per pool and one eight-leg deal per anchor/pool. All three recipes
use the common catalog and four-stream cross-balanced collection. The isolated
launcher disables local placement; Mini had no unfinished SDK attempts or
observed learner process at admission. The 48-hour allocation includes this
attempt and failures; no GPU spending is combined. Source/launch receipt is
`.runs/mtg122/allocation.json`, SDK authority `.runs/mtg122/launcher`.
Two declaration/remote-guard checks passed. Native build was observed running;
no calibration or scientific result is claimed yet. A remote build failure,
incomplete matrix or timing projection that cannot fit remains evidence, not
permission to replace the cohort silently.

Calibration attempt `ff9f4cd90d03` failed before learner construction: SDK archives
lack `.git`, but TrainingRun source identity required it. The existing verified
SourceBundle path now admits launcher-submitted exact source hashes and records
the remotely built native digest. Two tamper/admission tests passed. No fabricated
Git checkout or source-identity fallback was added. Source `330d365b` retries only
after this repair, retaining the original failure. Inclusive first-attempt charge
is 114.739966 seconds; retry is capped at the remaining 3485.260034 seconds of
the same one-hour calibration reserve. Exact retry admission remains in
`.runs/mtg122/calibration-retry-source.json`; the scientific cohort is not started.

## Scientific execution path prepared (2026-10-14)

The repaired Mini calibration `60cfb89655fa` was observed at 90 vanilla-pool
updates (310.48 active seconds), still running; this is neither its final timing
nor a scored result. The frozen campaign module now resolves the three cases
through Experiment/TrainingRegime, admits 1000–10000 common updates from complete
calibration at 1.5× cost margins, and preserves all 36,000 planned games. Nine
fits rotate pool order by seed; initial/mid/end deals are disjoint, midpoint/end
controls use the exact initial bytes, and parameter hashes check paired starts.
SDK publication retains every arena cohort. Missing fits/cells stop with failed
evidence rather than reducing counts. Declaration tests pass; scientific runner
has not executed and still needs measured admission and final reporting.

## Calibration complete; scientific cohort launched (2026-10-14)

Etude calibration `60cfb89655fa` succeeded: all three 100-update runs and48
complete replayed games. Original outputs are retained on Mini and copied to
`.runs/mtg122/calibration`. No calibration scores select scientific candidates.
Global-worst-cell cost applied to every game rejected the draft budget; the
pre-scoring protocol amendment instead reserves each of six pool/anchor panels
at its own worst measured cell/coordinator cost, with the same1.5× margin.
All36,000 games and three seeds remain. The calibrated target is1,200 updates,
midpoint600;26.833692 evaluator hours plus18 learner hours and600 seconds reserve
total45.000359 hours. No sparse timing guarantee is claimed.

Scientific run `030a6ea72338` was admitted through Etude on Mini, exact source
`16636408`, plan digest `f5bff8cd0a5b77fdd5898943f4bcb656e92d5800f4ccdc023b21c4cbfbf0500c`.
Its46.75-hour cap plus one-hour inclusive calibration reserve and15-minute
external setup/settlement reserve remain within48 hours. No GPU/local training.
The fixed protocol/plan live in `experiments/environment-complexity.md` and
`experiments/environment-complexity-plan.json`. Launch authority
and attempt receipt: `.runs/mtg122/science-launch.json`, SDK database beside it.
The original calibration convenience SDK checkpoint names collided between child
pools; original nested TrainingRun exports remain intact. Scientific child output
directories now isolate those copies. Four declaration/cohort tests passed.
Scientific learning, final analysis, editable notebook/HTML and archival are still
unfinished; admission is not a result. Do not edit the frozen remote source.

Scientific attempt `030a6ea72338` failed before learning because its SDK Step
constructor omitted the required meaning. The repaired constructor passed a real
tracker init/log/finish fixture (no games). Retry `063198864bfb` is live on Mini;
session `6e352edc3173`, attempt `6070eba01d26`. Exact source/cap/charged prior time
live in `.runs/mtg122/science-clock-repair.json`. Plan digest and all scientific
counts/seeds/recipes remain unchanged; no learned attempt was replaced. First
`l1-12251/training/run.json` confirms learning has begun. SDK root progress does
not aggregate child update clocks; nested TrainingRun receipts own per-fit work.
The failed first scientific attempt remains retained and charged.

## Offline scientific report (2026-10-14)

`complexity_report` reads frozen plans/TrainingRuns/arena results and generates
a create-once editable notebook, read-only HTML and hash-bound analysis JSON.
It preserves four role cells and independent-seed versus whole-deal uncertainty,
rejects protocol/source mismatches, and shows missing cohorts as unavailable.
Reported executor/evaluator components do not invent inclusive settlement.
Four report/declaration checks passed; real partial Mini evidence rendered
headlessly at `.runs/mtg122/report/comparison.html`. This is a saved snapshot,
not a live service or a learned result. No training/evaluation ran locally.
The frozen scientific source was not edited; shared protocol construction in
the checkout now also admits the offline reader. Final interpretation and
all-attempt cost/artifact settlement remain open.

## Corrected-world restart authorization (2026-10-15)

Jack Heart authorized restarting the study once both PR288 and PR289 land.
Fresh GitHub reads show PR288 merged at `78bf9429a628d1f4a2a0487199ccce1aa5ac7684`;
PR289 remains open. Restart is gated on both merges, integration and a new
source/world/architecture freeze, fresh initialization and bounded calibration.
PR288 fixes attack/decline aliasing; PR289 proposes w6/layout2 and further
object/action identity repairs. The running w4 cohort is bug-affected evidence,
not corrected-world results; preserve every attempt and charge its time. No
budget extension was requested or granted: the existing 48-hour Mini ceiling
(or alternative $25 GPU ceiling, not additive) remains binding. Reassess remaining
capacity before launch; ask if a complete rerun cannot fit. No restart or live
source modification occurred with this authorization. The current job has not
been cancelled; explicit cancellation was not yet confirmed.

## Baseline-first redirection and cancellation (2026-10-15)

Jack Heart deferred the card-pool comparison in favor of reproducing the best
supported training recipe on corrected Allies/Lessons first. Jack Heart explicitly
approved stopping the bug-affected Mini study and redirecting its remaining budget.
Both PR288 and PR289 are merged (PR289 `3b3e35fde9f7cc7819f391f657f47c7d0aec6576`).
Etude run `063198864bfb` now reports terminal `done/cancelled`; receipt is
`.runs/mtg122/cancel-w4.json`. The frozen campaign JSON retains its last `running`
status and is not rewritten: it records three completed seed12251 pool fits and
42574.162285 seconds at that progress point, not final inclusive settlement.
Remote originals remain retained; local copying targets `.runs/mtg122/science`.
No corrected-world job has launched. Baseline preparation must settle prior costs,
review the strongest supported recipe (ETU-118 is an evidence-backed candidate,
not yet declared the winner), integrate w6/layout2, and calibrate on Mini before
freezing counts. Fresh initialization, independent seeds, fixed greedy/random
anchors and deck-role cells remain intended; no historical score is a w6 baseline.
The 48-hour Mini ceiling is not reset. Complexity and curriculum remain deferred.

## Sustained horizon correction (2026-10-15)

Earlier horizon/budget deliberation remains at `cd72366b`; subsequent decisions below supersede it.

Jack Heart approved72 inclusive Mini hours. The replacement protocol fixes10000
updates ×3 seeds,3600 evaluation games (initialization, endpoint and paired frozen
control;25deals ×8legs ×2anchors). Calibration may reject capacity but cannot
shorten10000. The5000-update export is unscored. Prior short w6 run is cancelled
and retained; no optimizer/collector continuation is claimed. Reserve18h prior,
53.5h new job,0.5h settlement. Five focused declaration/count tests passed.

Sustained attempt `1ebb94c24ef2` (source `cd72366b`, session `0eaddc80a844`,
attempt `5d47abf32db7`) stopped before science at budget admission after100
successful calibration updates and16 games. Reserves:48.6918h training,4.8623h
evaluation,53.7208h total with overhead. Jack Heart subsequently requested launch
without further budget fine-tuning. Operational bound:80h inclusive (18h prior,
61.5h job,0.5h settlement). Reuse those exact calibration receipts, checked against
the unchanged recipe; no further calibration. Committed baseline plan fixes10000
updates ×3seeds and3600 games. Private calibration copy:
`.runs/mtg122/w6-calibration-retained`. Earlier attempts remain retained/charged.

The admitted baseline is now running: `335bd645dafe`, source `09dc662d`,
session `f89409f36a91`, attempt `a9878311d7d9`; receipt
`.runs/mtg122/w6-admitted-launch.json`. Native build and retained-calibration
admission passed; first scientific seed12261 has a10000-update target. No new
calibration or checkpoint continuation. Other attempts remain terminal and charged.

The5k seed12261 diagnostic completed208 replayed games:79/104 greedy,92/104
random; Lessons against greedy Allies12/26. Etude reports remain snapshots.
Historical GPU comparison is recorded in the baseline protocol, not a causal
speedup claim.

Jack Heart approved 500-update saves / 1000-update diagnostics for future seeds.
Queued `c1ec4899c7dd`, source `135c34a0`, retains seed1 for fixed evaluation and
runs12262/12263 with the new cadence. The44h cap and unchanged scientific plan
are documented in `experiments/corrected-world-baseline.md`. Local bounded Etude
controller PID50923 services the queue; it performs no training/evaluation.
Receipt/log: `.runs/mtg122/w6-cadence-launch.json`, `cadence-controller.json`,
`cadence-controller.log`. It was observed alive with successor queued behind
`335bd645dafe`; this is not successor launch or midpoint strength evidence.
Seven focused software checks passed; no local science ran.

Jack Heart approved a complementary GPU study: three seeds ×25k updates,
$25 inclusive cap, post-training evaluations on Mini without displacing its10k
study. GPU diagnostics:1k,2k,3k,4k,5k, then7.5k,10k,12.5k,15k,17.5k,20k,22.5k,25k.
Retain500-update saves and Mini future-seed1k diagnostics. GPU not launched; L4$0.59/h available. Correction: normal controller SSO
prefix listing PASSES; the prior denial used the restricted worker role. No new
IAM permission is established as necessary. Integrate SDK environment credentials
and remote-link expiry, preserving controller/worker separation. No rental.
