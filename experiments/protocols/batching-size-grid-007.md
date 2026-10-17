# Joint model size × stream count — 007

Jack Heart assigned this conversation the batching sweep, then requested one
combined grid on 2026-10-17. Inclusive campaign budget is now $50, including
previous studies and failed rentals. MTG-135 owns the grid and recommendations
for MTG-122. Existing corrected-world baseline executions remain unchanged.
This supersedes admission of the unlaunched size-only study006 replacements;
their failed attempts and completed diagnostics remain historical evidence.

## Question, controls and predictions

Which model/collection configuration gives reliable full-loop throughput at a
useful memory cost, and which deserves longer learning validation for MTG-122?
Test the full Cartesian grid: small/middle/large ×4/64/128/256/512 streams.
Small is MTG-122's width64/depth2/four-head masked-mean scalar model, exactly
matching the serialized early-learning-rate-mini-source agent. Middle is
width192/depth4; large width384/depth8. FF expansion stays4×width for all.
Preserve terminal rewards, product catalog/decks, w6, cross-balanced current-self,
64 transitions/stream, ordinary Ataraxos MOVE/Adam, float32, TF32 off, one CPU
producer/thread and one RTX5090. No optimizer-grouping or representation contrast.

Prediction: at least one64–512-stream cell per size provides≥2× inclusive
training transitions/second versus that size's four-stream cell. Larger models
are expected to benefit more from amortizing small requests, but512 need not be
optimal and may exceed memory limits. A completed negative result is useful.

## First allocation: complete-loop feasibility grid

One reserved feasibility seed13571; each of15cells targets131,072 learner
transitions. Updates are512/32/16/8/4 at4/64/128/256/512 streams respectively.
Keep every cell, including failures. This is a throughput/health screen, not a
model-strength ranking; four iterations give weak steady-state evidence.
Use separate processes and retain initial/final policy, optimizer, run records,
native/source identity, ordinary health diagnostics, active and inclusive timing,
peak CUDA memory and independent final checkpoint load/finite checks.

Ataraxos LR/tau are iteration based. Four streams keep the original schedule;
other cells use the existing endpoint-preserving IterationSchedule, mapping
their run onto the reference512-update horizon. This is explicit schedule
normalization, not unchanged training. At equal transition budgets there are
fewer optimizer steps and EMA updates at larger stream counts; EMA0.999 remains
per iteration. In this current-self recipe, collection and bootstrap targets use
the raw network, not EMA. Different EMA age affects its separate exported policy,
not the raw learning target; all proposed learning comparisons use raw exports.
Minibatch composition, filtering and policy staleness also change. Therefore
throughput alone cannot approve a new learning recipe.

Maximum$15 for15cell caps of$1, each at most45command minutes,1800training
seconds, and reserved retained-output time. No automatic failed-cell retries.
Startup/device failure stops further admission. OOM/failed cells stay missing;
no survivor-only claim of full-grid completion. One separate$1startup probe
retains provider system/container logs before the SDK startup cutoff. It reuses
the exact repaired two-update canary; its score never selects configurations.
Settled prior conservative charge$7.96 plus$2ancillary reserve plus$1probe plus
$15grid leaves$24.04 for measured validation/evaluation, subject to fresh receipts.

## Validation and recommendation boundary

Freeze the next allocation from measured costs before scored learning: repeat
promising throughput cells independently and compare learning at fixed collected
transition milestones, preserving all sizes in any size-strength comparison.
Use independent training seeds and paired greedy/random deal panels, reporting
initialization-relative effects separately from throughput. The shorter screen
cannot discard a slow-starting model for lack of skill or establish its ceiling.
Do not inherit25,000 iterations after expanding streams. Recommend exact model,
streams, transition horizon, schedule mapping and checkpoint cadence, with a
clear distinction between measured feasibility and validated playing strength.

Strongest confound: batch expansion changes the learning process even after
LR/tau endpoint normalization. A full-game learning comparison, rather than
matching throughput or finite gradients, discriminates that concern. Ordinary
CUDA non-repeatability and the earlier extreme parameter anomaly remain open;
finite final checkpoints do not clear either. All failed attempts count toward
the same$50cap. No other Task's budget or machines are borrowed.

## Added width/depth contrasts (2026-10-17)

Jack Heart requested64×4 and384×2 while the original small/middle four-stream
cells were running. Add both architectures across the same five stream counts,
seed13571 and131,072-transition target:25 feasibility cells in total. The
64×2/64×4 comparison changes depth at fixed width;64×2/384×2 changes width at
fixed depth. The384×2/384×8 pair adds a second depth comparison. Middle192×4
remains included. No playing-strength results motivated this expansion.

Added names are `small-deep` (64×4) and `large-shallow` (384×2). All other
recipe fields, schedule mapping, deadlines and failure rules stay as declared.
Original cells retain their original source revision; added cells use a new
revision that only adds these choices, their checks and this protocol amendment.
Record source per cell rather than claiming one source revision for all25.

Reserve another$10 for ten$1 cell caps. Conservative prior charge$8.22 plus
$25 feasibility caps and$2 ancillary reserve bounds exposure at$35.22, leaving
$14.78 uncommitted under the existing$50 total. Actual unused rental caps may
be released after verified deletion. Keep at most two concurrent rentals and
no automatic retries. Reconsider the later learning-validation allocation from
measured costs; this amendment does not authorize spending beyond$50.

## Startup transport and retained evaluation

The large four-stream cell and its first replacement both stopped before job
claim. The replacement used full-source gzip transport, which had passed a
separate two-update CUDA canary. Thus compression alone has not established
reliable startup, and the original CUDA initialization cause remains unresolved.
Keep those attempts; startup failure stops further scientific admission.

A separately capped canary tests an explicit source projection that omits only
`experiments/data/`, the historical measurement subtree. All other files retain
their original bytes, permissions and link targets. Preserve the full source
archive, retained/omitted file hashes, projected archive identity, complete cloud
readback and exact per-job source configuration. This is a declared packaging
change, not lossless compression of the full archive. Runtime code, content,
recipes and learning controls stay fixed. Require complete canary artifacts,
CUDA admission, independent policy load and provider deletion before further
grid admission. Startup cost comparisons must retain the transport difference.

For later learning panels, `experiments.runners.batching_evaluation` selects
raw exports by recorded learner transitions. It validates the entire requested
panel, including relocated artifact hashes, before playing games. Missing
milestones are errors; there is no rounding or substitution. Freeze the panel's
MonitorProtocol and study binding before scoring; initialization and later
points use paired deals against both anchors. This runner does not itself
choose the learning horizon, seed allocation or training configuration.
