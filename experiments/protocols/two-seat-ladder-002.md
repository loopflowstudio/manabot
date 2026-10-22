# Two-seat Allies/Lessons ladder pilot

Jack Heart requested a rented GPU experiment on 2026-10-21 with a $10 ceiling,
matching the running scale ladder while training both seats. This authorizes one
new attempt, not a replacement on failure or a $50 continuation.

The control is MTG-159's `scale-ladder-allies-lessons-v1`, source `e6a2aab1`:
w6 Allies/Lessons, width 64, two attention layers, scalar masked-mean value,
256 cross-balanced streams, 64 transitions per stream, Ataraxos move learning,
seed 15902, float32 CUDA and one learner CPU thread. Preserve its complete
learning settings, including absolute iteration LR/tau schedules and EMA .999.
Keep raw exports every 250 updates. Do not alter the live v1 run.

The new recipe is `experiments/data/two-seat-ladder/frozen-regime.json`.
Relative to the saved control JSON, only the ID, schema version and explicit
`learning_seats: both` differ. Both-seat collection finishes games after a
combined decision floor; learner rows can exceed 256 × 64. GAE and value targets
are computed separately across terminal seat trajectories before packed rows
are grouped into optimizer minibatches of at most 256 decisions. No padded rows
or unfinished seat trajectories train the model. This changes collection and
row order relative to v1; the iteration schedule is intentionally retained to
match the requested ladder, not migrated to a decisions clock in this attempt.

Rent one secure NVIDIA L4 at no more than $0.59/hour plus $0.02/hour storage,
6 vCPU and at least 62 GB RAM, matching the calibrated ladder machine class.
The standard remote launcher admits 11.9 hours (below its twelve-hour worker
credential cap), including 900 seconds setup, 180 checkpoint, 300 upload and
120 cleanup. Projected rental ceiling is $7.259, plus $0.10 for the guardian
proof. Together with attempt 001's $1.5312598409473894, admission is $8.89026.
Leave the remainder unallocated. The guardian's absolute deletion deadline and
the supervisor's earlier learning cutoff remain authoritative after disconnect.
Provider/network outages can affect actual cleanup; retain their receipts.

Use fresh job ID `mtg165-two-seat-002`; never reuse the cancelled 001 ID. Confirm
001's cleanup and fresh prices before creating 002. Keep provider-owned startup,
private S3 job intent, scoped worker credentials, artifact publication and
shutdown. Add a job-specific public SSH key for read-only bootstrap diagnostics;
retain the exact local launcher, key and provider receipts under the ignored
`.runs/mtg165-two-seat-002/`. No account credential is added for diagnostics.
Confirm a real learner update and an exported checkpoint before reporting that
training is running. If startup fails, retrieve its diagnostic log and stop this
attempt; no replacement rental is authorized.

This single budget-limited pilot predicts that both-seat training will retain
all generated policy decisions and export compatible raw/EMA checkpoints. It
does not predict a 2× speedup or certify improved strength. Report actual
learner decisions, environment decisions, updates, time and inclusive dollars.
The existing ladder's heuristic cohorts can later compare checkpoints; no
additional paid evaluation rental is included in this authorization.

## $50 ladder amendment (2026-10-21)

Jack Heart raised this experiment to $50, selected the ladder in
[PR #333](https://github.com/loopflowstudio/manabot/pull/333), authorized necessary
Etude changes, and selected the Mac mini as controller. This supersedes the
single-attempt $10 limit above. The current 002 lease remains unchanged.

Use the existing recipe, seed, machine class and 250-update checkpoint cadence.
The small tier is $10 within one day; the medium tier is $50 within seven days
of the original 002 admission. Both refer to the same learner lineage. Retain
actual learner and environment decisions separately, actual cumulative costs,
and earlier failed-attempt costs; this does not retroactively change v1.

The mini is reachable as `jack@mini-heart.local`. Keep 002 as the exact adopted
first job of cohort `mtg165-two-seat-50`. After each clean allocation pause,
verified final raw/EMA/Adam/run exports and confirmed provider cleanup permit
one declared successor. The explicit `last-complete` selector binds the actual
finalized update, never an intermediate checkpoint. Failures, incomplete exports,
unknown cleanup and spending overruns stop admission. No failed-job replacement
is included. CUDA collector/game/RNG streams restart, while model, Adam, EMA,
absolute iteration schedules and cumulative counters continue.

The prepared queue has five more 11.9-hour windows at $7.36 each, then one
4.9-hour window at $3.10. Including 002's existing $8.35 reservation, the prior
$1.5312598409473894 and $0.20 controller allowance, aggregate reservations are
$49.98125984094739. Each reservation includes its guardian proof allowance.
All windows share the original seven-day ceiling; each has its own immutable
shutdown deadline. Unspent reserves are retained, not automatically rerented.
The exact machine and recipe plans are in the private cohort intent.

Controller credentials are required on the mini itself; laptop credentials are
not sent to workers or embedded in plans. Keep its process under launchd, with a
permanent host/directory ownership binding. The controller uses a retained exact
checkout of worker source `fabb96db2` while its own version handles budget-pause
continuation. This bridge is needed because 002 predates Etude-managed launch;
Etude's SSH controller is configured for the mini without moving existing records.

Evaluation follows #333's heuristic anchor and paired 25-deal/mirror protocol,
with Lessons vs Allies first. CPU evaluation is separate from the training
allocation; this amendment does not allocate another paid evaluation rental.
Predict only retained two-seat decisions and exportable checkpoints. Strength,
throughput-per-dollar and the effect of restart boundaries remain measurements.
