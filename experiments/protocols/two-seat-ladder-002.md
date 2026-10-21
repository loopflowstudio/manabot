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

The new recipe is `experiments/regimes/scale-ladder-allies-lessons-both-v2.json`.
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
