# Evaluation-cost calibration for the batching/size grid

2026-10-17. Timing-only allocation under MTG-135's existing inclusive $50 cap.
This protocol does not admit a scientific learning comparison or use calibration
scores to select model capacity, stream count, seeds, horizon or evaluation count.

Use all five declared capacities: 64×2, 192×4, 384×8, 64×4 and 384×2. On one
RTX 5090 rental, build the pinned source once and pass ordinary CUDA admission.
For each capacity, execute one 64-stream update (4,096 transitions), seed13598,
with a120-second learner limit. These short trajectories supply verified,
ordinary initialization exports. Score initialization only, never the trained
endpoint. No laptop experiment or other Task's allocation is used.

For each initialization, play one complete eight-leg deal block against each
scripted-greedy and random anchor:80 games total across five capacities. Deal
seed1359800000 is shared across capacities and anchors, outside scientific
selection. Use ordinary stochastic one-pass CPU inference, one thread, both
cross-deck and mirror cells,120seconds/10,000commands per game, and exact replay.
Incomplete cohorts, admission errors or interrupted commands stop calibration.
Retain all failures. Do not replace a failed seed or count partial games as a
successful calibration. No automatic retry.

Reserve1.5 times the larger of full coordinator/replay seconds per game or the
slowest individual game's time, separately for every capacity/anchor pair.
Initialization timings can underestimate learned-policy game lengths; report
that limitation and retain hard deadlines in any later learning allocation.
The later allocation must freeze all scientific choices before scoring and
reserve its training, evaluation, startup and output-retention costs together.

Admission ceiling: one job, one attempt, at most90 command minutes and$2 total
rental allocation, using a quote no higher than$1.25/hour. Setup600seconds,
CUDA60seconds, each training command180seconds, each evaluation command600seconds,
all subordinate to the outer deadline with240seconds reserved for output
retention. Stop even if the full calibration cannot fit. At most two campaign
rentals may run concurrently; wait for a grid slot and use one scheduling owner.
Record the quoted rate and provider-confirmed deletion before releasing budget.
The protocol alone does not reserve or spend money; the ledger admission must
prove the inclusive campaign remains under$50.
