# Early learning screen from the fixed-transition feasibility grid

Frozen before scoring,2026-10-17. Use the already-trained grid checkpoints to
measure early learning while infrastructure recovery continues. This costs no
additional GPU rental and does not replace the later independent-seed, longer
learning comparison needed for MTG-122's training recommendation.

Eligibility is mechanical: a declared grid007 cell must have completed exactly
131072transitions at seed13571, retain its original scientific source and recipe,
pass its full manifest/checkpoint verification and coarse numerical audit.
Evaluate every eligible cell, initially the12 verified completed cells; use this
same panel for later completed cells, including both added shapes. Preserve failed,
missing and deadline-censored cells in coverage reports. Do not substitute a
shorter endpoint or treat an infrastructure failure as a model result.

For each eligible cell, evaluate both its raw initialization and exact131072
endpoint against scripted-greedy and random. Use four paired deal blocks,
seeds1357101000,1357101001,1357101002,1357101003, each containing all eight legs
including mirrors. This is32games per anchor/checkpoint,128games per cell.
Use the same deals across all cells and checkpoints. No best-checkpoint selection,
seed replacement or score-dependent stopping. Keep the ordinary stochastic policy,
one CPU inference thread,120second/10000command per-game limits, compatibility,
artifact hash, complete-cohort and replay checks.

Run on the local16-core128GiB host validated by route012, at most four independent
evaluation CLI processes. Each cell/anchor command evaluates initialization and
endpoint sequentially, with an1800second command limit. A batch of at most12cells
has a90-minute work limit plus240seconds for retained-output bookkeeping. Stop
pending and active commands on failure, retain partial evidence and do not retry
automatically. No training, paid CPU service, GPU rental or other Task's allocation
is involved. Both anchors and all role cells are required for an admitted result.

Report endpoint score, paired improvement from initialization and coverage for
each cell. Compare batching within a model and size within a stream count only at
this common experience horizon. Account for paired deal blocks; with only four
blocks and one training seed, uncertainty and seed sensitivity remain substantial.
These results can reveal an early learning/throughput tradeoff or justify a longer
horizon, but cannot establish a model's skill ceiling, broadly rank architectures,
prove noninferiority or clear the historical numerical anomaly. Any later
confirmation uses newly frozen independent training seeds and evaluation deals.
