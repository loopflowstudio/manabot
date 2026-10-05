# Recovering bounded manabot training

Recovery is opt-in for bounded CPU self-play stages, including multistage chains
that retain the latest collector or start a fresh learner. Set `schedule_clock`
to `iteration_fraction` and `recovery_max_microsteps` to an explicit journal
ceiling. Every stage must be `train_self_play`; supervised, compound, collection
and belief-training stages cannot opt into recovery. MPS/CUDA are unsupported.
Ordinary elapsed-time recipes remain unchanged.

```bash
uv run manabot train --regime recipe.json --seed 197 --out .runs/first
uv run manabot train --regime recipe.json --seed 197 --out .runs/continued --resume-from RUN_ID
```

Both output directories must share the same parent so the CLI selects the same
canonical `training.sqlite`. Use a new output directory and the stopped attempt's
stored ID. Only its latest committed boundary can resume, with one recovery
child per attempt. Continue the child after another failure. Completed attempts
cannot resume; an interrupted attempt with every stage completed can finalize
without replaying or exporting those stages again.

## Same-host contract

Private snapshots retain model parameters and buffers, Adam, evaluation EMA,
Python/NumPy/Torch and minibatch/sampling RNGs, native collector state, diagnostics,
collector count baselines, stage-local update progress and run-wide iteration.
Frozen-opponent identity is recipe-bound and its checkpoint digest is rechecked.
The iteration clock continues across linked stages; a fresh stage resets it.
Scientific schedules use the same update coordinates as uninterrupted execution.

Native state is reconstructed from its original seed and bounded action journal,
then every observation buffer is checked. The ceiling counts active environment
microsteps across vector streams, not learner transitions, and applies to the
collector's lifetime across linked stages. This is a bounded correctness path,
not a scalable native snapshot format.

Snapshots are published at stage entry, after each update, and after completed
export/admission. VerifyStore commits the snapshot reference and stage records
together. Completed prefixes retain their original artifact paths, digests,
counts and cost receipts; they are not re-exported into the child directory.
Uncommitted work may repeat, but its failed-attempt cost remains charged. Orphan
files never become authoritative merely because they exist on disk.

Recovery requires the same recipe, seed, source, runtime and host. Changed or
missing snapshot/completed-artifact bytes fail admission. Old source-bound
snapshots are historical evidence, not portable migrations. A local POSIX lease
rejects simultaneous writers; do not remove lease files because replacing the
inode defeats mutual exclusion. Foreign-host recovery and network-filesystem
leases are unsupported. Private snapshots contain hidden state and pickle
objects: never load untrusted files or serve them as model exports.

## Cost and failure semantics

`seconds` measures this attempt; `prior_seconds` charges ancestors once, and
`recovery_seconds` is a subset of this attempt's time. Sum run attempt costs, not
inherited completed-stage receipts a second time. Active-stage phase timings
measure only the current attempt, while counts describe committed scientific
work across retries. Completed stages retain their original receipts unchanged.

The watchdog uses macOS continuous time or Linux boottime, including sleep.
Whole-run allowances charge all prior attempts and reconstruction. Each stage's
watchdog retains its own consumed allowance across retries; a later stage does
not inherit an earlier stage's charge. Restoring an already completed collector
for the next stage charges the run's reconstruction cost without rewriting the
completed stage. Setup and finalization also belong to the run clock.

Caught failures retain measured time. Abrupt process death leaves a running
record; acquiring its saved lease on the same host permits settlement. The
canonical record becomes interrupted while the old `run.json` stays unchanged.
The unobserved calendar interval, including downtime, is conservatively charged
to the run and an active stage. A backwards calendar rejects settlement. These
estimated costs are not measured throughput; recovery assumes a reliable host
calendar and cannot reset an exhausted allowance.

The failure-injection suite compares uninterrupted and resumed learning state,
collector state, schedules, diagnostics and counters across stage entry, update,
completion and terminal boundaries. It also exercises abrupt subprocess exit,
exclusive ownership, incompatible artifacts and budget exhaustion. These are
software-correctness checks, not scientific training or strength evidence.

[ETU-99](https://linear.app/loopflow/issue/ETU-99) owns empirical hardware
calibration, seed/policy timing variation and cohort projections. The active
ETU-91 campaign is unchanged and does not gain recovery retroactively. The
[complete-loop calibration command](training-calibration.md) remains a separate
CPU accounting instrument; no new scientific run is required for this contract.
