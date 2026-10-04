# Recovering bounded manabot training

ETU-98 adds opt-in recovery for one CPU self-play stage. It preserves learner,
Adam, evaluation EMA, Python/NumPy/Torch and sampling RNGs, native collector
state, diagnostics and iteration coordinates. Native state is reconstructed
from the original seed and a bounded action journal, then every observation
buffer is compared before learning resumes. This is a correctness path for
bounded runs, not a scalable native snapshot format.

Set `schedule_clock` to `iteration_fraction` and `recovery_max_microsteps` to
an explicit journal ceiling in a TrainingRegime with exactly one self-play
stage. The ceiling counts active environment microsteps across vector streams, not learner transitions. An
ordinary elapsed-time recipe is unchanged and cannot opt into exact recovery.
Use the normal training CLI, and retain its local `training.sqlite`:

```bash
uv run manabot train --regime recipe.json --seed 197 --out .runs/first
uv run manabot train --regime recipe.json --seed 197 --out .runs/continued --resume-from RUN_ID
```

Both output directories must share the same parent so the CLI selects the same
canonical store. Recovery requires the same recipe, seed, source and runtime;
use a new directory and the stopped attempt's stored ID. Only its latest
committed boundary can resume. A single recovery child is allowed per attempt;
continue the child after a subsequent failure instead of choosing a different
seed or checkpoint. Completed attempts cannot resume.

A process-held POSIX lease prevents concurrent writers. Recovery rejects a live
lease, a missing lease file, a foreign hostname, incompatible identities or
corrupted snapshot bytes. Never remove lease files: a replacement inode would
invalidate mutual exclusion. These are trusted local files, not portable or
network-filesystem recovery artifacts. Private snapshots contain hidden state
and pickle objects; do not load untrusted files or serve them as model exports.

Caught failures retain their measured attempt time. Abrupt death leaves a
running record; only acquiring its saved lease on the same host permits
settlement. The canonical record becomes interrupted, while its last exported
`run.json` remains unchanged as evidence. The unobserved calendar interval is
explicitly recorded and conservatively charged, including downtime; it is not
measured training throughput. Backwards calendar movement rejects settlement.
This accounting assumes the host calendar is reliable across restarts.

`seconds` measures this attempt; `prior_seconds` charges ancestors, and
`recovery_seconds` includes native reconstruction. The watchdog uses macOS
continuous time or Linux boottime, including sleep. Resumed work consumes the
original watchdog allowance, including prior attempts and reconstruction.
Scientific schedules instead use completed-update coordinates. A budget too
small to cover downtime cannot be reset by recovery.

Seven focused cases include caught interruption, actual subprocess `os._exit`,
exact final model/optimizer equivalence, complete RNG/collector equivalence,
corruption, seed mismatch, live-writer rejection and watchdog exhaustion.
Existing executor and objective regression tests also pass (37 cases total).
The Ataraxos case verifies categorical critic, optimizer, EMA, collector, RNG
and iteration-schedule equivalence after interruption.
These checks establish recovery mechanics, not learning strength or a general
throughput estimate. CPU/MPS comparisons, multi-seed timing calibration,
clone/world-sampling costs, multi-stage recovery and complete-cohort projections
remain ETU-98 work. The active ETU-91 campaign is unchanged and does not gain
recovery retroactively.
