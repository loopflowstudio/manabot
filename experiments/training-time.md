# GPU training time: startup and steady work

2026-10-15. The basic training report exposed 62% unattributed time in a historical
41-minute L4 run. The retained update coordinates change its interpretation:
most of that time passed before the first update, not throughout collection.

Evidence: [source-bound measurements](data/training-time/historical-startup.json),
run `9162405354044aa7aa86fcc6bc842137`, source
`2555c0e2ab2fb01b1624b5f948a7b5813b0361d2`, job
`etu103-recovery-v2-pilot-1`. The run and evaluation metadata were downloaded from
the current job manifest and verified against its byte counts and SHA256 hashes.

| Observation | Measured time |
|---|---:|
| Complete recorded run | 2,465.680 s |
| First update coordinate | 1,501.394 s |
| Active collection + learning at first update | 2.569 s |
| Initial evaluator process | 1,369.203 s |
| Initial evaluation charged time | 1,369.130 s |
| Recorded interval between first and last updates | 961.225 s |
| Active time within that interval | 937.185 s (97.5%) |
| Full-run collection | 775.588 s |
| Full-run learning | 164.165 s |

The saved job requires initial evaluation admission. In the measured source,
`training/execution.py` waits for the admission file before constructing the
collector. `remote/supervisor.py` creates that file only after the initial
evaluation completes and its snapshot is durably published. This establishes a
blocking evaluation dependency. It does not measure the exact learner wait:
initialization, publication, polling and other startup work also occupy the
pre-update interval. The historical flamegraph therefore retains unattributed
time instead of assigning the entire residual to evaluation.

Collection accounts for 82.5% of the run's active time. That is the next boundary
to explain on a GPU: host observation preparation, policy execution and returned
results, native environment work, and rollout assembly. It does not establish
that neural inference or the simulator alone is the bottleneck. This recipe uses
64 streams and eight transitions per update, with a width-384, eight-layer model.
Earlier throughput experiments used width 64 and one layer. Those timings cannot
be substituted for this model; larger rollouts need their own measurements.
Post-first-update overhead is small in this run, which weighs
against persistence being the dominant explanation here.

Future ordinary runs now record `StageRecord.admission_wait_seconds`. The field
counts attempt-local learner wall time at the admission gate, including failed
waits; it is excluded from active training and partitioned once in the report.
Older records retain an unknown value. This adds a basic host timer, not PyTorch
tracing or evaluation instrumentation.

Ordinary self-play now supports optional collection subphase counters through
`execution.profile_collection`. They record observation preparation, policy/action
handling, rollout recording, opponent policy, environment step/encoding and batch
assembly. The report nests these under collection and retains an unmeasured
remainder. These are host intervals with existing device waits, not GPU kernel
durations. Learning remains a coarse measured phase. Detailed PyTorch traces stay
deferred. The new investigation has a separate $25 GPU allowance.

## Basic-profile-001: evidence delivery failed

The [first protocol](protocols/basic-training-profile.txt) admitted 28 cases on
one RTX5090, covering the small benchmark model and the larger historical shape.
Source `79ea7a0e`, run `70ec76cbf84d`. The job log reports an incomplete study, but
no cell records or training archive reached storage. All 28 planned cells remain
**unknown**, including the proof and observer-effect gates; there are no timing
medians to report. [Retained failure evidence](data/training-time/basic-profile-001.json)
includes every planned case, terminal state and provider-confirmed deletion.

The rental ended at its original 45-minute limit on 2026-10-16 at 02:19 UTC.
Etude recorded `lost / machine_lost`. Estimated rental cost is $0.894; accounting
reserves $1.09 conservatively, separately from the $2 ancillary allowance. Final
provider billing was not yet available. The provider confirmed the pod absent.

Only the claim, job traceback and setup log survived. The launcher uploads every
output before its completion receipt, using a single-object form capped at 5 GiB
and an HTTP timeout of 60 seconds. This explains why a finished local job can
still appear running while delivering artifacts. It does **not** identify the
actual failure: the archive's size, transfer rate and upload exception were not
recovered. Neither this delay nor its rental cost measures a training bottleneck.

The [focused follow-up protocol](protocols/basic-training-profile-002.txt) keeps
the historical width-384/eight-layer, 64-stream comparison: both deterministic
proofs and all three ordinary timer-on/off repetitions at the original work
counts. Timing metadata is independently readable and sorts before model data;
the complete evidence is split into 64 MiB compressed objects with ordered hashes.
Local tests reconstruct every original file and check independently readable
metadata. Training behavior and measured model exports are unchanged. This
repairs evidence delivery; it does not establish a GPU throughput improvement.
