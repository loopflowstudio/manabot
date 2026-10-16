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

The next study will use lightweight collection and learning subphase counters,
first verify attribution and observer effects, then compare representative
ordinary GPU training recipes. Detailed PyTorch traces remain deferred. The new
investigation has a separate $25 GPU allowance; no new rental was used for this
historical finding or the admission-wait measurement.
