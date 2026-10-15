# Learner admission and coupled cohort model

## Coupled zero-lag control

`scratch/cohort-flow-probe.py` now reuses the existing finite-request inference
simulator, followed by measured ordinary learning service. One global cohort leases
one64stream actor for8×64, or eight for8×512. Inference drains before one central
update; publication precedes the next lease. Extra actor capacity stays idle.
This models exclusive phase sharing on the GPUs, not concurrent inference/training
or independent Adam owners. Larger unmeasured learner shapes are rejected.

Seed10349 has8×64learner rows,1,099decisions,59requests and zero finished/truncated
games. Input auditing and three exact replays passed. Repeat1:28.606ms host work,
4.395ms final tail. Model/world contracts and hashes are checked; CPU/GPU trajectory
equality is not claimed.

Fourth-L4 completed reference learning service is0.2252/0.2361s per8×64 batch and
0.3690s per8×512; `gpu-probes/l4-004/learner-service.json` retains cold/repeated
samples.8×4096 has no complete-loop calibration. All shapes keep filtering populations
explicit; stationary raw rows/s and retained optimizer exposures/s are separate.

27cases and closed-form timing, lease, memory, horizon, publication, unsupported-shape,
request/CPU conservation checks passed. Results: `accelerator-probes/cohort-flow-detail.json`.
Earlier24case output and exact source snapshot remain. Conditional examples:

| Leased actors / collector CPUs / inference servers | Max batch | Raw learning rows/s |
|---|---:|---:|
|8 /8 /1|64|3,279|
|8 /8 /1|256|3,654|
|8 /8 /1|1024|3,517|
|8 /8 /8|64|6,265|
|8 /8 /8|256|4,571|
|8 /8 /8|1024|3,517|

At8servers, configuring4,096actors leaves4,088idle and exactly the same rates.
64servers also fail to improve this eight-actor cohort. At one server/batch256,
doubling host speed gains0.20%; assumed10ms aggregation plus10ms publication costs
1.75%. These dependencies prevent transferring an inference-only batching optimum
to useful learning. They do not establish distributed performance or strength.

Array accounting retains actor rollouts while copying one global batch:348,121,088
bytes at eight actors, including native array buffers. Idle capacity has no instantiated
collectors in this scenario. Native games, Python, models and staging are excluded.
Ten-second horizons retain unfinished phases; completed learning and published
versions are distinct. CPU-work totals explicitly exclude the partial final epoch.

Reproduce with a fresh output:
`uv run --no-sync python scratch/cohort-flow-probe.py --out scratch/accelerator-probes/cohort-new.json`.
Protocol: `cohort-protocol.txt`; capture helper only invokes existing capture/replay.
This pass used5.109commandseconds; legacy total238.718/240, no GPU allocation.

Limits: cloned stationary untrained tape, separate L4 rentals/drivers for inference
and learning, assumed links, one extra serving CPU thread/server. Zero aggregation/
publication is an optimistic network bound. Actual policy evolution, target CPU
contention, high-end GPU calibration and multi-GPU optimization remain unmeasured.
The strict zero-lag barrier is a compatibility control, not a selected production
architecture. Next validation needs matching physical service and traffic before
using modeled rates to choose a distributed design.
