# Learner admission and coupled cohort model

## Coupled zero-lag control

`scratch/cohort-flow-probe.py` now reuses the existing finite-request inference
simulator, followed by measured ordinary learning service. One global cohort leases
one64stream actor for8×64, eight for8×512, or64 for8×4096 when calibrated. Inference drains before one central
update; publication precedes the next lease. Extra actor capacity stays idle.
This models exclusive phase sharing on the GPUs, not concurrent inference/training
or independent Adam owners. Larger unmeasured learner shapes are rejected.

Seed10349 has8×64learner rows,1,099decisions,59requests and zero finished/truncated
games. Input auditing and three exact replays passed. Repeat1:28.606ms host work,
4.395ms final tail. Model/world contracts and hashes are checked; CPU/GPU trajectory
equality is not claimed.

Fourth-L4 completed reference learning service is0.2252/0.2361s per8×64 batch and
0.3690s per8×512; `gpu-probes/l4-004/learner-service.json` retains cold/repeated
samples. H100 now has completed8×4096 service at1.6677s, retained under`h100-005`. All shapes keep filtering populations
explicit; stationary raw rows/s and retained optimizer exposures/s are separate.

27cases and closed-form timing, lease, memory, horizon, publication, unsupported-shape,
request/CPU conservation checks passed. Results: `accelerator-probes/cohort-flow-detail.json`.
Earlier24case output and exact source snapshot remain. Conditional examples:

On L4, batch256 won with one server (3,654raw rows/s); batch64 won with eight
(6,265rows/s). Extra actor capacity leaves those rates unchanged under the same
cohort shape. Full sensitivities remain in the saved27case report. Hardware-specific
batch optima are conditional, not measured distributed throughput or strength.

Array accounting retains actor rollouts while copying one global batch:348,121,088
bytes at eight actors, including native array buffers. Idle capacity has no instantiated
collectors in this scenario. Native games, Python, models and staging are excluded.
Ten-second horizons retain unfinished phases; completed learning and published
versions are distinct. CPU-work totals explicitly exclude the partial final epoch.

Reproduce with a fresh output:
`uv run --no-sync python scratch/cohort-flow-probe.py --out scratch/accelerator-probes/cohort-new.json`.
Protocol: `cohort-protocol.txt`; capture helper only invokes existing capture/replay.
The original cohort pass used5.109seconds, closing legacy probes at238.718/240.
H100 integration used4.428seconds of a separate20second analysis slice within the
original300second accelerator allowance (`highend-integration-protocol.txt`).

Limits: cloned stationary untrained tape, named GPU service/learner receipts, assumed links, one extra serving CPU thread/server. Zero aggregation/
publication is an optimistic network bound. Actual policy evolution, target CPU
contention, multi-GPU optimization and physical distributed execution remain unmeasured.
The strict zero-lag barrier is a compatibility control, not a selected production
architecture. Next validation needs matching physical service and traffic before
using modeled rates to choose a distributed design.

H100 integration preserves all27L4 outcomes exactly. The same27case H100 matrix
at512streams prefers batch1024 on one server (5,465rows/s), versus L4's256.
A second27case H100 matrix uses64leased actors for global4096streams: one server
with batch1024 predicts12,097rows/s;64servers with batch64 predicts16,988. Thus64×
inference servers yield only1.40× admitted rows/s under this single-learner barrier.
Average server/learner reservation is6.3% in the latter case, not measured SM usage.
Modeled retained arrays are2,784,968,704bytes. Actual distributed behavior remains
unvalidated; the physical H100 run used one large collector, not64sharded actors.

Reproduce the larger modeled cohort with a fresh output:
`uv run --no-sync python scratch/cohort-flow-probe.py --card h100 --hardware scratch/gpu-probes/h100-005/capacity-report.json --learner-service scratch/gpu-probes/h100-005/learner-service.json --global-streams 4096 --out scratch/accelerator-probes/h100-new.json`.
