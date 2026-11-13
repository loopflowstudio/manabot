# PPO CUDA update throughput — MTG-338

## Decision: measurement blocked; candidate reverted

No L4 capacity was available at 2026-11-13 10:58:25 UTC. Authenticated launch
`947aca1850c9` (requested run `4e50a037bd12`) failed admission before a run was
recorded or a command started. No paired timing or numerical evidence exists.
The measure keep rule restored the runtime and candidate-motivated tests/harness
to `734f2cd5c`. This is not evidence that embedding lookup is slower or wrong.

The complete design, candidate and frozen comparison harness are recoverable at
`02c202956`. The hypothesis remains at least 20% less median update time on the
L4 Elemental w64l2 workload, with CPU bitwise equivalence and explicitly bounded
CUDA rounding. The [report](../experiments/ppo_cuda.md) records the pre-run bounds,
launch receipts and missing checks. Do not claim a candidate measurement from
the earlier baseline: 0.26437 optimizer steps/s, five repeats, 0.26366–0.26474.

Resume only when an L4 is available under the $4 launcher cap. Inject storage
from Doppler `manabot/prd` and RunPod from `etude/prd` together. Nothing is queued,
no rental was created by these failed admissions, and no process is left running.
