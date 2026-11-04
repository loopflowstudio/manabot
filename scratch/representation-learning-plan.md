# V7 learning comparison — accepted continuation

2026-11-04: Jack Heart requests actual replicated training after PR419, with
$25 total including earlier attempts. Results stop at demo; no model promotion.
The supplied continuation and questions retain prior decisions and evidence.

## Implementation

Replace the unexecuted v5 scientific declaration with explicitly named v7 pooled
and GRU treatments. Four independent seeds, 128 updates, 32 complete games per
update, same inputs/learner/curriculum; each seed trains 4,096 games. Both effect
and continuation reducers change together. Frozen canaries retain their source.
Use existing TrainingJob, execute_regime, checkpoint evaluator and monitor rows.
Commit the protocol before scientific execution. An excluded CUDA canary checks
practical admission and timing; its score cannot select horizon or architecture.

## Delete — do not maintain

- `experiments/state_representation.py` stale v5 scientific declaration and
  unexecuted three-arm schedule. Preserve excluded receipts and the old recipe
  through git at `4b4656d33`; retain seed-level inference and strict admission.
- No production model or ABI deletion: this study uses the delivered v7 owner.

## Allocation and remaining work

Provider billing now reports $0.07258259388618171 for the deleted old L4 pod
(superseding $0.07083924 estimate). L40 is unavailable; RTX4090 available at
$0.89/hour at 07:20 UTC. Budget $1 for excluded CUDA admission, $16 for the
scientific cohort, $6 ancillary reserve, $0.073 prior billing rounded up;
$1.927 remains unallocated. No MTG-213 money or machines. Read-only admission
receipt `.runs/representation/resume-admission.json`.

V7 CPU GRU admission completed one32-game update in19.04seconds. RunPod
refused4090 launch8dd337797e0a after source preparation, acquiring no machine.
Protocol now freezes whole-cohort CPU execution in two four-process batches
(two seeds × both arms each), one thread/process,16GiB watchdog each.

Remaining: execute cohort and retain raw endpoints/receipts,
evaluate fixed deals, report independent-seed uncertainty and costs, verify
teardown, publish evidence. Stop on capacity/data/numerical errors; no scientific
seed retries or score-dependent extensions. CPU fallback requires prospectively
freezing a whole-cohort device choice, never mixing device-selected survivors.

Check: `uv run --extra dev --extra artifacts pytest tests/training/test_state_representation_study.py -q` — 2 passed. Gate owns broader affected checks.
