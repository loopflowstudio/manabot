# Learner admission and coupled cohort model

`scratch/cohort-flow-probe.py` couples the existing finite-actor inference queue to
measured ordinary learning, under a synchronous zero-lag barrier. A global8×64,
8×512 or8×4096 batch leases1/8/64 actors; extra configured actors stay idle.
Inference drains before one central optimizer update; publication precedes reuse.
No concurrent inference/learning or multi-GPU Adam is assumed.

`--serving-costs scratch/gpu-probes/serving-007/service-model.json` selects the
prospectively checked full serving boundary, with matching `--hardware` from the
same folder and `--card h100`. The loader rejects incompatible model/world/input
contracts or forward evidence and reports request/row extrapolation. Without this
option the original model-plus-link control remains unchanged.

Current comparisons use run007 inference and run005 ordinary learner service:

|Global streams|Servers|Model+links best rows/s|Complete serving best rows/s|
|---|---:|---:|---:|
|512|1|5284|4223|
|512|8|6036|5610|
|4096|1|11923|8874|
|4096|64|16732|16272|

One-server optima use batch1024; multi-server optima use64. For4096streams,
64×servers give only1.83× raw learner throughput under the central barrier.
Doubling collector CPU speed changes8874→8880rows/s(0.061%). This ranks modeled
constraints, not actual distributed performance or strength. Distinguish raw rows,
optimizer exposures, surfaced decisions and published policy versions.

The8×64 seed10349 tape has512learner rows,1099decisions,59requests and no finished
or truncated games. Input audits and three exact replays passed; repeat1 host work
is28.606ms, including4.395ms final tail. H100 has a completed8×4096 learner receipt;
L4's matching loops timed out, so that learner shape remains unavailable on L4.

Retained array accounting assumes leased rollouts survive a global-batch copy:
348121088bytes for8actors;2784968704 for64. Idle capacity has no instantiated
collectors. Native games, Python, models, staging and allocator slack are excluded.
Finite horizons separately count updates, publications and unfinished phases;
CPU totals exclude partial final epochs. Source/recipe checks reject unsupported
learner shapes and changes in model/world/observation contracts.

`accelerator-probes/serving-integration/` retains54complete and54same-forward control
cohort cases, admission checks and27unchanged L4 outcomes. All nine prospective
physical predictions were reproduced exactly after extracting the shared model.
`serving-integration-results.txt` gives exact reproduction commands, the6case
4096actor/64server inference-only sweep, budgets and proof limits. That sweep has
262144streams; it must not be conflated with this4096-global-stream learner cohort.

Models still clone one untrained trace and assume independent per-server CPU,
memory bandwidth and links. Zero aggregation/publication is an optimistic network
bound;10ms sensitivities are assumptions. Learner and serving measurements come
from different rentals. Target CPU contention, policy evolution, distributed
execution, all-reduce and strength per compute remain unmeasured. Complete prior
notes and L4/H100 historical comparisons are preserved by hash in the artifact index.
