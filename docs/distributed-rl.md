# Distributed RL: research and mini feasibility evidence (ETU-108)

Research began 2026-10-05 on `be6260d9136e49069a12864514b849811f32b32d`;
mini receipts were reviewed on 2026-10-06. Jack Heart requested research and a
disposable prototype in which laptop and mini contribute to one learner. Mini
single-host probes now demonstrate collection, inference, training and a separate
replay-verified workflow. They do not implement that distributed prototype or
complete ETU-108. Speedup is an observation, not the acceptance condition.
The evidence-report follow-up ran no benchmarks and accessed no ETU-106 checkout.

## Findings and immediate recommendation

Separate actor collection, inference, learner updates, and frozen evaluation as
logical boundaries. Keep their physical placement configurable. Retain one
optimizer/checkpoint owner for the first prototype. Measure local inference and
trajectory transfer before deciding whether inference crosses Tailscale on every
decision. An inference service is an architectural option, not a prerequisite.

Do not select asynchronous stale-sample learning just because transport is
asynchronous. Existing PPO and Ataraxos move code preserve a frozen collection
policy per batch; neither implements V-trace. A bounded-version collection lease
is a useful transport API now, but the admissible lag belongs to the learning rule.
The old synchronous-round proposal is not an accepted decision. Zero-lag batches
are the conservative compatibility baseline; bounded-lag PPO and a separately
implemented V-trace learner remain competing experimental choices.

Matched comparison remains unavailable: mini has completed feasibility probes,
but laptop compute and transfer measurements remain outstanding. The original
name-resolution failure is historical, not a current access diagnosis. The base
already supported historical mean, masked mean and value-token aggregation plus
extra attention layers. The initial fixed probe selected historical mean and one
layer by default; the representative run selects width-64 masked mean explicitly.
These are configuration choices, not missing model implementations.

`Agent.forward` calls `forward_distribution`, which gathers visible/optional
belief objects, appends a valid neutral value token when configured, and runs
shared attention plus any extra layer. Action focus uses only the original
objects, preserving indexes; the token still influences policy representations
through shared attention. `_value_from_objects` dispatches historical fixed-slot
mean, masked mean after projection, or the trailing token through the existing
critic weights. `forward` converts WDL logits to signed expectation. Compound
play has a separate decoder/critic path; AgentSpec rejects aggregation/depth
variants there. The presence of MeanPoolingLayer in the constructor alone cannot
establish which critic path executes. These are source findings, not new runtime
or strength evidence.

## Primary evidence map

Sources were opened on 2026-10-05. Recommendations below are adaptations to
manabot, not claims that these papers validate two Macs or this game's estimators.

| Family | Evidence and mechanism | Implication here |
| --- | --- | --- |
| Asynchronous actor gradients | [Mnih et al., A3C (2016)](https://arxiv.org/abs/1602.01783): parallel actor-learners asynchronously update shared parameters. | A historical CPU baseline; asynchronous gradients are a different design from shipping trajectories to one optimizer. Do not graft remote Adam updates onto current training. |
| PPO | [Schulman et al. (2017)](https://arxiv.org/abs/1707.06347): alternate collection and clipped-surrogate minibatch optimization. | Saved action likelihoods support reuse within the update; clipping does not reconstruct missing behavior or remove historical state-distribution mismatch. |
| Distributed PPO | [Heess et al. (2017), section 3](https://arxiv.org/pdf/1707.02286): distribute collection and gradient computation; synchronous gradient averaging worked better than asynchronous updates in their experiments. | Evidence for a baseline, not a universal synchronization theorem. Remote gradients add collective and optimizer-consistency obligations unnecessary for two small models. |
| DD-PPO | [Wijmans et al. (2020), method and appendix E](https://arxiv.org/html/1911.00357v2): synchronous distributed updates with early rollout termination for stragglers; gradients all-reduced. | Unequal machines need not wait for identical rollout lengths. However unequal sample counts require explicit weighting and correct tail bootstraps; quitting a segment cannot fabricate a terminal result. |
| IMPALA | [Espeholt et al. (2018), sections 3–4](https://proceedings.mlr.press/v80/espeholt18a/espeholt18a.pdf): actors send trajectories; decoupled learning uses V-trace. | A principled asynchronous candidate, requiring a new estimator contract rather than only a remote collector. |
| SEED RL | [Espeholt et al. (2020), section 3 and appendix A.7](https://arxiv.org/html/1910.06591v2): central accelerator inference, streaming RPC and batching, with remote environments. | Removes actor model distribution but adds a round trip per step. Tailscale latency, batching delay and model size must determine suitability. Central inference still permits policies to change within an unroll. |
| Large self-play PPO | [OpenAI Five (2019), sections 3–4 and appendix M](https://arxiv.org/html/1912.06680v1): separate rollout, inference and optimizer resources; measured sensitivity to staleness and sample reuse. | Their system targeted roughly 0–1 parameter-version lag; that is empirical context, not a transferable safe threshold. Track consumed unique samples separately from exposures. |
| League self-play | [Vinyals et al., AlphaStar (2019), methods](https://storage.googleapis.com/deepmind-media/research/alphastar/AlphaStar_unformatted.pdf): opponent populations and distinct worker roles; hybrid V-trace policy/TD-lambda value learning. | Version the opponent and selection policy separately. AlphaStar's training-only opponent-information critic is not authorization to bypass manabot's viewer-safe critic contract. |
| Damped self-play | [Ataraxos supplement S3.4, equations (5)–(6), Table S7](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41586-026-11036-y/MediaObjects/41586_2026_11036_MOESM1_ESM.pdf): clipped action ratios, reverse KL to behavior and magnet, filtered lambda targets, one epoch grouped by simulator step. | Keep the full behavior distribution, categorical collection estimates, filtering population and iteration clock. The cited objective does not specify arbitrary asynchronous trajectory admission. |

Implementation documentation makes the placement choices concrete.
[SEED RL's repository](https://github.com/google-research/seed_rl) implements
V-trace and R2D2 under the same architecture: transport placement is not an
algorithm identity. [TorchBeast's README](https://raw.githubusercontent.com/facebookresearch/torchbeast/main/README.md)
distinguishes local CPU-actor MonoBeast from the more involved PolyBeast runtime;
its queues/batching and separate environment servers are useful examples, not
dependencies to install here. Its [V-trace implementation](https://raw.githubusercontent.com/facebookresearch/torchbeast/main/torchbeast/core/vtrace.py)
retains behavior and target action likelihoods, discounts, rewards, values and
bootstrap value and forms corrected value and policy-gradient targets.
[RLlib EnvRunner documentation](https://docs.ray.io/en/latest/rllib/rllib-env.html)
distinguishes worker count from vectorization and resource allocation: increasing
environment concurrency and increasing inference batch size are separate knobs.
No Ray, Redis, gRPC service or cluster scheduler is needed for this contribution.

### What correction does and does not buy

For IMPALA, write `r_t = pi(a_t|x_t) / mu(a_t|x_t)`,
`rho_t = min(rho_bar, r_t)`, `c_t = min(c_bar, r_t)`.
The corrected TD residual uses `rho_t`; earlier states receive it through
products of `c_t`. The policy advantage uses the corrected next-state target.
Finite `rho_bar` changes the evaluated policy toward the behavior policy;
`c_bar` controls trace propagation. This is not ordinary GAE with a clipped PPO
loss. The paper's tabular fixed-point argument is not a deep-network convergence
guarantee or a cure for arbitrary lag. These distinctions follow
[IMPALA section 4](https://proceedings.mlr.press/v80/espeholt18a/espeholt18a.pdf).

Manabot adds a further issue: opponent actions occur between learner decisions.
Even perfect learner-action ratios do not correct a changing opponent-induced
transition kernel. Pin a frozen opponent per game for an initial diagnostic, or
explicitly retain the self-play opponent version at each segment and describe
that nonstationarity. Neither choice silently becomes the other. A current-self
baseline must remain available for comparison.

For the current damped rule, row-wise reverse KL requires probabilities for
every legal action, not only the sampled log probability. Altering worker-local
quantile filtering changes which states get gradients. Compute the selected
population centrally over the declared batch, preserve timestep ordering, and
advance the learning/EMA schedule once per learner iteration, not once per
arriving worker packet. Distributed arrival order must not become a new learning
hyperparameter accidentally. These recommendations follow the actual local
contracts; asynchronous correction for its categorical lambda target remains an
unresolved algorithm change.

## Local system model and reusable seams

* `manabot/training/models.py`: `TrainingRegime` owns resolved AgentSpec, world,
  match, observation and stage configuration. `Execution` admits CPU float32
  and one worker here. Do not add a remote device string to bypass validation.
* `manabot/sim/net_opponent.py`: `SeatRoutedCollector.collect` emits
  `RolloutBatch` with `[time, stream, ...]` observations, actions, logprobs,
  behavior distributions, values, end markers and exact next observations.
  Only learner-seat decisions train; opponent decisions and native microsteps
  have separate counters. Streams pause before sampling their next action.
* `manabot/training/ataraxos.py`: detached behavior/reference/target inputs,
  positive legal support and collection-time WDL estimates. `transition_gae`
  consumes end-of-transition markers; stock episode-start GAE is incompatible.
* `manabot/training/execution.py` plus `VerifyStore`: attempts, costs, stages,
  ordinary export admission and single-writer execution. JSON is an export.
  A distributed adapter should feed this learner path rather than invent a
  competing checkpoint schema or shared writable SQLite file.
* `experiments/runners/calibrate_training.py`: existing complete CPU collect,
  update, export, arena and exact-replay proof. Parent CPU time excludes arena
  workers; replay time is a subset of arena time. Existing calibration's seed
  and tiny model are smoke fixtures, not hardware or scientific calibration.
* Tests worth reusing later: `tests/training/test_calibration.py`,
  `test_ataraxos_regime.py`, `test_ataraxos_gradients.py`, and collector boundary
  tests. None verifies remote packet admission or disconnect recovery yet.

## Proposed contracts beyond the Macs

These are recommendations for the later prototype, not implemented APIs.

1. **Actor lease.** Run/attempt, actor incarnation, shard sequence, independent
   environment/action seeds, learner seat/deck assignment, source/native runtime,
   world/setup/ABI, resolved recipe and behavior artifact digest. Freeze actor
   weights for a declared segment; install new bytes atomically after validation.
   A version integer orders updates but does not replace the weight digest.
2. **Trajectory envelope.** Viewer-only tensors with every key/dtype/shape,
   legal-mask and action mapping identity, selected action/log probability,
   full behavior distribution, scalar or WDL collection values, rewards,
   transition-end flags, episode/seat identity, exact bootstrap observation,
   and opponent identity. Keep replay tapes in the evidence sidecar; tensors
   alone cannot prove authoritative Command replay. Preserve chronological
   sequences through truncation; reject unsupported belief/compound memory
   rather than dropping it from transport.
3. **Admission.** Validate digest, identities, finite values, legal support,
   sequence continuity and declared lag. Bound both bytes and packet count.
   Pause collection at the existing next-observation boundary when credits run
   out. Stale, duplicate and malformed packets receive explicit dispositions.
   Faster workers must not silently displace all samples from mini or one deck.
4. **Learner ownership.** One process owns Adam, live parameters, EMA, schedule
   and checkpoint writes. Admit immutable packets into a local journal. Publish
   weight files by temporary write, digest verification and atomic rename.
   Evaluation takes immutable ordinary exports and separate paired seeds;
   live weights never drift during an evaluation game.
5. **Retry/accounting.** Use `(run, actor incarnation, shard sequence)` plus
   payload digest as idempotency key. Lost acknowledgement retries identical
   bytes and does not increase admitted samples; same key/different bytes is
   corruption. Track produced, received, admitted, rejected and uniquely
   consumed rows plus optimizer exposures and unfinished games independently.
   Account both hosts' occupied time and CPU time, including failed work.
6. **Crash boundary.** Exactly-once optimization requires the consumed-packet
   journal and optimizer checkpoint to commit together. For a disposable first
   implementation, stop the attempt after ambiguous learner death; do not
   promise restart by replaying acknowledged data into uncertain weights.
   Actor disconnect/reconnect with idempotent delivery can be supported without
   claiming arbitrary learner recovery. Preserve failed attempts separately.

Useful now: typed envelopes, behavior provenance, a single optimizer owner,
bounded credits, unique-sample accounting and frozen evaluation. Keep transport
(files versus socket), actor/inference placement, segment length, worker quotas
and lag policy reversible. Defer multi-learner all-reduce, fault-tolerant shared
storage, autoscaling, a general actor framework, remote recurrent-state migration
and league scheduling until measurements require them.

## Benchmark evidence and remaining measurements

**Original research-only budget (2026-10-05):** no training runs; one name-resolution/access
attempt; dependency-free inventory and syntax/guard checks only. Later harness
invocations allow one CPU thread, at most 110 seconds per child with five seconds
for kill/reap. Microbenchmark timing windows default to three seconds (maximum
ten). This is a per-diagnostic cap, not an allocation for repeatedly benchmarking.
Value-token training keeps priority. No ETU-91 files or running services changed.

| Observation | Result |
| --- | --- |
| Laptop inventory, `sysctl` and `uname` | Apple M4 Max; arm64; 16 logical CPUs; 137,438,953,472 bytes RAM (128 GiB) |
| `lf home ssh mini home id`, one attempt | Exit 1: `ssh: Could not resolve hostname mini: nodename nor servname provided, or not known`; then `failed to write preamble to ssh`, broken pipe |
| Mini at initial research pass | Access failed before a remote result; superseded by the retained measurements below |
| Tailscale RTT/bandwidth | Unmeasured; name-resolution failure is not a latency measurement |
| Local environment | `.venv/bin/python` absent; no install or native build attempted |
| Inventory host load | One/five/fifteen-minute load averages 62.20 / 58.99 / 48.41; this is contention context, not attribution to a particular job |
| Original research contribution | No compute workloads; subsequent mini results below remain single-host evidence |

The original [inventory receipt](../experiments/data/etu108/inventory.json) and
[missing-environment rejection](../experiments/data/etu108/missing-environment.json)
are retained byte-for-byte. Their scratch paths, hashes and `prepared` inventory
status describe the earlier harness; the listed commands were not executed.
The mini error is a retained textual observation, not a network receipt.

`experiments/runners/distributed_benchmark.py` is a standard-library supervisor;
`experiments/runners/distributed_workloads.py` contains optional repository-dependent workers.
Each attempt gets a new directory, frozen command/source hashes, host metadata,
stdout/stderr and result. Failed exits/timeouts remain visible. Its resource
summary records child CPU seconds and OS-reported child max RSS, not a sampled
aggregate memory peak, energy consumption or throttling estimate. No stored
credentials or environment dump. Host load is context, not contention correction.

### Completed mini attempts (2026-10-06)

The [compact evidence extract](../experiments/data/etu108/mini-20261006.json)
retains exact source/runtime/world/ABI/configuration and artifact identities,
per-stage counters, resource observations, the failed traceback and SHA-256 hashes
of the supplied files. It was extracted from
`/Users/jack/src/etude.agent-9039d61b/.runs/etu108-mini-20261006/`, including
`summary.json`, all five initial attempt directories and
`etu108-mini-representative-1/{result.json,recipe.json,run/run.json,verified-summary.json}`.
Original remote paths remain provenance; raw checkpoints/tensors are not committed.
Artifact verification is reported from the retained receipt, not rerun here.

Mini is Apple M1, 8 logical CPUs, **16 GiB** (17,179,869,184 bytes),
macOS 15.5 arm64; training records identify Torch 2.10.0. All probes use CPU,
one thread. Initial attempts used `d154a4b27aaf11812125a5a1140d1ddf766dffcd`;
inference retry and representative training used PR #237's merged source
`1941b831d9f6a48e590fac68a91aef533453df45`. Initial supervisor caps were
110 seconds each; timing windows were three seconds. The representative process
cap was 240 seconds, with a 220-second regime and 100 seconds per stage.
These receipts describe completed attempts, not a new allocation.

| Attempt suffix | Observation / denominator | Attempt wall seconds | Child CPU seconds / max RSS bytes |
| --- | --- | ---: | ---: |
| simulator-1 | 100,564 surfaced decisions / 3.00004 s = 33,520.87/s | 12.455 | 6.536 / 317,128,704 |
| inference-1 | Failed at collector cleanup; no timing measurement | 2.015 | 1.637 / 279,068,672 |
| inference-2 | 9,216 forward observations / 3.01714 s = 3,054.55/s | 5.338 | 4.762 / 339,542,016 |
| train-1 | 128 learner transitions; both collect/update iterations skipped optimization, zero exposures | 5.444 | 4.034 / 373,489,664 |
| complete-1 | 8/8 complete replay-verified arena games; separate calibration recipe | 30.054 | 29.899 / 444,039,168 |

The initial inference failure was `AttributeError` on
`SeatRoutedCollector.close()`. PR #237 releases the collector reference in
`finally`: its native environment follows object lifetime. The failed attempt
remains intact; inference-2 is a new attempt, not replacement evidence.
Simulator measurements include wrapper/random-sampling overhead, not pure Rust
ticks. Forward timing excludes initialization and collection and repeats a fixed
real batch of 64 on an untrained 11,316-parameter width-16 WDL model.

Tiny train-1 used seed 108 and 4 streams × 16 transitions × 2 iterations,
with 2.856 training seconds. Both filters were empty; it is not gradient-throughput
evidence. Complete-1 used calibration seed 197, 512 learner transitions and 256
optimizer exposures over two stages; training took 2.597 seconds. Evaluation
including replay took 25.801 seconds, of which replay was 3.929 seconds (do not
add it again). This different direct-self-play objective and tiny workflow do not
measure the representative model's strength.

### Representative single-host training

Run `e0007af0847a4f45abbf42b12f937c63`, seed 10831, used w4 authored
UR Lessons versus GW Allies with sideboards, semantic input, width 64, four
attention heads, one layer, scalar value with masked mean pooling. Its Ataraxos
move recipe uses current-self behavior, .75 quantile, zero advantage floor and
actor_critic filtering. This is neither the tiny WDL probe nor a value-token run.

Two linked stages each completed 10 updates × 4 streams × 64 transitions:
**5,120 learner transitions in 68.60 training seconds / 70.73 process seconds**.
There were 10,706 environment decisions, 51 completed training games and 1,280
optimizer sample exposures, with no empty-filter skips. Collection used 26.760
seconds; learning (including targets/filter/minibatch overhead) used 40.776;
exports used 0.063. Summed stage CPU time was 68.134 seconds. The derived rate
is 74.63 learner transitions per training second; it is not useful learning
progress or a simulator rate. Six artifacts (raw, EMA and optimizer for each
stage) appear in the verified receipt; optimizer artifacts are not playable
policies. No representative arena evaluation was performed.

Actual RAM was 16 GiB, while the inherited recipe declared **32 GiB** per-stage
execution memory. This mismatch is retained: completion does not demonstrate
memory enforcement or validate capacity for that budget. Maximum sampled stage
RSS was 497,991,680 bytes (below 500 MB); it is not an exact aggregate peak.
The representative supervisor did not record whole-process CPU/max-RSS fields;
stage resource receipts must not be substituted for those missing quantities.

One short run per successful workload gives no cross-run uncertainty, hardware
ranking, speedup, learning-strength, GPU or distributed-training result. Mini can
execute this bounded local path. Matched laptop workloads, transfer costs and
both machines contributing to one learner remain necessary before placement or
synchronization is selected. The tiny empty-filter result is a reason to inspect
optimizer exposures rather than treat completed iterations as gradient work.

Harness modes and their scopes:

| Mode | Denominator and scope |
| --- | --- |
| `inventory` | No Torch/native imports or training; host and source receipt only |
| `simulator` | Four native streams, random legal actions, current authored setup; surfaced decisions/s including wrapper/tensor/sampling overhead, not pure Rust ticks/s |
| `inference` | Width-16 semantic WDL Agent; batch 64 drawn from a real 4-stream, 16-step self-play collection; forward observations/s, initialization and collection excluded |
| `train` | Existing Ataraxos recipe narrowed to two updates, 4 streams × 16 transitions/update; executor phase clocks isolate actual gradient-update time and sample exposures; raw/EMA exports through ordinary admission |
| `complete` | Existing CPU calibration (its own direct-self-play recipe and seed), export plus eight exact-replayed arena games; outer cap can cut it short and is not relaxed on failure |

The inference batch is one small, untrained trajectory sample, not a representative
state distribution for every checkpoint. It measures neither actor throughput nor
useful learning. Train-mode learning time includes target/filter/minibatch
overhead; a pure backward-kernel test is still missing. Complete mode uses a
different objective from the Ataraxos micro workload and must be labeled as such.
The retained mini attempts exercise these paths; their single-run coverage is not
a general runtime or hardware certification.

### Reproduction and stopping

From this checkout, without installing project dependencies:

```bash
uv run --no-project --python 3.12 --no-python-downloads experiments/runners/distributed_benchmark.py \
  --mode inventory --out .runs/etu108-inventory-new
```

After the priority experiment releases capacity and a matching environment is
already provisioned, run one selected mode in a fresh directory:

```bash
uv run --no-project --python 3.12 --no-python-downloads experiments/runners/distributed_benchmark.py \
  --mode inference --seconds 3 --timeout 110 --out .runs/etu108-inference-1
```

Replace `inference` with `simulator`, `train` or `complete` for the other modes.
The supervisor invokes `uv run --no-sync` inside the existing project environment.
Ctrl-C stops its own child process group and retains an interrupted result;
the deadline does the same for timeout. Use a new output path for every attempt.
An external hard kill of the supervisor itself is outside this cleanup promise.

Remote mini execution succeeded in the retained receipts. Future placement/remote
task execution must use `lf`; run the same
harness at the pinned source on mini. Do not copy a Mac native extension blindly
or infer the remote repository path. Do not launch another coding agent merely
to obtain benchmark numbers. The local command above is the workload entry point,
not a tested `lf` remote-execution recipe.

### Remaining matched measurement plan

Measure both hosts independently with identical code/runtime/content, AgentSpec,
batch/stream sizes and seeds; preserve cold startup separately from warm timing.
Repeat only under a separately declared aggregate budget. Report device and thread
counts, memory and host load beside the counters. Select intended value-token/depth
variants through existing AgentSpec fields and
add PPO recipe contrasts before representative comparisons; the current fixed
harness exposes neither contrast. Do not retrofit any future results.

For transfer, measure application echo RTT (median/p95) on a persistent connection,
then bidirectional uncompressed transfer of 1 KiB, 1 MiB and actual serialized
rollout/weight payloads with digest checks. Separate connection startup,
serialization, transfer and deserialization. Record direct versus relayed route
where available; subtract nothing to manufacture an engine rate. Cap the later
transfer diagnostic at 30 seconds and 64 MiB per direction, within its 110-second
supervisor allowance. This protocol is prepared; no transfer worker is implemented.

Compare laptop-only, mini-only and two-host execution at the same unique learner
sample quota and learning settings. Attribute collection, inference, waiting,
transfer, update, export and replay separately. Aggregate actor decisions/s,
unique admitted learner transitions/s, optimizer exposures/s and time-to-export
are different metrics. Time to useful policy progress requires a declared
evaluation curve and independent training seeds; a two-minute diagnostic cannot
establish it. Report it unavailable rather than extrapolating simulator speed.

## Later prototype acceptance and coordination

Before launching, choose placement from measured local inference/batch cost,
payload size and network RTT. If per-decision RPC dominates, start with inference
beside actors; if centralized batching wins, retain exact behavior version per
row. Compare zero-lag collection with a bounded-lag alternative only after its
learning-rule treatment is explicit. Neither architecture is selected by this note.

The minimal successful prototype must show overlapping laptop/mini actor work,
nonzero unique admitted rows from each in one optimizer lineage, changed finite
weights, ordinary checkpoint reload, and a small complete replay-verified arena.
Inject disconnect after receipt but before acknowledgement; retry the same packet
and prove that consumed row counts and updates do not double. Also disconnect
mid-payload and prove no partial admission. Fail visibly on learner ambiguity.
These checks and the actual distributed implementation remain outstanding.

ETU-101 should receive the existing run/actor IDs and counters, queue/lag/throughput
series, failures and artifact identities as the monitoring contract; this note
does not introduce W&B configuration. ETU-99 should receive per-attempt host wall
and CPU time with shared run IDs, including discarded samples and overlap, before
allocating further calibration. These are proposed handoffs; neither owner was
contacted and no authorization or accounting reconciliation is claimed. The
active ETU-91 campaign and its budget are not borrowed for this prototype.

Reusable result: the estimator/transport distinction and explicit provenance,
admission and accounting boundaries. Disposable result: the benchmark supervisor
and fixed smoke workloads. Unresolved: machine placement, synchronization,
safe nonzero lag for each rule, actor quotas under heterogeneity, representative
model size, matched runtime provisioning and the aggregate prototype budget.

Prior contribution checks: Python 3.12 AST parsing passed; inventory and missing-venv guard passed;
isolated subprocess fixtures passed success, exit-7 failure and timeout/SIGKILL
retention in 0.71 seconds; Torch/native workload checks deferred until a local
environment and priority-safe capacity are available. The first syntax-check
invocation omitted the `python` subcommand and failed in uv argument parsing;
the corrected command passed without installing dependencies.


## Design review for the repository conversation

Recommend one optimizer/checkpoint owner with explicit actor, inference and frozen
evaluation boundaries. Actor-local batched inference is a useful first candidate:
it avoids a network round trip per decision. Central inference becomes attractive
only when measured batching/device gains exceed transfer and queue delay. Keep
placement reversible and bind exact behavior artifacts in either design.
Single-host mini costs are measured; two-host placement remains unselected.

Policy lag is an independent algorithm decision. Zero-lag collection can exercise
current PPO and Ataraxos contracts without changing their estimator, but does not
require equal-sized synchronous worker rounds. Nonzero lag needs a declared
admission rule and comparison: PPO clipping is not V-trace, and Ataraxos additionally
needs full behavior distributions, central filtering and its learner-iteration
clock. V-trace is a separate implementation/treatment, including opponent-version
and same-viewer bootstrap semantics. Transport concurrency alone is no evidence
that either existing learner tolerates stale data.

The review leaves placement, quotas and nonzero lag unresolved until measurements.
The reusable deliverable is the ownership/data contract and bounded supervisor;
the fixed workloads are disposable probes. Two-host contributions, updates,
checkpoint reload, replay evaluation and disconnect accounting still define the
prototype finish line. ETU-108 remains open. This is agent technical review;
no human architecture approval or completed distributed system is claimed.


Delivery checks (2026-10-05): six dependency-free supervisor tests pass via
`uv run --no-project --python 3.12 --no-python-downloads python -m unittest discover -s tests/infra -p test_distributed_benchmark.py`;
focused Ruff lint/format and Python 3.12 AST checks pass. Fixtures cover inventory,
missing environment, successful child, exit-7 failure, launch failure and timeout
with child reap. They do not import Torch/native code or certify its workloads.
The existing CI Python unit job includes these tests. These historical delivery
checks were dependency-free; the subsequent mini runtime evidence is recorded
above. The evidence-report pass only parsed and checked retained files.
