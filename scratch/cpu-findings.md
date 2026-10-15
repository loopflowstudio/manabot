# Useful Managym CPU throughput — findings for review

Requested by Jack Heart; Task **lf-8f83a20**, Wave rules. Research pass dated
2026-10-14 PDT (probe receipts use 2026-10-15 UTC). Source checkout:
`9576dc40ef486966246e20d9fdf58e5d741f1c40`.

**Recommendation:** retain the selected full-clone rules representation. The
first optimization experiment should reuse learned semantic-catalog embeddings
during inference while the policy weights are frozen. In one current authored
collection profile, catalog encoding consumed **61.2%** of wall time; the entire
native vector-step boundary consumed **1.25%**. A GPU rules rewrite or finer
grained environment threading addresses the smaller measured fraction here.
This is a prioritized hypothesis, **not an implemented or measured speedup**.

For the current random-leaf search teacher, pursue a separate diagnostic of the
structured apply/observation/receipt boundary before any representation change.
It consumed about **86%** of the three-root profile; exact forks consumed about
**0.16%**. These fractions belong to different workloads and must not be combined.

This pass delivers findings and a prospective protocol. Production source,
historical receipts, other experiments, Work state, and PRs were not changed.
No distributed learner, new Task, remote job, or paid resource was launched.

## Evidence and historical lineage

Exact local Task reads resolved the supplied MTG identifiers. Their complete
records are in [cpu-probes/](cpu-probes/); synchronization reported disconnected.
They establish historical scope and PR lineage, not fresh external Task status.

| Requested evidence | Retained source | Observation and proof limit |
|---|---|---|
| MTG-41, PR159 | [RUL-9 played workloads](../experiments/rul-9-played-workloads-v1.md) | Live Command p95 150.492 ms missed its 100 ms budget while inner semantic Command p95 was 4.291 ms. Training teacher passed at 4.520 roots/s, 578.5 traversals/s. Rules transitions alone did not explain the product miss. |
| Follow-up to that workload | [RUL-12](../experiments/rul-12-release-stack-budget-v1.md), [contended receipt](../experiments/rul-12-release-stack-budget-v1.contended-host-load.md), [recovered-then-contended receipt](../experiments/rul-12-release-stack-budget-v1.control-recovered-then-contended.md) | Passing receipt: live p95 30.659 ms, training 10.897 roots/s. Retained contention evidence includes live p95 131.500 ms and training 1.928 roots/s on unchanged source/binary. Do not select only the passing run or interpret these as a representation comparison. |
| MTG-48, PR137 | [Selected BranchDriver teacher](../docs/benchmarks/selected-branchdriver-teacher-v1.md) | Real compiled teacher retained full clone; selected versus legacy reference delivered 1.64× interactive and 1.38× saturated decision throughput. The selected structured native seam avoids legacy JSON/index lookup and a redundant rollback clone. This benefit already shipped. |
| MTG-51/52/54, PR119/107/72 | [Decision](../docs/benchmarks/search-branching-decision-v1.md), [contract](../docs/benchmarks/search-branching-contract-v1.md), raw W2-182/198/199 JSON | Matched full-clone/undo/COW whole-rollout comparison rejected both alternatives as defaults. All three stored payload hashes were recomputed successfully; this is integrity checking, not revalidation against current source. |
| MTG-53, PR78 | [search-state contract tests](../managym/tests/search_state_contract.rs), [witness implementation](../managym/src/search_state.rs) | Exact forks, source/sibling isolation, nested rollback, seeded replay, viewer boundaries, object-incarnation rejection. Current bounded checks are reported below, including timeout. |
| MTG-70, PR81 | [W2-208 allocation boundary](../experiments/w2-208-content-pack-clone-allocations.md) | Adding 4,364,032 immutable definition bytes changed neither allocation counts nor requested bytes for 1,024 GameState/Game clones. This establishes ownership, not an end-to-end speedup. |
| MTG-73, PR64 | [W2-179 evidence boundary](../experiments/w2-179-content-pack-local-diagnostic.md) | Immutable sharing is implemented. Original unmatched before/after numbers were withdrawn; there is no contract-valid ContentPack performance comparison to restore. |
| MTG-108 | [ETU-108 research and receipts](../docs/distributed-rl.md), [matched extract](../experiments/data/etu108/matched-20261006.json) | Reused its existing local benchmark and TrainingRegime/TrainingRun path. One retained laptop semantic run spent 20.733 s collecting, 21.397 s learning, 0.053 s exporting within 45.325 training seconds. No distributed execution was measured here. |
| Earlier training baseline | [March SPS closeout](../experiments/sps-closeout.md) | Historical 183k env-only SPS and 97.2% inference share refer to its old passive-opponent/model workload. Rayon/manual threading were neutral to worse there. They are counterevidence to assuming automatic multicore gains, not current universal constants. |

The matched branching comparison used source
`4354b41a80706409cdc23b26979ca2679f88b62a8d48516e18fc7b9cd3ab8beb`,
revision `2d804a766757cace5c3e0d41c3a95f36b7e31cc6`. Relative to full clone,
clone-plus-undo achieved 0.911×/0.946× flat throughput and roughly 1.50×/1.55×
RSS. Event-page COW achieved 0.843×/0.926× flat throughput. COW's retained
saturated RSS fell to 0.462×, but its full set of preregistered gates still
failed. Preserve that useful memory result alongside the rejection. Allocation
totals unsupported by the benchmark remain null, not zero.

## Current execution model

**Training:** `TrainingRegime → execute_regime → SeatRoutedCollector → Agent →
PyVectorEnv → native VectorEnv → Env/Game → Observation/encoder → NumPy buffers`.
The collector stores learner transitions and behavior distributions, builds a
`RolloutBatch`, then the configured learning rule updates the model and exports
identified artifacts. Current Ataraxos-move updates and PPO are distinct paths;
the probe uses the former, not an assumed PPO loss.

- [VectorEnv](../managym/src/agent/vector_env.rs), especially `step`, `step_into`
  and `for_each_env` (lines 124, 197, 236), iterates environments serially.
  Auto-reset and seat/opponent routing add work within the call. There is no
  Rayon iterator or native worker pool on this path.
- [PyO3 buffer binding](../managym/src/python/vector_env_bindings.rs),
  `run_into_buffers` (line 397), validates owned buffers and releases the GIL.
  GIL release permits independent callers to run; it does not make this loop
  concurrent. Each worker would need exclusive games/buffers and stable seeds.
- [Python VectorEnv](../manabot/env/vector_env.py) reuses NumPy/Torch CPU views;
  dtype/device differences still copy. [SeatRoutedCollector](../manabot/sim/net_opponent.py)
  separately selects learner/opponent rows, copies observations into pending
  transitions, and copies them into final `[time, stream, ...]` arrays. NumPy
  advanced row indexing in the sampler is also a copy, despite the underlying
  buffers being shared.
- [Agent](../manabot/model/agent.py), `_gather_object_embeddings` line 359,
  calls [SemanticCardEncoder.definitions](../manabot/model/semantic_cards.py)
  on every forward. That method embeds and runs a GRU over the complete fixed
  catalog. It is shared within one forward, not memoized across forwards.
  Catalog content is immutable; its learned embeddings change with weights.

**Search:** two existing consumers must be distinguished.
[RolloutPool](../managym/src/agent/rollout_pool.rs) holds
`worlds × root actions × rollouts` live cloned games, batches acting-viewer
observations for policy inference, and iterates slots serially. Its random
tail runs natively without observation encoding. “Simultaneous” means retained
states/batched inference, not concurrent rule execution.

The production [PUCT teacher](../manabot/sim/mcts.py) instead loops worlds and
traversals in Python through [SelectedFullCloneBackend](../manabot/sim/search_branch.py).
It retains independent nodes and uses typed structured Commands. Its random
leaf loop calls `sample_policy_index` and `apply_policy_choice` per decision.
[The binding](../managym/src/python/bindings.rs), line 2265 onward, validates
offers/preconditions, steps the rules, constructs an observation and info, and
serializes a receipt even when audit is off. The random-leaf caller discards
the observation and returned receipt. This is a possible avoidable cost, but
its separate magnitude is not measured. The selected apply binding retains
the GIL; vector-buffer GIL release must not be generalized to this consumer.

`FullCloneDriver` uses `Game::clone`; a benchmark mark is a full snapshot and
rollback replaces it. Production PUCT reports zero marks/rollbacks. Undo and
event-page COW remain implemented diagnostic drivers, not missing features.
`Arc<ContentPack>` and shared `Arc<CardDefinition>` already keep immutable
definitions out of mutable clones; sharing is within a process, not an assumed
cross-process shared heap.

Changes since the frozen receipts include w4 rules, decision schema v6,
possible-world v2, sideboard/known-hand semantic input, optional policy history,
and seed/journal support for training recovery. [WORLDS.md](../WORLDS.md) explains
why unchanged tensor dimensions alone do not establish comparability.
The native benchmark now names `*-w4-v1` fixtures: midgame expects **2** root
actions and heavy expects **120/24/505** cards/permanent slots/events. The
frozen v1 document expects **6** and **120/28/498**, respectively. The native
constant still says `manabot.search-branching.v1`. A new comparable run needs
an explicitly reconciled/versioned contract; merely fixing the source inventory
is insufficient.

## Bounded measurements and identity

[Limits](cpu-probe-limits.md) were recorded before execution, with later small
probe refinements recorded before those calls. Limits: 600 setup seconds,
240 aggregate measurement/correctness seconds, ≤90 seconds per attempt,
CPU only, one Torch/BLAS thread, ≤2 worker processes, sampled 6 GiB RSS guard.
Cargo compilation used two jobs. The debug test binary used its default test
threading and is correctness evidence only.

Host: Apple M4 Max, 16 physical/logical cores, 128 GiB RAM, macOS 26.0.1 arm64;
Python 3.12.12, Torch 2.10.0, Rust/Cargo 1.98.1, uv 0.9.22. Native extension was
built locally in release with system allocator; SHA-256
`f1a2622036d899bfde12b153d20950f02631bb6e11b24b9c862d9903c286a190`.
The tree's tracked diff remained empty.

Authored model probes use w4, policy input v1, decision v6, possible-world v2,
UR Lessons/GW Allies including sideboards; content digest
`dca9ab421ff18dc33546a35a0555c9817c302267f8fe21ff75d247e1600ce124`,
compiled IR `7a70679a4f4952976a286c4a31ffa90318bd0569745aace26353348fe6ee1edd`.
Complete setup/schema/runtime identities live in each workload's `identity.json`.
The representative recipe digest is
`4fcc602b0d7b414356e7c59933c46433e060805e7aad023865eef640eac7d8f8`;
training Python source digest is
`ba2fb33b20bde8c883062a1d4acc8048bee002ea303d3eb29dee2c02f04bfc04`.

All models begin untrained; no historical checkpoint was loaded. The update
probe produced run `408c9ef3c1e34fd5abcaa71eb06f788f` and local raw/EMA/optimizer
exports. [checkpoint-manifest.json](cpu-probes/checkpoint-manifest.json) records
every generated `.pt` path, byte count and SHA-256. None is a strength-certified
candidate and none was published.

| Probe | Observed result | Boundary / limit |
|---|---|---|
| Existing simulator mode, seed 108, 4 streams | 44,476 surfaced decisions / 1.00006 s = **44,474/s** | Includes random sampling, Python/tensor transport, native observation/reset; not pure rules ticks. |
| Existing inference mode, seed 108 | 5,760 observations / 1.00952 s = **5,706/s** | Fixed real batch of 64; 11,316-parameter width-16 WDL model; forward only. |
| Existing generic breakdown, 16 envs × 64 steps | **2,593 SPS**; inference 95.5%, native step 4.0% | Default generic model/passive opponent; not the authored semantic training recipe. Script does not seed model initialization, so this is not an exact replay baseline. |
| Tiny ETU-108 train mode, seed 108 | 128 learner transitions, 2 iterations, **0 optimizer exposures**, 2 empty-filter skips | Completed smoke; cannot measure gradient throughput. |
| Bounded semantic update, seed 10831 | 512 learner transitions, 1,348 env decisions, 4 terminal games, 128 optimizer exposures, 85 optimizer steps, 0 empty skips | 138,434 parameters, width 64, 4 attention heads, scalar/masked-mean value; 2 linked iterations of 4×64. No strength inference. |
| Isolated semantic collection, seed 10831 | 256 learner transitions, 591 env decisions, 2 terminal games, **1.08277 s** | Same width-64 architecture, fixed untrained weights, 4×64; 318 forward calls and 186 native batch calls. |
| Current selected PUCT, seed 1197, prefixes 0/16/48 | 24 traversals in **0.07260 s** across 3 roots; 54 forks, 4,247 applies, 0 caps/fallbacks/marks/rollbacks | 8 traversals/root, 2 worlds, 2,000-step cap; uniform prior/random leaves, no neural inference. Selected timings include cProfile. |

The first simulator/inference/tiny-train attempts overlapped the local release
benchmark build. Later update, isolated collection and search profiles did not.
Host one-minute load was roughly 6–10 during attempts; other work was left alone.
No thermal, energy, clock-frequency or idle-host correction is available.
Each workload has one timing attempt, not a confidence interval. Search's
unprofiled reference was used for parity, not a matched speed ratio.

### Attribution: measured components and inseparable costs

The semantic update's **5.41482 training seconds** comprise 2.54373 collection,
2.02789 learning (targets/filter/update included), 0.09763 export, and 0.74557
other setup/orchestration time. It used 5.32452 summed stage CPU seconds.
Process attempt time was 7.98670 s; sampled process-tree peak was 468.5 MiB.
Across that process, cProfile recorded 845 catalog calls taking 1.88573 s,
1.75468 s in GRU kernels, 1.413 s in backward, and 0.03598 s in native vector
calls. These nested/inclusive clocks must not be added as disjoint buckets.

The isolated collection gives the cleaner no-update attribution:

| Component | Seconds | Fraction of collection wall | Meaning |
|---|---:|---:|---|
| Agent forwards | 1.019 | 94.1% | Inclusive of catalog and attention below |
| Semantic catalog definitions | 0.66302 | **61.2%** | 318 repeats with unchanged catalog and weights |
| Native vector calls | 0.01355 | **1.25%** | Fused simulation, observation, encoding, reset, buffer write and crossing |
| Observation row tensor preparation | 0.00992 | 0.92% | Collector row selection and tensor construction |
| Final rollout assembly | 0.00231 | 0.21% | Excludes earlier pending-observation copies |

Stored batch arrays totaled **9,323,812 bytes**, including 9,046,016 observation
bytes; native buffers totaled 141,384 bytes. These are live array sizes, not peak
allocation traffic. Python objects, copies during collection, allocator slack,
and model/native state are excluded. Sampled process-tree peak was 300.8 MiB.

In search, native `apply_policy_choice` accumulated **0.06244 s** and
`fork_exact` **0.000114 s**. Each apply fuses rules, structured validation,
observation projection, Python conversion and receipt serialization. The profile
cannot assign that 0.06244 s entirely to simulation or entirely to the crossing.
Search process peak was 260.7 MiB including imports; this is not memory per branch.
No new retained-pool saturation measurement or allocator profile was obtained.
Undo timing is not applicable to these production calls (zero operations), and
current diagnostic undo performance remains unmeasured.

### Failures and correctness witnesses

- `branching-smoke` failed before any measured cell: the declared compile-time
  source inventory omits `content/semantic/v1/generated/jeong_increment.ir.json`.
  [stderr](cpu-probes/branching-smoke.stderr.log) is retained. No guard was bypassed,
  source changed, or historical artifact overwritten.
- The three search roots remained byte-identical under `search_witness_json`;
  selected and reference matched visits, Q values, root values, caps and node
  counts. Exact prefix tapes and root witnesses are in [search.json](cpu-probes/search.json).
  The ordinary measurement was not audited at every child; the separate focused
  tests cover audited world/child/leaf Commands and stale/poisoned paths.
- `uv run --no-sync pytest tests/sim/test_selected_branchdriver_teacher.py -q`:
  **10 passed** in 4.30 s.
- Debug Cargo run: all **3 ContentPack tests** and **9 of 12 search-state tests**
  printed success, including full-clone contract/replay, viewer privacy,
  hidden-world visibility, sibling isolation and stale identity rollback.
  The process was terminated at its declared **60-second cap**, with deeper
  clone-plus-undo equivalence, COW equivalence, and exhaustive nested undo cases
  unfinished. This is not a passing full suite and not evidence they failed.
  [Raw output](cpu-probes/debug-contract.stdout.log) and [timeout receipt](cpu-probes/debug-contract.receipt.json)
  preserve that distinction. No source change is being landed.

Costs: **78.502 summed setup wall seconds**, **98.985 summed probe/correctness
wall seconds**, within both declared limits. These are command durations, not
summed CPU or total conversation time; some setup overlapped probes. The maximum
sampled tree RSS was 938.2 MiB during compilation, below the 6 GiB guard.
Paid-resource spend **$0**; energy and full-session CPU cost are unmeasured.
The 200 ms guard sampling can miss short peaks and double-count shared pages.
It is deliberately not the canonical branching contract's 5 ms RSS method.

## Amdahl comparison and decision

For fraction `f` sped up by `s`, overall speedup is `1 / (1-f+f/s)`.
The following are ceilings/conditional projections from one profile, not measured
parallel or optimized results. An entire native call is an upper bound on its
pure simulation component.

| Candidate change | Measured fraction | Ideal 2× component | Ideal 4× component | Infinite-component ceiling |
|---|---:|---:|---:|---:|
| Native vector region in isolated collection | 1.25% | 1.006× | 1.009× | **1.013×** |
| Catalog encoding in isolated collection | 61.2% | 1.441× | 1.849× | **2.580×** |
| Native step region in short whole-training probe | 0.665% | 1.003× | 1.005× | **1.007×** |
| Exact forks in short search profile | ≈0.16% | ≈1.001× | ≈1.001× | **≈1.002×** |

Cache lookup, invalidation and the first catalog computation cost time, so the
catalog ceiling is unattainable. Applying the 61.2% collection fraction to the
update probe's 47.0% collection share gives an illustrative **1.40×** maximum
whole-training speedup if only collection catalog work disappeared. This mixes
two short probes and assumes stable fractions; it is a planning hypothesis.
Autograd catalog work in learning must not be removed by inference caching.

1. **First:** test explicit inference-session catalog reuse. This targets an
   observed repeated computation without changing world semantics, gradients,
   observation meaning or the selected branch representation.
2. **Then, if needed:** benchmark larger/coalesced inference batches. Current
   learner/opponent seat splitting produces 318 forwards for 186 vector calls.
   Batched API calls already reduce crossings but are not multicore execution.
   Combining seats must preserve independent sampling/accounting and policy
   identity; it is not automatically equivalent for different opponents.
3. **Search-specific:** split structured apply into rules, observation/conversion
   and receipt costs using the existing teacher. A bounded native random-tail
   API that preserves every structured Command and audit witness is plausible;
   bypassing legality, hiding a fallback or changing the tail policy is not.
4. **Defer:** multicore vector stepping, new undo/COW selection, historical
   observation reconstruction, CUDA. Independent process actors can exploit
   cores, but duplicate models/catalogs and create IPC/inference costs. There is
   no current matched 1/2/4-worker scaling curve, and the historical threading
   result was negative. Let MTG-108 own distributed learning and policy-lag
   contracts; this Task need not build another runner.

The paper's relevant systems claim was source-checked against
[Sokota et al., Nature, Engineering details and Simulator and rollout buffer design](https://www.nature.com/articles/s41586-026-11036-y).
It describes a preallocated CUDA simulator/rollout-history object, reconstructing
observations and masks on demand, independent resets and search resets, and a
target of roughly ten million board updates/s. Those ideas motivate measurement;
they do not provide a matched CPU-versus-CUDA speedup transferable to Managym.
The paper explicitly limits ablations to its final configuration without retuning.
No GPU investment conclusion is inferred from the historical total-cost comparison.

## Prospective bounded protocol — not executed or authorized by this report

After findings review accepts the next experiment, use the same existing
ETU-108 runner, collector, training executor and RUL-2 search instruments.
Do not create a second training/evaluation framework.

**Scope and cap:** local CPU only, 300 aggregate measured wall seconds,
≤60 seconds per attempt, one Torch/BLAS thread, no simultaneous measured jobs,
6 GiB sampled RSS stop, $0 paid resources. Freeze current source/binary/world,
catalog, schemas, recipe and checkpoint identities before comparing variants.
Use seeds 10831–10833, 4×64 collection, and the same two-iteration semantic update.
Any extension beyond that budget requires a separately accepted protocol.

**Candidate:** explicit reuse only inside a fixed-weight, no-grad inference
session. Invalidate after optimizer/EMA/load-state changes and on catalog,
device, dtype or mode changes. Never reuse a detached inference value during a
gradient-enabled forward; never reuse an old autograd graph across updates.
Keep the reference path selectable for differential checks.

**Measurements:** alternate baseline/candidate ordering across the three seeds;
retain all attempts, including empty filters/timeouts. Time collection, catalog,
other inference, native step, batch assembly, learning and export separately.
Record surfaced/learner decisions, forward batch histogram, terminal/cap counts,
optimizer exposures/steps, bytes copied/retained and sampled RSS. Report paired
ratios and all three samples, not a strong confidence claim from three seeds.
Use warmup and identical inference observations for kernel comparisons; keep
end-to-end trajectories as a separate workload.

**Acceptance gates:** identical legal masks, viewer-safe input identities,
sampled actions with the same RNG, logits/values at predeclared numerical tolerance,
and unchanged collection receipts. Compare post-update parameters/gradients
against the uncached training path. Demonstrate invalidation on loading an EMA
or new checkpoint. Provisional performance target: at least 20% median collection
gain and 10% whole-training gain, no paired training regression over 5%, and no
RSS increase over 10%. Failure is a useful result; retain full reference behavior.

**Search and representation safeguards:** before any later search fast path,
repeat RUL-2 selected/reference witnesses on the compiled authored world, every
mutation site, fixed viewers, hidden-world materialization, stale-offer atomic
rejection, RNG continuation, root/sibling isolation and replay. Any state-storage
change additionally needs nested rollback of zones, choices, triggers, allocation
watermarks and observation/history state, including the unfinished deeper debug
checks. Fix source-closure inventory and freeze a genuinely new w4 benchmark
contract before claiming a canonical new clone/undo/COW or multicore result.

## Reproduction and remaining gaps

[evidence-index.json](cpu-probes/evidence-index.json) holds attempts, source and
payload hashes, extracted profiler counters, identities and costs. Each
`*.receipt.json` retains exact argv, wall limit, host load and RSS samples;
matching stdout/stderr logs retain failures. `.pstats` files retain raw call
profiles; `*-profile.txt` are readable extracts. The small scripts under scratch
only bound commands or call existing collector/PUCT APIs; they do not implement
an optimization or replace a runner.

From this checkout and its prepared environment, representative direct commands
are below. Use new output directories and the recorded supervisor limits for a
fresh run; the commands are reproducibility instructions, not a launch request.

```bash
uv run --no-sync experiments/runners/distributed_benchmark.py --mode simulator --seconds 1 --timeout 75 --out scratch/cpu-probes/simulator-new
uv run --no-sync experiments/runners/distributed_benchmark.py --mode inference --seconds 1 --timeout 75 --out scratch/cpu-probes/inference-new
uv run --no-sync python -m cProfile -o scratch/cpu-probes/update-new.pstats -m experiments.runners.distributed_workloads --mode train --seconds 1 --timeout 85 --recipe scratch/cpu-probes/update-recipe.json --seed 10831 --out scratch/cpu-probes/update-new
```

Unresolved: uncontended repetitions; mature trained-policy profiles; native
simulation versus observation/serialization attribution; current retained-pool
memory per branch; allocator traffic; current undo/COW performance; multicore
scaling; cache invalidation/parity and actual benefit; power/thermal cost; and
strength per compute. The failed canonical harness and unfinished deep tests
remain explicit gaps. No claim of complete acceptance, a selected implementation,
or improved playing strength follows. **Stop at findings review.**
