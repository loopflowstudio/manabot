# Sparse collection requests and native wrapper work

2026-10-15, while020 runs unchanged. Read-only follow-up to verified018 artifacts;
no new rental, local performance probe, production edit or changed acceptance gate.

## Observations

Both018Torch profiles at4096streams record199forward calls and67,275acting rows.
Calls of at most64rows:142calls carrying815rows (1.21% of acting rows).
67calls contain exactly one row. This histogram is not an ordered request trace:
it cannot establish that every small request occurs in the final tail, or match
a particular request size to a particular native call.

Chronological annotation windows independently show substantial late native work:

| Profile | Starting forward call (1-based) | Seconds until batch assembly | Remaining native seconds | Remaining forward seconds |
|---|---:|---:|---:|---:|
| reference |100|2.287592|1.344082|0.667761|
| prepared |100|1.932207|1.309898|0.293276|
| reference |150|1.191285|0.712463|0.338191|
| prepared |150|1.005394|0.693731|0.146105|

These are instrumented host intervals, not throughput or predicted savings.
Computation: sort collection/forward_distribution windows by ts; cut at the
indicated call's start; subtract cut from collection/_build_batch start; sum
native/forward durations starting at or after cut. No overlapping nested windows
are added together. Existing full collection totals remain8.921549/8.131573seconds.

Sources: verified018 worker-evidence/torch-{reference,prepared}-s4096/
w64-d1-loop-b32768-s4096/profile.json and trace.json. The prepared trace SHA256 is
0d93ded2aa79546f8781b7acaa37f29478695276b66be25659c77f707c41af6a.
Separate cProfile records145native calls in that second collection.

Measured source3afa1ac26f277f70dea997e288c28dcf3b77e66f and current source both show:

- managym/src/agent/vector_env.rs: VectorEnv::step_into clones the entire
  Vec<Vec<PlayerConfig>> before entering the loop and checking active rows.
- managym/src/state/player.rs: PlayerConfig derives Clone and owns String names,
  BTreeMap<String,usize> decklists/sideboards and optional String content pack.
  This is a deep copy, not shared immutable storage. At4096streams with two
  players and145calls, the source executes1,187,840PlayerConfig clones in that
  profiled collection, independent of how many rows are active.
- Inactive rows still skip actual Env stepping and observation encoding.
  python/vector_env_bindings.rs resets their reward/terminal fields; the
  per-vector result still includes empty info maps. Do not claim inactive worlds
  are simulated.
- step_with_autoreset_on_env uses player_configs only when a terminal/truncated
  game needs reset. The configs are read-only inputs to that path.
- run_into_buffers releases the GIL. The full-copy operation therefore also runs
  in each persistent producer shard; total config copies do not shrink with
  worker count. This does not prove allocator contention or explain019's scaling.

## Interpretation and separating evidence

Small GPU requests are numerous, but the late intervals also contain material
native host work. Making only their neural forward cheaper cannot remove that
native work. Repeated immutable configuration copying is a specific producer
boundary hypothesis, distinct from changing game rules or simulator algorithms.
Its cost has not been isolated. Other explanations include real active-game
stepping/encoding, bindings and scans, host scheduling and allocation contention.

After020, if decision-bearing, the smallest direct test is a native profile or a
matched frozen-source versus borrowed-config implementation on the same actual
GPU host. Preserve reset seeds, row/error ordering, masks, observations and full
learned-state gates. A field-split borrow can avoid cloning without introducing
an additional cache/Arc owner. This is a candidate test, not an implementation or
an admitted experiment. Resolve020 first and retain its complete evidence.

## Settled implementation and evidence

Direct rollout storage was rejected in020 and removed. Native configuration
borrowing was implemented, locally verified and measured in022. Exact chronology
and local checks are retained in .runs/gpu-bottleneck/collection-tail-before.md;
source/receipts and results are published in experiments/worker-scaling.md and
experiments/data/worker-scaling/gpu-022.022measured90d7894a; previous storage
sourcee41edb26 and restored Python collectora07b3cd3 remain reproducible in git.

022passed throughput gates at4096streams but failed ordinary numerical adoption
gates. All deterministic proofs passed. No tolerance was waived; no optimization
is accepted solely from throughput. Paired builds retained hashes of both native
libraries and all source inputs.021failed before training because Cargo was absent
from the parent PATH;022used the shared worker-gpu-env.sh fix. Both pods are deleted.
The new2026-10-15 investigation has a separate$25budget and targets measured
training bottlenecks; earlier prospective admission instructions are historical.

## Additional serving overhead, bounded by retained evidence

While022 runs, read-only audit of018's verified collection.pstats found598
PolicyInference.check calls for199prepared forwards (one scope-entry plus three
per request: prepare, Agent._gather_object_embeddings, catalog). These traverse
parameter/buffer versions and module modes; model_dump_json itself contributes
only0.008469s inclusive across599calls. Full checks contribute0.381104s inclusive
inside7.910221s of cProfile collection; _state accounts for0.290821s nested within
that. Do not add these nested costs or mistake profiled Python overhead for an
unprofiled saving. Reference has onecheck,0.000586s.

Source boundaries: manabot/model/inference.py check/prepare/catalog and
manabot/model/agent.py _gather_object_embeddings. Removing every check would remove
only~4.8% of this instrumented collection and would break the policy-version
contract; duplicative validation may be reducible but is not established as the
dominant remaining cost. This evidence deprioritizes a check-only optimization as
the next major intervention. No source change, new timing probe or rental.
