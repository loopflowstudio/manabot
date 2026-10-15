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
