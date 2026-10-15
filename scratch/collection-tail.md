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

## Selected implementation after020

020completed all32cells and failed performance/adoption gates. Its immutable
evidence and exact implementation sourcee41edb26 remain retained. The original
single-GPU goal authorizes continuing the producer data-flow work; this does not
change game simulation algorithms or the original$50budget.

Delete—do not maintain: production _RolloutStorage/direct-write path, its exclusive
storage differential tests, collector_reference loader, --reference-collector and
--rollout-storage runner modes. Restore the existing pre-storage collector and
producer experiment owner froma07b3cd3. Keep frozen collector bytes/receipt and
all020protocol/results as historical evidence; reproduction uses the measured SHA.
Keep the public update-boundary test, producer/recovery tests and behavior.

Borrow immutable player configs in native step/reset iteration by splitting
field ownership. The per-env callback receives its existing config slice; no
cache, new owner, concurrency primitive or rules change. Existing error ordering,
active-mask semantics, terminal auto-reset and seed strides remain identical.
Leave restoration's infrequent owned setup replacement alone.

Local correctness allowance:120seconds of focused test execution, excluding a
bounded native rebuild (600seconds). Run debug vector tests, add a mixed-config
masked/autoreset comparison against scalar environments, rebuild root cp312release
and run existing producer/recovery plus ordinary collector tests. No laptop
performance probe. No new GPU admission until exact source publication, bounded
matched protocol, paired native-binary provenance, prior-pod absence and budget
preflight. GPU numerical diagnostics remain a separate experimental question.

Implementation complete locally: three Python owners restored byte-for-byte from
a07b3cd3; exclusive loader/test and storage CLI paths removed. Public update-boundary
test retained. Rust borrows configs in step/reset_all and passes each immutable
slice into the buffered for_each_env callback. No source mutation remains needed
inside native stepping. New masked mixed-setup test independently compares scalar
observations/rewards/dones/autoreset seeds, including different sideboards.

Validation:9Rustdebugtests pass(0.89s;5.67s including compile); root cp312release
rebuild7.85s;20Pythonproducer/recovery/inference/collector tests pass11.29s
(13.21s commandwall). Ruff, cargo fmt and diff checks pass. No performance probe
or new rental. Taskstatus shows no Flow process; only this Session and its tooling
were active in the Task observation. Raw large Task status retained privately;
future reads must filter completed processes before printing anything.

Next measured boundary: paired native builds on one rented RTX5090 host. Build
the frozen pre-borrow native source and candidate with the same toolchain/flags,
retain both binary/source hashes, and run the same Python collector/recipe for
both. Never overwrite a loaded shared-library inode; select via fresh child
processes and atomic installation before import. Source receipts must identify
which native source produced each binary. Control data remain immutable.
Before admission, implement/validate that provenance and freeze the protocol;
no021 is reserved or running. Repeated backward on a saved actual Ataraxos loss
graph remains the smallest separate diagnostic for ordinary Adam variation.

Paired-build instrumentation now uses native_variants.py and worker_experiment
--native-variants. Exacte41edb26vector_env.rs is immutable fixture/receipt. Build
both variants with identical release command, record everyRustinput+binaryhash,
restore source/library in finally, and install selected libraries by atomic
replacement before fresh children import. Every cell binds its actual library in
the source bundle and execution witness; full libraries remain in GPU artifacts.
Prospective protocol: experiments/protocols/native-config-borrowing.txt,32cells,
same gates/deadlines as020, originalbudget unchanged. No admission yet.
Local instrument-validation cap:120seconds focused correctness (not performance),
plus600seconds for two native builds. Check real fresh-child complete learning
and collection parity across both native variants and1/4workers; require loaded
hash matches selected receipt and reject a tampered binary before installation.

Instrument validation passed: both real native builds compile locally; four
fresh-process CPU cells(cloned1/borrowed1/borrowed4/cloned4) each complete3updates
with exact fullcollection and learned-state witnesses(11.40s total). Libraryhashes
were checked after each child imported its selected build; candidate restored.
Three selector/provenance/failure-restoration tests and four existing inference
tests pass(7total,4.82s). Ruff/CLI/diff checks pass. No GPU claim from these tests.
Local receipts .runs/gpu-validation/native-pair-local/builds.json and
native-pair-correctness/receipt.json; these are CPU correctness only.

021operational helper branches are prepared but not admitted. They retain build
failure logs with a manifest even if the training worker never starts. Expected
archive analyzer is analyze-configs.py, not the old storage/producer analyzers.
It must additionally verify per-cell selected native bytes/source provenance
against both retained binaries before interpreting performance/parity.

021failed before all32training cells: native paired-build process lacked Cargo in
PATH because setup's environment exports died with its subprocess. Verified547byte
archive/2manifestfiles retains exact build.log; poddeleted,wholeestimate$0.05436.
Repair adds worker-gpu-env.sh as shared sourced environment for setup and parent
job; four selector/environment tests pass, including stripped PATH and nested
child lookup. No native/learning code change or deadline/gate relaxation. New022
helper branches exist but no reservation/admission until publication/readback and
preflight. Retained021summary explicitly marks all32cells not_started.

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
