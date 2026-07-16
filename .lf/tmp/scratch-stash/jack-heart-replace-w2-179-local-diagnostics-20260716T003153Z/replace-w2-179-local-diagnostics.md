# W2-196: Replace W2-179 local diagnostics with contract evidence

## Decision

Rewrite `experiments/w2-179-content-pack-local-diagnostic.md` as an evidence-
boundary note. Delete the local performance workloads, result tables, warm
samples, derived byte/size calculations, numeric performance conclusions, and
the local checksum claim instead of relabeling any of them. Do not publish
replacement numbers: the repository has one W2-182 post-ContentPack contract
run at its recorded source state, not two contract-identical runs from which a
W2-179 before/after comparison could be made.

This is one documentation-only serial PR. Rebase through `lf` before editing
and again immediately before headless `lf pr land -c` when orchestration is
available. Do not create another Task Session, worktree, or PR.

## User-visible outcome

Developers and reviewers reading the W2-179 experiment see the shipped
definition/state seam and an honest statement that no contract-valid
before/after performance evidence exists. They are directed to W2-182 for the
historical full-clone baseline and measurement contract, without being offered
non-comparable W2-179 numbers or a current-main regression-gate claim.

Engine, Python, observation, fixed-action, serialization, and search users
observe no behavior or ABI change. `ContentPack`, stable `CardDefId` values,
shared immutable definitions, deterministic state hashes, and legacy
compatibility fields remain as shipped.

## Source of truth

- The implementation source of truth for the seam is `ContentPack`,
  `CardDefId`, `GameState::content`, and deterministic state hashing under
  `managym/src/`. `managym/tests/content_pack_tests.rs` and
  `managym/tests/match_state_hash_tests.rs` are executable proofs of sharing,
  stable IDs, deterministic traces/hashes, and legacy serialization behavior.
- `docs/benchmarks/search-branching-contract-v1.md` is the performance
  measurement contract. `scripts/bench_branching.py` owns execution,
  verification, and summary derivation. W2-182 owns the checked-in raw JSON and
  Markdown summary.
- The W2-179 Markdown file is a derived explanatory record. It must not become
  a second measurement owner or copy selected W2-182 values into a synthetic
  comparison.

The checked-in W2-182 artifact is historical evidence at its recorded source
digest. Its payload can remain internally consistent while current-main
verification fails the canonical source-digest check after later landings;
that expected source drift means it is not a current regression gate.

## Affected surfaces and consumers

- Change only the W2-179 experiment narrative, removing every local numeric
  performance claim and preserving the qualitative seam/compatibility facts.
- Keep the W2-182 contract, harness, raw artifact, generated report, schema,
  fixtures, and artifact ownership unchanged.
- Keep Rust engine structures and all fixed-action, Python, observation,
  serialization, replay, and search consumers unchanged.
- Keep Project KR interpretation honest: W2-179 proves definition sharing and
  stable identity; W2-182 contributes one historical full-clone baseline. This
  Task does not turn either into a two-run performance comparison.

## End-to-end proof

Concrete scenario: a reviewer opens the W2-179 experiment after this Task.
They can identify the shared `ContentPack`/`CardDefId` seam, find no local
before/after throughput, latency, allocation, byte-size, RSS, or checksum
claim, follow the W2-182 contract/artifact ownership reference, and run the
focused seam/hash tests without any compatibility change.

Verification target:

1. Inspect the final W2-179 document and its Task diff to confirm that all
   local results, warm samples, numeric derivations, and performance
   conclusions were deleted rather than renamed, and that W2-182-owned files
   and `managym/src/` did not change.
2. Run `cd managym && cargo test --test content_pack_tests --test match_state_hash_tests`.
3. Run `uv run pytest tests/bench/test_branching_benchmark.py`.
4. Do not use `uv run scripts/bench_branching.py verify` as a current-main
   regression gate for the checked-in historical artifact: its canonical
   source digest is intentionally stale after later landings.

The observable finish line is an evidence-clean W2-179 narrative plus passing
seam/hash and harness tests, with no engine or ABI diff.

## Absent and error states

- With fewer than two contract-identical current-source raw runs, the W2-179
  performance comparison is absent and the document says so plainly. One
  baseline is never treated as a before/after pair.
- A replacement claim is invalid if either run changes any required deck,
  fixture, deterministic seed, warmup/measured count, reset/termination rule,
  timing boundary, fresh-process 5 ms aggregate-RSS method, hardware/source
  metadata, raw-result schema, or the equivalence/canonical-hash gate.
- A failed equivalence check, checksum check, artifact check, worker, sampler,
  or required metadata field invalidates performance publication; it does not
  authorize a partial or directional claim.
- A current source-digest mismatch on the historical W2-182 artifact means
  only that it cannot gate current main. Do not rewrite, regenerate, or
  reinterpret that artifact in this Task.
- Missing W2-182 evidence does not fall back to the deleted W2-179 local
  numbers; the correct state remains “no contract-valid comparison.”

## Operational boundary

This Task is a documentation correction and does not run the expensive
canonical benchmark, mutate the checked-in W2-182 raw artifact, contact a
network service, or add runtime latency. If a future Task publishes a
comparison, both drivers must be measured from fresh processes on one source
state and may compare only `step-v1`, `clone-v1`, and `flat-single-64-v1` under
the exact `manabot.search-branching.v1` contract and raw schema.

## Exclusions

- Generating a second benchmark run or changing W2-182 artifacts/reporting
- Selecting full clone, clone-plus-undo, page-COW, or another representation
- Altering `ContentPack`, `CardDefId`, match-state hashing, engine behavior, or
  any ABI
- Expanding the benchmark contract, fixtures, cells, schema, or regression
  policy
- Claiming that Runtime State Foundation KRs are complete
