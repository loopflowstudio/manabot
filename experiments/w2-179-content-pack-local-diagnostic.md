# W2-179: ContentPack evidence boundary

## Shipped seam

W2-179 separated immutable card definitions from mutable match facts without
changing the fixed action system or search algorithm. Card definitions live in
a process-wide, schema-versioned `ContentPack` with deterministic `CardDefId`
indices. Match-local `Card` records retain object identity, ownership,
definition identity, the legacy registry-key compatibility value, and a shared
definition handle.

`GameState::content` shares the pack across environments and search branches.
`ContentPack::definition_entries` exposes stable `(CardDefId, CardDefinition)`
input for content hashing without coupling consumers to pack storage, and the
deterministic match-state hash uses definition identity rather than card names,
pointer addresses, or formatting output.

The existing fixed-action, Python, observation, replay, serialization, and
search ABIs remain compatibility surfaces. W2-179 did not add or select a new
branching representation.

## Executable behavior evidence

The seam is covered by focused tests rather than the deleted local performance
samples:

- `managym/tests/content_pack_tests.rs` checks stable IDs across independently
  built packs, legacy registry-key equality, shared pack and definition
  identity across matches and search clones, copy-on-write isolation for the
  scenario mutation seam, and deterministic seeded fixed-action traces.
- `managym/tests/match_state_hash_tests.rs` checks deterministic hashes across
  seeded traces and cloned allocations, sensitivity to meaningful mutable
  facts and content changes, RNG stability, typed `CardDefId` stack identity,
  legacy serialization, and the absence of card-name dispatch from the hash
  boundary.

These tests establish behavior and compatibility. They are not performance
measurements or allocation profiles.

## Performance evidence boundary

The original local before/after samples did not use the subsequently published
`manabot.search-branching.v1` contract. Their workloads, measurement
boundaries, memory method, and raw-data provenance were not contract-identical,
so the numeric claims have been removed rather than relabeled.

[W2-182's contract](../docs/benchmarks/search-branching-contract-v1.md),
[raw artifact](data/w2-182-search-branching-v1.json), and
[generated summary](w2-182-search-branching-v1.md) own the full-clone baseline.
That checked-in artifact is one historical run at its recorded source state,
not a W2-179 before/after pair. Later source changes make its canonical source
digest stale against current main, so it must not be described as a current
regression gate even though its recorded payload remains internally
consistent.

There is therefore no contract-valid W2-179 performance comparison to publish.
One baseline cannot establish the performance effect of the ContentPack seam.

A future comparison would require two contract-identical raw runs from one
current source state and may compare only `step-v1`, `clone-v1`, and
`flat-single-64-v1`. Both runs must preserve the exact contract deck and
fixtures, deterministic seeds, warmups and measured counts, reset and
termination rules, timing boundaries, equivalence and canonical-hash gates,
fresh-process aggregate RSS sampling every 5 ms, hardware and
source metadata, and raw-result schema. A failed or missing gate leaves the
comparison absent. W2-182 remains the harness, raw-artifact, and generated-
summary owner.

## Conclusion

W2-179's supported claim is architectural and executable: environments and
search branches share immutable, versioned card definitions through stable
typed identity while existing behavior and compatibility seams remain intact.
No performance conclusion or branching-representation decision is claimed
here.
