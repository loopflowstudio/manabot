# W2-193: deterministic MatchState hash and identity boundary

## User-visible outcome

Rust engine, replay, and search callers can ask `GameState` for a versioned
deterministic hash of the complete mutable match state. Equivalent matches have
the same hash even when their vectors, `Arc`s, and content packs live at
different addresses. A clone retains the hash, a fixed seeded action trace
matches an independently constructed trace at every checkpoint, and any
rules-relevant mutation changes the hash.

`CardDefId` is the typed immutable-definition identity inside mutable state.
Legacy numeric registry keys and card names remain derived compatibility
projections for the current Python, observation, and fixed-action ABIs; neither
is an authority for the state hash or runtime card dispatch.

## Source of truth

`GameState::deterministic_hash` is the runtime authority. Hash contract v1 is
BLAKE3 over canonical JSON with:

- the hash-contract version, `ContentPack::schema_version`, and
  `ContentPack::content_digest()`;
- every `GameState` field, destructured exhaustively so a future field addition
  fails compilation until the hash contract handles it;
- physical cards represented only by object ID, `CardDefId`, and owner;
- non-string-keyed ordered maps converted to key-sorted entry vectors;
- vectors and zone contents kept in semantic order; and
- the next eight words of a cloned RNG, so hashing does not advance the match.

The committed architecture note documents versioning, canonical ordering, and
compatibility boundaries. Changing field inclusion, ordering, encoding,
algorithm, or identity interpretation requires a new contract version.

## Affected surfaces and consumers

- `GameState` gains a Rust-only deterministic-hash API and hash value type.
- Stack-resident definition identity becomes `CardDefId`; legacy observation
  serialization and `source_card_registry_key` output remain numerically and
  structurally compatible.
- Rust integration tests become the executable equality, sensitivity, and
  no-name-dispatch contract.
- Python bindings, observation JSON, fixed actions, benchmark manifests,
  benchmark workloads/results, and the frontend do not change.

## Absent and error states

Canonical serialization is infallible for the closed internal state schema; a
serialization failure is an engine invariant violation. There is no fallback
to `Debug`, pointer identity, an unstable map order, or card-name lookup.
Content-pack schema or content changes deliberately produce a different hash.

Compatibility card names and legacy numeric registry keys are excluded because
they are derived from `CardDefId`; divergent compatibility projections are
invalid state but do not redefine semantic identity. Per-card `Arc` allocation
and copy-on-write definition address are also excluded; immutable definition
meaning is represented once by the shared content identity.

## Operational boundary

Hashing is deterministic and side-effect free. It performs an untimed canonical
serialization and BLAKE3 digest on demand; this Task establishes correctness,
not a throughput or incremental-hashing target. It starts no subprocess and
performs no network or filesystem I/O.

## End-to-end proof

One Rust integration suite constructs two games independently with the same
deck and seed, drives the same fixed-action sequence, and compares hashes after
every transition. The same suite checks clone/allocation independence, content
schema and content sensitivity, meaningful mutable-fact sensitivity, RNG
non-mutation, and the `CardDefId`/no-card-name source boundary.

Verification target:

```bash
cd managym
cargo test --test match_state_hash_tests
cargo test
cargo clippy --all-targets -- -D warnings
uv run maturin build --release -i ../.venv/bin/python
```

After the required extension copy, the existing Python ABI regression suite is
run through `uv`.

## Exclusions

- No W2-182 benchmark harness, scenario, raw result, report, RSS sampling,
  throughput threshold, or benchmark-ownership change.
- No ObjectRef/incarnation redesign, undo journal, clone-plus-undo, page-COW,
  branching-representation selection, or incremental hash cache.
- No Python, observation, fixed-action, frontend, card catalog, or unrelated
  rules behavior change.
