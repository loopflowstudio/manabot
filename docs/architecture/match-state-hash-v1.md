# MatchState deterministic hash v1

`GameState::deterministic_hash()` is the comparison oracle for mutable match
facts. It returns a `MatchStateHash` carrying contract version `1` and a BLAKE3
digest. It is intended for replay checkpoints, differential tests, and search
correctness checks. It is not an observation hash, a save format, or an
incremental transposition-table implementation.

## Canonical payload

Version 1 hashes canonical JSON with fixed struct-field order. The payload
starts with:

1. hash contract version;
2. shared `ContentPack` schema version; and
3. shared `ContentPack` content digest.

It then includes every field in `GameState`. The implementation destructures
`GameState` without `..`, so adding a field fails compilation until its v1 or
later canonical treatment is chosen.

Physical cards are encoded as `(ObjectId, CardDefId, PlayerId)`. Immutable card
definitions are represented once by the content identity, never by copied
definitions on each physical card. `CardDefId` is the semantic identity at this
state boundary. Card names and legacy registry-key values are compatibility
projections and are not canonical state identity.

The remaining facts include permanent slots, card/permanent links, current
object incarnations and LKI, players, exact zone and reverse-membership order,
turn and priority state, stack objects, combat, mana and caches, committed and
pending event ledgers, triggers and delayed links, suspended decisions, the
trigger enqueue counter, and the allocation watermark. The RNG is represented
by eight `next_u64` values from a clone; hashing never advances the live RNG.

## Ordering and representation

- Struct fields use their declared canonical order.
- Vectors, zones, stack entries, events, and choices retain semantic order.
- `BTreeMap`/`BTreeSet` order is ascending key order.
- Maps whose key cannot be a canonical JSON object key are encoded as sorted
  entry vectors. This applies to per-turn ability counters, combat blocker
  assignments, and LKI.
- Numeric typed IDs serialize by value. They are not converted through card
  names, `Debug`, pointer values, or native layout.
- Pointer addresses, vector capacities, allocator state, `Arc` strong counts,
  timing, RSS, action-space projections, and observation compatibility fields
  are excluded.

Stack-resident immutable-definition references use `CardDefId` internally.
Their serialized field remains `source_card_registry_key`, and the observation
adapter still projects the same integer, so Python, observation, fixed-action,
and the existing W2-182 benchmark snapshot ABI do not change.

## Versioning

Increment `MATCH_STATE_HASH_VERSION` before changing any of these:

- included facts or their order;
- collection ordering or canonical serialization;
- the digest algorithm;
- RNG representation; or
- the meaning or type domain of an encoded identity.

A content schema or content digest change intentionally changes the state hash,
even when mutable match facts are otherwise equal. Hash consumers must compare
both the version and digest; `Display` renders `v<version>:<hex-digest>`.

Version 1 makes no promise that a future engine release will reproduce a v1
hash after a schema change. Long-lived replay formats must store their own
schema/content provenance and treat this hash as a versioned checkpoint oracle.

## Executable contract

Run:

```bash
cd managym
cargo test --test match_state_hash_tests
```

The suite proves independently built same-seed fixed-action traces, clone and
allocation independence, mutation/content sensitivity, RNG non-mutation,
typed stack identity compatibility, and the source guard against card-name,
legacy-registry, pointer, or `Debug` formatting inputs in canonical hashing.
