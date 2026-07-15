# Experience protocol v1 artifacts

`managym/src/experience.rs` is the authority for the currently integrated v1
adapter boundary. It generates `experience-v1.schema.json`, a Draft 2020-12
schema for one `ProtocolV1ConformanceBundle`: a complete recovery envelope and
the command selected from that exact frame.

Regenerate the schema after changing the Rust types:

```bash
cd managym
cargo run --example export_experience_protocol -- ../protocol/experience-v1.schema.json
```

The checked-in `fixtures/bolt-target.json` is consumed directly by all three
languages:

- Rust strictly deserializes and round-trips it, rejects another protocol
  version, checks its revision/prompt/offer bindings, and fails if the schema
  was not regenerated.
- Python validates both the shared fixture and a live `GameSession` recovery
  plus command against the Rust-generated schema.
- TypeScript validates the shared fixture with AJV and consumes the result
  through the frontend's `RecoveryEnvelope` and `Command` types.

This is intentionally a narrow claim. It certifies the merged adapter's
recovery-to-command seam, including the legacy hero-view projection and empty
presentation list. It does not yet certify command outcomes as fixtures,
non-empty `PresentationEvent.kind` payloads, the future semantic table
projection, or generated TypeScript declarations. Those gaps should become
new fixtures as their producers land rather than being represented here as
already complete.

The Rust design uses `u64` revisions and sequence IDs while TypeScript currently
represents them as `number`. The certified fixture is far below
`Number.MAX_SAFE_INTEGER`; the wire contract must gain an explicit safe-integer
limit or string encoding before long-lived counters can approach that boundary.
