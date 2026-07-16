# Experience protocol v1 artifacts

`managym/src/experience.rs` is the authority for the currently integrated v1
adapter boundary. It generates `experience-v1.schema.json`, a Draft 2020-12
schema for one `ProtocolV1ConformanceBundle`: a complete recovery envelope and
the command selected from that exact frame. `gui/experience_protocol.py` is the
explicit Pydantic representation used by the Python adapter; automated tests
compare its fields, required/optional keys, tagged variants, and enums to the
Rust-generated schema.

Regenerate the schema after changing the Rust types:

```bash
cd managym
cargo run --example export_experience_protocol -- ../protocol/experience-v1.schema.json
```

The checked-in `fixtures/bolt-target.json` is consumed directly by all three
languages:

- Rust strictly deserializes and round-trips it, rejects another protocol
  version, checks its revision/prompt/offer bindings, requires nullable keys,
  closes tagged payloads to unknown fields, and fails if the schema was not
  regenerated.
- Python round-trips both the shared fixture and a live `GameSession` recovery
  plus command through explicit Pydantic models and the Rust-generated schema.
  Its generated model schema is compared to the Rust authority for every
  concrete object plus all tagged variants and string enums.
- TypeScript validates and JSON-round-trips the shared fixture with AJV,
  compares its version, enums, object fields, required keys, and tagged variants
  to the schema, and validates one statically typed example of every supported
  presentation kind.

This is intentionally a narrow claim. It certifies the merged adapter's
recovery-to-command seam, including the legacy hero-view projection and a
non-empty six-kind presentation tail (`cast`, `targeted`, `resolved`, `damage`,
`destroyed`, and `died`). The envelope cursor names the first event in that
ordered tail, so Rust, Python, and TypeScript also certify exact event
addressing. It does not yet certify command outcomes as fixtures,
the future semantic table projection, or generated declarations. Those gaps
should become new fixtures as their producers land rather than being
represented here as already complete.

The Rust design uses `u64` revisions and sequence IDs while TypeScript currently
represents them as `number`. The certified fixture is far below
`Number.MAX_SAFE_INTEGER`; the wire contract must gain an explicit safe-integer
limit or string encoding before long-lived counters can approach that boundary.
