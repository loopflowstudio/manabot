# Experience protocol v1 artifacts

`managym/src/experience.rs` is the authority for the currently integrated v1
adapter boundary. It generates `experience-v1.schema.json`, a Draft 2020-12
schema for one `ProtocolV1ConformanceBundle`: a complete recovery envelope and
the command selected from that exact frame. `etude/experience_protocol.py` is the
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

## Testing-house control v1

`etude/testing_house_protocol.py` owns the Game control envelope around the
unchanged match protocol. It closes the one-pilot/one-watcher request
vocabulary, server-derived roles and capabilities, participant leases,
personal-until-shared GAM-6 scenario references, and participant-local Study
events. It never embeds match legality or strategy evidence; both participants
remain bound server-side to the same player-0 rules projection.

Regenerate the checked schema after changing the Pydantic authority:

```bash
uv run --python 3.12 --locked python scripts/export_testing_house_protocol.py
```

`fixtures/testing-house-control-v1.json` contains pilot/watcher access and one
example of every control request and event. Python validates the fixture,
schema, and server dispatch as one closed operation set; TypeScript validates
the same fixture with AJV and compares its union vocabulary to the schema
discriminator. This fixture contains no credentials or strategy payload and
does not alter `fixtures/advice-curated-decision.json`.

## Canonical replay decision index v1

`managym/src/canonical_replay.rs` owns the safe projection contract and address
grammar; `gui/replay_index.py` owns the complete mixed-view Game record and
authorized restoration. A complete replay contains every deliberate human and
opponent-policy decision in one contiguous global ordinal sequence. Automatic
passes and rules resolution advance authority revisions and presentation tracks
but never receive decision ordinals.

The complete artifact is persisted only inside a trace and never checked into a
client fixture. `canonical-replay-player-0.json` and
`canonical-replay-player-1.json` are separate viewer-safe projections from one
deterministic UR Lessons versus GW Allies match. Their ordinal union equals the
private metadata-only chronology in `canonical-replay-authority-metadata.json`.
Neither safe artifact contains the other viewer's frames, offers, commands, or
private hand identities.

An `erd1.` address is unpadded base64url over a fixed JSON array. Every numeric
identity is a canonical decimal string, so JavaScript never rounds a `u64`.
The address binds replay, match, global ordinal, viewer, revision, prompt,
offer, command, presentation cursor, and a SHA-256 digest of the exact
viewer-safe frame/offer/command row.

Regenerate and verify the shared artifacts with:

```bash
uv run --extra dev python scripts/generate_replay_fixtures.py
cargo run --locked --manifest-path managym/Cargo.toml \
  --example export_canonical_replay -- protocol/canonical-replay-v1.schema.json
cargo test --locked --manifest-path managym/Cargo.toml --test canonical_replay_tests
uv run --extra dev pytest -q tests/gui/test_replay_index.py
npm --prefix frontend test -- --run src/lib/replay-index.test.ts
```

## Study artifact v1

`managym/src/study.rs` owns the separate, closed study evidence contract and
generates `study-v1.schema.json`. Its shared
`fixtures/study-curated-decision.json` carries one Game-issued `erd1` address
from the player-0 canonical replay projection. The artifact embeds the exact
viewer-safe `ExperienceFrame`, the selected `InteractionOffer`, and the played
`Command`; validators bind those copies back to the address and consumers never
reconstruct that history from current state.

The artifact pins content and asset pack, engine build, model checkpoint,
match state, viewer, decision, prompt, offer, source replay, and analysis
budget identities. Policy mass, search value, visits, sampled-world
robustness, uncertainty, and provenance are separate typed fields keyed to
the same exact command alternatives. Cross-field validators in Rust, Python,
and TypeScript reject identity drift and opponent hand identities. The closed
schema rejects unmodelled sidecars such as RNG seeds.

Regenerate and verify it with:

```bash
cargo run --locked --manifest-path managym/Cargo.toml \
  --example export_study_protocol -- protocol/study-v1.schema.json
cargo test --locked --manifest-path managym/Cargo.toml --test study_protocol_tests
uv run pytest -q tests/etude/test_study_protocol.py
npm --prefix frontend test -- --run src/lib/study-protocol.test.ts
```

This certifies portable restoration and evidence labelling only. V1 contains
no landmark selector, search execution, branch state, UI, annotation or share
service, hindsight lens, or client-side legality model.
