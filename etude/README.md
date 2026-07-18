# etude

The Etude Fantasia experience server: authoritative play, presentation, and
study. The Svelte client in [frontend/](../frontend/) renders what this
package decides; the client never invents rules meaning.

## Authority model

One versioned experience protocol drives direct play, replay, and decision
inspection as projections of the same authority. The server owns legality,
hidden information, and state; the client receives `ExperienceFrame`s and
`PresentationEvent`s and returns `Command`s bound to the exact frame that
offered them. Recovery envelopes make sessions reloadable — including fully
offline. Schemas and three-language conformance fixtures live in
[protocol/](../protocol/README.md).

## Modules

- **`server.py`**: FastAPI WebSocket server (`/ws/play`) — sessions,
  authoritative game loop, viewer-safe observation serialization
- **`experience_protocol.py`**: Pydantic representation of protocol v1,
  tested field-for-field against the Rust-generated schema
- **`presentation.py`**: semantic presentation events (combat, turn
  transitions, spell sequences) derived from engine events
- **`curated_pack.py`**: the frozen curated matchup asset pack
- **`study_protocol.py`**: viewer-safe study artifacts and decision evidence
- **`study_branch.py`**: source-bound historical forks, native structured
  execution, typed failure, and consuming exact-return receipts
- **`trace.py`**: trace persistence with hand redaction (traces land in
  `etude/traces/`; override with `ETUDE_TRACES_DIR`)
- **`villain.py`**: opponent policies for the hero to face
- **`enums.py`**: wire enums kept separate so the play runtime stays minimal

## Run it

```bash
./scripts/play          # certified one-command path from fresh checkout
./scripts/verify-clean-machine   # the clean-machine proof
```

Development server without the launcher:

```bash
uv run uvicorn etude.server:app --port 8000
```

Tests: `uv run --extra dev pytest tests/etude`. The end-to-end browser
proofs (launch, offline reload, accessibility, release prompts) are under
[frontend/e2e/](../frontend/e2e/) and documented in
[docs/clean-machine-play.md](../docs/clean-machine-play.md) and
[docs/experience-proof.md](../docs/experience-proof.md).
