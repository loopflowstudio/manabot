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
- **`trace.py` / `attempts.py`**: viewer-safe trace projection and SQLite human
  attempts/feedback (`ETUDE_TRACES_DIR/play.sqlite`, default `etude/traces/`).
  New records have one SQLite authority; legacy JSON remains readable.
  `attempt_players` identifies both seats with stable public IDs, types, recorded
  names, decks and bot versions. Table credentials and browser player credentials
  remain private authorization values. Shared history applies permissions in SQL
  before filtering/pagination; only completed shared replays are readable by
  nonparticipants. `ETUDE_PLAY_RECORD_ORIGIN` distinguishes declared automated
  validation from human play; absent provenance is unknown.
- **`villain.py`**: opponent policies for the hero to face
- **`enums.py`**: wire enums kept separate so the play runtime stays minimal

Keep training and belief dependencies lazy at their execution boundaries.
Ordinary play must import, advance and record without NumPy or Torch; type-only
belief imports belong under `TYPE_CHECKING`. Study validates the existing
`ed2` address against the pre-command frame, and validates the played Command
against its landmark separately. Retained `erd1` addresses still bind the
recorded offer and Command.

## Run it

```bash
./scripts/play          # certified one-command path from fresh checkout
./scripts/play --demo learn  # checked prefix into Learn, then ordinary play
./scripts/verify-clean-machine   # the clean-machine proof
```

The Learn demo opens `/?demo=learn`: choose **Take a Lesson**, **Discard and
draw**, or **Decline Learn**. The first two modes show card previews and a
Back control; only selecting a card commits the choice. **New Game** repeats
the same prefix. The demo fixes UR Lessons in seat 0 against Random GW Allies;
its [evidence and remaining integration work](../docs/rules/learn-lesson.md)
are separate from complete-game replay certification.

Development server without the launcher:

```bash
uv run uvicorn etude.server:app --port 8000
```

The [local trained-challenger workflow](../docs/local-trained-challenger.md)
covers training, the built same-origin server, feedback, and read-only SQL.
Configure `ETUDE_PLAY_CANDIDATE` on the server, never in player input.

Replacement admission must keep the old GameSession authority intact until the
new environment resets successfully. Invalid native deck setup can raise
PyO3's `PanicException` (a direct `BaseException`, with no importable Python
class); contain that failure only at disposable-environment construction, while
letting process interrupts escape. Do not broaden gameplay error suppression.

Tests: `uv run --extra dev pytest tests/etude`. The end-to-end browser
proofs (launch, offline reload, accessibility, release prompts) are under
[frontend/e2e/](../frontend/e2e/) and documented in
[docs/clean-machine-play.md](../docs/clean-machine-play.md) and
[docs/experience-proof.md](../docs/experience-proof.md).
