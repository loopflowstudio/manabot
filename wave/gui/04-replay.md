# 04: Trace Replay

**Finish line:** A replay mode in the frontend loads a recorded trace and lets you step through it with timeline controls. Same board rendering as live play.

## What to build

Frontend replay UI that consumes the trace API (shipped in `gui/server.py`). Reuses the board rendering from `frontend/src/routes/+page.svelte` — either by extracting a shared `GameBoard` component or adding a replay mode to the existing page.

## Backend contract (shipped in Stage 1)

Replay API endpoints already exist in `gui/server.py`:

```
GET /api/traces
  → [{"id", "timestamp", "winner", "end_reason", "num_events"}, ...]

GET /api/traces/{trace_id}?reveal_hidden=false
  → full Trace object (events redacted by default, reveal_hidden=true for debug)
```

Trace events contain `pre_observation`, `actions`, `action` (index chosen), `action_description`, `reward`, `post_observation`, and `actor` ("hero"/"villain"). Observations use the same shape as the live WebSocket protocol.

Traces are saved to `gui/traces/` as JSON. The `MANABOT_GUI_TRACES_DIR` env var overrides the directory.

## Frontend data structures

```typescript
interface ReplayState {
  trace: Trace
  currentStep: number
  playing: boolean
  speed: number  // steps per second
}
```

## Key components

- Trace list page: fetch `/api/traces`, show summary table, click to open replay
- Replay viewer: reuses board rendering from `+page.svelte`, fed observations from trace events
- Timeline controls: play/pause, step forward/back, speed slider
- Step backward is array indexing (no reverse computation)
- Auto-play advances on timer, pauses on game-over

## Future: sim.py trace recording

Also record traces from `sim.py` runs (add a flag/hook to the simulation code) so RL games can be replayed.

## Done when

- Play a game, trace is saved
- Open replay mode, select a trace
- Step through observations forward and backward
- Auto-play with adjustable speed works
- Same visual rendering as live play
