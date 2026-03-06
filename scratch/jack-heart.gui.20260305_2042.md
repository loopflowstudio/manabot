# GUI: Stage 1 — Backend

Design doc for the first stage. See `wave/gui/01-backend.md` for full spec.

## What to build

FastAPI WebSocket server wrapping managym.Env for interactive human play. Auto-plays villain turns. Records traces as JSON.

## File structure

```
gui/
  __init__.py
  server.py         # FastAPI app, WebSocket endpoint, GameSession
  trace.py          # Trace data structures, save/load
  villain.py        # passive_policy, random_policy
tests/gui/
  test_server.py    # WebSocket integration test
```

## Done when

- `uvicorn gui.server:app` starts and serves WebSocket at `/ws/play`
- Full game playable over WebSocket (send actions, receive observations)
- Trace saved to `gui/traces/` on game completion
- `pytest tests/gui/test_server.py` passes
