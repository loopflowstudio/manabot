# GUI: Stage 1 — Backend (Validation)

## How to verify

```bash
# Rust bindings expose card name
python -c "import managym; print(managym.Card.__dict__)" | grep name

# Server starts
uvicorn gui.server:app --port 8000

# WebSocket endpoint accepts connections and plays a game
pytest tests/gui/test_server.py -v

# Trace file written after game completion
ls gui/traces/*.json

# Replay API returns trace summaries + full trace payload
pytest tests/gui/test_trace_api.py -v
```

## Quick play test

```bash
# Start server in one terminal
uvicorn gui.server:app --port 8000

# In another terminal, connect via wscat or similar:
# wscat -c ws://localhost:8000/ws/play
# Send: {"type": "new_game"}
# Observe: hero observation + action list
# Send: {"type": "action", "index": 0}
# Repeat until game_over
```

## Known limitation

Tests were validated with `py_compile` and lightweight smoke checks but not run with `pytest` in the sandbox (missing deps / `uv run` panic). Full test run needed on a machine with the dev environment.
