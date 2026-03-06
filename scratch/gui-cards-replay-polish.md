# Cards, Replay, and Polish — Validation

## Verify

1. `cd frontend && npm run build` succeeds
2. `cd frontend && npm test` passes (action-map, game.svelte, replay.svelte tests)
3. `pytest tests/gui/test_server.py` passes (log field on observation and game_over)
4. Start the server and frontend, play a full game — cards show Scryfall images, tapped permanents rotate, hover preview works
5. Open `/replay`, select a trace, step through with timeline controls — back/forward, play/pause, speed, scrubber all work
6. Opponent selector switches between passive and random
7. Game log shows hero and villain action descriptions
8. Clicking cards in hand and permanents on the battlefield selects actions; action panel remains usable for every legal move
9. Opponent hand shows card backs (not card contents) in both live play and replay

## Try it

```bash
# Backend
pip install -e ".[dev]" && pip install -e managym
uvicorn gui.server:app --reload

# Frontend (separate terminal)
cd frontend && npm install && npm run dev
# Open http://localhost:5173

# Play a game, then visit http://localhost:5173/replay
```
