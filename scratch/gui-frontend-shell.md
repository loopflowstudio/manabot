# Frontend Shell — Validation

## How to verify

Terminal 1: `cd managym && uvicorn gui.server:app --reload`
Terminal 2: `cd frontend && npm run dev`

## Done when

1. `cd frontend && npm install && npm run dev` serves the app at `localhost:5173`
2. App connects to `ws://localhost:8000/ws/play` via Vite proxy
3. Clicking "New Game" starts a game with the default deck
4. Board renders: life totals, hand (with card names/types), battlefield (with P/T, tapped state), graveyard, library count, turn/phase info
5. Action panel shows available actions as clickable buttons
6. Hovering an action highlights related cards/permanents on the board
7. Clicking an action sends it to the backend and the board updates
8. Game-over state is displayed with winner and "Play Again" option
9. A human can play a full game from start to finish in the browser
10. If the socket drops and reconnects within 15 minutes, the game resumes in-place
11. If resume fails/expired, UI clearly reports non-resumable state and offers New Game
12. Frontend has at least basic automated coverage for message parsing and store transitions (observation, game_over, error, disconnect/resume)
