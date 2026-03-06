# 02: Frontend Shell

**Finish line:** A SvelteKit app in the browser connects to the backend WebSocket, renders the board as text, and a human can play a full game to completion by clicking actions.

## What to build

SvelteKit + TypeScript app that connects to the Stage 1 WebSocket server (`/ws/play`), renders game state, and lets the player select actions. No card images yet — card names, types, and stats as text.

## Backend contract (shipped in Stage 1)

The backend is in `gui/server.py`. Key details:

**WebSocket protocol** — connect to `ws://localhost:8000/ws/play`:

Client sends:
```json
{"type": "new_game", "config": {"hero_deck": {...}, "villain_deck": {...}, "villain_type": "passive"}}
{"type": "action", "index": 3}
```

Server sends:
```json
{"type": "observation", "data": {<serialized obs>}, "actions": [<action descriptions>]}
{"type": "game_over", "data": {<serialized obs>}, "winner": 0}
{"type": "error", "message": "..."}
```

Observation `data` shape (from `serialize_observation` in `gui/server.py`):
```json
{
  "game_over": false,
  "won": false,
  "turn": {"turn_number": 1, "phase": "PRECOMBAT_MAIN", "step": "...", "active_player_id": 0},
  "agent": {"life": 20, "hand": [...], "battlefield": [...], "graveyard": [...], "library_count": 40},
  "opponent": {"life": 20, "hand": [...], "battlefield": [...], "graveyard": [...], "library_count": 40}
}
```

Action descriptions shape:
```json
{"index": 0, "type": "PRIORITY_CAST_SPELL", "card": "Grey Ogre", "focus": [42], "description": "Cast spell: Grey Ogre"}
```

Config can be omitted in `new_game` — server uses a default deck (Mountain/Forest/Llanowar Elves/Grey Ogre).

## Frontend data structures

```typescript
interface GameState {
  turn: { turn_number: number; phase: string; step: string; active_player_id: number }
  agent: PlayerState
  opponent: PlayerState
  actions: ActionOption[]
  gameOver: boolean
  winner: number | null
}

interface PlayerState {
  life: number
  hand: CardState[]
  battlefield: PermanentState[]
  graveyard: CardState[]
  library_count: number
}

interface CardState {
  object_id: number
  name: string
  registry_key: number
  types: string[]
  power: number | null
  toughness: number | null
}

interface PermanentState extends CardState {
  tapped: boolean
  damage: number
  summoning_sick: boolean
}

interface ActionOption {
  index: number
  type: string
  card: string | null
  focus: number[]
  description: string
}
```

## Key components

```
<App>
  <GameBoard>
    <PlayerArea side="opponent">     // top
      <LifeTotal>
      <Zone type="hand" hidden>     // show card backs + count
      <Zone type="graveyard">
    </PlayerArea>
    <Battlefield>                   // center
      <PermanentRow side="opponent">
      <PermanentRow side="agent">
    </Battlefield>
    <PlayerArea side="agent">        // bottom
      <Zone type="hand">           // show actual cards
      <Zone type="graveyard">
      <LifeTotal>
    </PlayerArea>
  </GameBoard>
  <ActionPanel actions={actions} onSelect={sendAction} />
</App>
```

## Constraints

- SvelteKit with Vite (built-in)
- WebSocket store manages connection lifecycle (connect, reconnect, message parsing)
- State updates replace the full game state on each observation (no incremental patching)
- Action panel highlights which cards/permanents are relevant to each action (using focus IDs)
- Backend uses `_mini_fastapi.py` fallback if `fastapi` isn't installed — either `pip install fastapi uvicorn` or run via `uvicorn gui.server:app`

## Done when

- `npm run dev` serves the app
- Connects to backend WebSocket, starts a game
- Board renders with card names, life totals, zones
- Clicking an action sends it to the backend
- Can play a full game to completion in the browser (text-only rendering)
