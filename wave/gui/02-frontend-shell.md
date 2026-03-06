# Stage 2: Frontend shell — React app with WebSocket and board layout

## What to build

A React + TypeScript app that connects to the backend WebSocket, renders the game board, and lets the player select actions. No card images yet — just structured layout with card names, types, and stats as text.

## Data structures

```typescript
interface GameState {
  turn: { turn_number: number; phase: string; step: string; active_player: string }
  hero: PlayerState
  villain: PlayerState
  actions: ActionOption[]
  gameOver: boolean
  winner: number | null
}

interface PlayerState {
  life: number
  hand: CardState[]        // only hero sees own hand contents
  battlefield: PermanentState[]
  graveyard: CardState[]
  library_count: number
}

interface CardState {
  id: number
  name: string
  mana_cost: string       // e.g. "2R"
  types: string[]          // e.g. ["Creature"]
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
  type: string             // "play_land", "cast_spell", "pass_priority", etc.
  card: string | null      // card name if relevant
  description: string      // human-readable
}
```

## Key components

```
<App>
  <GameBoard>
    <PlayerArea side="villain">     // top
      <LifeTotal>
      <Zone type="hand" hidden>     // show card backs + count
      <Zone type="graveyard">
    </PlayerArea>
    <Battlefield>                   // center
      <PermanentRow side="villain">
      <PermanentRow side="hero">
    </Battlefield>
    <PlayerArea side="hero">        // bottom
      <Zone type="hand">           // show actual cards
      <Zone type="graveyard">
      <LifeTotal>
    </PlayerArea>
  </GameBoard>
  <ActionPanel actions={actions} onSelect={sendAction} />
  <GameLog entries={log} />
</GameBoard>
```

## Constraints

- Use Vite for build tooling
- WebSocket hook manages connection lifecycle (connect, reconnect, message parsing)
- State updates replace the full game state on each observation (no incremental patching)
- Action panel highlights which cards/permanents are relevant to each action (using focus objects)

## Done when

- `npm run dev` serves the app
- Connects to backend WebSocket, starts a game
- Board renders with card names, life totals, zones
- Clicking an action sends it to the backend
- Can play a full game to completion in the browser (text-only rendering)
