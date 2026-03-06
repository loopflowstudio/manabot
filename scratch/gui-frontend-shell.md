# Frontend Shell

## Problem

The backend WebSocket server (`gui/server.py`) is complete for basic play — it serializes game state, auto-plays villain turns, and records traces. But there's no way to actually play a game in a browser, and reconnect currently drops the game. A human needs a browser UI to see the board and click actions, plus a resumable session model for transient socket disconnects.

## Approach

Build a SvelteKit 5 + TypeScript app in `frontend/` that connects to `ws://localhost:8000/ws/play`, renders the full board state as styled text (no card images), and lets the player select actions from a panel. Extend the backend protocol with resumable sessions (in-memory only, 15 minute TTL). Use one page, one live WebSocket per client, full-replace state updates.

If session-resume implementation grows beyond a bounded effort in this stage, fall back to explicit non-resumable behavior (disconnect banner + New Game only) and defer resume to the next stage.

### Project setup

- SvelteKit with Svelte 5 (runes), TypeScript strict mode
- Tailwind CSS 4 for styling — fast iteration, consistent spacing, dark theme by default
- Lives in `frontend/` at repo root
- Vite dev server proxies `/ws/*` and `/api/*` to `localhost:8000`
- Package manager: npm (standard, no additional tooling)

### TypeScript types (corrected to match actual backend)

The wave item's TypeScript interfaces had several mismatches with `gui/server.py`. Corrected types derived from `_serialize_card`, `_serialize_permanent`, `_serialize_player`, and `serialize_observation`:

```typescript
// Matches _serialize_card (server.py:50-69)
interface CardState {
  id: number;
  registry_key: number;
  name: string;
  zone: string;
  owner_id: number;
  power: number;
  toughness: number;
  mana_value: number;
  types: {
    is_creature: boolean;
    is_land: boolean;
    is_spell: boolean;
    is_artifact: boolean;
    is_enchantment: boolean;
    is_planeswalker: boolean;
    is_battle: boolean;
  };
}

// Matches _serialize_permanent (server.py:72-85)
interface PermanentState {
  id: number;
  name: string | null;
  controller_id: number;
  tapped: boolean;
  damage: number;
  summoning_sick: boolean;
  power: number | null;
  toughness: number | null;
}

// Matches _serialize_player (server.py:88-127)
interface PlayerState {
  player_index: number;
  id: number;
  is_active: boolean;
  is_agent: boolean;
  life: number;
  zone_counts: Record<string, number>;
  library_count: number;
  hand: CardState[];
  graveyard: CardState[];
  exile: CardState[];
  stack: CardState[];
  battlefield: PermanentState[];
}

// Matches serialize_observation (server.py:130-147)
interface Observation {
  game_over: boolean;
  won: boolean;
  turn: {
    turn_number: number;
    phase: string;
    step: string;
    active_player_id: number;
    agent_player_id: number;
  };
  agent: PlayerState;
  opponent: PlayerState;
}

// Matches describe_actions output (server.py:186-201)
interface ActionOption {
  index: number;
  type: string;
  card: string | null;
  focus: number[];
  description: string;
}

// Wire messages from server
type ServerMessage =
  | {
      type: "observation";
      data: Observation;
      actions: ActionOption[];
      session_id?: string;
      resume_token?: string;
    }
  | { type: "game_over"; data: Observation; winner: number | null }
  | { type: "error"; message: string };

// Client->server resume message
type ClientMessage =
  | { type: "new_game"; config?: Record<string, unknown> }
  | { type: "action"; index: number }
  | { type: "resume"; session_id: string; resume_token: string };
```

### Architecture

**State management:** A single Svelte 5 rune-based store (`game.svelte.ts`) holds the current `Observation`, `ActionOption[]`, game-over status, and connection state. Every server message replaces the full state — no patching.

**WebSocket layer:** A `socket.svelte.ts` module manages the connection lifecycle (connect, auto-reconnect with backoff, message parsing, send). Exposes `connect()`, `sendNewGame()`, `sendAction(index)`. Calls into the game store on message receipt.

On reconnect, the client attempts `resume` first when `session_id` + `resume_token` exist in memory/sessionStorage. If resume fails (expired/invalid), UI switches to disconnected state with explicit New Game CTA.

**Backend session registry:** `gui/server.py` keeps an in-memory session map:

- key: `session_id`
- value: `{session, resume_token, last_seen_at, expires_at}`
- TTL: 15 minutes from last activity
- single active websocket attachment per session; latest socket wins

Session close no longer finalizes trace immediately on socket disconnect. Traces finalize on game over, explicit replacement by a new game, server error, or TTL expiration cleanup.

**Component tree:**
```
+page.svelte
├── ConnectionBar          — ws status, "New Game" button
├── TurnBanner             — turn number, phase, step
├── GameBoard
│   ├── PlayerArea (opponent, top)
│   │   ├── LifeTotal
│   │   ├── LibraryCount
│   │   ├── HandZone (hidden — show count only)
│   │   └── GraveyardZone
│   ├── Battlefield (center)
│   │   ├── PermanentRow (opponent)
│   │   └── PermanentRow (agent)
│   └── PlayerArea (agent, bottom)
│       ├── LifeTotal
│       ├── LibraryCount
│       ├── HandZone (visible — show card details)
│       └── GraveyardZone
├── StackZone              — cards on the stack (if any)
├── ActionPanel            — clickable list of available actions
└── GameOverOverlay        — winner announcement + "Play Again"
```

**Focus highlighting:** Each `ActionOption` has a `focus` array of object IDs. When hovering/selecting an action in the `ActionPanel`, permanents, hand cards, and player panels/life badges with matching IDs get a highlight class. This uses a reactive `Set<number>` in the game store.

### Visual design (text-only, dark theme)

- Dark background (`slate-900`), light text
- Cards rendered as small rectangles with name, P/T (if creature), tapped indicator
- Tapped permanents rotated 90deg via CSS transform
- Summoning-sick permanents dimmed
- Opponent hand shows card backs (count badge)
- Agent hand shows full card info
- Life totals prominent (large font)
- Action panel is a vertical list of buttons on the right side
- Focus-highlighted cards get a colored border (`blue-400`)
- Phase/step shown as a banner between player areas

### File structure

```
frontend/
├── package.json
├── svelte.config.js
├── vite.config.ts
├── tsconfig.json
├── tailwind.config.ts
├── src/
│   ├── app.html
│   ├── app.css              — Tailwind imports
│   ├── lib/
│   │   ├── types.ts          — all TypeScript interfaces
│   │   ├── socket.svelte.ts  — WebSocket management
│   │   └── game.svelte.ts    — game state store
│   └── routes/
│       └── +page.svelte      — single page with all components
```

All components inline in `+page.svelte` or extracted to `lib/components/` only if they exceed ~80 lines. Start inline, extract if needed. No premature component splitting.

### Dev workflow

Terminal 1: `cd managym && uvicorn gui.server:app --reload`
Terminal 2: `cd frontend && npm run dev`

Vite proxy config handles routing `/ws/play` and `/api/*` to the backend.

## Alternatives considered

| Approach | Tradeoff | Why not |
|----------|----------|---------|
| React + Next.js | Larger ecosystem, more boilerplate | SvelteKit is specified in the wave vision. Svelte 5 runes are simpler for reactive game state. |
| Plain HTML + vanilla JS | Zero build tooling | No TypeScript, no component model, harder to maintain as features grow. |
| Svelte 4 (stores) | More docs/examples available | Svelte 5 is current default. Runes are cleaner for this use case. |
| CSS modules / plain CSS | No extra dependency | Tailwind is faster for prototyping a layout-heavy UI. Utility classes read clearly in templates. |
| Multiple SvelteKit routes | Separate pages for play/replay | Single page is simpler. Replay is out of scope for this stage. |
| Separate component files from start | Cleaner file tree | Premature splitting. Start inline, extract when components grow. |

## Key decisions

1. **Types match the real backend, not the wave item spec.** The wave item's `CardState` had `object_id` and `types: string[]`. The actual backend sends `id` and `types: {is_creature: bool, ...}`. We use the real shapes.

2. **Dark theme by default.** MTG games look better on dark backgrounds. Matches the "developer tool" aesthetic.

3. **Inline components first.** Everything starts in `+page.svelte`. Extract only when a component crosses ~80 lines. This keeps the initial PR small and avoids premature abstraction.

4. **No card images in this stage.** The wave vision includes Scryfall images, but this stage is text-only. Card names + P/T stats + type indicators are sufficient for playability.

5. **Proxy instead of CORS.** Vite dev server proxies WebSocket and API requests to the backend. No CORS configuration needed on the backend.

6. **Full state replacement.** Every server message replaces the entire game state. No diffing, no incremental updates. The observation is small enough that this is fine.

7. **Focus highlighting via hover.** Hovering an action highlights its focus objects. This is the minimum viable way to connect actions to board state without click-to-select-then-click-to-act flows.

8. **Session resume v1 (in-memory) is part of this stage.** Resume uses `session_id` + `resume_token`, stored in memory/sessionStorage on the frontend; backend keeps in-memory session state with 15 minute TTL.

9. **Race risk accepted for v1.** No action idempotency key in this stage. Duplicate sends during reconnect races are a known risk and are deferred.

10. **Bounded fallback:** if resume is not landable within a tight implementation window, ship explicit non-resumable disconnect UX (Option 1) and defer full resume.

## Scope

- **In scope:** SvelteKit project setup, WebSocket connection, game state rendering (all zones), action selection, focus highlighting (including player targets), game-over handling, new game flow, dark theme styling, resumable sessions (`resume` protocol + backend in-memory registry + TTL cleanup), and frontend tests for socket/store behavior
- **Out of scope:** Card images (Scryfall), replay mode, deck builder, multiple routes, mobile layout, trained model opponents, human vs human

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
