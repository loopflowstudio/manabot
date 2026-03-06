# Cards, Replay, and Polish

## Problem

The frontend shell renders a playable game, but cards are text-only, there's no way to review past games, and the UX lacks opponent selection, a game log, and mana display. These three wave items (03-cards-scryfall, 04-replay, 05-polish) are tightly coupled: replay reuses the board renderer, card images improve both live and replay views, and polish features (game log, opponent selector) apply to both modes. Shipping them together avoids intermediate states where replay renders text cards or the game log only works in one mode.

Wave goals advanced:
- "See card images from Scryfall, tapped/untapped permanents, life totals, mana, zones"
- "Record game traces (observations + actions) from any game mode"
- "Replay traces with step-forward/back and play/pause controls"
- "Action selection via clickable UI elements (cards in hand, permanents, action buttons)"

## Approach

One PR that delivers card rendering, replay mode, and polish features. The frontend gets a proper component architecture (extracted from the current monolithic `+page.svelte`), Scryfall images, a `/replay` route, and UX additions.

### Card rendering with Scryfall images

Extract `Card.svelte` and `CardBack.svelte` components. Cards use Scryfall's `/cards/named?format=image` endpoint with the card name.

```typescript
function scryfallImageUrl(name: string, size: 'small' | 'normal' = 'small'): string {
  return `https://api.scryfall.com/cards/named?format=image&exact=${encodeURIComponent(name)}&version=${size}`;
}
```

- Hand/graveyard/battlefield cards: `small` (146x204) image
- Hover preview: `normal` (488x680) in a floating overlay
- Opponent hand: card backs (solid color rectangles)
- Tapped permanents: `rotate(90deg)` (already handled in CSS)
- Summoning-sick: reduced opacity (already handled)
- Damage counter: overlay badge on damaged creatures

Browser-native caching handles the 9-card pool. No custom image cache needed.

### Replay mode

New route at `/replay` (SvelteKit file-based routing: `frontend/src/routes/replay/+page.svelte`).

**Trace list view:** Fetches `GET /api/traces`, renders a table with timestamp, winner, event count. Click opens replay.

**Replay viewer:** Extracts a shared `GameBoard.svelte` component from the current `+page.svelte`. The replay page feeds it observations from trace events. Timeline controls:

- Step forward/back (array index into `trace.events`)
- Play/pause with adjustable speed (1x, 2x, 4x steps/sec)
- Event scrubber (slider showing position in trace)
- Current action description displayed as annotation
- Auto-pause on final event

Data flow: `trace.events[currentStep].observation` feeds `GameBoard`, and `trace.events[currentStep].action_description` displays in an annotation bar. Step backward is just decrementing the index — no reverse computation.

**Trace fetching:** Simple `fetch()` calls to the existing REST endpoints. No WebSocket needed for replay.

### Polish features

**Opponent selector:** Dropdown before "New Game" button that sets `villain_type` in the config. Two options: "Passive" and "Random". Default: "Passive".

**Game log sidebar:** Scrollable log of action descriptions from the current game. Each time an observation arrives, diff it against the previous one to detect:
- Cards changing zones (played, cast, destroyed, drawn)
- Life total changes
- The action description from the server

Implementation: maintain an array of log entries in `GameStore`. Each hero action appends the action description. Villain actions are implicit in the observation diff (cards appeared on battlefield, life changed). The trace events already have `action_description` — for live play, the server doesn't currently send action descriptions for villain turns in the observation response. We'll add a `log` field to the wire message that includes all villain action descriptions that occurred during auto-play.

**Backend change for game log:** Modify `_auto_play_villain` and `_wire_message` to accumulate villain action descriptions and include them in the response:

```python
# In _wire_message response:
{"type": "observation", "data": {...}, "actions": [...], "log": ["Villain: Pass priority", "Villain: Play land: Mountain"]}
```

**Mana display:** Out of scope. `managym.Player` (PyO3 bindings) only exposes `life` and `zone_counts` — no mana pool. Adding it requires Rust changes to `PyPlayer` in `managym/src/python/bindings.rs`. Noted as future work.

### Component extraction

Current `+page.svelte` (284 lines) becomes:

```
src/
  lib/
    components/
      Card.svelte          — card with Scryfall image, tap/focus state
      CardBack.svelte      — hidden card (opponent hand)
      GameBoard.svelte     — the board layout (extracted from +page.svelte)
      ActionPanel.svelte   — action buttons sidebar
      GameLog.svelte       — scrollable action log
      Timeline.svelte      — replay controls (step, play/pause, scrubber)
      HoverPreview.svelte  — full-size card preview on hover
    game.svelte.ts         — (existing) game state store
    replay.svelte.ts       — replay state store
    socket.svelte.ts       — (existing) WebSocket controller
    scryfall.ts            — image URL helper
    types.ts               — (existing) + replay types
  routes/
    +page.svelte           — live play (uses GameBoard + ActionPanel + GameLog)
    +layout.svelte         — nav between play and replay
    replay/
      +page.svelte         — trace list + replay viewer
```

## Alternatives considered

| Approach | Tradeoff | Why not |
|----------|----------|---------|
| Ship cards, replay, polish as 3 separate PRs | Each PR is smaller but creates intermediate states (replay with text cards, log without replay support) | Coherent experience > smaller PRs. ~1000 LOC total is manageable. |
| Custom image proxy/cache for Scryfall | More infrastructure, handles rate limits | 9 cards + browser cache = no rate limit risk. Over-engineering. |
| Replay via WebSocket (server replays game) | Server-side replay, consistent protocol | REST fetch of trace JSON is simpler, no connection management, step-back is trivial with array indexing |
| Game log via pure client-side observation diffing | No backend changes | Misses villain action descriptions. Adding `log` to wire message is minimal backend work. |

## Key decisions

1. **Combined PR.** These three items share components (GameBoard) and would create awkward intermediate states if split. Total scope is ~800-1000 LOC of meaningful code.

2. **Scryfall direct URLs, no proxy.** The card pool is 9 cards. Browser caching handles everything. No backend image proxy.

3. **REST-based replay, not WebSocket.** Fetch the full trace as JSON, navigate client-side. Simpler, enables instant step-back, no server state.

4. **Backend `log` field for villain actions.** Minimal change: accumulate descriptions during auto-play, attach to response. Gives the game log complete information without client-side guessing.

5. **Mana display: attempt if managym exposes it, skip if it requires Rust changes.** Need to check `managym.Player` for mana pool access. If not available, note it as a future item rather than blocking the PR.

## Scope

- In scope:
  - Scryfall card images for all zones (hand, battlefield, graveyard)
  - Hover preview for full-size card
  - Card backs for opponent hand
  - Component extraction from monolithic +page.svelte
  - Replay route with trace list and viewer
  - Timeline controls (step, play/pause, speed, scrubber)
  - Opponent type selector (passive/random)
  - Game log sidebar with action descriptions
  - Backend `log` field with villain action descriptions
  - ~~Mana display~~ — deferred, requires Rust changes to expose mana pool in PyO3 bindings

- Out of scope:
  - sim.py trace recording (noted in 04 as future work)
  - Trained model opponents
  - Deck builder
  - Mobile layout
  - Human vs human
  - Custom Scryfall caching beyond browser defaults

## Done when

- All cards render with Scryfall images (9 cards, no broken images)
- Tapped permanents rotated, summoning-sick dimmed, damage shown
- Hover shows full-size card preview
- Opponent hand shows card backs
- `/replay` route lists saved traces
- Clicking a trace opens replay with step forward/back, play/pause, speed control
- Opponent selector (passive/random) works on new game
- Game log shows action descriptions for both hero and villain turns
- `npm run dev` serves everything, `npm run build` succeeds
- Existing tests pass, new replay store has tests

## Measure

Not applicable — this is feature delivery, not performance work.
