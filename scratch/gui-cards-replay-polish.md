# Cards, Replay, and Polish

## Problem

The frontend shell is playable but still feels like scaffolding: cards are text-only, replay is missing, opponent selection is missing, and the only way to act is through the action list. The wave wants a coherent browser experience where cards look like cards, past games are inspectable, and the board itself participates in action selection.

These items belong together:
- replay should reuse the same board renderer as live play
- card images improve both live play and replay
- game log and opponent selector apply in both modes
- clickable board interactions depend on shared card/permanent components

Shipping them together avoids transitional states where replay is text-only or the board is still passive.

## Approach

One PR that delivers:
- shared board components
- Scryfall-backed card rendering
- replay UI at `/replay`
- opponent selection and game log
- clickable card/permanent UI that maps to the existing backend action list

Mana display remains deferred because the current Python bindings do not expose mana pool state.

## Core decisions

1. **Keep the combined PR.** This is one architectural slice, not three independent ones.
2. **Use Scryfall direct image URLs.** Browser caching is enough for the tiny card pool.
3. **Replay is frame-based, not raw-event-indexed.** Build replay frames from trace events once on load.
4. **Live game log is authoritative first, derived second.** Use chosen action descriptions and backend villain logs as the source of truth, then optionally append derived zone/life notes.
5. **Board interactions are first-class.** Cards in hand and permanents are clickable and route through the existing action descriptions; the action panel remains as a fallback and debugger.
6. **Hidden information stays hidden.** Shared board components render opponent hand count without relying on opponent hand contents.

## Key types and APIs

### Existing wire protocol additions

```typescript
type ServerMessage =
  | {
      type: 'observation'
      data: Observation
      actions: ActionOption[]
      log?: string[]
      session_id?: string
      resume_token?: string
    }
  | {
      type: 'game_over'
      data: Observation
      winner: number | null
      log?: string[]
    }
  | { type: 'error'; message: string }
```

### Frontend state additions

```typescript
interface PlayerState {
  player_index: number
  id: number
  is_active: boolean
  is_agent: boolean
  life: number
  zone_counts: Record<string, number>
  library_count: number
  hand_hidden_count?: number
  hand: CardState[]
  graveyard: CardState[]
  exile: CardState[]
  stack: CardState[]
  battlefield: PermanentState[]
}
```

```typescript
interface ReplayFrame {
  observation: Observation
  actionDescription: string | null
  actor: 'hero' | 'villain' | null
}
```

```typescript
interface BoardActionTarget {
  objectId: number
  actionIndexes: number[]
}
```

### Backend contract for live logs

`gui/server.py` adds `log: string[]` to responses after villain auto-play. This field is included on both `observation` and `game_over` so the final villain action is preserved.

Example:

```python
{
  "type": "observation",
  "data": {...},
  "actions": [...],
  "log": ["Villain: Play land: Mountain", "Villain: Pass priority"]
}
```

## UI architecture

```text
src/
  lib/
    components/
      Card.svelte
      CardBack.svelte
      CardImage.svelte
      CardBadge.svelte
      HoverPreview.svelte
      ZoneRow.svelte
      PermanentRow.svelte
      PlayerArea.svelte
      GameBoard.svelte
      ActionPanel.svelte
      GameLog.svelte
      Timeline.svelte
      OpponentSelector.svelte
    game.svelte.ts
    replay.svelte.ts
    socket.svelte.ts
    scryfall.ts
    action-map.ts
    log.ts
    replay.ts
    types.ts
  routes/
    +layout.svelte
    +page.svelte
    replay/+page.svelte
```

`GameBoard.svelte` is passive with callbacks:

```typescript
interface GameBoardProps {
  observation: Observation
  focusedIds: Set<number>
  clickableTargets?: Map<number, BoardActionTarget>
  onSelectTarget?: (objectId: number) => void
}
```

No socket logic in board components.

## Card rendering

Use Scryfall named-image URLs:

```typescript
function scryfallImageUrl(
  name: string,
  size: 'small' | 'normal' = 'small'
): string {
  return `https://api.scryfall.com/cards/named?format=image&exact=${encodeURIComponent(name)}&version=${size}`
}
```

- hand / graveyard / battlefield: `small`
- hover preview: `normal`
- opponent hand: `CardBack`
- tapped permanents: rotated
- summoning sick: reduced opacity
- damaged creatures: overlay badge
- failed image load: fall back to text card shell with name and stats

## Replay mode

Route: `frontend/src/routes/replay/+page.svelte`

### Trace list

- `GET /api/traces`
- table with timestamp, winner, end reason, event count
- selecting a row loads the trace and viewer

### Replay frame model

Do not bind the board directly to `trace.events[currentStep].observation`.

Build `ReplayFrame[]` once:
- frame 0 = `trace.events[0].observation`, or `trace.final_observation` when there are no events
- frame `i + 1` = post-state after event `i`
  - use `trace.events[i + 1].observation` when present
  - use `trace.final_observation` for the last event

This yields:
- initial state before any action
- correct board/action annotation pairing
- reachable final game-over frame
- trivial step-back via array indexing

### Timeline controls

- previous / next
- play / pause
- speed: 1x, 2x, 4x steps per second
- scrubber slider
- annotation bar with `actionDescription` and `actor`
- auto-pause on final frame

## Live game log

The game log has two layers:

1. **Authoritative action log**
   - when the hero clicks an action, append `Hero: {description}`
   - when the server responds, append each `Villain: ...` entry from `message.log`

2. **Derived event notes**
   - diff previous and next observations to detect:
     - life total changes
     - permanents entering/leaving battlefield
     - cards moving to graveyard
     - stack changes
   - append concise secondary notes beneath the authoritative action

The authoritative layer is required. Derived notes are additive polish, not the primary source of truth.

## Clickable board actions

The backend still exposes the canonical `actions[]` list. The frontend builds a target map from action focus IDs:

- if an action targets a card in hand, that card becomes clickable
- if an action targets a permanent, that permanent becomes clickable
- if multiple actions map to one object:
  - if there is only one sensible action, click sends it directly
  - if there are several, click narrows/highlights the action panel to those actions and requires a second explicit action click

Rules:
- action buttons always remain available
- hover on buttons still highlights focus targets
- hover on clickable cards/permanents highlights the matching actions
- pass / no-target actions remain action-panel only

This advances the wave goal without inventing a second action protocol.

## Hidden-information handling

Replay uses redacted traces by default. Opponent hand contents are therefore absent in replay.

Shared board rule:
- hero hand renders real cards from `hand`
- opponent hand renders card backs from `hand_hidden_count ?? zone_counts.HAND ?? hand.length`

If the backend only provides redacted replay payloads, the replay adapter should populate `hand_hidden_count` from `zone_counts.HAND`.

## Opponent selector

`new_game` config already supports:

```json
{"villain_type": "passive"}
{"villain_type": "random"}
```

UI:
- dropdown next to New Game
- persisted in local component/store state
- default `passive`

## Mana display

Deferred. Current bindings expose life and zone counts, not mana pool state. Doing this now would pull Rust/PyO3 changes into a frontend-oriented PR.

## Tests

### Frontend

- `game.svelte.ts`: log accumulation, clickable target map, filtered actions
- `replay.svelte.ts`: replay frame building, play/pause stepping, final-frame pause
- component tests for `GameBoard` click behavior if lightweight enough
- existing socket tests updated for `log`

### Backend

- `tests/gui/test_server.py`
  - `log` is included on observation responses with villain auto-play
  - `log` is included on `game_over` when the final villain action ends the game
  - replay endpoints still redact hidden info by default

## Alternatives considered

| Approach | Tradeoff | Why not |
|----------|----------|---------|
| Split cards / replay / polish into separate PRs | Smaller diffs, more temporary states | The shared board extraction is the real seam |
| Replay driven directly from raw trace events | Less derived state | Off-by-one board/annotation pairing and awkward final-state handling |
| Pure observation-diff game log | No backend change | Misses exact villain intent and is less trustworthy |
| Replace action panel with board clicking entirely | Cleaner-looking UI | Loses a canonical fallback for ambiguous or no-target actions |

## Scope

### In scope

- Scryfall images across live and replay
- hover preview
- opponent hand card backs
- shared board component extraction
- replay route and viewer
- timeline controls
- opponent type selector
- authoritative game log plus derived event notes
- backend `log` field for villain actions
- clickable hand/permanent interactions routed through existing `actions[]`

### Out of scope

- mana pool display
- trained model opponents
- deck builder
- mobile-specific layout
- human vs human
- custom image proxy/cache
- sim.py trace recording

## Done when

- all cards render with Scryfall images, with text fallback on image failure
- tapped permanents rotate, summoning-sick permanents dim, damage badges show
- hover preview works
- opponent hand shows card backs in live play and replay
- `/replay` lists traces and opens a viewer
- replay supports back/forward, play/pause, speed control, scrubber
- replay reaches the final board state correctly
- opponent selector starts passive or random games
- live game log shows hero and villain actions, with derived notes for major state changes
- hand cards and permanents are clickable where actions exist
- action panel remains usable for every legal move
- `npm run build` succeeds
- relevant frontend tests and backend GUI tests pass
