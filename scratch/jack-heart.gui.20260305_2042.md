# GUI: Stage 1 — Backend

## Problem

There's no way to interact with managym as a human. The engine is wrapped for RL training (vectorized envs, encoded observations, tensor outputs) but there's no interface for playing a game, inspecting game state, or debugging card behavior interactively. A human-facing WebSocket server is the foundation for the GUI wave — every later stage (frontend, card rendering, replay) depends on it.

Who benefits: developers debugging card behavior, anyone wanting to play against the engine, future replay/analysis tooling.

## Approach

FastAPI WebSocket server that wraps `managym.Env` directly (not the training wrappers). One WebSocket connection = one game session. The server auto-plays villain turns and sends observations only when it's the hero's turn to act (or on game over). Every game is recorded as a trace (JSON).

### Architecture

```
Client (WebSocket)
  ↕ JSON messages
GameSession (per connection)
  ↕ raw managym.Observation objects
managym.Env (Rust)
```

### File structure

```
gui/
  __init__.py
  server.py         # FastAPI app, WebSocket endpoint, GameSession
  trace.py          # Trace dataclasses, save/load, serialization helpers
  villain.py        # passive_policy, random_policy (operate on raw obs)
tests/gui/
  test_server.py    # WebSocket integration tests
```

### Rust change: expose card names

The PyO3 `Card` binding exposes `registry_key: int` but not `name: str`. The Rust `Card` struct has `name: String`. Add `pub name: String` to `PyCard` in `bindings.rs` — ~5 lines of Rust. Without this, the GUI can only show card IDs, which is useless for human play.

Also add `name` to the `toJSON()` card serialization for trace readability.

### Data flow

1. Client sends `{"type": "new_game", "config": {...}}`
2. Server creates `managym.Env`, calls `env.reset([hero_config, villain_config])`
3. If villain acts first, auto-play villain until hero's turn
4. Send `{"type": "observation", ...}` with serialized game state + available actions
5. Client sends `{"type": "action", "index": N}`
6. Server calls `env.step(N)`, auto-plays villain, sends next observation
7. On game over, send `{"type": "game_over", ...}`, save trace

### Villain policies

Operate on raw `managym.Observation` (not encoded numpy arrays like the RL policies). Simple logic:

```python
def passive_policy(obs: managym.Observation) -> int:
    """Pass priority when possible, otherwise first action."""
    for i, action in enumerate(obs.action_space.actions):
        if action.action_type == ActionEnum.PRIORITY_PASS_PRIORITY:
            return i
    return 0

def random_policy(obs: managym.Observation) -> int:
    """Uniform random from available actions."""
    return random.randrange(len(obs.action_space.actions))
```

These differ from `single_agent_env.py` policies which operate on encoded numpy dicts. We need raw-observation versions because the GUI server skips the ObservationEncoder entirely.

### Observation serialization

Convert `managym.Observation` to a JSON-friendly dict with human-readable fields:

```python
def serialize_observation(obs: managym.Observation) -> dict:
    return {
        "game_over": obs.game_over,
        "won": obs.won,
        "turn": {
            "turn_number": obs.turn.turn_number,
            "phase": PhaseEnum(obs.turn.phase).name,
            "step": StepEnum(obs.turn.step).name,
            "active_player_id": obs.turn.active_player_id,
        },
        "agent": serialize_player(obs.agent, obs.agent_cards, obs.agent_permanents),
        "opponent": serialize_player(obs.opponent, obs.opponent_cards, obs.opponent_permanents),
    }

def describe_actions(obs: managym.Observation) -> list[dict]:
    """Build human-readable action descriptions with card name context."""
    results = []
    card_names = _build_id_to_name(obs)  # map object IDs to card names
    for i, action in enumerate(obs.action_space.actions):
        card_name = card_names.get(action.focus[0]) if action.focus else None
        results.append({
            "index": i,
            "type": ActionEnum(action.action_type).name,
            "card": card_name,
            "focus": list(action.focus),
            "description": _format_action(action, card_name),
        })
    return results
```

### Trace format

```python
@dataclass
class GameConfig:
    hero_deck: dict[str, int]
    villain_deck: dict[str, int]
    villain_type: str       # "passive" | "random"
    seed: int | None = None

@dataclass
class TraceStep:
    observation: dict       # serialized observation
    actions: list[dict]     # available actions with descriptions
    action: int | None      # action taken (None for final obs)
    reward: float | None

@dataclass
class Trace:
    config: GameConfig
    steps: list[TraceStep]
    winner: int | None      # 0 = hero, 1 = villain, None = draw/truncated
    timestamp: str          # ISO 8601
```

Traces saved as `gui/traces/{timestamp}_{hero_vs_villain}.json`.

### WebSocket protocol

Server → Client:
```json
{"type": "observation", "data": {<serialized obs>}, "actions": [<action descriptions>]}
{"type": "game_over", "data": {<serialized obs>}, "winner": 0}
{"type": "error", "message": "..."}
```

Client → Server:
```json
{"type": "new_game", "config": {"hero_deck": {...}, "villain_deck": {...}, "villain_type": "passive"}}
{"type": "action", "index": 3}
```

### Default decks

Hardcode a default deck for quick start (the simple training deck from MatchHypers):

```python
DEFAULT_DECK = {
    "Mountain": 12, "Forest": 12,
    "Llanowar Elves": 8, "Grey Ogre": 8,
    "Lightning Bolt": 8, "Counterspell": 0,
    "Plains": 6, "Island": 6,
}
```

If `new_game` omits deck config, use defaults for both players.

## Alternatives considered

| Approach | Tradeoff | Why not |
|----------|----------|---------|
| Reuse `SingleAgentEnv` + `ObservationEncoder` | Leverages existing code | Encoded numpy arrays are wrong format — we need card names, human-readable zones, not RL tensors. The encoder strips semantic info. |
| HTTP polling instead of WebSocket | Simpler protocol | Games are inherently stateful and turn-based. WebSocket is the natural fit — no polling delay, server can push state changes. |
| Add card names via Python-side registry mapping | No Rust changes | Fragile — requires maintaining a separate name map synced with Rust. Adding `name` to PyCard is 5 lines and permanently correct. |
| Use `obs.toJSON()` directly as the wire format | Zero serialization code | JSON output lacks card names (only `registry_key`), doesn't include action descriptions, and isn't structured for display. |

## Key decisions

1. **Raw `managym.Env`, not training wrappers.** The GUI needs card names, zone names, and game structure — not flattened numpy arrays. This means new villain policies that work on raw observations, but they're trivial (5 lines each).

2. **Expose card `name` in PyO3 bindings.** This is a small Rust change (~5 lines in `bindings.rs` + `__init__.pyi`) that unblocks the entire GUI. Without it, we'd need a fragile Python-side registry. The `name` field already exists on the Rust `Card` struct.

3. **One session per WebSocket connection.** No session management, no multiplayer routing. A connection is a game. Disconnect = game ends. This keeps the server stateless beyond the connection lifetime.

4. **Auto-play villain in the step loop.** Same pattern as `SingleAgentEnv._skip_opponent()` but operating on raw observations. The human only ever sees their own decision points.

5. **Traces record serialized observations, not raw.** Traces should be self-contained and human-readable. Storing the serialized dict (with card names and action descriptions) means traces are useful without access to the card registry.

## Scope

**In scope:**
- FastAPI WebSocket server at `/ws/play`
- `managym.Env` → JSON observation serialization with card names
- Human-readable action descriptions
- Auto-play villain (passive/random policies on raw obs)
- Game trace recording to `gui/traces/`
- Rust change: add `name` field to `PyCard`
- Integration test: start game, play actions, verify trace

**Out of scope:**
- Frontend / UI (Stage 2)
- Card images / Scryfall (Stage 3)
- Trace replay endpoint (Stage 4)
- Trained model opponents
- Human vs human multiplayer
- Deck builder / card database

## Done when

```bash
# Rust bindings expose card name
python -c "import managym; print(managym.Card.__dict__)" | grep name

# Server starts
uvicorn gui.server:app --port 8000

# WebSocket endpoint accepts connections and plays a game
pytest tests/gui/test_server.py -v

# Trace file written after game completion
ls gui/traces/*.json
```

Advancing wave goals:
- "Play a full game of Magic against a passive or random opponent through a browser" — backend half
- "Record game traces (observations + actions) from any game mode" — trace recording
- "Action selection via clickable UI elements" — action description API that the frontend will consume
