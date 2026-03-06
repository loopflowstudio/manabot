# GUI: Stage 1 — Backend

## Problem

There's no way to interact with managym as a human. The engine is wrapped for RL training (vectorized envs, encoded observations, tensor outputs) but there's no interface for playing a game, inspecting game state, or debugging card behavior interactively. A human-facing WebSocket server is the foundation for the GUI wave — every later stage (frontend, card rendering, replay) depends on it.

Who benefits: developers debugging card behavior, anyone wanting to play against the engine, future replay/analysis tooling.

## Approach

FastAPI WebSocket server that wraps `managym.Env` directly (not the training wrappers). One WebSocket connection = one game session. The server auto-plays villain turns and sends observations only when it's the hero's turn to act (or on game over). Every game is recorded as a trace (JSON), including each internal action event (hero + villain) for high-fidelity replay/debugging.

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
  test_trace_api.py # Replay API integration tests
```

### Rust change: expose card names

The PyO3 `Card` binding exposes `registry_key: int` but not `name: str`. The Rust `Card` struct has `name: String`. Add `name` end-to-end in the observation model:

- `managym/src/agent/observation.rs` (`CardData` + `card_json`)
- `managym/src/python/bindings.rs` (`PyCard` + conversion + `toJSON()`)
- `managym/__init__.pyi` (`Card.name: str`)

Without this, the GUI can only show card IDs, which is useless for human play.

### Data flow

1. Client sends `{"type": "new_game", "config": {...}}`
2. Server creates `managym.Env`, calls `env.reset([hero_config, villain_config])`
3. If villain acts first, auto-play villain until hero's turn
4. Record every internal engine step in trace events (including villain auto-steps)
5. Send `{"type": "observation", ...}` with serialized game state + available actions
6. Client sends `{"type": "action", "index": N}`
7. Server calls `env.step(N)`, auto-plays villain, emits next hero decision point
8. On game over, send `{"type": "game_over", ...}`, finalize trace and save

Live client updates remain hero-only. Event-level fidelity is trace-only (for replay/debug).

### Reliability/safety constraints

- Reject actions when no active session exists, and when index is out of range.
- Auto-play loop uses a hard step cap per request (fails with explicit error instead of infinite loop).
- On disconnect: close env, save partial trace with `end_reason="disconnect"`.
- Server-side exception boundary always sends `{"type":"error"}` before closing socket.

### Trace fidelity model

Trace captures one event per engine step:

- `actor`: `"hero"` or `"villain"`
- pre-step observation + actions
- selected action index (+ description snapshot)
- immediate reward from `env.step`
- post-step observation

This preserves villain decisions and intermediate combat/stack transitions that are invisible in hero-only snapshots.

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
            "phase": PhaseEnum(int(obs.turn.phase)).name,
            "step": StepEnum(int(obs.turn.step)).name,
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
            "type": ActionEnum(int(action.action_type)).name,
            "card": card_name,
            "focus": list(action.focus),
            "description": _format_action(action, card_name),
        })
    return results
```

Use Python `IntEnum`s from `manabot.env.observation` for naming (`PhaseEnum`, `StepEnum`, `ActionEnum`). PyO3 enum objects do not expose `.name`.

### Trace format

```python
@dataclass
class GameConfig:
    hero_deck: dict[str, int]
    villain_deck: dict[str, int]
    villain_type: str       # "passive" | "random"
    seed: int | None = None

@dataclass
class TraceEvent:
    actor: str              # "hero" | "villain"
    pre_observation: dict
    actions: list[dict]
    action: int
    action_description: str
    reward: float
    post_observation: dict

@dataclass
class Trace:
    config: GameConfig
    events: list[TraceEvent]
    final_observation: dict
    winner: int | None      # 0 = hero, 1 = villain, None = draw/truncated
    end_reason: str          # "game_over" | "disconnect" | "error"
    timestamp: str          # ISO 8601
```

Traces saved as `gui/traces/{timestamp}_{hero_vs_villain}.json`.

### Replay API (included now)

```python
@app.get("/api/traces")
async def list_traces() -> list[dict]:
    # [{"id","timestamp","winner","end_reason","num_events"}]

@app.get("/api/traces/{trace_id}")
async def get_trace(trace_id: str, reveal_hidden: bool = False) -> dict:
    # redacted trace by default; full trace when reveal_hidden=true
```

Trace ID is filename stem. Endpoints are read-only and local-file backed (`gui/traces/`).
Reject path traversal by requiring `trace_id` to match filename stem (`^[A-Za-z0-9_.-]+$`) before path join.

`reveal_hidden=false` (default) redacts hidden zones in replay payloads:
- hero-hand details are hidden during villain-turn events
- villain-hand details are hidden during hero-turn events

`reveal_hidden=true` returns full debug trace.

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
    "Mountain": 12,
    "Forest": 12,
    "Llanowar Elves": 18,
    "Grey Ogre": 18,
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

5. **Traces are event-level, not hero-turn snapshots.** This preserves true action history for replay/debugging and Stage 5 game log derivation.

6. **Replay API ships in this stage.** We expose read-only trace listing/loading now so Stage 2 frontend can wire replay UI without backend redesign.

7. **Traces are stored with full engine perspective, but replay supports hide/reveal.** API defaults to redacted (`reveal_hidden=false`) and can return full debug payload (`reveal_hidden=true`).

## Scope

**In scope:**
- FastAPI WebSocket server at `/ws/play`
- `managym.Env` → JSON observation serialization with card names
- Human-readable action descriptions
- Auto-play villain (passive/random policies on raw obs)
- Event-level game trace recording to `gui/traces/`
- Replay API: `GET /api/traces`, `GET /api/traces/{trace_id}?reveal_hidden=true|false`
- Rust change: add `name` field to `PyCard`
- Add Python dependencies: `fastapi`, `uvicorn`, and test dependency `httpx`
- Integration tests:
  - WebSocket: start game, play actions, verify hero-only protocol + trace output
  - HTTP: list traces, fetch redacted trace, fetch revealed trace

**Out of scope:**
- Frontend / UI (Stage 2)
- Card images / Scryfall (Stage 3)
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

# Replay API returns trace summaries + full trace payload
pytest tests/gui/test_trace_api.py -v
```

Advancing wave goals:
- "Play a full game of Magic against a passive or random opponent through a browser" — backend half
- "Record game traces (observations + actions) from any game mode" — trace recording
- "Action selection via clickable UI elements" — action description API that the frontend will consume
