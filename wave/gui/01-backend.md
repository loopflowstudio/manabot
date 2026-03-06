# Stage 1: Backend — FastAPI + WebSocket game server

## What to build

A FastAPI server that wraps managym.Env and exposes game play over WebSocket. Human sends actions, server sends observations. Games are recorded as traces.

## Data structures

```python
@dataclass
class GameConfig:
    hero_deck: dict[str, int]    # card_name -> count
    villain_deck: dict[str, int]
    villain_type: str            # "passive" | "random"
    seed: int | None = None

@dataclass
class TraceStep:
    observation: dict            # raw observation serialized
    action: int | None           # action taken (None for final obs)
    reward: float | None

@dataclass
class Trace:
    config: GameConfig
    steps: list[TraceStep]

# WebSocket messages (JSON)
# Server -> Client:
#   {"type": "observation", "data": <serialized obs>, "actions": <action descriptions>}
#   {"type": "game_over", "data": <final obs>, "winner": int}
#
# Client -> Server:
#   {"type": "action", "index": int}
#   {"type": "new_game", "config": <GameConfig>}
```

## Key functions

```python
# gui/server.py
class GameSession:
    """Holds a managym.Env and manages a single game."""
    env: managym.Env
    trace: Trace
    villain_policy: Callable  # passive or random

    def reset(config: GameConfig) -> dict:
        """Start a new game, return initial observation."""

    def step(action: int) -> dict:
        """Apply human action, auto-play villain if needed, return next observation."""

    def serialize_observation(obs: managym.Observation) -> dict:
        """Convert raw managym observation to JSON-friendly dict with card names."""

    def describe_actions(obs: managym.Observation) -> list[dict]:
        """Convert action space to human-readable descriptions."""
        # e.g. {"index": 0, "type": "play_land", "card": "Forest", "description": "Play Forest"}

# Villain policies (reuse from single_agent_env.py or reimplement simply)
def passive_policy(obs) -> int: ...
def random_policy(obs) -> int: ...

# Trace persistence
def save_trace(trace: Trace, path: Path) -> None: ...
def load_trace(path: Path) -> Trace: ...
```

## Constraints

- Use `managym.Env` directly, not the training wrappers (VectorEnv, ObservationEncoder). We want raw observations with object IDs and card names.
- The server must serialize managym observation objects to JSON. Need to verify what fields are accessible through PyO3 bindings (card names, mana costs, etc.).
- Villain turns should be auto-played before sending the next observation to the client — the human should only see observations where it's their turn to act (or game over).
- Traces are saved as JSON files.

## Done when

- `uvicorn gui.server:app` starts
- WebSocket at `/ws/play` accepts a `new_game` message and streams observations
- Human can send action indices and receive next observations
- Game completes and trace is saved to `gui/traces/`
- `pytest tests/gui/test_server.py` passes: start game, play to completion, verify trace
