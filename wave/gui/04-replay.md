# Stage 4: Trace replay

## What to build

A replay mode that loads a recorded trace and lets you step through it. Timeline controls: play/pause, step forward/back, speed slider. Same board rendering as live play.

## Data structures

```typescript
interface ReplayState {
  trace: Trace
  currentStep: number
  playing: boolean
  speed: number  // steps per second
}
```

## Key functions

```python
# Backend: replay endpoint
@app.post("/api/traces")
async def list_traces() -> list[TraceSummary]: ...

@app.get("/api/traces/{trace_id}")
async def get_trace(trace_id: str) -> Trace: ...
```

```typescript
// Frontend: replay controls
function ReplayControls({ state, onStep, onPlay, onPause, onSpeed }: ReplayProps)
function useReplay(trace: Trace): ReplayState  // hook managing playback timer
```

## Constraints

- Replay uses the same GameBoard component as live play — just fed observations from the trace instead of the WebSocket
- Step backward is just indexing into the trace array (no reverse computation)
- Auto-play advances on a timer, pauses on game-over
- Backend serves traces from `gui/traces/` directory
- Also record traces from sim.py runs (add a flag/hook to the simulation code)

## Done when

- Play a game, trace is saved
- Open replay mode, select a trace
- Step through observations forward and backward
- Auto-play with adjustable speed works
- Same visual rendering as live play
