# GUI

## Vision

A web-based GUI for managym that lets you play Magic games interactively and replay recorded simulations. SvelteKit + TypeScript frontend, FastAPI + WebSocket backend, Scryfall card images.

Two primary modes:
- **Play**: Human vs passive/random opponent (human vs human later)
- **Replay**: Watch recorded game traces with timeline controls

The card pool is tiny (5 basic lands, Llanowar Elves, Grey Ogre, Lightning Bolt, Counterspell) which keeps the rendering surface small.

Not here: trained model opponents, deck builder, mobile layout, multiplayer networking.

## Strategy

The backend (WebSocket server, observation serialization, trace recording, replay API) is built on raw `managym.Env` — not the RL training wrappers — so the frontend gets card names, zone structure, and human-readable action descriptions directly. One WebSocket connection = one game session. Villain turns are auto-played server-side; the client only sees hero decision points.

Key architectural decisions:
- **Raw Env, not training wrappers.** Card names, zones, game structure — not flattened numpy arrays.
- **Card `name` exposed in PyO3 bindings.** Small Rust change that permanently solves the card identity problem.
- **Event-level traces.** Every engine step (hero + villain) is recorded for replay fidelity.
- **Replay API ships with the backend.** Frontend stages can wire replay without backend redesign.
- **`_mini_fastapi.py` fallback.** Server works without installing FastAPI (offline/sandbox environments).

## Goals

- Play a full game of Magic against a passive or random opponent through a browser
- See card images from Scryfall, tapped/untapped permanents, life totals, mana, zones
- Record game traces (observations + actions) from any game mode
- Replay traces with step-forward/back and play/pause controls
- Action selection via clickable UI elements (cards in hand, permanents, action buttons)

## Risks

- Scryfall rate limits (10 req/sec) — mitigated by browser caching and tiny card pool
- WebSocket state management for human vs human (two concurrent connections to same game) — deferred
- The observation doesn't include the opponent's hand contents (hidden info) — the shared board renders opponent hand backs from counts; a later pass could redact hidden hand contents server-side
- Session resume protocol assumptions: session-expiration traces finalize with `end_reason = "session_expired"`; resume failures return `type: "error"` keeping socket open; `session_id`/`resume_token` only attached to `observation` responses, not `game_over`
- Mana pool display blocked on Rust/PyO3 bindings not exposing mana pool state

## Metrics

- Time from `npm start` to playable game: under 30 seconds
- Full game (human vs passive) completable in browser
- Trace replay of a 50-step game renders correctly
- Card images load for all 9 implemented cards
