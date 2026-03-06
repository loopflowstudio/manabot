## Vision

A web-based GUI for managym that lets you play Magic games interactively and replay recorded simulations. React + TypeScript frontend, FastAPI + WebSocket backend, Scryfall card images.

Two primary modes:
- **Play**: Human vs passive/random opponent (human vs human later)
- **Replay**: Watch recorded game traces with timeline controls

The card pool is tiny (5 basic lands, Llanowar Elves, Grey Ogre, Lightning Bolt, Counterspell) which keeps the rendering surface small.

### Not here
- Trained model opponents (requires loading model weights, wandb integration)
- Deck builder / card database browser
- Mobile-optimized layout
- Multiplayer networking (human vs human over the internet)

## Goals

- Play a full game of Magic against a passive or random opponent through a browser
- See card images from Scryfall, tapped/untapped permanents, life totals, mana, zones
- Record game traces (observations + actions) from any game mode
- Replay traces with step-forward/back and play/pause controls
- Action selection via clickable UI elements (cards in hand, permanents, action buttons)

## Risks

- managym observations are designed for RL (numeric IDs, encoded types) not display — need to bridge from raw obs to human-readable state. The `last_raw_obs` on the Rust Env has card names but we need to verify what's accessible through PyO3
- Scryfall rate limits (10 req/sec) — mitigated by browser caching and tiny card pool
- WebSocket state management for human vs human (two concurrent connections to same game) — deferred to later stage
- The observation doesn't include the opponent's hand contents (hidden info) — need to verify the GUI correctly shows only public information

## Metrics

- Time from `npm start` to playable game: under 30 seconds
- Full game (human vs passive) completable in browser
- Trace replay of a 50-step game renders correctly
- Card images load for all 9 implemented cards
