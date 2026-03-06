# Stage 5: Polish — game modes, game log, UX

## What to build

Round out the experience: human vs random opponent, game log sidebar showing what happened each step, visual feedback for actions, and general UX polish.

## Features

- **Human vs random**: Select opponent type when starting a game (passive or random)
- **Game log**: Sidebar showing step-by-step history — "Villain plays Mountain", "Hero casts Lightning Bolt targeting Grey Ogre", "Grey Ogre is destroyed"
  - Derive from observation diffs (cards moving zones, life changes, etc.)
- **Phase/step indicator**: Show current phase and step prominently (e.g., "Combat - Declare Attackers")
- **Mana display**: Show available mana pool for hero
- **Action highlights**: When hovering an action, highlight the relevant card/permanent on the board
- **New game button**: Start a fresh game without page reload

## Done when

- Can select passive or random opponent before starting
- Game log shows meaningful descriptions of each game action
- Phase indicator updates as the game progresses
- Mana pool visible during main phases
- Hovering actions highlights relevant board elements
