# 05: Polish — Game Modes, Game Log, UX

**Finish line:** Opponent type is selectable, a game log sidebar narrates what happened each step, and phase/mana/action-highlight UX is polished.

## What to build

Round out the experience: opponent selection, game log sidebar, visual feedback for actions, and general UX polish.

The backend already supports `villain_type` in the `new_game` config (`"passive"` or `"random"`). Trace events include `action_description` which can seed the game log. Phase/step names come through as strings in the observation.

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
