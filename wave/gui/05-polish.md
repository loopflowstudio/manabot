# 05: Polish — Game Modes, Game Log, UX

**Finish line:** Opponent type is selectable, a game log sidebar narrates what happened each step, and mana display is visible during main phases.

## What to build

Round out the experience: opponent selection, game log sidebar, mana display, and general UX polish.

The backend already supports `villain_type` in the `new_game` config (`"passive"` or `"random"`). Trace events include `action_description` which can seed the game log. The frontend shell (`frontend/src/routes/+page.svelte`) already has phase/step display and action focus highlighting — those don't need re-implementing.

## Already shipped (in frontend shell)

- Phase/step indicator (turn banner with turn number, phase, step)
- Action highlights: hovering an action highlights relevant cards/permanents via `focus` IDs
- New game button (works without page reload)
- Focus highlighting on player panels/life badges

## Remaining features

- **Human vs random**: Select opponent type when starting a game (passive or random) — add a dropdown or toggle before "New Game" that sets `config.villain_type`
- **Game log**: Sidebar showing step-by-step history — "Villain plays Mountain", "Hero casts Lightning Bolt targeting Grey Ogre", "Grey Ogre is destroyed"
  - Derive from observation diffs (cards moving zones, life changes, etc.)
  - May require backend changes to include action descriptions in observation responses
- **Mana display**: Show available mana pool for hero — requires backend to expose mana pool in observation serialization

## Done when

- Can select passive or random opponent before starting
- Game log shows meaningful descriptions of each game action
- Mana pool visible during main phases
