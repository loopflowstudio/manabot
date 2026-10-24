# MTG-181 seed 18101 head-to-head (copied 2026-10-23)

Local side check run from the MTG-157 conversation, outside MTG-181's frozen protocol.
Script: `scratch/mtg181-h2h/h2h.py` (run from a checkout with MTG-181's `experiments/elemental_grid.py`).
Game rows: `.runs/mtg181-h2h-18101/<new>v<old>/<new_deck>--<old_deck>--seat<s>/games.json`.
Design: newer checkpoint (x deck) vs older (y deck), 10 games per cell, seat per game from RNG 18190101.

| Pair | Games | Newer score [95% Wilson] |
|---|---:|---|
| 200 v 0 | 1,000 | 0.966 [0.953, 0.976] |
| 400 v 200 | 1,000 | 0.571 [0.540, 0.601] |
| 600 v 400 | 1,000 | 0.539 [0.508, 0.570] |
| 600 v 200 | 1,000 | 0.597 [0.566, 0.627] |

Heuristic grid (50/cell): update 0 ≈ 0.21; update 200 0.787 [0.775, 0.798].
