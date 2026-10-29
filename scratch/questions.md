# Task e2826b questions and assumptions — 2026-10-29

- **Assumption — protocol version stays `jev-mtg183-v3`:** the heuristic
  opponent and `planned_games` are optional fields whose defaults reproduce
  v3, so MTG-212's retained protocols still validate. Bump the literal if a
  reviewer wants heuristic campaigns distinguishable by version alone.
- **Assumption — 100 planned games, not budget stopping:** the first 100 games
  of MTG-212's frozen shuffle (seat counts as drawn), so the cohort is fixed
  in advance; the $2 ledger still stops admission if games run costlier.
- **Assumption — heuristic identity:** `heuristic_sha256` hashes the heuristic,
  `sim/players.py` and the semantic card sources (file name + bytes); the
  source tree digest already covers the Python.
- **Stacked on MTG-212:** this branch merges its unlanded branch; land MTG-212
  first. Comparisons with MTG-212 should use the same 100-game prefix of its
  campaign 3, not its full prefix.
