# MTG-301 assumptions (headless)

- Larger size: random-init w384l8 (hidden 384, 8 layers, 8 heads), falling back to w256l8 on OOM.
- The option-3 patch was rebased onto 92560a1e0 (conflicts resolved to v9-only logic;
  state_records.py dropped since main deleted it). CPU gate: scratch/option3-cpu-numerics.json.
- Bucketed run source e55157ac5 pushed as tag `mtg301-bucketed-run` at publish (2026-11-12).
  `mtg301-bucketed-run` (no remote contains it). Publishing should push that ref (e.g. a tag
  `mtg301-bucketed-run`) so the report's source commit stays reachable; left to the publish step.
- Report reorder (concept review): Method was not placed in the brief's order, so it moved into the
  closing appendix; the bounds definition (wall / GPU busy / waits → sync-removal bound) moved into (a),
  and the in-situ update table moved into (b) with the rollout-recording and host-stall notes. Reversible.
