# MTG-301 assumptions (headless)

- MTG-302's bucketed arm was not ready at rental time (its Graph was still at `implement`
  on 2026-11-12 ~02:10); this run publishes without it, per the steer.
- Larger size: random-init w384l8 (hidden 384, 8 layers, 8 heads), falling back to w256l8 on OOM.
- The option-3 patch was rebased onto 92560a1e0 (conflicts resolved to v9-only logic;
  state_records.py dropped since main deleted it). CPU gate: scratch/option3-cpu-numerics.json.
- Bucketed run 1999b57daf19 ran from `e55157ac5`, which exists only on local branch
  `mtg301-bucketed-run` (no remote contains it). Publishing should push that ref (e.g. a tag
  `mtg301-bucketed-run`) so the report's source commit stays reachable; left to the publish step.
