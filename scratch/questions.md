# Open choices (assumed, reversible)

- Display-name rule: size word given by the experiment, then width, then
  attention layers only when they differ from the first job's. Reproduces
  "Small (width 64)" / "Large (width 160, 8 layers)" without naming heads.
  Alternative: the experiment writes the parenthetical itself.
- Module name `manabot.training.report` (beside `panels.py`).
- Helpers do no tag validation of their own: Etude rejects an empty
  selection and notes an unused tag on the page.
- Schedule drawn through the damping horizon (values hold after it), not the
  10,000-update ceiling, so the steep part is readable; a run past the horizon
  gets Etude's "no recorded updates within the schedule" note.
- Docs corrected rather than left aspirational where Etude 0.2.16 lacks the
  behavior (earlier-self side, setup pooling, build errors); Etude Tasks
  eb0db3 and 9f66c2 carry the gaps.
