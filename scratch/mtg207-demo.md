# MTG-207 demo — card page through `write_page`, 2026-10-27

Headless; the agent stood in for Jack Heart. Design and measurements:
`scratch/mtg207.md`. Assumptions: `scratch/questions.md`.

## What was run

- At HEAD 32b05920d, `/tmp/mtg207/demo_page.py` opened `card_db` on the
  converted MTG-183 blocks (read only) and called `write_page` for
  `CardSelection(experiment="mtg183-elemental-eval", subjects=("u005950",),
  opponents=("heuristic",))`. Result: 1.9 s wall, 2.5 s CPU, a 5.7 MB
  self-contained HTML page (`/tmp/mtg207/demo/u005950.html`, title
  "mtg183-elemental-eval · card data", data embedded in the page).
- `/tmp/mtg207/compare.py` (exact equality): fresh u005950 page data against
  base 4ab16a140 gives 1,452 variants, 0 differ. Whole-experiment page saved
  after the change (`/tmp/mtg207/after/whole.json`) against base gives 78,741
  variants, 0 differ.
- `tests/eval/test_card_evaluation.py`: 42 passed.
- The whole page was not run again (≈ 266 s CPU on a machine at load
  average ~140). The earlier timing in `scratch/mtg207.md` stands.

## Judgment (agent, for Jack Heart)

This matches the acceptance in Jack Heart's operator steer (2026-10-28): values
are identical on every combination, and the speedup is real (small selection
under 3 s; whole page 557 → 266 s CPU). The whole page still misses the ≤ 10 s
goal, which belongs to the follow-up Task (options 1–3 in `scratch/mtg207.md`).
No browser was available, so nobody looked at the rendered page. Only its
bytes and its data were checked.

Next: advance to delivery. The PR body should carry the measurement table and
the three follow-up options.
