# MTG-135 protocol017 final result — delivered 2026-10-19

The complete longer size/batching comparison finished on the Mini: 22 fits,
two training seeds, four raw checkpoints, 18,304 replay-verified games, and the
automatic audit. The rendered report is in `report/evidence.md`. All 53
manifest entries pass SHA256 verification. The full 587 MB package, including
per-panel evaluation evidence, is at
`.runs/mtg135-handoff/learning-017-final/`. That copy matched its source byte
for byte across 9,553 files. Its per-file hash list is `full-package.sha256`,
SHA256 `3824e75bc1fa74892f2c68824f270c1fa06cbd8538e21994669d13ce40e1a416`.
Source commit: `3c3681d2b26a7fc8167928fa63fe24a6cd3d058a` (full source pack:
`.runs/mtg135-handoff/protocol017-source/`).

## Operator review of the result

- **Frozen-rule selection: 64×2 at 256 streams** (`report/recipes/small-s256.json`).
  Both seeds pass at 524,288 transitions. Its worst-seed time to qualification
  is 5.2 minutes.
- **For MTG-122's selected 64-stream plan, use 64×2** (`report/recipes/small-s64.json`).
  It qualifies at 262,144 transitions, with a worst-seed time of 6.9 minutes.
  Next is 64×4 at 64 streams, with 8.5 minutes. At 64 streams, 192×4, 384×2 and
  384×8 did not qualify.
- **The strongest finding is speed.** At equal transitions, fits took 109–111
  minutes at 4 streams, 12–16 minutes at 64, and 5–7 minutes at 256 for every
  model except 384×8.
- **Batch size trades update opportunities for wall time.** Curves at 64 streams
  rise by 131k–262k transitions and then flatten. Curves at 256 streams are
  slower per transition and are still rising at 524k. Measured by wall time,
  256 reaches the same endpoint in about 40% of the time. This compares whole
  recipes: optimizer steps, filtering and staleness all change together.
- **Endpoints cluster together.** Except for 384×8 at 64 streams, endpoints are
  46–60% against greedy and 74–89% against random. With 104 games per anchor
  and two seeds, the data does not support a skill ranking. Larger models show
  no advantage at this horizon.
- **384×8 at 64 streams is anomalous.** It used about 57k of 131k selected
  exposures and skipped some empty groups. It reached only 29–32% against
  greedy. The same model at 256 streams used about 122k–128k exposures and
  reached 46–56%. Do not read this as "deep models fail"; the cause is not
  established.
- Minor rendering issue: the report's opening sentence says "even though"
  sustained qualification appeared at 524,288. That is the full horizon, so the
  wording is a template artifact. The facts are correct.

## Integration cautions (unchanged)

These cautions, from `../mtg-135-mtg122-integration-20261018.md`, still apply:

- The receiving checkout lacks `iteration_schedule`.
- `ComplexityPlan`'s 1,000-iteration minimum alters the tested horizon.
- The product catalog's checkpoints and transport IDs cannot be interchanged
  with the research catalog's.
- Mini evaluation time scales strongly with model size. That is another reason
  to prefer 64×2.

Choosing 256 streams would be an alternative to MTG-122's accepted 64-stream
direction, not an automatic amendment. This handoff changes no MTG-122
authorization, cohort, budget or control.
