# MTG-207 assumptions — 2026-10-27

Inherited MTG-206 notes are at `4ab16a140:scratch/questions.md`.

- **Design chosen without review.** The Task left design open; headless, the
  simplest exact path was taken: batch `estimate` by deal count and keep the
  page one self-contained file computing every combination. Per-pair payloads
  were not adopted: without a server, every combination must still be
  computed when the page is written, so splitting saves browser memory, not
  render time.
- **254 MB page.** The whole-experiment JSON is 254.2 MB at base and after
  (unchanged bytes of values). Whether a browser should load that is a page
  design question left for review; values were not rounded, since the
  acceptance compares them exactly.
- **etude-sdk 0.2.11** (etude #72) merged 2026-10-28 02:09Z but was not on
  PyPI or GitHub releases at 03:24Z (still 0.2.10 at 03:40Z), so the pin stays 0.2.10 (confirmed by Jack Heart's operator steer 2026-10-28). Projection with
  Etude main's `estimate` swapped in is recorded in `scratch/mtg207.md`.
- **Machine load.** Load average 90–240 on 16 cores throughout; wall times
  are inflated several-fold, CPU less so.
