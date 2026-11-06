# c5fba4 open questions and assumptions — 2026-11-05

The previous contents described MTG-243 (naming), inherited from the parent
commit; they remain at `8e5d152ab:scratch/questions.md`.

Material assumptions made headless by the first implement step (reversible
until launch):

- **Larger size = width 256, 4 layers, 8 heads (15× compute parameters).** The
  brief's 290k base count does not match the current v9 count (648k excluding
  the five 4096-bucket tables); the ratio is computed on the current code. A
  larger choice (e.g. width 384, ~34×) would cut the large size's updates on
  USD38; reconsider after a first per-update cost reading.
- **Input v9 with 512 facts** rather than v10's v7/128: v1–v7 now reject layer
  payloads that TLA cards can produce, and v9 carries naming identity.
- **Same learning rate for both sizes.** Larger transformers often want lower
  rates; 1e-4 max is already conservative, and a change would confound scale.
- **Equal wall-clock budget, not equal updates.** Equal-update comparisons are
  the shared milestones; equal compute is the endpoint head-to-head.
- **16 pairs (64 games) per comparison**, twice v10's, for ±~0.09 intervals.
- **cpu32 for evaluation passes**, as in MTG-186 and the learning-signal study;
  1.75 h keeps a pass inside USD2 at USD0.96/hour.
- **600 s per game**, five times MTG-213's limit, sized from one-observation
  latency on this Mac rather than a rented-CPU measurement.
- **No execution in this step.** Launching paid training needs the protocol
  reviewed and frozen at a merged revision.
