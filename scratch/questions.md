# Assumptions (MTG-347 kickoff, 2026-11-13)

- "Self-play" for the frequency count is read as mirror games per deck. A
  trained checkpoint is used where one loads on the current world; heuristic
  and random mirrors run everywhere. Simpler option taken: do not train a
  policy just to count.
- The Task asks for a small learning comparison; the 2026-11-13 comment hands
  learning-effect claims to MTG-265. Both kept: one exploratory A-vs-automatic
  comparison under USD25, reported without a claim.
- The automatic-spend switch for that comparison is branch scaffolding.
  Whether it lands is for review of the recommended design.
- "Payment outcome as one decision" is the leading hypothesis only; no design
  is chosen before the counts.
- Frequency slice: select the first ten 40-card TLA frontier population
  entries, in file order, with their source IDs and exact main/sideboard lists.
  This keeps the sample bounded and does not rewrite the frozen regime.
- Over-tap attribution is observational, not a prediction of post-fix play;
  the non-over-tapped subset retains the original decision denominator.
