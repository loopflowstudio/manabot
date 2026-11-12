# MTG-306 status

Done 2026-11-12: fill (laptop) and L4 timing (Etude run a1bcb3bf17a2, ~$0.16, machine ended).
Results and conclusion are in experiments/token_stream.md; data published under
experiments/data/token_stream/ (manifest committed).

Learning stage not run: the decision rule's speed arm (>=1.5x per update) and fill arm both
fail on measured numbers (1.07x end to end at w64), so strength cannot change the outcome;
the full comparison would cost ~$20 against the $15 cap. Recommendation: MTG-302 bucketing.
On-demand `mtg306-l4` exists in the launcher DB (max 1); unused now.
