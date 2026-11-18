# Frontier Small u1952 on main

Exploratory compatibility check of the original checkpoint from Etude run
`59718fe34945`. Does it still play complete games against the first-64 heuristic
on the recorded TLA lists, and on lists containing cards released into managym
after its training run? Both arms run locally on the mini. The spending cap is USD 3.

## Cohorts

The historical arm uses the original 16 deck pairs in their recorded order,
including the pair with two identical lists. Each pair plays its recorded
deal and a second deal at that seed plus 16. Every deal crosses the two decks
and the two seats in four legs. A draw scores one half. All games must finish
and pass Command replay. The 95% interval resamples whole deck pairs, keeping
both deals and all eight legs together.

The new-card arm takes the first eight complete trophy lists in retained
17Lands source order that managym admits today and whose main deck contains
a definition missing from the writer catalog. The sideboard is admitted
alongside the main deck. Adjacent lists form four pairs and use the same deal
schedule as the first four historical pairs. This measures the whole pilot on
those lists. It cannot isolate the contribution of a new card embedding.

The heuristic changed at `452d7b5357` and `86ad8e9c6a`. A score difference
combines possible drift in rules, policy inputs and opponent play.
