# Questions

Kickoff ran headless. Four choices were made without anyone present; each is
listed with what was passed over in "Decisions made for review" in
`scratch/plan.md`:

1. Hold the PR unmerged for 0.3.0 (like #612) rather than land a shimmed
   version on 0.2.x.
2. Move decision-memory names out of `record_vocabulary()` onto decision
   memory's own surface, rather than keep them there with `token=None`.
3. Keep the name `catalog_tokens()` and pair it with `runtime_tokens()`.
4. Generate legacy `execution/*` and phantom-token retirements rather than
   freeze a literal list.

Open: no stored decision-memory checkpoint was found among the pinned pilots,
so the phantom-token retirement path may have only a constructed test. Is
there a retained memory checkpoint worth pinning?
