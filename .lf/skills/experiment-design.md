---
requires: a question and the current integrated source
produces: the experiment module (`experiments/<name>.py`) and the opening of its report
action_style: exploratory
---
Design the smallest experiment that answers the question.

1. Classify the run as a **finding** or **exploratory**. A finding will be
   cited as a claim about a method or playing strength (in the README's "What
   we know", the research program or the paper). Everything else—calibration,
   throughput, capacity, pilots, smoke runs, prep work—is exploratory. When
   unsure, treat it as exploratory; a finding that grows out of it gets its
   own run.
2. **Exploratory:** write `experiments/<name>.py` and one line at the top of
   `experiments/reports/<name>.md` saying what the run is for. No prediction
   needed.
3. **Finding:** open `experiments/reports/<name>.md` with one paragraph—the
   question, what is measured against what, the strongest confound and a cost
   cap—and commit it before any scored run. Name the control, the matching
   clock (learner decisions, hours or dollars), independent training seeds
   and paired evaluation deals. Say what decision the result informs and
   which readings favor each option without pre-committing the verdict: the
   incumbent is the default, and a candidate justifies its cost once the data
   is in. Write an expectation as an expectation, never as a threshold that
   decides the verdict beforehand.
4. Keep the experiment to its Python module at a commit: `training_jobs()`,
   `eval_jobs(state)` and `report()`, using recipes from
   `manabot.recipes.builders` launched through Etude. Existing checkpoints
   need no placeholder training. Write no protocol doc, serialized regime,
   digest check or test suite for one experiment's schedule; reusable
   machinery belongs in `manabot/` or Etude, tested once there.
5. If the experiment needs software beyond its module, land that as an
   ordinary PR from this checkout first.

Ask about a judgment that changes what the result would mean; otherwise
choose and record the choice in the report as an assumption.
