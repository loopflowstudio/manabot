---
requires: a question and the current integrated source
produces: the experiment module and the opening of its report
---
Design the smallest experiment that answers the question.

1. Decide whether the result is a finding or exploratory. A finding is a result
   that will be cited as a claim about a method or about playing strength: in
   the README's "What we know", the research program or the paper. Everything
   else is exploratory: calibration, throughput, capacity, pilots, smoke runs
   and any run whose purpose is to make later work possible. When unsure, it is
   exploratory; a finding that grows out of it gets its own run.
2. Exploratory: write `experiments/<name>.py` and one line at the top of
   `experiments/<name>.md` saying what the run is for. It needs no prediction.
3. Finding: open `experiments/<name>.md` with one paragraph holding the
   question, what is measured against what, the strongest confound and a cost
   cap, and commit it before any scored run. Name the control, the matching
   clock (learner decisions, hours or dollars), independent training seeds and
   paired evaluation deals. Say what decision the result will inform and which
   readings would favour each option, without fixing the decision: the
   incumbent is the default, and a candidate justifies its cost after the data
   is in. An expectation may be written down as an expectation; never a
   threshold that decides the verdict before the result exists.
4. An experiment is its Python module at a commit: `runs()`, `eval(state)` and
   `report()`, with recipes from `manabot.recipes.builders`, launched through
   Etude. Existing checkpoints need no placeholder training. Do not write a
   protocol document, a serialized regime, a digest check or a test suite for
   one experiment's schedule. Machinery a second experiment would reuse belongs
   in `manabot/` or Etude, tested once there.
5. When the experiment needs software beyond its module, deliver that as an
   ordinary PR from this checkout, then continue here.

Design has no separate review. Ask in the conversation about a judgment that
changes what the result would mean; otherwise choose, and record the choice in
the report as an assumption.
