---
requires: an updated experiment report and its run results
produces: a recorded conclusion, or the specific missing portion to run
action_style: conversational
---
Review with a person what the experiment showed and whether it is done.

1. Read `experiments/reports/<name>.md` from the top as a reader. It must
   answer its questions in plain words before tables, with provenance in a
   closing appendix; a report opening with counts, ledgers or execution
   status isn't done.
2. Check that every claim matches the numbers, exploratory numbers aren't
   presented as findings, a finding answers the question committed before its
   runs, and its recommendation reads the result against the incumbent rather
   than a pre-set bar. A merged PR, a complete schedule or a tiny update
   doesn't show learning. An inconclusive result can finish an honest
   experiment.
3. Record what the person decided in their own words rather than turning a
   casual reply into approval. When a finding changes what we know, update
   the README and research program in the same delivery. If a needed judgment
   is unavailable, record it in the report and stop.

When deciding whether to continue:

- **Advance** when the question is answered and the report is ready.
- **Iterate** only for a specific missing or fixable portion of the same
  question within already-approved spending; a different question is a new
  experiment.
