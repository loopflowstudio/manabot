---
requires: completed runs and their results
produces: updated `experiments/reports/<name>.md`
action_style: procedural
---
Write what the runs showed in `experiments/reports/<name>.md`, the experiment's only report.

1. Section order: question, method, results, what they support,
   recommendation, caveats. List each run's commit, run ID and cost, and keep
   the dated change lines.
2. **Exploratory:** label the report "Exploratory" and state what was
   learned, the numbers and what changes next. Don't cite exploratory numbers
   as findings.
3. **Finding:** decide after reading the numbers, not from a pre-set
   threshold. Compare against the incumbent (what runs today): whether the
   candidate justifies its cost, where its advantage appears and disappears,
   the noise floor and what the numbers can't settle. State the
   recommendation with its reasoning and what would change it. Report
   whole-deal paired differences with uncertainty, using independent training
   seeds as the unit of replication. Keep missing games in the denominator as
   missing—never zero, never dropped. Inconclusive is a valid result. Name
   the strongest confound and keep refutations.
4. Separate learned skill from deck strength, rules changes and exposure, and
   policy-only compute from search-assisted compute.
5. Write for a reader, not as a process log. Answer each question in plain
   words before any table: the effect in points with its interval and what it
   means. Use one chart or small table per question, labelled in plain words
   ("Uniform hands, 16 shuffles"), not condition keys. Show cost in plain
   units (seconds per decision, games per hour) beside strength. Put ledgers,
   admission checks, run state, hashes and receipts in a closing appendix,
   never the opening. Read the rendered report before finishing.

Put result files in the artifact store and the report and manifest in git.
Start no new run from here.
