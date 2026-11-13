---
requires: completed runs and their retained results
produces: the experiment's one report, updated
---
Write what the runs showed in `experiments/<name>.md`, the experiment's only
report.

1. Order: question, method, results, what they support, recommendation,
   caveats. List each run with its commit, run ID and cost, and keep the dated
   change lines.
2. Exploratory: say what was learned, the numbers and what it changes next,
   under the word "Exploratory". Its numbers are not cited as findings.
3. Finding: decide after reading, not before. Compare the result against the
   incumbent, what we run today, and say whether the candidate justifies its
   cost, where its advantage appears and disappears, what the noise floor is
   and what the numbers cannot settle. Give the recommendation with its
   reasoning as a recommendation, and name what would change it; do not apply
   a threshold chosen before the run as the verdict. Report whole-deal paired
   differences with uncertainty; training seeds are the unit of evidence for a
   method. A missing game stays in the denominator as missing, never as a zero
   and never dropped. Inconclusive is a valid result. Name the strongest
   confound and keep refutations.
4. Keep learned skill apart from deck strength, rules changes and exposure,
   and policy-only compute apart from search-assisted compute.
5. Write for a reader, as science rather than a process log. Answer each
   question in plain words before any table: the effect in points with its
   interval and what it means. Use one chart or small table per question,
   labelled in words ("Uniform hands, 16 shuffles"), not condition keys. Give
   cost in plain units (seconds per decision, games per hour) beside strength.
   Ledgers, admission checks, run state, hashes and receipts go in a closing
   appendix, never in the opening. Read the rendered result before handing it
   over.

Result files go to the artifact store; the report and manifest go in git. The
report starts no new run.
