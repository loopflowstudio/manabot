---
requires: completed runs and their retained results
produces: the experiment's one report, updated
---
Write what the runs showed in `experiments/<name>.md`, the experiment's only
report.

1. Order: question, prediction (findings only), method, results, verdict,
   caveats. List each run with its commit, run ID and cost, and keep the dated
   change lines.
2. Exploratory: say what was learned, the numbers and what it changes next,
   under the word "Exploratory". Its numbers are not cited as findings.
3. Finding: judge the prediction as it was written. Report whole-deal paired
   differences with uncertainty; training seeds are the unit of evidence for a
   method. A missing game stays in the denominator as missing, never as a zero
   and never dropped. Inconclusive is a valid verdict. Name the strongest
   confound and keep refutations.
4. Keep learned skill apart from deck strength, rules changes and exposure,
   and policy-only compute apart from search-assisted compute.

Result files go to the artifact store; the report and manifest go in git. The
report starts no new run.
