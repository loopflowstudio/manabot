---
requires: frozen protocol and retained results including missing outcomes
produces: reconciled analysis and supported claims in the owning report
---
Analyze the experiment against its frozen schedule and criteria.

1. Use the protocol's existing reader and analysis owner. For MTG-245 use
   `experiments/growing_pool_analysis.py`; reconcile both declared seeds and all
   expected game slots, exports, callbacks and replay identities. Missing evidence
   never shrinks the denominator or becomes a zero result.
2. Report whole-deal paired gains, uncertainty and health evidence at the declared
   coordinates. Keep policy-only and search-assisted compute separate and
   distinguish deck strength, rules changes and exposure from learned skill.
3. Report training exposure independently of evaluation draws/actions. An absent
   measure remains absent; neither deck membership nor schedule completeness
   satisfies it. Preserve unresolved acceptance in the evidence review.
4. Separate execution completeness, health, learning, generalization and native
   support. Inconclusive is a valid scientific result. Explain negative findings,
   retained failures, cost and limits without changing the frozen decision rule.

5. Write the report for a reader, as science rather than a process log. Open
   with the questions and a plain answer to each: the effect in points with its
   interval and what it means. Follow with one chart per question, labelled in
   words ("Uniform hands, 16 shuffles"), not condition keys; add a small table
   only where it says something the chart cannot. Put cost in plain units
   (seconds per decision, games per hour) beside strength, then the limits in a
   few sentences. Ledgers, admission checks, run state, hashes, receipts and
   interruptions go in a collapsed appendix at the end; no timestamps or
   execution status above it. Check the rendered page before publishing.

Update the existing report, not a second summary store. Analysis authorizes no
new execution or retrospective parameter changes.
