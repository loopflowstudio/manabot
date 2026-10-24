---
requires: MTG-177 brief and current rules engine
produces: whole-rules audit report, complete rule index, bounded draft changes and CI cost evidence
---
Deliver MTG-177's complete rules audit and concrete draft improvements. Read the
fresh Task brief and repository instructions; its report is outside the roughly
10,000-line combined draft code/test/tooling/documentation budget.

1. Pin the official Wizards Comprehensive Rules source, effective date and
   digest, and the audited code revision. Inventory every numbered rule/subrule.
   Inspect existing rules conformance/index tools before inventing a new one.
2. Audit each inventory entry against actual code and behavioral test evidence.
   Record implementation status, scope/applicability, exact ownership/dispersion,
   appropriate comments and test/gap references. Use unknown for an unresolved
   finding, never infer correctness just from matching names. Ensure no numbered
   entries are silently missing. Keep unsupported mechanics distinct from bugs.
3. Draft high-value deterministic coverage and important rule/invariant comments
   within the separate approximate 10k-line change budget. Use tiny fixtures and
   bounded adversarial combinations. Avoid broad production redesigns and report
   larger proposals. Coordinate with MTG-174 and optional-payment Task 4ba5dd;
   do not duplicate their payment-model rewrite. Jack accepted one payment model
   shared by offers and execution. The known Prism/Treasure failure is a useful
   review case, not proof of the GPU failure's exact root.
4. Inspect CI and measured suite durations. Assess sharding, shared builds,
   caching, test tiers and required-check aggregation. Report total runner-minutes
   as well as wall latency. Make reversible draft CI changes only where evidence
   supports them. No intentional stress/load testing on main, long flakiness
   loops, paid compute, branch-protection edits or weakening existing coverage.
5. Run focused debug Rust checks and uv-managed Python checks appropriate to the
   changes. Establish explicit local time/node limits; retain failures and gaps.
   Do not run all expensive suites repeatedly. Record exact check selection and
   measured incremental cost. New tests need independent expected behavior,
   rather than merely checking that two callers use the same implementation.
6. Deliver a readable prioritized report and machine-readable complete rule
   index, separate draft-line accounting, concrete committed draft changes and
   validation/CI cost evidence. Keep the report useful for deciding future rules
   investments. Retain all work in this checkout. Report what remains unknown.
   Do not merge, arm auto-merge or publish a ready PR; this is draft delivery.

Continue through the report AND concrete draft changes; a plan or partial index
alone does not fulfill the request. Stop only on completed draft delivery or a
specific external blocker, without manufacturing unsupported audit claims.
