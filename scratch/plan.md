# MTG-318 plan

Done: Etude run `1a2e6cb86345` (~$0.27, 2026-11-12) at revision `8b753ade3`;
report `experiments/collection_tail_graphs.md` (no path accepted, so no
`Execution` switch and no code change); data published, manifest committed.
The report records every assumption earlier kept in `scratch/questions.md`
(decision-rule thresholds, host, re-captured inputs, compile flags).

`experiments/collection_tail_graphs.py` stays byte-identical to `8b753ade3`:
the report cites that revision as the freeze. It defines no train/eval/report,
so Etude does not discover it and `tests/eval/test_experiment_files.py` needs
no entry (its exact-set discovery test passes).

Remaining: delete `scratch/` in the pre-land commit, publish the PR.

Concept review (2026-11-12): the change adds no product concept, API or
configuration; the report answers the Task's three questions (graphs,
compilation, coalescing) and the recording gap under its pre-registered rule.
The pre-registered "captured" and "coalesced" arms appear as the trunk graph
(whole-forward capture failed at `buckets.py:307`) and as the calls == steps
check; the report says why, so the frozen text stays. Nothing to change.
