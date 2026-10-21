# Paper restructure

## Accepted outcome

Jack Heart's October 21, 2026 Task direction: open with 2–4 pages of demonstrated results and failures, then methods as cited recipes for an RL-literate reader. Both `main.tex` and `drafts/01-refuted-ledger.tex` subtitles must read exactly “Written by AI. Not yet intended for distribution.” Build with the existing paper Makefile and describe the structure in the README. The paper remains an internal draft.

## Reconciled approach and implementation

The active paper now opens with three pages of selected results, including failed acceptance criteria and the reset-seeding correction; methods begin on page 4. Results retain their original worlds, recipes, seeds, opponents, uncertainty and report provenance. Historical measurements do not certify w9, FiLM, human strength or a general advantage over PPO. Methods name code owners and cited influences, with local adaptations distinguished from exact reproduction. Report and correction-ledger links share source pin `a04dcdc2`.

The generic difficulty introduction, stale platform census, challenge tables and claim ledger have been replaced. Unused table/math packages and the unused results label are gone. The README is a reading/build/source map. The first archive changes only its requested subtitle; frozen reports remain untouched.

The October 21 upstream change `f15196e1` independently introduced a current report, retained the July source/PDF as `drafts/02-platform-and-challenges`, and renamed the active output to `manabot.pdf`. The integrated rewrite retains that archive, output name and the union of bibliography entries. There are no intervening changes to runtime code, experiment reports or methods documentation between source pin `a04dcdc2` and this upstream base; the pinned implementation descriptions remain applicable.

Prior build evidence records seven total pages, exact subtitles, three results pages, no final layout/citation warnings, resolved draft citation keys and report paths, a successful isolated first-archive compilation and an unchanged July archive. Focused reconciliation confirmed the baseline-v2/FiLM description against its code and the move-learning recipe against its existing fidelity guide. No implementation mismatch requiring repair was found.

## Remaining work and evidence limits

No implementation work or product decision remains identified. Visual judgment remains with review; the headless checks do not establish visual quality or strategic strength. Publication and Task completion remain separate delivery actions.

The associated Wave is `intelligence`; its local objective, memory and child directories are absent from this checkout. The Task's Project context retains separate goals for repeatable training, full-game improvement and human comparison, none certified by this paper. Local Wave memory could not be reconciled; historical chapter files are not substitutes for current memory.

Check: `make -q -C paper` and `git diff --check HEAD` passed; active PDF is current and the full diff has no whitespace errors. Prior build/page/citation evidence retained above; visual review remains with review.

## Concept decision

The October 21 concept review leaves Jack Heart's accepted experience and model unchanged. An RL-literate reader opens the active PDF, reads bounded findings and failures, follows a source-pinned report to audit a claim, then reads the current recipes and their influences. A result, a current recipe and a historical draft remain distinct records: world, seed, opponent and cost identities belong to the result, not to whichever implementation is newest.

Failure and recovery preserve that contract: a reader encountering the July seat claims can follow the active paper's reset-seeding explanation to F1–F3, retain the archived claim as history and recover the corrected interpretation. A counterexample would be carrying an earlier result forward as evidence for w9/FiLM, or silently replacing the report after correction; the active text explicitly rules out both. The README already expresses the reading/build interaction, so no proposed usage or skill changes are needed.

Across layers, TeX owns the narrative, retained reports and correction records own measured evidence, source pins identify the described implementation, and the existing Makefile derives the active PDF. No runtime types, APIs, experiment persistence or execution lifecycle need to change for this outcome. No unresolved product choice or next implementation direction was found. Next proof remains review of the built paper's readability and layout; headless build checks do not establish those qualities.
