# Paper update — MTG-173

## Accepted direction

Jack Heart, October 26, 2026: “lets take another pass at updating the paper based
on latests results”. Preserve the October 21 results-first structure: 2–4 pages
of demonstrated results/failures, then cited recipes for RL-literate readers;
both required distribution subtitles remain exact. No training or evaluation
allocation is part of this update. Earlier implementation/review notes are at
`1860caf8f:scratch/78a9c4-restructure-the-paper.md`.

## Evidence and implementation

Latest inspected main: `558e0186b`, October 26. Main's history had been rewritten
since this branch's original base, producing broad add/add sync conflicts.
The reconciled tree takes current main outside the retained paper rewrite and
this Task's notes. No runtime or frozen-report changes are intended.

New evidence: MTG-181 three-seed ten-deck learning and strengthened-heuristic
addendum; four-seed seat-comparison-v1; MTG-186 specialist head-to-head and
fresh-deal card cohort; heuristic revision/search study; Qwen single functional
game. The three external reports were copied from the shared artifact cache
after checking their manifest SHA-256 hashes. Normal fetch was unavailable in
the stale local environments (`etude` / `etude.run` missing); no auth was changed.
MTG-183 has a protocol and report generator but no final retained report in the
inspected source. Its expected endpoint is not a result.

Delete/replace: stale w9/source pin, baseline-v2/one-seat default, retired Agent
and training/search paths, supervised policy training and standalone learnability
claims. Preserve historical reports, archive files and exact experiment identities.
New methods cover baseline-v3, complete-game both-seat credit, maintained learner
ownership, policy input v2, Etude and the bounded text-selection contract.

Evidence limits: old/new heuristics are different anchors; frozen Elemental
policies are additive/v1, not FiLM/v2; Wilson game intervals do not resample
shared deals or training seeds; matched updates are not matched transitions or
cost. Qwen proof is one small game on the pre-final renderer. No general method,
human-strength, transfer or saturation claim is added.

## Validation and delivery

Verified: warning-free `make -C paper`; eight PDF pages, three results pages
and methods beginning on page 4; all eight rendered pages inspected; six
citation keys, 15 pinned source links, README file links and both subtitles
checked; `git diff --check` and `make -q -C paper` pass. Relative to current main,
the final tree changes only the paper and this Task note. Updated TeX, PDF and
README are available for local review. PDF artifact-marker helper is absent in this repository;
use the existing TeX build and system Poppler for validation. Prior PR #336 was
closed unmerged; this request does not reopen or land it.

## October 26 shipping gate

Gate reviewed `bd032ffd8` against `origin/main` at `558e0186b`; `git ls-remote
origin refs/heads/main` confirmed that base is still current. The six-file diff
contains five paper files and this note only. No paper corrections were needed.
The three retained reports, heuristic study and LLM proof support the quoted
results; the draft preserves the seed/deal, cost, opponent and renderer limits.
Current baseline and damped-learning constants agree with the pinned code.

- `git diff --check origin/main` — passed.
- `make -C paper` and `make -q -C paper` — passed; committed output is current.
- `uv run --no-sync python` (inline gate audit) — passed: six citation keys,
  15 concrete pinned source/report targets (excluding the macro definition),
  README relative paths, both exact subtitles, and all three report SHA-256/size
  entries agree with the manifest.
- `make -C <temporary paper copy>` — clean build passed without warnings;
  `pdftotext -layout` output matches the committed PDF exactly, with eight pages
  and methods starting on page 4. The paper sources and PDF remain unchanged;
  the prior inspection of all eight rendered pages remains the visual evidence.
- `git diff --name-only -z origin/main | bash .github/ci/select-tests.sh` — all
  runtime suites correctly unselected for this documentation-only change.

No checks deferred, no unresolved gate blockers, and no training or scientific
evaluation launched. Return to the primary conversation for `lf land --wait-and-fix`;
publication, landing and Task completion have not been performed by this gate.
