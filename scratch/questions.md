# MTG-242 decisions — 2026-11-04

Jack Heart authorized one engine change and a reviewable PR, stopping at review;
no merge or training experiment. The accepted scope, evidence and remaining
integration work live in [mtg-242-mana.md](mtg-242-mana.md).

Material assumption: the Task directive is the implementation plan; no separate
accepted design was supplied. Main integration preserves this scope. Inherited
T7 and MTG-213 notes are historical context, not training authorization:
`4463e57f5a92ebd3b60e2fb50c046b84ac2c0372:scratch/questions.md` and
`8e8d702f807d61675fc1bca46417eee69316daa0:scratch/questions.md`.

## Compatibility and dependency boundaries

Frozen inputs keep their meanings; fresh construction stays v6. V9 is an
explicit model successor that joins mana classes over byte-identical v8 wire
rows, so native buffer binding needs no new width discriminator. Saved v8 keeps
its original model. This does not settle MTG-244's conflicting schema; its
published PR #427 at `6165e0fe48e7c4d500e724b030b6dc11fce749f6` is a stable
handoff for reconciliation, not combined acceptance.

MTG-241 remains a live writer with no superseding integration handoff for its
no-live-copy request. Its retained batches and suspended-return witness exist
at `d83376181`; integration must adapt shared completion to post-cost variable
output, selected taps and the original parent. The synchronous witness here
at `68a2094dc` is distinct. No replacement frame protocol or missing product
judgment is inferred from that dependency.

## Measurement assumptions

Jack Heart authorized bounded performance and model probes before dependency
integration, including comparison against main and mitigation of attributable
regressions. Read-only `git archive` snapshots with separate Cargo targets are
measurement inputs, not managed worktrees or competing Tasks. Exact revisions,
workloads, digests, timing distributions and contrary evidence remain in the
plan and `research/mana-risk-2026-11/`. Initial 32-game and later 128-game
comparisons do not replace dedicated Engine Budget acceptance. Unsupported
baseline mechanics have absolute costs only; no standing reference was adopted.

Jack Heart clarified that no net speedup is required. Keep the existing 10%
guardrail and disclose tails; noisy samples cannot establish a pass. Model
forward/backward probes establish capacity and gradient paths, not RL learning
or playing strength. Completion and schema integration remain acceptance work.
Earlier detailed assumptions are retained at
`13c9ae522e9e7304fb0db738c7efe729c36ae5e3:scratch/questions.md`.

## Main sync context — 2026-11-04

Incoming MTG-213 capacity-recovery and v10 execution notes remain historical
context at `51b4bd274c34398dc0bf0066dfc3933f26911f87:scratch/questions.md`.
They do not authorize training or change MTG-242’s review boundary. The sync
retains main’s permanent-capacity regression tests alongside this branch’s
v8 native/Python buffer parity coverage.
