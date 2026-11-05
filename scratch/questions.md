# MTG-244 decisions — 2026-11-04

Jack Heart approved a successor policy input: "new input version is totally
fine". The implemented version is explicit v8; preserving fresh v6 and frozen
saved inputs is the operator's rollout choice, not a requested default migration.
The earlier blocker is resolved. Unrepresentable older-input states fail at the
shared encoder boundary before inference; no omission, implicit upgrade,
checkpoint reshaping or card exclusion list.

Jack Heart also required measurement and attempted mitigation of performance and
learning risks before review readiness. [mtg-244-risk-validation.md](mtg-244-risk-validation.md)
owns evidence and limits; [mtg-244-layers.md](mtg-244-layers.md) owns implementation,
accepted witnesses and remaining work. No human decision is missing. The code
Graph publishes for review, then stops at its interactive boundary: no merge or
training experiment launch.

Implementation assumptions:
- W17 remains open; this Task does not freeze a world. Its new
  `semantic-kernel-w17-v2` corpus preserves v1 evidence.
- Input admission precedes the explicitly configured native action prefix on
  both encoder paths. Default overflow remains error. Incoming MTG-213 pilot
  decisions do not authorize a pilot launch here.
- Local timing receipts under contention are exploratory. Complete-payment
  allocation/operation attribution is retained. Clean ordinary/dependency and
  full-policy comparisons require the dedicated `layer-risk.yml` producer;
  standard Engine Budget runs only the original TLA step/search pair.
- Historical source archives are measurement inputs, not Task worktrees or
  compatibility paths. The pinned before revision receives instrumentation and
  identical fixtures only. Publication belongs to the later Graph step and
  enables the workflow's first PR-triggered run; adding the workflow does not
  establish acceptance or approve a regression.

Historical context: parent T7 decisions remain at
`4463e57f5a92ebd3b60e2fb50c046b84ac2c0372:scratch/questions.md`; incoming pilot
context at `1546aae3fa7b207b2c5e8ac92a6571af036ddf83:scratch/questions.md`.
The full preceding decision note is blob
`8a6de1149523ed8d99031fcc4d1b4797e6bcc287` (`git show <blob>`).

Latest incoming pilot context is retained at
`ee85d6fc82ed631d7b460e68c73e7ac27f0458ff:scratch/questions.md`.
It records prospective v10 permanent capacity 64 and explicit per-game
capacity recovery at .25 with window/warmup 100/100, preserving executed v9.
These MTG-213 decisions remain historical context for this layers Task;
they do not authorize a pilot launch or change its review boundary.
