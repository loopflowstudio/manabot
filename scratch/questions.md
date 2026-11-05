# MTG-243 assumptions — 2026-11-04

Accepted decisions, evidence and remaining work live in
[mtg-243-naming.md](mtg-243-naming.md). Jack Heart's speed direction continues the
same Task/PR against `b31998cc3`, ending at interactive review. Merge, auto-merge,
paid execution and strength experiments remain unauthorized.

Material implementation choice: keep the existing GRU, weights and serialized
buffers; pack integer tokens before embedding and execute at most 64 recurrent
steps per call. Carry hidden-state gradients across every boundary. This bounds
packed backward slice copies without changing model capacity or requiring a
checkpoint migration. It does not establish GPU throughput or training strength.

Native profiles overlap active MTG-241/242/244 ownership. Preserve those domains
and the unchanged budget reference; finish the highest-payoff measured learner
optimization here. Native measurements are unchanged-code controls. External
scheduling noise stays in raw evidence, separately from process CPU cost.

Additional production names, fresh v8 defaults and broad training remain separate
decisions. Historical detail is at `b31998cc3:scratch/mtg-243-naming.md` and the
pre-curation uncommitted plan in `.lf/logs/naming-speed/plan-before-curation.md`.

Sync resolution: main's MTG-213 pilot assumptions remain available at
`e113602a2dc1c866af513ff0953598db390db9f2:scratch/questions.md` and its delivered
results in `experiments/tla-trophy-pilot.md`. They describe a separate Task and
supply no execution or delivery authority for MTG-243. Keep this file scoped to
the naming branch; preserve the incoming pilot report and artifact manifest.
