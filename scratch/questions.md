# Current revision — 2026-11-05

Jack Heart confirmed ordinary sequential native group → name decisions, each
using the regular policy head and normal rollout; only the final name commits
the effect. Remove active neural compound decoding and its exclusive learning
paths. This supersedes proposals for internal routing or a hybrid neural head.
Read CURRENT ACCEPTED REVISION at the top of mtg-243-naming.md. Continue this same
Task/PR to interactive review; no merge or strength campaign. Compound decisions,
including mana payment, belong to a separate experimental-validation Task.

Material implementation assumptions (2026-11-05, reversible, for review):
- V1–v7 never carried name identity; they see group and name prompts as
  ordinary anonymous modal choices within capacity (previously 344 anonymous
  names). Rejecting them would stop fresh-default v6 games that cast The Rise of
  Sozin. Only v8, whose contract is one whole-vocabulary choice, rejects groups.
- V8 stays as an accepted, explicitly rejecting configuration rather than being
  deleted, because the accepted revision lists v1–v8 as frozen.
- Group identity is its public sorted span; the model embeds it as the exact mean
  of member name vectors. Member programs are not encoded for group prompts.
- The native compound-offer lowering (`Game::compound_offers`/`compound_commands`
  and their Python bindings) stays: it is an engine surface with Rust rule tests
  beside MTG-241's domain and a candidate input for study acc0fe. After the
  neural decoder's removal no Python code calls the bindings.
- Branching is the native constant 64, bound into v9 checkpoints; it is not a
  user configuration.

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

Sync resolution 2026-11-05: main `264041f99` replaced this file with MTG-245's
handoff (available at `264041f991afd3bd451cefa4f50a761577a5eb50:scratch/questions.md`);
it confers no authority on MTG-243. Main deleted docs/ARCHITECTURE.md and
docs/search-and-beliefs.md and moved policy inputs to
docs/observation-and-model.md, so the catalog naming section now lives there.
