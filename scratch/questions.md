# Current revision — 2026-11-05

Jack Heart confirmed ordinary sequential native group → name decisions, each
using the regular policy head and normal rollout; only the final name commits
the effect. Remove active neural compound decoding and its exclusive learning
paths. This supersedes proposals for internal routing or a hybrid neural head.
Read CURRENT ACCEPTED REVISION at the top of mtg-243-naming.md. Continue this same
Task/PR to interactive review; no merge or strength campaign. Compound decisions,
including mana payment, belong to a separate experimental-validation Task.

Material implementation assumptions (2026-11-05, reversible, for review):
- V1–v8 carry no name identity; they see group and name prompts as ordinary
  anonymous modal choices within capacity. Rejecting them would stop
  fresh-default v6 games that cast The Rise of Sozin. (Superseded by the sync
  resolution below: the branch-only whole-vocabulary `[A,2]` v8 was dropped.)
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

Sync blocker found at gate, 2026-11-05: main `423dd2fcd` (MTG-244, merged
22:52 UTC after reviewed head `833ef4120`) took the number **policy input v8**
for layer state (4096 × 3 effect details, fact kinds 11–14, 512 local facts).
This branch also defines v8 (`name_candidates[A,2]`, rejects groups) and v9
(v7 layouts + `name_candidates[A,4]`). PR #424 is CONFLICTING (config/bindings
version ranges, encoder errors, `__init__.pyi`, AGENTS.md, Academy fixtures,
conformance replays, parity digests). Gate did not perform the sync; it belongs
to land's sync step. Assumption for that resolution (reversible until merge):
main's v8 keeps its number and meaning; this branch's naming input becomes the
next explicit version, composing main's v8 layout with `name_candidates[A,4]`;
the branch-only `[A,2]` input was never on main and should not claim v8 (drop it
or renumber it explicitly — never reinterpret main v8 rows). Re-record fixtures
and parity digests from the merged engine; never hand-merge digests. Fresh v6
defaults stay. Re-run debug cargo tests, Engine Budget and integration on the
merged head; earlier green CI does not cover it.

Sync resolution applied 2026-11-05 (merge of `423dd2fcd`): main's v8 keeps its
number and layer meaning. The branch-only `[A,2]` v8 is dropped. V9 is main's
v8 layout (4096 × 3 details, 512 default facts, fact kinds 11–14, layer
admission) plus `name_candidates[A,4]`; the checkpoint binding always records
`public-groups-v1`. V1–v8 see naming prompts anonymously. Academy fixtures,
authored-match/parity receipts and their pins were regenerated from the merged
engine (`generate_authored_match_receipt.py`, both parity `write`s,
`refresh_academy_fixtures.py`). The naming learnability witness now allows up
to eight bounded supervised steps for greedy selection, because v9's v8-based
model initializes differently; its loss-decrease and gradient checks are
unchanged.
Main's `semantic-kernel-w17-v2` corpus was recorded under state hash v15; this
branch's hash contract 16 moved every case's final hash and trace digest.
Re-recorded with `semantic_conformance -- record`; command counts, outcomes,
manifest and Phase rows are unchanged.

# Task b015d5 — training seconds per update (PR #435, merged `45cc961a2`)

History of the measurement and concept reviews: `git show 645b9be21:scratch/questions.md`.
Live items:
- Inherited, unrepaired: `test_resource_limit_receipt_replays_its_retained_encoding_boundary`
  fails because `tests/fixtures/tla-pilot-permanent-capacity.jsonl` retains a schema 7
  frame while native emits `SEMANTIC_DECISION_VERSION` 8 (`binding_mismatch`). The test
  is in no CI selection. It needs its own bug Task; the fixture is retained evidence.
- `HostFacts.cpu_model` reads `/proc/cpuinfo` (null off Linux); the GPU stays in run
  identities, not `TrainingRun.host`.
- Proposed infrastructure Wave memory (not written; `.lf/wave/` is read-only here):
  rented-host CPU class moves update time 1.6-1.8x at a fixed GPU label, so compare
  `host.cpu_model` and native stepping before attributing a slowdown; a packed
  ~50,000-row column makes per-time-step device recursions launch-bound.

## experiment-design b015d5 (2026-11-06): training throughput

Design: `experiments/protocols/training-throughput-b015d5-v1.md` (draft). Reversible
assumptions, for Jack Heart's review:
- Control = 07c490's `elemental_variants.elemental_recipe` (world relabel + capacity
  recovery), so the design waits for PR #436 instead of declaring a second copy of the
  same variant. If #436 stalls, this file can declare the identical two changes under
  its own ID.
- Reuses 07c490's 50 `ELEMENTAL_PAIRS` evaluation deals, so scores read against its
  baseline curve. They stay evaluation-only here; that protocol called them reserved, so
  sharing them is a choice for review.
- Packing is judged without strength runs, because processes share only hardware.
  Spending strength runs on it would buy nothing a digest cannot show.
- Proposed reservation $24.40; the remaining $25.60 of the cap is held back pending an
  amendment.
- An A4000 probe is included as the larger-CPU, cheaper-GPU shape. Etude's `a4000`
  template lists several rented machines that may be ghost records; check
  `etude machines` before launch.

## implement b015d5 (2026-11-06): step-1 probe software

Built: `scripts/bench_training_update.py` `--json` (whole receipt), `--recipe` as a file
or `module:function(regime_id, updates)` declaration, `--cuda-counts`;
`experiments/training_throughput.py` (phase driver, 1 Hz samplers, `/dev/shm` journal,
`--collect` cleanup, `runs()`); `tests/training/test_training_throughput.py`.
Reversible choices:
- PR #436 is still open, so `elemental_variants` is not on main. The probe names the
  control as the declaration string `experiments.elemental_variants:elemental_recipe`;
  it resolves only at a commit containing #436. The CPU test uses the frozen 183 file
  instead so it does not depend on #436. Launch pins a commit that has both.
  Checked: the declaration loads from #436's file (w17, 256 streams, both seats,
  `recover_capacity` true).
- Children drop every `ETUDE_*` variable and track locally; only the driver is the
  launched run. Each child's `TrainingRun` receipt is the evidence.
- A phase is skipped when the previous completed phase's largest child peak x its child
  count exceeds `memory_fraction` (.85) of machine memory. Peak is Linux VmHWM, else the
  largest sampled RSS.
- Any failed phase fails the run after all phases have run; evidence is kept.
- Quote 2026-11-06 at `8d178f40e`: `--rent l4 --machines 2` plans $2.06 (reservation
  $2.10); `--rent a4000` refused, no stock at quote time. Re-quote before launch.

