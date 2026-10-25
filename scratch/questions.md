# LLM harness questions and assumptions — 2026-10-24

Current accepted direction, protocol, engineering settings and retained proof
live in `scratch/llm-finetune.md`. This checkout owns MTG-185; MTG-184 owns the
representation. No Wave is associated. Jack Heart requested both Task sessions
in cmux, then authorized harness pursuit after the HTML review. The later
Qwen3.5-9B BF16 selection and provisional context resolve the historical model
choice blocker; measured L4 capacity remains outstanding.

- Corpus/world and an authorized GPU execution environment remain open. No paid
  allocation or full-weight download is selected. The provisional complete-input
  limit is 4,096 tokens, 32 response tokens and 120 seconds including loading;
  optimize from evidence without truncating inputs or repairs.
- MTG-184 must settle current-state-only versus visible history and review real
  prompts before freezing templated English, aliases and card-text deduplication.
  Empty native events and absent policy history do not establish tensor-policy
  input parity. Resolve missing meanings at the shared observation authority.
- Interface-only fitting needs an objective designed from untouched-model failures.
  Random targets teach randomness; expert action targets teach strategy.
- Trying more models, reasoning-output support and quantization remain follow-ups.
  “Sample models” does not select stochastic decoding. Reasoning needs native
  channel/final-answer boundaries and explicit budgets; the current exact-reference
  protocol and single protocol-only repair remain unchanged.

## Historical context

Inherited MTG-174/176/180/182 notes remain at
`912771b9484bddf282bd11183d2f14dfbbbb8d67:scratch/questions.md`,
`912771b9484bddf282bd11183d2f14dfbbbb8d67:scratch/elemental-baseline-preflight.md` and
`912771b9484bddf282bd11183d2f14dfbbbb8d67:scratch/elemental-mana-repair.md`.
They do not select work for MTG-185.

The pre-Graph sync conflicts were resolved preserving both the upstream sim/eval
migration and harness CI/README additions. Earlier implementation and CPU proof
remain the baseline; do not replay them. Inherited card-evaluation plans/evidence
remain at `1e601527094601e5f152d866fbdf9f924c33e954:scratch/17lands-for-bots.md`,
that commit's `scratch/17lands-for-bots-evidence.md` and `scratch/questions.md`.
Their cross-run/Etude direction and outstanding gates belong to that work.

## Publication scope — 2026-10-24

Publish the standalone harness, its configuration, dependency/CI wiring, tests
and documentation. Preserve all scratch edits, additions and inherited deletions
locally rather than including historical working notes in the product PR.
The Task steer permits publication with measured L4 capacity outstanding; that
is still required for Task completion. Cached-model verification of the 5.3.0
runtime is a compatibility check, not replacement or relabeling of the earlier
4.57.6 CPU evidence.

Publication verification: `lf sync` succeeded after a transient index lock.
Ruff lint and format checks passed. The isolated `dev,llm` run passed all 36
harness tests. The affected harness/CI/release/package suite passed 84 tests and
found a retired `manabot/arena/__pycache__` directory left by the upstream move;
the cache was preserved at `.runs/llm/publication-smoke/retired-arena-cache`, and
all six package-boundary tests then passed. The exact main integration command
collected 618 tests without collisions. No Rust source or world changed.

The 5.3.0 CLI compatibility receipt is
`.runs/llm/publication-smoke/decision-transformers-5.3-online.json`:
pinned SmolLM2, `A0\nA1` repaired to `A0`, 88/124 input tokens,
6/3 output tokens, 18.72 seconds, null GPU memory. The offline-only failure
receipt is retained alongside it. Qwen pretrained CUDA/L4 inference, largest
complete prompt/repair capacity and game integration remain unverified.

## Review sync — 2026-10-24

After publication, main advanced to `8842490249484dd7085229bcb922a50d061c61fb`.
`lf sync` merged the code cleanly as `5a6a98136`; automatic scratch restoration
failed because upstream also removed inherited notes and changed questions.
Restored this Task's questions, design and HTML from retained stash
`8b46fceda264589af9a86e2b75e2f643e0005f8e`, without reintroducing already-removed
historical files. Upstream questions remain at that main commit and are copied
to `.runs/llm/sync-20261024/upstream-questions.md`; their unrelated work is not
selected here. The retained stash remains available.

Post-sync verification: `uv run --no-sync pytest tests/llm
tests/semantic/test_elemental_world.py tests/test_package_boundaries.py -q`
passed 59 tests; three supervisor tests exhausted their 2-second deadlines before
worker evidence arrived. The isolated `uv run --no-sync pytest
tests/llm/test_llm_runner.py -q -x` rerun passed all five supervisor tests in
7.54 seconds. No code or timeout was changed to obtain the pass. Record the
initial timing failures rather than claiming a clean combined run.
