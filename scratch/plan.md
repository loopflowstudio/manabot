# MTG-328 plan

Docs written for the intended behavior (read these before implementing):
- docs/observation-and-model.md — "Bucketed padding" and "Live-token attention"
- docs/training.md — Run training (`--padded-shapes`), Stage operations (`auto`)
- docs/evaluation.md — Shared inference, "Forward layout"
  (docs pass 2: training-only default wording, `EvalAllocation` example,
  fixed `#live-token-attention` anchor; no behavior change described)

Built (implement pass 1):
- `Execution.bucketed_shapes`/`live_tokens`: `bool | "auto"` (default auto),
  off omitted; `live_tokens=True` off CUDA fails; `resolved_*` properties.
- `TrainingRegime`: saved stages omitting them read False; `resolve_forward_layout`
  (on for CUDA train_self_play) before `references`;
  `authored_dump` restores auto; builders/experiments/study rebuild with it.
- CLI/recipes: `--padded-shapes`; `--bucketed-shapes` warns (removed 0.3.0).
- `EvalAllocation.bucketed_shapes`/`live_tokens`: opt-in bools (default off),
  outside `EvalCompute`, `live_tokens=True` refused off CUDA; serving loads
  with them. Revised after the L4 measurement below (was `auto` on for CUDA).
  Search players and range likelihood load padded, as on main. Merged main
  after #495 and #498 (pooled stream removed).

Width rule: `auto` is on for CUDA `train_self_play` at every width, with no
threshold. The brief asked to measure below the paying width first; the
MTG-319 table (experiments/live_token_attention.md) already covers w160l8:
both on is 2.3 s learner + 2.9 s collection per pass, against 2.9 + 2.4 s for
live tokens over full tables and 7.8 + 9.2 s padded, so both-on is fastest in
total at both measured widths. The 0.5 s collection cost at w160l8 is
documented in docs/observation-and-model.md as the known small-width cost.

CPU live-token cost (for the PR body; `experiments/eval_forward_layout.py cpu`,
MTG-319's retained w160l8 inputs, 4 threads on a 16-CPU laptop, median of 3
timed passes after 1 warm-up, `experiments/data/eval_forward_layout/cpu.json`):
- Collection forward (244-256 rows): padded 459 ms, live tokens 855 ms (1.9x
  slower), live over buckets 793 ms (1.7x slower); single calls reached 3.5 s.
- Learner forward+backward (23-84 rows): padded 2,252 ms, live 954 ms, live
  over buckets 1,249 ms. The small learner minibatches are not slower on CPU.
- Passes vary by up to 2x on this host, so read it as "collection roughly
  doubles", not a precise ratio. The refusal off CUDA stands.

L4 evaluation pass (experiments/eval_forward_layout.md, run 2e3990779ba4,
USD 0.11): w384l8, 128 games, ~9 rows per served forward. Wall 118 s padded and
118 s with both on; forward 23.7 ms padded against 27.3 ms with both on (15%
slower); identical decisions and score. This contradicts outcome 3's premise
that evaluation speeds up, so the evaluation default is padded and the
switches are accepted explicitly (questions.md 2, 8).

Remaining:
- MTG-324 has not landed (still iterating on its learning A/B; main has no
  `precision="auto"`). When it lands, merge main and combine its ladder with
  `resolve_forward_layout`/`authored_dump`/`_read_saved_stage` (questions.md 5).
- MTG-325 (moving) is moving the torch forward to `manabot/policy` +
  `manabot/backends/torch` and ActorCritic over `prepare.Prepared`. Its merge
  will touch where `bucketed_shapes`/`live_tokens` enter the forward; keep one
  path (constraint: "switches reach the backend forward through one path").

## Delete — do not maintain

Removed `Execution._from_auto` and its custom `__eq__`: Pydantic's
`model_fields_set` distinguishes resolved defaults from explicit writes,
including edits after resolution. `authored_dump` restores only those defaults.
Folded `_read_saved_layout` and `_read_saved_self_play` into `_read_saved_stage`;
one reader preserves saved layout, seat and closure defaults. No pending cuts.

Check: `uv run --extra dev --extra artifacts pytest tests/training/test_bucketed_execution.py tests/training/test_builders.py -q` — 23 passed.
Check: `uv run --extra dev --extra artifacts --extra notebook --extra reports pytest tests/training/test_regimes.py -q` — 18 passed; focused Ruff lint/format and `git diff --check` pass.
