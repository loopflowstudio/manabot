# MTG-329 implementation

Move execution management to Etude; keep Manabot's learning loop, recipes,
world-bound policies and game evidence. No replacement executor, shim or SDK
stub. Pydantic remains the required config protocol (config-api-direction.md).
The accepted design is design-train-api.md; its proposed commands, namespaces
and implementation status are historical. Current commands use `etude train`,
the live Run is in `etude.train.tracking`, and pyproject.toml/uv.lock use
released Etude 0.4.0. EXP-56 landed; no development override remains.

## Implemented

- `training/loops/self_play.py:self_play` registers raw/Adam/EMA/counts with
  Etude, steps, logs, saves and publishes immutable world-bound policies.
  Fixed schedules, checkpoint_source, paired deals, game-error/replay evidence
  and numerical incidents remain. `loops/minimal.py` is the small example.
- Deleted execution.py, recovery/admission/learning_state, job/runner/study,
  evaluation queue/monitor worker, TrainingRecipe/RunContext, Tracker/LoopState,
  local_launch and exclusive tests. Record/report readers retain frozen bytes.
  Retired regimes name f4e1822faf46bbc9cdc43909293f8cbc8de4a300; old state.pt
  folders name f577362135562e569033fe33e48c7ef0411a829d. No old-folder loader.
- Backend capture takes SelfPlay + Run. Frozen MLX/MPS capture commands refuse
  at the revision boundary; saved-data analysis survives. fetch_artifact owns
  cache validation; dashboard locking has no recovery mode.
- Synced main's log-only packed actions, native chooser policies, CUDA/MPS
  BF16 admission and Academy packaging checks. Release smoke retains Etude
  training and now checks installed Academy bot discovery.

## Remaining acceptance and validation

1. Gate first rebuilds the native extension against synced main, then runs the
   affected checks from docs/development.md. The sync check recorded 9 passed,
   3 failed: all failures reached the installed pre-sync VectorEnv rejecting
   `player_policies` from `rollout/collection.py:open_collector`. This explains
   the observed boundary failure; it does not prove collection works after a
   rebuild. Earlier checks in implementation.md predate this merge. Include
   installed CPython 3.12/3.13 release smoke for the new packaging assertions.
2. Etude must deliver local Job.inputs/environment. Rechecked installed 0.4.0
   `etude/local.py:_prepare`: neither is carried. The real policy→sampler probe
   failed `input not given: 'start'`; `test_recipe_etude.py` manually supplies
   that input and proves provenance, not CLI carriage. After the upstream fix,
   prove the actual two-command handoff in docs/training.md. No Manabot fallback.
3. Demo reviews recovery losses and the belief dependency contract: explicit
   completed CheckpointRecord handoff is implemented; automatic same-seed
   policy→sampler scheduling is still unmet, not silently dropped. Choose
   whether the explicit workflow is sufficient or automatic scheduling remains
   required; that needs product judgment and Etude support.
4. Launched smoke/retry/restore and changed-config `continue` remain unproved.
   Use a bounded launch under the approved cost rule; no paid probe has run.
   Earlier local SIGINT/SIGTERM resume and raw/EMA bitwise parity are recorded
   in implementation.md, not evidence for launched recovery or exact resume.
   Games in flight, random streams and in-run source repair are not recovered;
   review these tradeoffs before treating acceptance as complete.

Frozen replay schema 7 versus runtime 8 remains contrary evidence: gate must
report it without regenerating frozen bytes. This pass changes the plan only;
publishing, landing and Task movement belong to the caller.

## 2026-11-14: experiment-file API (`training_jobs` / `eval_jobs`)

Etude resolves from commit 9e8fec1268ed (`[tool.uv.sources]`); requirement
strings stay `etude-sdk>=0.4.0`, and the published `manabot` package imports
only names released 0.4.0 has.

- Ported: `experiments/elemental.py` (job `elemental`), `seat_comparison.py`
  (`seat-comparison-one`/`-both`), `tla_frontier.py` (one job per size),
  `batched_baseline.py`, and new `experiments/belief.py` (`policy`, then
  `sampler` from stored state). `elemental_long_eval.py` became
  `elemental_play.py`, holding only what `elemental` plays with.
- Deleted: the MTG-183 follower/rent/db/report, MTG-186, curriculum readers,
  inference baselines and the LLM campaign drivers, with their tests and
  fixtures. `manabot/llm/`, its tests and setup scripts stay. Last commit with
  the deleted code: b49df79b49.
- `experiments/llm_claude.py` and `llm_qwen.py` are eval-only declarations over
  `manabot/verify/llm_match.py`, checked in not runnable (Etude EXP-70).
- Blocked on Etude: `etude train experiments/belief.py --job sampler` fails
  `input not given: 'start'` (`etude/local.py:_prepare` carries no job inputs);
  an `EvalJob` cannot choose its command or uv extras, so the frontier pass
  sets `UV_NO_SYNC=1` through `environment`, unproved on a rental.

## 2026-11-14: repin to Etude main 6090a771a43f

`[tool.uv.sources]` pins Etude main 6090a771a43f (#113, #115, #116); no Etude
release carries these names, so requirement strings stay `etude-sdk>=0.4.0`
and a Manabot release must wait for the Etude release that does
(`checkpoint_id` is gone on main; `CheckpointRef(address=...)` is new).

Applied: `read_job_file` -> `read_training_jobs`; `CheckpointRef(key=
checkpoint_id(...))` -> `address=` (legacy manabot run ids keep their exact
`RUN/checkpoints/NAME` key, so retained matches keep their identity; the 0.2.5
migration test still resumes playing nothing); frontier and Claude evals
declare `extras=` instead of `UV_NO_SYNC=1` (Qwen keeps it: its setup places a
prebuilt engine without the workspace); `launcher-20.db` in docs; the frontier's
launcher check skips runs the pass has only queued.

Gaps re-tried (temporary `ETUDE_HOME`):
1. `etude train experiments/belief.py --job policy` then `--job sampler`:
   fixed; sampler trained from `start`. #116 not adopted: Etude's docs keep
   checkpoints on `start=`, outs are for other stored files.
2. `etude train experiments/belief.py` twice: fixed (pass 1 policy, pass 2
   reports it finished and trains the sampler, pass 3 trains nothing).
3. round-robin `enter=`: fixed (synthetic eval-only probe played 1 match, a
   second pass printed `nothing to play`). No kept file uses a round robin.
4. outside checkpoint: fixed; `launch llm_claude.py --quote` and
   `llm_qwen.py --quote` (with `--extra llm`) plan 100 matches. MTG-183's final
   is a manabot run, not an Etude policy, so it stays an outside address.
5. eval-only smoke: fixed (probe smoke passed eval and report).
6. EvalJob extras: fixed on main; frontier and Claude use `extras=`.
7. docstrings: Etude-side.

Remaining: LLM files stay not runnable (EXP-70: secret, API dollar limit,
job-lifetime server, per-game limit; MTG-360). Qwen's setup installs no
`evaluation` extra (duckdb), unproved. Quotes read an empty temporary store, not
r2; the frontier's stored runs live in `launcher-19.db`, which launcher-20 does
not read, so its eval pass against r2 is unproved. AGENTS.md still names
`etude.declarations.read_job_file` (needs a person's edit). No smoke of a
training file (each prepares the engine build).
