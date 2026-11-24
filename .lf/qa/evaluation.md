Read when the change touches evaluation, search or simulation (`manabot/eval`, `manabot/sim`).

- Local-update search changes: `uv run --extra dev python
  scripts/bench_local_update.py --reference main --worlds 4 16` checks in
  seconds that receipts still equal main's, field for field; the
  `--rollout-evidence`/`--native-encoder` arms check that full evidence and
  per-row encoding still match the compact, batched defaults. To play
  games, build two players per arm with `make_player` and drive `act`,
  `prepare_step`, `env.step_semantic` and `observe_step` as
  `manabot/sim/native_game.py` does; the default `decision_seconds=1` times
  out full evidence at 16 worlds × depth 20 (about 0.9 s on the
  initialized fixture), so set the budget explicitly for full-evidence arms.
  `_bench_match` and `_initialized_checkpoint` from the bench script give the
  match and checkpoint; with equal player and reset seeds, every evidence and
  encoder arm must pick identical Commands for a whole game (two 93-decision
  games per arm at 4 worlds × depth 20 took about 90 s).
- Card-capture rejections in a retained evaluation (`match_failed: …
  CapturedCards`): the arena key's `evaluation.json` `error` names the
  validator message, and `failures/` stays empty for it. Find every rejected
  match with `grep -l '<message>' */evaluation.json` under the arena output
  directory, copy each key's `games/deals/*.jsonl` (all legs of one deal) into
  `.runs/`, and replay each line through `capture_verified_game(GameTrace(
  data=json.loads(line)))` → `retain_evidence` → `trace_card_tables`; no
  checkpoint inference is needed. MTG-415: 51 MTG-408 trophy matches (204
  games) replayed in about 3 minutes.
- Old checkpoints in a local evaluation: a checkpoint from a TLA
  frontier run (such as u1952 at `.runs/qa/final.pt` or the
  `experiments/best_player_on_main.py` artifact) refuses the CLI's default
  Allies/Lessons setup. Write a `DeckMatchup` JSON (for example
  `new_card_pairs()[0].matchup.model_dump_json()` from that experiment) and
  pass it with `--matchup`; one deal of four games took about 75 s. Read
  `writer_revision` and `stored_programs` from `<out>/request.json`'s
  `candidate`; `evaluation.json` is under `<out>/panel/`. To act as a returning
  user, strip those two fields from a request and repeat it with `--request`.
- Served evaluation without a GPU: `evaluate(request, out_dir=...,
  execution=EvalExecution(device="cpu", deadline_unix=time.time() + 900))`
  runs the same serving child (and its checkpoint ABI checks) as CUDA; omitting
  `execution` plays scalar, which never compares ABI hashes. An experiment's
  `_registration` takes an Etude `Model` whose `checkpoint.key.address` and
  `artifact.uri` are a `file://` URI with `sha256`/`size_bytes`. To compare
  with `main`, `git worktree add --detach /private/tmp/<x> <merge-base>`, copy
  this checkout's `managym/_managym.cpython-313-darwin.so` into it (only when
  the Rust is unchanged), and run the driver with
  `PYTHONPATH=/private/tmp/<x>`; use the resolved `/private` path, or
  `_source()`'s `relative_to(ROOT)` fails. Remove the worktree afterwards.
- Benchmarks on `play_pairing` and the card page: as the author, train
  `uv run etude train manabot/recipes/local.py --experiment NAME --until updates=2`
  (about 8 s) with `ETUDE_HOME` under `.runs/`, then write two files in a
  `.runs/` directory on `PYTHONPATH`: a callback module (`play(match, agents)`
  returning `play_pairing(...)` with `AuthoredMatchup("ur-lessons-vs-gw-allies",
  "ur_lessons", "gw_allies", world=managym.WORLD_SETUP)`), and `NAME.py`
  importing it, whose `eval_jobs` seats each run's last checkpoint against
  `Code(name="heuristic")` in a `Benchmark` with a literal
  `Environment(name="fixture", files="0"*64)` (ignored files cannot be digested). A callback
  defined in the discovered file itself is `<run_path>` and cannot be
  imported. `uv run --extra evaluation etude eval NAME.py --source "$ETUDE_HOME/track"`
  with `ETUDE_OUTPUT_DIR` set to an existing directory played two deals in about 4 s; then run the three
  `manabot.eval.cards` commands in `docs/evaluation.md#card-data`. For games
  stored before a change, put the base's `pairing.py` in that directory under
  another module name. Changing an existing seating's configs raises
  `seating_conflict`; add a new seating instead.
