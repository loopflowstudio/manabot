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
- Editable Etude: a plain `uv run` re-syncs the venv to the locked
  `etude-sdk` and silently drops an editable checkout. Reinstall with
  `uv pip install -e <etude checkout>` and run everything with
  `uv run --no-sync` (or `.venv/bin/python`). For pytest, set `UV_NO_SYNC=1`
  so the CLI tests' own `uv run` subprocesses keep the editable SDK too.
- Vector trophy evaluation (`play_pairings`): drive
  `experiments/eval_throughput.py` as in its report's Reproduce section
  with `--deals 4` (16 games): CPU scalar about 55 s, MPS/MLX vector-64
  about 10–12 s on the M4 Max, `--slots 3` exercises backfill. Compare arms
  leg for leg by reading every `arena/*/games/deals/*.jsonl` record's
  `games[].actions`, `digests` and `end` keyed by
  `context.evaluation.(deal_seed, leg)`; float32 arms must be identical,
  bf16 may differ. The CLI's Match differs from a managed one, so player
  seeds (and games) differ between the CLI and a managed run of a deal.
- Managed `play_many` locally: `etude.eval.tournament.evaluating(
  Environment(...), experiment=..., matches=[Match(..., config=
  held_out_deal(i).model_dump(), games=4)], directory=out / "track")` then
  `run.play_many(experiments.tla_trophies.play_many, agents=agents_for(p))`
  with `ETUDE_OUTPUT_DIR` set; the script needs an `if __name__ ==
  "__main__"` guard because Etude spawns its game process. Two deals take
  about 10 s on MPS. Explicit-match runs never resume: rerunning one
  replays everything under a new name. A background job started with `&`
  ignores SIGINT; to simulate Ctrl-C restore
  `signal.signal(signal.SIGINT, signal.default_int_handler)` first.
- Returning experiments (changes to `experiments/tla_trophies*.py`,
  `pairing.py` or anything an `Environment` hashes): check that each stored
  experiment still plans without conflict and without replaying, without
  playing a game. Replace `etude.controller.advance._play_here` with a stub
  that prints `len(spec.matches)` and returns `SimpleNamespace(played=0)`
  per spec, then call `advance(read_experiment(Path(file),
  experiment=name), source=storage_from_uri("r2://manabot/jobs/<name>"))`
  under `doppler run -p manabot -c prd`. Each pass takes seconds. A healthy
  pass prints "nothing to play" for every stored tournament.
  `tournament_conflict` means an environment identity moved (compare
  `stored_runs(...).tournaments[name].tournament.environment` with the
  declaration's). Check `tla_trophies_floor`, `tla_frontier_snapshots` and
  `experiments/tla_trophies_colab.py` under its stored experiment name
  `tla_trophies_colab_500k_v2` (about 129 tournaments, all "nothing to
  play"); the file's own name has no stored runs and plans 64 new matches. `tla_trophies` already conflicts on `main`.
- To verify live resume, run a named `Benchmark` with one
  `CheckpointSlot` seating through `uv run --no-sync etude eval FILE --source DIR`.
  Add a distinct checkpoint to the same declaration and rerun. Expect one new
  played deal with the new checkpoint, the old result block's checksum to
  match and all four new legs to replay. A third pass reports
  `nothing to play`. Use a temporary track directory and retain each run's
  arena output separately.
- Precision profiling (`experiments/eval_precision_profile.py`) without a
  GPU: a fresh checkout needs `uv venv --python 3.13`, then `uv sync --extra
  evaluation --extra artifacts --extra dev --no-install-package managym` and a
  `managym.pth` holding the checkout path in `.venv/lib/python3.13/site-packages`
  (the in-tree `.so` serves when the Rust is unchanged since it was built).
  Newer Etude checkouts drop `etude.eval.game.Matches` and fail the import, so
  install the report's pinned revision from an archive, leaving the Etude
  checkout alone. Create `.runs/etude-d876a7b1`, extract into it with
  `git -C <etude> archive d876a7b1 | tar -x -C .runs/etude-d876a7b1`, then run
  `uv pip install --no-deps -e .runs/etude-d876a7b1`. `etude fetch` and
  `manabot.data fetch` need `doppler run -p manabot -c prd --`. On CPU a
  one-deal `capture` took about 20 s (315 calls). Replay supports `float32`
  on every device, both bf16 variants on CUDA and MPS, and `tf32` on CUDA.
  Rendering `.runs/eval-precision-profile/l4-a468879a7a` with `render_report`
  must reproduce that directory's `l4-tables.md` byte for byte. A refused
  replay (bf16 on CPU, another checkpoint's arena batches, or an empty grid)
  must leave no `--output-dir`.
