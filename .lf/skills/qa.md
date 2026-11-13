---
requires: a change or behavior to assess
produces: a change that works for the people who will use it, with what you saw and what is still wrong
action_style: procedural
---
Use the change the way the people it is for will use it, and fix what breaks.

You are its first user. Nobody should open a demo or merge a PR and find it broken
on the first click; if they would, this step failed.

## Use the real thing

Run the real product, not its tests: the built app, the installed CLI, the live
page in a browser, the API over the network, the simulator or device. Drive it
with computer use, a browser, a terminal or whatever reaches it the way a person
would. Internal setters, test-only endpoints, mocks and fixtures do not count.

Getting there is your job. Build it, start its services, create accounts and
realistic data, sign in, set up whatever state a real user would already have.
When no route exists, make one. Hand back only what truly needs a person, such
as their own credentials, a physical device, a payment or a production approval,
and say exactly what that left unchecked.

Make the scenario as real as you can reach:

- Arrive the way a person arrives: a fresh install or first visit, then a
  returning user with existing data from before this change.
- Do the main job end to end, then the things people actually do: wrong input,
  empty and large data, going back, interrupting and resuming, a slow or failed
  network, two things at once.
- Check what it leaves behind as well as what it shows: files, rows, messages,
  processes, and what another part of the product now reads.
- Look at it. Read the output, check the screenshot, notice what is confusing,
  slow or ugly, not only what is wrong.

## Fix and repeat

Repair what breaks within the change's scope, then go through the experience
again from the start. Keep going until it works for the people it is for. A
change to what the product should be is not yours to make; describe it
instead. When asked for a read-only review, report instead of
fixing.

Run tests only where they tell you something the real path cannot reach, and
keep any you add about what a person would notice.

## Keep what you learned

How to launch and drive this product belongs in this skill. When you had to work
out a step it does not describe, or a step it describes no longer works, write
the working version here with `lf skill customize qa`, replacing what was wrong.

Show what you did in `scratch/qa.html`: each scenario, what you did, what you saw
with screenshots or transcripts, and what you fixed. Finish with what works now,
what you could not reach and why, and what is still wrong, worst first. Do not
publish, land or choose the next Graph step from here.

## This repository

- Loopflow workflow changes (`.lf/graphs/`, `.lf/skills/`): the user surface is
  `lf graph list`, `lf graph show <name>` (and `--json` for nodes and
  `returns_to` edges), `lf help <skill>` for the text an agent receives, and
  `uv run python scripts/check_loopflow_workflows.py`. Check every command,
  flag and doc anchor a skill names against the real tool (`--help`, `rg`).
  Launching the graph starts agents and possibly benchmarks; inspecting and
  compiling it is sufficient for workflow-only QA. When the Task authorizes an
  actual-workload measurement, run the guide's benchmark example as written into
  an ignored `.runs/` path, then read the receipt fields the skills name. The
  default CPU `bench_training_update.py` took about 13 minutes and 8 GB resident
  in one 16-CPU Mac reading; budget for the workload before starting it. Keep its
  process supervised through completion or cleanup. Measurements require the
  Task's scope and resource authority; never create artificial contention to
  test them (see `TESTING.md`).
- Academy table: `./scripts/play --port <unused> --frontend-port <unused>`; wait
  for the `ACADEMY_PLAY_READY` line (it names Python, native ABI and module
  path), open the frontend URL, press New Game, play a land and Pass Turn.
  Stop with SIGINT to the `scripts/play.py` process; it stops both servers.
  Without a connected browser, run a Node Playwright script from `academy/ui`
  (its `playwright` package and cached Chromium resolve there; delete the
  script afterwards). Click only `[data-testid=action-panel] button` offers:
  other buttons open the deck picker or start a New Game. Triggered abilities
  appear under "Stack · top first" as `Card — <trigger>: <effects> ·
  triggered ability · Seat N`; read that text while the stack is non-empty.
- First-time clone: copy tracked plus untracked non-ignored files
  (`git ls-files -co --exclude-standard`) to an empty `/tmp` directory and run
  `ACADEMY_CLEAN_API_PORT=… ACADEMY_CLEAN_FRONTEND_PORT=… ./scripts/verify-clean-machine`
  there; it builds cold, runs the browser proof and writes
  `.cache/clean-machine/proof.json` (`launch.elapsed_to_playable_ms`). Shared
  uv/Cargo/npm download caches stay warm locally, unlike the CI reference.
- Local-update search changes: `uv run --extra dev python
  scripts/bench_local_update.py --reference main --worlds 4 16` checks in
  seconds that receipts still equal main's, field for field; the
  `--rollout-evidence`/`--native-encoder` arms check that full evidence and
  the Python encoder still match the compact, native defaults. To play
  games, build two players per arm with `make_player` and drive `act`,
  `prepare_step`, `env.step_semantic` and `observe_step` as
  `manabot/sim/native_game.py` does; the default `decision_seconds=1` times
  out full evidence at 16 worlds × depth 20 (about 0.9 s on the
  initialized fixture), so set the budget explicitly for full-evidence arms.
  `_bench_match` and `_initialized_checkpoint` from the bench script give the
  match and checkpoint; with equal player and reset seeds, every evidence and
  encoder arm must pick identical Commands for a whole game (two 93-decision
  games per arm at 4 worlds × depth 20 took about 90 s).
- Model or training-path changes: `uv run manabot train --recipe local --device
  cpu --out .runs/<name>` takes seconds and writes `run.json` with per-phase
  seconds (`learning_seconds`, collection and forward splits) for an A/B. Play
  checkpoints with `uv run manabot sim --hero-deck-file <f> --villain-deck-file
  <f> --set sim.hero=<ckpt.pt> --set sim.villain=<ckpt.pt> --set
  sim.num_games=10 --set sim.num_threads=1`. The deck files must equal the
  checkpoint's `world_binding["setups"]`, written as `N Name` lines with a
  `Sideboard` section, because the default sim match is refused. Fixed seeds
  make equal win, timeout and mean-step counts a check that every sampled
  action matched. Ten games took about two minutes on a loaded laptop.
- MLX backend (`manabot.backends.mlx`, Apple Silicon with the `mlx` extra only):
  run it on the Mac that owns the measurement, not on a shared laptop. Convert a w160l8
  checkpoint (the published `experiments/data/mlx_pilot/capture/initial.pt`;
  `local` recipe checkpoints use FiLM and are refused), load it with
  `load_mlx_policy` and step a `VectorEnv` with `MatchConfig(world=WORLD_SETUP)`,
  admitting each batch with `admit_observation` and comparing against torch
  `PolicyInference` (torch values are `[B]`, MLX `[B,1]` from `policy(...)` and `[B]` from
  `policy.evaluate`; untrained-policy argmax
  differences are exact ties). `python -m manabot.backends.parity <initial.pt>
  --capture <capture dir>` on the published subset takes about 50 s on an M2.
- Library consumers: follow `docs/development.md#build-and-verify` into an
  ignored `.runs/` output (the engine staging is `<staging>/managym`), then
  `uv build --wheel --python 3.12 <managym sdist>` for the other interpreter
  (reuse a `CARGO_TARGET_DIR`). In a fresh `/tmp` venv with PYTHONPATH unset and
  a PATH holding only a `uv` symlink plus `/usr/bin:/bin` (no Cargo), install the
  engine wheel alone and run `scripts/release/engine_smoke.py` and the
  `docs/engine-python.md` commands verbatim; then add both wheels at the minimum
  Torch/Etude pins and run a copy of `scripts/release/smoke.py` (CPU training,
  about 20 s). Also `uv init` a project and `uv add` both `file://` wheels for
  latest resolution. Returning users: install PyPI `manabot==0.2.0` in a
  `--seed` venv and upgrade with plain `pip` (wheel paths plus `--find-links`
  and the CPU Torch index), since uv's one-step upgrade hides file-ownership
  problems that pip's install order exposes.
- Engine rules changes (zones, tokens, copies, SBAs): play many games through
  `managym.Env` with `PlayerConfig(name, {card: count}, world=managym.WORLD_SETUP)`
  decks built around the cards involved, and a scripted policy over
  `obs.action_space.actions`. Actions carry no `description`; match
  `action.focus` ids against `env.observation_for_player(env.current_agent_index())`
  card ids to pick a card. Read `observation_for_player(p).agent_cards`
  (`zone`, `is_token`) at every decision. A token in the graveyard during a
  mana ability's color choice is correct: SBAs wait for priority. Academy's
  game end is a "Game Over" dialog with "Play Again".
- "No behavior change" engine claims: `git worktree add --detach /tmp/<x>
  <before>`, build it with `uv run --extra play maturin build --release -i .venv/bin/python -m
  /tmp/<x>/managym/Cargo.toml` (about 4-5 min), drop its `.so` into
  `/tmp/<x>/managym/`, and run one seeded script against each tree
  (`sys.path.insert(0, root)` before `import managym`). Per decision, write a
  hash of both seats' `toJSON()` plus `env.state_digest()`; `cmp` the files.
  Use `random.Random(seed).randrange(env.action_count())` for actions.
  `managym.authored_deck_keys(pack)` / `authored_deck_setup(pack, key)` give
  the authored decks. Remove the worktree afterwards.
- Record `uptime` when the Academy table stalls or Vite fails startup. MTG-320
  QA observed both at load averages around 300, but did not establish a cause.
  Repeat on an idle host before attributing a stall to the change or to load.
- Python-only library changes (no engine change): `uv build --wheel --package
  manabot -o .runs/<dir>` builds in seconds; install it into a fresh `/tmp`
  venv with `uv pip install <wheel> --index-url https://pypi.org/simple
  --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy
  unsafe-best-match` (the matching `managym` comes from PyPI), then run user
  scripts from `/tmp` with `<venv>/bin/python -I`. Outside a checkout,
  `save_checkpoint` prints a harmless `fatal: not a git repository`.
- Training guide (`docs/training-loop.md`): run its example command verbatim
  (about 20 s on CPU), save and reload the policy as it says, call both
  maintained rules directly on a collected batch, and compare the tables'
  signatures with `inspect.signature`.
- Training recipes: write a regime JSON from Python
  (`TrainingRegime(...).model_dump_json()`) and run
  `uv run manabot train --regime <file> --seed N --out /tmp/<dir>/<run>` with a
  fresh `--out` each time; success prints nothing, so read `<run>/run.json`
  (`status`, `stages[].artifacts`, `selected_artifact`). Keep it small (4 streams,
  64 transitions, 2 updates; a few seconds on CPU). Artifact fetches need
  `doppler run --project manabot --config prd --`; set `MANABOT_DATA_CACHE` to
  an empty directory to act like a new machine. To A/B an `Execution` switch
  the CLI lacks (such as `live_tokens`), start from
  `manabot.recipes.local.local_smoke(steps=3, device="cpu")`, set
  `stages[0].execution.<field>`, re-validate, and dump one regime per arm with
  the same `--seed`. Equal layouts then give `run.json` files that differ only in
  `regime_digest`, and `model_state_dict` within about 1e-6. On CPU,
  `bucketed_shapes` makes learning about 7× slower than padded, so time
  bucketed arms against a bucketed control. Rebuild a regime for another
  device from `authored_dump()`, not `model_dump()`, so `auto` fields resolve
  again. To resume, the regime needs `recovery` (e.g.
  `{"checkpoint_updates": 2}`); without it `--resume-from` passes the digest
  check, then fails with "no committed recovery boundary". Resume with the same
  regime file, `--seed` and a new `--out`. `wall_seconds` counts the parent's
  time. macOS has no `timeout`; use the tool's own timeout.
- TLA Limited decks (the MTG-282 pool corpus) overflow default observation
  capacities within a few dozen games; copy the `observation` block and the
  `game_errors` policy (`recover_capacity: true`) of
  `experiments/regimes/tla-frontier-c5fba4-w64.json`. The default 300 s
  `wall_seconds` ends a 40-update run on a laptop as `interrupted`.
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
- CI workflow or test-timing changes: the user surface is hosted CI on the
  pushed branch: `gh workflow run ci.yml --ref <branch>`. Read job and step
  timestamps with `gh run view <id> --json jobs`; read logs with
  `gh run view <id> --job <job-id> --log`. Compare install time with peers in
  the same run and inspect dependency logs for unexpected downloads.
  See `docs/development.md` (Job time) for the budget.
- Ctrl-C a run: start it with `subprocess.Popen(..., start_new_session=True,
  preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_DFL))` and
  `os.killpg(pid, SIGINT)`. A job started with `&` from a non-interactive shell
  ignores SIGINT, so `pkill -INT` appears to hang.
- Experiment reports and figures: arrive as a new reader with
  `MANABOT_DATA_CACHE` set to an empty directory, then `doppler run --project
  manabot --config prd -- uv run --extra artifacts python -m manabot.data fetch
  <paths>`. In zsh, pass a path list as an array (`"${P[@]}"`), since zsh does
  not word-split `$P`. Fetch only the files the figure code reads; whole
  evidence folders can exceed 1 GB. Redraw twice and compare digests, because a
  figure that changes bytes on every redraw also changes its manifest entry.
  Render SVGs with `rsvg-convert -z 2 -b white` and look at them. The L4 profile
  (`experiments/l4_step_profile.py`) has a CPU smoke job,
  `ETUDE_OUTPUT_DIR=<dir> MTG301_WORK=<dir> MTG301_CANDIDATE=<c> uv run python
  experiments/l4_step_profile.py job --smoke`. For `live` it took about 6
  minutes and 2.2 GB and runs every stage as a real child process.
