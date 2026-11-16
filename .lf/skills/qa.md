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
- Engine changes that must not alter play (performance work in `managym/src`):
  build `main`'s release extension from a detached worktree with its own
  `CARGO_TARGET_DIR`, unzip the wheel's `.so` into that worktree's `managym/`,
  and play the same seeded random games on both engines by switching
  `PYTHONPATH` (print `managym.__file__` to confirm which loaded). Per game,
  hash `obs.toJSON()` at every decision and step `rng.randrange(len(
  obs.action_space.actions))`; equal digests mean equal offers, taps and
  events. Use the authored setups (`managym.authored_deck_keys`) plus the TLA
  pool corpus human builds (`experiments/data/mtg282/pools-v2/corpus.json`,
  in the artifact store: fetch it with `manabot.data fetch` when a checkout
  lacks it; `PlayerConfig(name, pool["human_build"])`), which bring waterbend, Treasure
  and any-color sources the curated benchmark decks lack; 2,400 games took
  about two minutes per engine. A few games end in `PyAgentError` on `main`
  too; the error must match, not be absent. `maturin build` without
  `--release` gives an extension with `debug_assert!` live, about 10x slower;
  unzip it to its own directory, never over `managym/`.
- Academy browser scripts: offers are `[data-testid=action-option]:not([disabled])`
  (buttons are disabled while a Command is in flight, and `Pass Turn (F6)` in
  the panel is not an offer). Tapped lands show only in a `fullPage`
  screenshot, not in the board's text.
- Academy policy/session recovery: build the static SPA before browser servers
  (`ACADEMY_STATIC_BUILD=1 npm --prefix academy/ui run build`). Then use
  `npm --prefix academy/ui run test:e2e -- --config playwright.policy.config.ts`
  for policy refusal, both Watch seats, original-Command reload and backend
  restart; `--config playwright.search.config.ts` covers explicit Search retry
  and saved evidence. Ports are scoped by `ACADEMY_POLICY_PORT` and
  `ACADEMY_SEARCH_PORT`. The release config covers unified replay, session
  continuation and laptop navigation. Keep generation/builds finished while
  browser proofs run; rebuilding replaces chunks under a preview server.
  Combat remains ordinary per-creature offers, including nonparticipation.
  A swapped human is native seat 1; check player-target role labels against
  actual native seats. Floating mana has its own disclosure outside player
  status; status retains full policy identity and provenance.
- Configured launch (`managym academy play|replay`, `managym.academy.launch.serve`):
  from a checkout first build the static SPA with
  `ACADEMY_STATIC_BUILD=1 npm --prefix academy/ui run build`, then set
  `ACADEMY_FRONTEND_BUILD=$PWD/academy/ui/build` (otherwise `site.missing`).
  Plain `npm run build` uses adapter-auto and does not create that directory. Pass `--no-open --port <unused> --traces-dir /tmp/<dir>`;
  port 8000 is often taken. Open READY's `open` address in Playwright. For a
  multi-game evaluation trace, call `manabot.eval.match.play_cell` with
  `tests.eval.arena_fixture.arena_fixture()` anchors, several `deal_seeds` and
  `legs=(0, 1)` (six games took about 80 s under load). To test Ctrl-C, start the
  CLI with `start_new_session=True` and SIGINT restored, as below; with `nohup … &`
  SIGINT is ignored and the server looks hung.
- Academy table: `./scripts/play --port <unused> --frontend-port <unused>`; wait
  for the `ACADEMY_PLAY_READY` line (it names Python, native ABI and module
  path), open the frontend URL, press Start game, play a land and Pass Turn.
  Stop with SIGINT to the `scripts/play.py` Python process (`pgrep -f
  'scripts/play.py'`); SIGINT to the `./scripts/play` shell wrapper's PID
  leaves both servers running. To play specific cards, open "Import decklist
  files", check "Use imported decks for the next game", set both files
  (`setInputFiles` with counted lines such as `4 Heartbeat of Spring`), press
  "Check decklists", wait for `Ready —`, pick the `passive` opponent and press
  Start game; imported decks take any number of copies.
  Under heavy host load the first frontend probe in `scripts/play.py` keeps a
  fixed 30-second deadline whatever `--ready-timeout` says, so the launch
  fails with `ready.timeout`. Then start the two servers as it does: from the
  root, `uv run --python 3.13 --locked --only-group play-runtime uvicorn
  managym.academy.server:app --host 127.0.0.1 --port <api>`; from `academy/ui`,
  `ACADEMY_API_PORT=<api> npm run dev -- --host 127.0.0.1 --port <ui>
  --strictPort`; poll `/` and `/api/traces` until both return 200. Demos:
  open `/?demo=learn|precision|precision-mana`, wait a second after the
  `connected` badge, then press Start game (an immediate click can leave "No
  game in progress").
  Without a connected browser, run a Node Playwright script from `academy/ui`
  (its `playwright` package and cached Chromium resolve there; delete the
  script afterwards). Click only `[data-testid=action-panel] button` offers:
  other buttons open the deck picker or start a game. Triggered abilities
  appear under "Stack · top first" as `Card — <trigger>: <effects> ·
  triggered ability · Seat N`; read that text while the stack is non-empty.
  The session strip always shows **Rematch** (mid-game it asks to stop the
  game), so detect the end by a second Rematch button, in the "Game Over"
  dialog (`getByRole('button', {name: 'Rematch', exact: true}).count() >= 2`),
  and bound play loops by time, since the bot can win first. Live bot work
  ("Searching · Seat 2") shows in the table masthead. Opening `/games?session=`
  needs the owner credential from the same browser profile; a fresh browser
  context gets 403. Games-page Replay links are labelled `Replay game <id>`;
  select them with `locator('a', {hasText: /^Replay$/})`. The game-over
  dialog moves focus and scrolls the table, so screenshot it as it is.
  To play specific cards, open **Import decklist files**, check "Use imported
  decks for the next game" (the `#hero-deck-source`/`#villain-deck-source`
  textareas appear only then), fill both, press Check decklists, then New
  Game; `POST /api/decklists/validate` answers the same check. During the
  opponent's turn the offers stay rendered but disabled, so click only enabled
  buttons, or every click times out while the bot plays on. Hover a card name
  to read `[data-testid=card-rules-text]`. Search 64 with a fast creature deck
  usually wins before a slow card is used; give the opponent a light deck.
- Live network recovery: browser offline emulation may leave an existing
  WebSocket connected. To prove recovery, put a local TCP forwarder between the
  browser and the real server, destroy its active socket pairs, stop listening
  until the disconnected/reconnecting badge appears, then listen again. Keep
  the backend process alive so its real session survives. Compare match,
  revision, frame hash, projection and offers after recovery; do not retain
  resume credentials in receipts. Close the forwarder after the browser exits.
  Accepted Commands carry their next frame in `update.frame`; recovery carries
  it in `recovery.frame`. Capture both when observing the wire. At game end the
  result deliberately receives focus and may scroll; assert document stability
  during active decisions. After Play Again, select the completed replay by its
  result label, since the newest entry is the unfinished next game.
- First-time clone: copy tracked plus untracked non-ignored files
  (`git ls-files -co --exclude-standard`) to an empty `/tmp` directory and run
  `ACADEMY_CLEAN_API_PORT=… ACADEMY_CLEAN_FRONTEND_PORT=… ./scripts/verify-clean-machine`
  there; it builds cold, runs the browser proof and writes
  `.cache/clean-machine/proof.json` (`launch.elapsed_to_playable_ms`). Shared
  uv/Cargo/npm download caches stay warm locally, unlike the CI reference.
- Engine cost on large boards (`docs/development.md#stress-states`): build
  once with `cargo build --release --features benchmark --example bench_states
  --manifest-path managym/Cargo.toml`, then call
  `managym/target/release/examples/bench_states` directly (all 17 states ran
  in about 11 s on an M-series laptop). Compare two commits by building the
  other one in a `/private/tmp` worktree with its own `CARGO_TARGET_DIR` and
  diffing each row's `work` field; the closing digest also moves whenever
  cards are added, so it is not a behavior check across a merge from main.
  `scripts/bench_states.py` uses the installed extension.
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
  cpu --out .runs/<name>` writes `run.json` with per-phase
  seconds (`learning_seconds`, collection and forward splits) for an A/B. The
  complete-game default is bounded by its deadline, not a fixed duration. For
  installed-wheel QA use `scripts/release/smoke.py`: it trains one tiny policy
  on the native lethal-target root, reloads `self-play-raw.safetensors`, and
  discovers that exact path through `managym.academy.bots`. Run from outside
  the checkout with `uv run --no-project --python <venv>/bin/python <script>`.
  Play checkpoints with `uv run manabot sim --hero-deck-file <f> --villain-deck-file
  <f> --set sim.hero=<checkpoint.safetensors>
  --set sim.villain=<checkpoint.safetensors> --set sim.num_games=10 --set sim.num_threads=1`. The deck files must equal the
  checkpoint's `world_binding["setups"]`, written as `N Name` lines with a
  `Sideboard` section, because the default sim match is refused. Fixed seeds
  make equal win, timeout and mean-step counts a check that every sampled
  action matched. Ten games took about two minutes on a loaded laptop. Sim
  logs a model failure as `Error getting action`, plays a random move and
  still prints win rates: count those lines before trusting a result. The
  regime's decks are `run.json` `regime.match` (`hero_deck`, `hero_sideboard`).
- MLX backend (`manabot.backends.mlx`, Apple Silicon with the `mlx` extra only):
  run it on the Mac that owns the measurement, not on a shared laptop. The
  published `experiments/data/mlx_pilot/capture/initial.pt` is now refused by
  the parity CLI's world binding (written at `b62ac3127`). Make a current
  capture instead with `manabot.backends.torch.capture.capture(out, regime,
  seed, experiment=...)`, using the regime of
  `tests/model/test_backend_capture.py` with `hidden_dim=384`,
  `attention_layers=8`, `num_attention_heads=8`, 8 streams and 32 transitions
  (about 2 minutes on CPU; `hidden_dim=64` takes 20 s). Convert its
  `initial.pt` with `uv run python -m manabot.backends.convert` (`local` recipe
  checkpoints use FiLM and are refused), load it with
  `resolve("mlx").load_policy` and step a `VectorEnv` with `MatchConfig(world=WORLD_SETUP)`,
  preparing each batch with `policy.prepare(obs, inference=True)` and evaluating
  with `policy.evaluate` (logits `[B,A]`, values `[B]`); untrained-policy argmax
  differences are exact ties. `uv run python -m manabot.backends.parity <initial.pt>
  --capture <capture dir> --backend mlx --precision <dtype> --minibatches 4`
  takes seconds; `--json` keeps per-minibatch `loss_torch`/`loss_backend`, which
  show when a relative-loss failure comes from a near-zero loss.
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
  `--seed` venv and upgrade with
  `uv run --no-project --python <venv>/bin/python -m pip`
  (wheel paths plus `--find-links`
  and the CPU Torch index), since uv's one-step upgrade hides file-ownership
  problems that pip's install order exposes.
- Release installed-Academy proof (`scripts/release/academy_smoke.py`, the
  Release packages job): `./scripts/verify-installed-academy` builds the staged
  sdist, its wheel, a no-build-tools venv, the smoke and the wheel Playwright
  spec (about 10 minutes under load). It deletes its directory on exit; to rerun
  an older smoke against the same wheel, copy the script into `.runs/` with the
  `trap` line replaced by `echo $proof_dir`. Socket readers must skip
  `work_status` and early accepted `command_outcome` receipts.
- Installed Academy (`managym[academy]` wheel): build as above, then in a fresh
  `/tmp` directory `uv venv --python 3.13 --seed` (without `--seed` there is
  no `pip`) and
  `uv run --no-project --python <venv>/bin/python -m pip install '<wheel>[academy]'`,
  with `node`/`npm`/`cargo`
  shadowed by failing stubs on PATH. Re-running that on an already installed
  identical wheel file exits 0 silently and skips the extra; install by name
  with `--find-links <dist>` instead. When the candidate's version is already
  on PyPI, `uv run --no-project --python <venv>/bin/python -m pip install
  --find-links <dist> manabot` takes PyPI's copy (no
  `managym.academy.bots` entry point); `--force-reinstall --no-deps` the local
  wheel. Never patch files inside the installed `managym`: manabot's training
  refuses a changed source bundle. Supervise `managym academy` from a Python
  `Popen(start_new_session=True)` that writes `p.wait()`, delete the old log
  before relaunching, and wait for its `ACADEMY_PLAY_READY`. `manabot train
  --recipe local --device cpu --out .runs/<x>` in that directory gives trained
  opponents within a minute. Another project's server may hold `*:8000`; use
  your own port.
  With a current installed policy, set `ACADEMY_PLAY_CHECKPOINTS=<absolute path>`
  before launch. Read `/api/opponents` for its digest, then open
  `/?hero=ur_lessons&villain=gw_allies&opponent=<digest>` for the local recipe's
  bound decks (Elemental UR/GW are different lists and correctly refuse it).
  Click Start game, then the legal Play Island offer and Pass Turn. Wait until
  an action offer is enabled and automation has settled before recording
  `main[data-frame-revision]`; reload must restore that revision. A merely
  visible offer can still belong to an automated pending transition.
- Engine rules changes (zones, tokens, copies, SBAs): play many games through
  `managym.Env` with `PlayerConfig(name, {card: count}, world=managym.WORLD_SETUP)`
  decks built around the cards involved, and a scripted policy over
  `obs.action_space.actions`. Actions carry no `description`; match
  `action.focus` ids against `env.observation_for_player(env.current_agent_index())`
  card ids to pick a card. Read `observation_for_player(p).agent_cards`
  (`zone`, `is_token`) at every decision. A token in the graveyard during a
  mana ability's color choice is correct: SBAs wait for priority. Academy's
  game end is a "Game Over" dialog with "Rematch".
  Battlefield permanents are `agent_permanents`, with their own ids (the ones
  activation offers focus) and an integer `registry_key`; join them to a name
  through `agent_cards` rows that have the same `registry_key`.
- Policy vocabulary/admission changes (`record_vocabulary.rs`, token kinds,
  `manabot/policy/admission.py`): the consumer is training's catalog policy.
  Script a directed `managym.Env` game that reaches the runtime program,
  `space.encode` each decision at `ObservationConfig(policy_input_version=10)`
  and pass it through `manabot.verify.record_coverage._embed` with a tiny
  catalog `ActorCritic`, as `coverage_sweep` does. Find decisions carrying a
  token by scanning encoded buffers for its u32 as adjacent 16-bit halves
  (hi, lo). For Koh's `HaveAbilitiesOf`: Swamps plus Koh against Forests plus
  Giant Spider, Koh cast once, always targeting with its ETB, the villain
  playing randomly, and one Pay-1-life activation per turn. The choice
  is a Modal decision only with two or more cards exiled with Koh (one is
  skipped as forced). 3 of 6 games reached it (8 decisions, about 7 minutes at load ~110).
  For the negative, build a mutated release wheel with
  `-o /tmp/<dir>/wheels`, restore the source at once, unzip the wheel into
  `/tmp/<dir>/site` and run with `PYTHONPATH=/tmp/<dir>/site` (print
  `managym.__file__`); admission must then fail at the first such decision.
- "No behavior change" engine claims: `git worktree add --detach /tmp/<x>
  <before>`, build it with `uv run --extra play maturin build --release -i .venv/bin/python -m
  /tmp/<x>/managym/Cargo.toml` (about 4-5 min), drop its `.so` into
  `/tmp/<x>/managym/`, and run one seeded script against each tree
  (`sys.path.insert(0, root)` before `import managym`). Per decision, write a
  hash of both seats' `toJSON()` plus `env.state_digest()`; `cmp` the files.
  Use `random.Random(seed).randrange(env.action_count())` for actions.
  `managym.authored_deck_keys(pack)` / `authored_deck_setup(pack, key)` give
  the authored decks. Remove the worktree afterwards. Include the TLA pool
  human builds too: in MTG-320, 138 authored and Azula games matched while 33
  of 120 pool games exposed a regression (Kyoshi Battle Fan's token attach).
  When games diverge, replay one to the first differing decision on both
  engines and diff the parsed `toJSON()`; permanents' `attached_to`, `power`
  and the cast action's `focus` names locate the card.
- Record `uptime` when the Academy table stalls or Vite fails startup. MTG-320
  QA observed both at load averages around 300, but did not establish a cause.
  Repeat on an idle host before attributing a stall to the change or to load.
- Python-only library changes (no engine change): `uv build --wheel --package
  manabot -o .runs/<dir>` builds in seconds; install it into a fresh `/tmp`
  venv with `uv pip install <wheel> --index-url https://pypi.org/simple
  --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy
  unsafe-best-match`. If the checkout uses engine APIs newer than the same-version
  PyPI wheel, supply the current engine wheel too: a version match alone does
  not establish source compatibility. A wheel from `maturin develop` may contain
  a `.pth` pointing back to the checkout; it is not an installed-wheel proof.
  Use a release wheel from the build-and-verify procedure, then confirm both
  packages' `__file__` paths are inside the fresh venv. Run user scripts from
  `/tmp` with `uv run --no-project --python <venv>/bin/python python -I`.
  Training CLI QA requires the staged manabot wheel from
  `scripts/release/prepare.py`: its `.manabot-source.json` supplies verified
  source provenance outside Git. A direct workspace wheel lacks that receipt
  and training fails at `source_commit`; do not patch the installed files.
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
  For a public safe-pause proof, pass an existing `pause_path` to
  `execute_regime(..., backend=resolve("torch"))`; catch `TrainingPaused` and
  read the retained run from the store. It is recorded as `interrupted` with
  a committed snapshot. Continue by ID through the CLI, then compare the
  final `load_update(..., backend=resolve("torch").state)` payload against
  uninterrupted execution, including Adam, EMA and collector sampling RNG.
- Deck-source or `elemental-uniform` regimes from `local_smoke`: drop `match`
  and set `deck_source` plus `agent.world = managym.WORLD_SETUP`, or keep an
  authored Elemental `match` and set `agent.world = "elemental-world-v2"`;
  either way validation refuses the local recipe's agent world. Give parallel
  runs separate `--out` parents, since `training.sqlite` lives beside `--out`
  and concurrent runs lock it. To see what training games actually were, wrap
  `SeatRoutedCollector.collect` in a driver calling `execute_regime` and read
  `self._env.current_player_configs()`, `self.stream_matches` and
  `self._buffers` (e.g. `decklists`, `[streams, 2, cards, 5]`) after each
  batch. The native `VectorEnv.set_training_decks(decks, distinct)` differs
  from the collector wrapper's `(decks, digest, sampling)`.
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

- Live Academy QA without fixtures: build first, then start the ordinary
  `managym.academy.server:app` with `ACADEMY_TRACES_DIR` pointing into an ignored
  `.runs/` directory and `ACADEMY_PLAY_RECORD_ORIGIN=automated_validation`.
  Run `ACADEMY_API_PORT=<api> npm run preview -- --host 127.0.0.1 --port <ui>
  --strictPort` from `academy/ui`. Browser frame probes must read accepted
  Commands from `message.update.frame` as well as `message.recovery.frame`;
  otherwise a reload comparison uses a stale opening frame. For a short real
  target/recovery game, import `20 Mountain\n40 Lightning Bolt` against
  `40 Plains`, select Passive, play Mountains and cast Bolts at Villain.
  Replay's newest entry can be an unfinished Play Again attempt: select the
  completed result explicitly before checking Next frame. Playwright offline
  mode did not close an existing localhost WebSocket in the MTG-355 run; it
  alone does not prove disconnect/reconnect behavior.
- For a genuine local connection interruption, put a TCP forwarder between the
  browser and the ordinary same-origin Academy CLI server. Stop its listener and
  destroy both sockets of every forwarded pair; observe disconnected/reconnecting
  and disabled offers, then restore the listener. Compare the pending frame and
  recorded event identities before and after. Keep credentials out of receipts.
  Also reload the whole page: a delta reconnect alone cannot detect a persisted
  presentation cursor suppressing the tail when in-memory history is empty.
  Fresh reload can recover older facts absent from an authored demo's initial
  prefix; require committed facts to remain, with unique rows and no old motion.
  Let keyboard Home scrolling settle before measuring a reading anchor. Prompt
  kinds in frames are lowercase; `data-action-space-kind` uses uppercase names
  such as `CHOOSE_TARGET` and `LOOK_AND_SELECT`. Identical card copies have
  separate cast offers, so choose an explicit occurrence in browser scripts.
  In Replay, open Consequences with a real pointer click at the final frame;
  setting `<details>.open` cannot detect a hand overflowing over its summary.
  Check the hand ends above history. Recorded terminal results have no next-game
  action and must leave keyboard navigation available; the live Play Again
  dialog retains its focus behavior.
- Play/Inspect QA: fresh storage starts in Inspect. In Play, activate the exact
  hand `[data-object-id]` by pointer or Enter, then commit its labeled
  `[data-offer-id]`; selection, Inspect card, mode switches and Escape send no
  Command. Only an unambiguous cast/play-land source supports drag: move at
  least 8 CSS px and release in `[data-testid=play-drop-area]`. Verify outside
  release and Escape during capture send nothing and leave no selection after
  the compatibility click. Observe WebSocket frames without rewriting them and
  compare each deliberate Command's match, revision, prompt and offer.
  Open History and settings before selecting a card and before a payment choice
  at 1280×800: inspect screenshots and compare `.decision-controls`, `.offer-list`
  and `[data-testid=automation-status]` rectangles. Parent panel bounds alone
  miss overflowing children. Repeat with 100 offers; the list must scroll
  independently while controls remain visible. Use Edit stops, change a whose-turn
  checkbox, wait for the authority update, then reload; check settings and pending
  frame as well as the separate persisted Play preference. A second tab must
  have no resumed game. Initial automatic history may precede a launch's received
  tail; compare that tail exactly to the saved source and the full saved prefix
  to Replay, retaining explicit unavailable-prefix labels.
