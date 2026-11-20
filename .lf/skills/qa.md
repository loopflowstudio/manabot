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
  events. Use the authored setups (`managym.authored_deck_keys` of
  `elemental-world-v2` and `ur-lessons-vs-gw-allies`; pack directory names are
  refused) plus the TLA pool corpus human builds
  (`experiments/data/mtg282/pools-v2/corpus.json`, fetched with `manabot.data
  fetch` when absent; `PlayerConfig(name, pool["human_build"])`), which bring waterbend, Treasure
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
- Checkpoints as Academy opponents: launch `managym academy play` with
  `MANABOT_RUNS_DIR=<folder of .pt/.safetensors>` and `ETUDE_HOME=<home of an
  etude train run>`; `GET /api/opponents` lists what the picker shows (an
  unreadable file is omitted, not listed; a refused one is listed with
  `available: false` and its reason). The list is read at launch: restart
  the server after adding checkpoints. The launch opens inside a game, so
  **Start game** on the New game page and **Start next game** raise a browser
  confirm ("Stop the current game and start a new session?"). Playwright
  dismisses it by default and the click then does nothing: register
  `page.on('dialog', d => d.accept())` first. Check the header names the
  chosen policy before reading the opponent's plays. When this checkout has no
  `academy/ui/node_modules`, keep the script under `.runs/` and load Playwright
  from a sibling checkout with `createRequire('<sibling>/academy/ui/')('playwright')`.
- Game records in Academy (`managym.recording`, `managym academy replay
  <file|dir> --game N`): write records with the `docs/engine-python.md` Record
  games example (a complete game, and one closed after a few steps for
  interrupted), then open each from an engine-only wheel venv in a fresh
  Playwright context: the launcher's `open` address carries no credential, so
  a 403 there is a real failure. `[data-testid=recorded-decision-evidence]`
  holds Value/Update/Policy or "not retained". `etude train
  manabot/recipes/local.py --config '{"record_games": 0.25}'` prints the games
  command. Every self-play recipe takes the same flat keys: `etude train
  manabot/recipes/elemental.py --config '{"updates": 2, "streams": 4,
  "record_games": 1, "statistics": true}' --until updates=1` records eight
  Elemental games and statistics in about 9 s on CPU; a changed key on
  `--resume` must refuse with `run_identity_conflict`. `play_cell` with two legs
  writes `games/<cell>.jsonl`. Count `read_records(<dir>)` against the games
  played: duplicates mean a second file in that directory. A different-world
  record can be made by editing `identities.world`/`card_digest` in a copy.
  Stall checks: `"stall_seconds": 0.3` in the local recipe config and
  `touch <out>/diagnostics.request` exercise the watchdog without load; read
  `escalations.jsonl` for one line per real stall.
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
  (`import { chromium } from '@playwright/test'`; a fresh worktree first needs
  `npm ci` there, about a minute; Chromium is cached; delete the script
  afterwards). Click only `[data-testid=action-panel] button` offers:
  other buttons open the deck picker or start a game. Triggered abilities
  appear under "Stack · top first" as `Card — <trigger>: <effects> ·
  triggered ability · Seat N`; read that text while the stack is non-empty.
  The configured-launch table (`managym academy play`) has no Rematch in its
  header, so detect the end by the right panel's "Game over — no next
  decision." text (or one Rematch button), not by counting two Rematch
  buttons, and bound play loops by time, since the bot can win first. Live bot work
  ("Searching · Seat 2") shows in the table masthead. Opening `/games?session=`
  needs the owner credential from the same browser profile; a fresh browser
  context gets 403. Games-page Replay links are labelled `Replay game <id>`;
  select them with `locator('a', {hasText: /^Replay$/})`. The game-over
  dialog moves focus and scrolls the table, so screenshot it as it is.
  For a demo with specific cards, check that art for an uncached card loads
  through the same launch the person will use. Hover a card name to read
  `[data-testid=card-rules-text]`. During an active decision, move the pointer
  between offers and tab through them. Keep the prompt, committed facts and
  hints distinct, and check that changing a preview leaves the action buttons
  in place. For a multi-step payment, make a contribution and check that the
  panel shows what committed, what remains payable and what is still floating.
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
- Native Command QA: use an ordinary seeded game and its current semantic frame;
  submit one legal Command, then another command id with the old revision and
  an absent offer id with the current revision. Native rejection raises
  `managym.AgentError` (not only `ValueError`). Compare both `state_digest()`
  and `semantic_decision_frame_json()` before/after each rejection.
- Engine cost on large boards (`docs/development.md#stress-states`): build
  once with `cargo build --release --features benchmark --example bench_states
  --manifest-path managym/Cargo.toml`, then call
  `managym/target/release/examples/bench_states` directly (all 17 states ran
  in about 11 s on an M-series laptop). Forks carry the payment cache, so
  check that a layer's `work` (`calls`, `cache_hits`) still shows the search
  it claims to time rather than a hit warmed by preparation. Compare two commits by building the
  other one in a `/private/tmp` worktree with its own `CARGO_TARGET_DIR` and
  diffing each row's `work` field; the closing digest also moves whenever
  cards are added, so it is not a behavior check across a merge from main.
  `scripts/bench_states.py` uses the installed extension.
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
- Trained-policy Play/Watch/Search: train a checkpoint at this checkout
  (`uv run etude train manabot/recipes/local.py`, about 30 s) and register
  `{"provider": "manabot", "artifact": "<abs path>/out/policies/<update>-raw.safetensors", "config": {}}`
  with `managym academy --policy <file>.json play|watch`. A checkpoint trained
  before syncing main can pass the picker yet fail Start with "Failed to load
  the selected opponent." (catalog vocabulary grew); retrain rather than debug
  it. Watch resume tokens live in that tab's storage, so reopen a live Watch
  trace in the same browser context. A completed Play game's evidence opens
  from `/replay?trace=<id>` (ids from `/api/traces`) via "Inspect bot search".
- Model or training-path changes: use `uv run etude train manabot/recipes/local.py`
  for the local CPU smoke. Inspect update rows for per-phase collection and
  learning seconds. When saved training state gains a field, check that
  emitted `coordinates/` values are scalar numbers. When adjusting input or
  embedding scales, perturb the feature at initialization and measure the
  resulting change in the full object token with its usual streams enabled.
  For installed-wheel QA, `scripts/release/smoke.py` trains one policy on the native
  lethal-target root through `etude train`, reloads its
  `out/policies/00000002-raw.safetensors`, and discovers that exact path
  through `managym.academy.bots`. Run from outside the checkout with
  `uv run --no-project --python <venv>/bin/python <script>`. Play
  checkpoints with `uv run manabot sim --hero-deck-file <f> --villain-deck-file
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
  before relaunching, and wait for its `ACADEMY_PLAY_READY`. `uv run etude
  train manabot/recipes/local.py` with `ETUDE_HOME` in that directory gives
  trained opponents under `track/<experiment>/out/policies/` within a minute. Another project's server may hold `*:8000`; use
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
  `encode_observation(space, obs)` each decision at `ObservationConfig(policy_input_version=10)`
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
  Catch `BaseException` per game: `PanicException` is not an `Exception`.
  Python action types print as `ActionEnum.PriorityPlayLand` (no int
  constructor); compare `str(a.action_type)`. When a change deliberately alters
  the default action space, play a canonical policy on both engines. That policy
  drops the intended offers, draws `rng` only when two or more choices remain and
  hashes `(chooser, space kind, [(type, focus, ability_index)], turn, step)`.
  The remaining divergence is what the change did not mean to touch (MTG-385:
  324/328 matched, and the rest were the same intended correction).
- Record `uptime` when the Academy table stalls or Vite fails startup. MTG-320
  QA observed both at load averages around 300, but did not establish a cause.
  Repeat on an idle host before attributing a stall to the change or to load.
- Python library changes: build manabot with `uv build --wheel --package
  manabot -o .runs/<dir>`. When the checkout has unreleased engine changes,
  stage the engine with scripts/release/prepare.py as in docs/development.md
  and build its wheel from the staged managym directory. Install both local
  wheels in the fresh venv, including the evaluation extra for corpus SQL.
  The PyPI engine can carry the same version and lack modules used by the
  checkout. Run the user script from /tmp with
  `uv run --no-project --python <venv>/bin/python python -I <script>`.
- Training guide (`docs/training-loop.md`): run its example command verbatim
  (about 20 s on CPU), save and reload the policy as it says, call both
  maintained rules directly on a collected batch, and compare the tables'
  signatures with `inspect.signature`.
- Training recipes: run `uv run etude train manabot/recipes/local.py --experiment smoke`.
  Inspect `~/.etude/track/smoke/` (or `$ETUDE_HOME/track/smoke/`):
  `run.json`, logged rows, `out/state/current.json` and the save folder it
  names (`state.json` with the framework state files), and
  immutable `out/policies/`. A second new run with that name must refuse.
  For a planned stop, use a larger config update count with `--seconds 1`;
  expect exit 75, then `--experiment smoke --resume` to finish the same run. To extend a
  finished run, use `--experiment smoke --resume --until updates=N`; without a later target
  it must refuse. `etude continue --run RUN --until updates=N` makes a child
  run for a launched parent and leaves the first unchanged.
  Device and other recipe fields are passed with `--config JSON`; the seed
  is the run's `--seed`, not a config field. Check raw/EMA checkpoint tensors
  with matched small inputs. Fetch artifacts through Doppler without printing
  credentials. `manabot train`, its `--out` identity and `state.pt` are retired.
  After a train and a `--resume`, run `uv run --extra reports manabot
  training-report --source $ETUDE_HOME/track --out <f>.html` and read the
  page: learner transitions and transitions per active second must be numbers,
  not "unavailable" (they come from the loop's declared clocks). Screenshot it
  with the `academy/ui` Playwright script above (`npm ci` first in a fresh
  worktree). A run without a `training_seconds` clock, such as belief's
  sampler, honestly shows "unavailable" with four warnings.
  Set `ETUDE_HOME` to an ignored `.runs/` folder so QA runs stay out of
  `~/.etude`. `etude smoke experiments/elemental.py --keep --revision HEAD
  --updates 3200 --hours 11.9 --usd-per-run 9` trains for about 3 minutes, then
  evaluates for about 30 minutes on a laptop (600 one-game matches). A Python
  error in a game fails smoke eval within seconds, naming the error; only native
  failures become invalid rows, so still read the eval rows:
  `etude.eval.query.statsdb("elemental", "<kept>/track")`, then group `games`
  by `unit`, `valid` and `failure` (duckdb over
  `<kept>/track/elemental/*/blocks/*/games.parquet` also works; there is no
  pyarrow). The `failure` column is omitted when every value is null. Rerun only eval and report with
  `--step eval --step report --source <kept>/track`. That eval-only report has
  no training runs, so its heuristic panels stay empty; for the full picture,
  copy the training and eval run folders into one track and run
  `etude report experiments/elemental.py --source <it> --out <dir>`. Local eval
  evidence goes to `.runs/elemental/arena/<match-key>/attempt-*`; a retry adds
  an attempt beside earlier failed ones, so nothing needs deleting. To see a policy from `etude train` in Academy, start
  `./scripts/play` with the same `ETUDE_HOME`; discovery searches Etude's local
  track.
- TLA Limited decks (the MTG-282 pool corpus) overflow default observation
  capacities within a few dozen games; copy the `observation` block and the
  `game_errors` policy (`recover_capacity: true`) of
  `experiments/regimes/tla-frontier-c5fba4-w64.json`; `experiments/tla_frontier.py` trains that record on the current loop.
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
- CI workflow or test-timing changes: the user surface is hosted CI on the
  pushed branch: `gh workflow run ci.yml --ref <branch>`. Read job and step
  timestamps with `gh run view <id> --json jobs`; read logs with
  `gh run view <id> --job <job-id> --log`. Compare install time with peers in
  the same run and inspect dependency logs for unexpected downloads.
  See `docs/development.md` (Job time) for the budget.
- Training game journals (`diagnostics: detailed`): run `etude train manabot/recipes/local.py --experiment NAME --config '{"updates": 200, "diagnostics": "detailed"}'` from a supervised `Popen(start_new_session=True)`, poll `game-journals/operation.json` and `os.killpg(..., SIGKILL)` while `collection` is pending after update 1; copy `vector-N.jsonl` and replay it with `manabot.training.state.game_errors --attempt 1`. Kills land at `ready` boundaries; a pending native step is reachable only through the `os._exit` child fixtures. Compare on/off runs by safetensors tensor bytes (metadata differs by `run_id`; the dev venv has no `safetensors` package, so read the little-endian u64 header length, the JSON header and each tensor's `data_offsets` slice directly); `SelfPlay.workers=2` gives sharded files.
- Standalone statistics/recording: stage and install an engine-only wheel into a
  fresh venv, then run the two Python examples in `docs/engine-python.md` outside
  the checkout. Play the same seeded authored game with neither option, statistics
  only, recording only and both; compare observation digests and summary counts.
  Exclude `generation` from cross-Env summary comparison: it intentionally names
  each distinct game. Check terminal drain-once and replaced partial summaries.
  Copy a journal and run `uv run --no-project --python <venv>/bin/python python -m
  managym.replay <copy> --seconds 30 --statistics`; confirm source bytes remain
  unchanged. Empty/missing/corrupt inputs must fail, and a torn final append must
  warn while recovering the last complete boundary. `scripts/release/engine_smoke.py`
  additionally checks that manabot, Torch and Etude are absent.
- Ctrl-C a run as a terminal does: fork it on a pty (`pty.fork()`, then
  write `\x03` to the master). The first SIGINT is Etude's saved stop (exit
  75); a second is a hard interrupt. Without a tty, uv 0.9.22 forwards a
  process-group SIGINT to its child, which gets two and ends `failed` with no
  save, so `os.killpg` on `uv run etude …` does not show what a person sees.
  Signal `.venv/bin/etude` directly, or use the pty. A job started with `&`
  from a non-interactive shell ignores SIGINT.
- Experiment reports and figures: arrive as a new reader with
  `MANABOT_DATA_CACHE` set to an empty directory, then `doppler run --project
  manabot --config prd -- uv run --extra artifacts python -m manabot.data fetch
  <paths>`. In zsh, pass a path list as an array (`"${P[@]}"`), since zsh does
  not word-split `$P`. Fetch only the files the figure code reads; whole
  evidence folders can exceed 1 GB. Redraw twice and compare digests, because a
  figure that changes bytes on every redraw also changes its manifest entry.
  Render SVGs with `rsvg-convert -z 2 -b white` and look at them.
  `fetch` skips files that already exist in the checkout ("already present"),
  whatever the cache holds. In the authoring checkout, fetch in a `git worktree
  add --detach /private/tmp/<x> HEAD` instead, running `uv run --project
  <checkout>` there, then `cmp` its files with the checkout's. Recompute the
  report's table cells from the fetched CSVs.

- Combat drafts (`[data-testid=combat-draft]`, `POST
  /api/sessions/<id>/combat-preview`): the draft appears only after the
  preview answers, so wait for it before clicking creatures; without a preview
  (a mocked socket with no route) the table keeps one creature at a time.
  `[data-draft]` on a card is `can`, `in` or `armed`; confirm only once the
  primary has `data-draft-ready=true`. Collect creature ids before clicking:
  a locator over `[data-draft=can]` shifts as cards are drafted.

- Payment previews (`[data-testid=payment-preview]`, `POST
  /api/sessions/<id>/payment-preview`): the panel exists only once a payment
  offer has been previewed in the current decision, so read it with a count
  check. Arrival focus previews the first offer as soon as a game starts; a
  `page.route` for failure or delay must be installed before Start game.
  Compare `[data-payment-source]` ids with the frame's tapped permanent ids
  after the commit. `trigger_order_server` with `payment_fixture: true` in the
  `new_game` config gives a ready Lightning Strike payment with nine offers.
  Rebuilding the SPA stops running `npm run preview` servers; restart them.
  For a fixture-free game, `managym academy play --deck <f> --opponent-deck
  <f> --opponent passive --seed 7` with `18 Mountain / 18 Forest / 12
  Badgermole Cub / 12 Lightning Bolt` against `60 Plains` reaches an
  all-sources cast (turn 3), a real source-choice prompt with triggered mana
  (turn 5, after earthbending the Mountain) and a Bolt target → payment flow
  (turn 7). Cub's earthbend trigger asks for a land target after it resolves.
  Each new browser context starts a new game from the same launch URL.
  **Open replay** is a link, not a button. Read the resume credentials from
  `sessionStorage['etude.gui.resume']` to probe the route from the page.
- Search (search-capable policy in Play/Watch): `uv run etude train
  manabot/recipes/local.py` gives a compatible
  `out/policies/<update>-raw.safetensors` in about 90 s; save `{"provider": "manabot",
  "artifact": "<abs path>", "config": {}}` as `<alias>.json` and launch `uv run managym
  academy --policy <alias>.json play --opponent <alias> --search
  '{"worlds":1,"depth":1}' --seed 7` (or `watch --player <alias> --player
  random --search '[{...},null]'`, then press Start watching). Seed 7 puts the
  bot on the play. The initial budget is 1×1 with a 30 s ceiling. With the
  provider's cached card catalog, full games at 1×1 (Watch, both seats) and
  4×2 (Play) completed with median decisions near 0.5 s and 1 s at load ~150;
  16×20 still exceeded 30 s. If one evaluation takes seconds, profile it:
  re-encoding the card catalog per forward (`_encode_programs`, `torch.gru`)
  means the session is not using `inference_scope()`. To time decisions
  without a browser, drive `create_watch`/`step_watch` in-process as
  `tests/academy/test_search_baseline.py` does. Retained evidence is in
  the attempts table's `trace_json`
  (`events[].bot_decision.mirror`). To probe reconnect, click an offer that hands
  priority to the bot and reload about 1.5 s later while the table shows
  `Search W×D· N s`; the game must stay `active` and resume at the
  committed state. A reload before the first response has no resume
  credentials and launches a new game by design.
- Live Academy QA without fixtures: build first, then start the ordinary
  `managym.academy.server:app` with `ACADEMY_TRACES_DIR` pointing into an ignored
  `.runs/` directory and `ACADEMY_PLAY_RECORD_ORIGIN=automated_validation`.
  The dev `.venv` has no uvicorn/fastapi. Start the API with
  `UV_PROJECT_ENVIRONMENT=<ignored .runs dir>/play-venv uv run --python 3.13
  --locked --only-group play-runtime uvicorn …` from the root, which imports the
  checkout's `managym`. Without `UV_PROJECT_ENVIRONMENT`, `--only-group` syncs the
  shared `.venv` down to that group. Saved games are rows in
  `<traces>/play.sqlite` `attempts.trace_json` (`config` omits default
  settings). In zsh, pass multi-word script arguments explicitly, because
  `set -- $m` does not split. After a deliberate float, if Pass is the only legal
  action, forced-decision skipping passes it and the mana expires. To see a pool
  in the table, float from one of two untapped sources.
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
- Checkpoint compatibility (loader and Academy listing): put the
  checkpoints in one directory, plus broken copies made with `torch.load` /
  `torch.save` after editing one `world_binding` field (`world_digest`,
  `learning_schema_sha256`, `naming_branching`). Start
  `MANABOT_RUNS_DIR=<dir> ACADEMY_FRONTEND_BUILD=… uv run managym academy
  --no-open --port <unused>` (bare `academy`; `watch` requires `--player`);
  `GET /api/opponents` lists each entry's `available` and `reason`, and the
  Seat 2 bot select's "Registered policy" shows them as disabled options.
  An unexplained-key copy: set `semantic_cards.record_token_keys[0]` in a v9
  `.pt` state dict to `(3 << 32) | 12345`. In the browser, select
  `[data-testid=opponent-select]` → `policy`, then the
  `[data-testid=checkpoint-select]` option, and press "Start game" (the setup
  form has no "New Game" button); the end dialog offers "Rematch".
  `uv run python -m manabot.backends.convert` must accept both a
  PyTorch zip and a container saved under a `.pt` name (`save_checkpoint`
  always writes a container); convert the pinned pilots and u1952, then load
  each `.safetensors`.
  `manabot sim` cannot seat `heuristic` as villain; seat the checkpoint twice.
- PR CI failures while the run is still in progress: `gh run view --log-failed`
  refuses; read `gh api --allow-escape-sequences
  repos/loopflowstudio/manabot/actions/jobs/<job-id>/logs` and grep `^FAILED`.
  Check the PR's checks before QA ends; tests elsewhere in the tree can still
  assert the behavior the change replaced.
- Training corpus and checkpoint ancestry: use a fresh ETUDE_HOME for
  `uv run --extra evaluation etude train manabot/recipes/local.py --config
  '{"statistics":true}' --until updates=1`. Resume with the same home and
  recipe, `--resume --until updates=2`, omitting the config. Use the guide
  in docs/evaluation.md on the resulting checkpoints and compare Island
  draws and choices with terminal records from
  `read_statistics(home / "track/local/out/statistics")` at capture update
  one less than the outcome update. For ancestry, train a one-update recipe
  in the same home with SelfPlay.pretrained set to the update-one policy in
  the parent's Etude checkpoint record. The child corpus should contain the
  parent's first update and the child's first update. Query historical
  frontier settings against the sealed run declaration as well as its counts.
  Run the guide as written: extract the four `python` blocks under
  "Querying the training corpus" with a regex and replace only the track
  root (and, to run every query on the file-form corpus, drop the three
  frontier `read` lines), so an import or query the guide no longer supports
  fails. The first block starts from a policy file,
  `<home>/track/<run>/out/policies/0000000N-raw.safetensors`; try each file a
  run wrote, a moved copy, a string path, a storage root without the run, and
  a file from an earlier pass's `.runs/` (older files record no experiment
  and take `experiment=`). For a consumer outside the
  checkout, build both wheels from the warm target
  (`uv run --extra play maturin build --release -i .venv/bin/python -m
  managym/Cargo.toml -o <dist>` and `uv build --wheel --package manabot -o
  <dist>`, under a minute), install the engine wheel plus
  `manabot[evaluation] @ file://<wheel>` in a `/tmp` venv with the CPU Torch
  index, and run the same script with `<venv>/bin/python -I` from `/tmp` on
  absolute storage paths. `<venv>/bin/etude train
  <site-packages>/manabot/recipes/local.py --until updates=1` trains there.
  An engine wheel built before a merge from main does not match the merged
  Python; rebuild it.
  `etude train <any file>.py` runs a recipe file from an ignored `.runs/`
  directory too, named by its stem: copy `manabot/recipes/local.py`'s `train`
  and `model_copy` the settings for a child run (`pretrained=<parent
  policies/0000000N-ema.safetensors>`, `match` with `decklists="closed"`), or
  rebuild `SelfPlay(**dump)` with `match=None`, a `UniformDecks(...,
  decklists="mixed")` deck source and `model.world = managym.WORLD_SETUP` for
  per-game disclosure. Runs written by an earlier commit of the same branch are
  returning data; read them again after a table rename.
- Collector changes: scripted environments in `tests/training`
  (`collector._env = ...` in `test_credit_assignment.py`,
  `test_two_seat_learning.py`, `test_game_errors.py`) stand in for the vector
  environment, so an attribute the collector newly reads fails there, not in
  the tests of the change. Run those files; CI does not run
  `test_game_errors.py`. Read a failed job while its run is still in progress
  with `gh api --allow-escape-sequences
  repos/<owner>/<repo>/actions/jobs/<job-id>/logs`.
- Decision memory is opt-in on raw engines: `managym.Env(...)` and
  `VectorEnv(...)` record no viewer decision log unless built with
  `decision_memory=True` (the training wrappers derive it from
  `max_decision_records > 0`). A script, fixture or product path that builds a
  raw engine and reads `decision_memory_json()`, decision-record buffers or the
  LLM text history sees an empty log otherwise, with no error. Grep new raw
  constructions when memory consumers change.
- Staged experiment launches (`etude launch ... --quote`): run each documented
  command verbatim under `doppler run -p etude -c prd -- doppler run -p manabot
  -c prd --preserve-env=RUNPOD_API_KEY --`; the locked client reads the launcher
  DB (2026-11-17). A quote prints the selected runs and the committed ceiling,
  then refuses or prices; confirm with `etude runs --experiment <name>` that
  nothing launched. `etude report` exits 1 until the experiment has a run.
- Decision memory in self-play (`ModelConfig.decision_memory`): the smallest
  real run is `tests/training/test_memory_self_play.py`'s `settings()`: Gray
  Ogre decks, the four `max_decision_*` capacities, and
  `SelfPlay.{collection,bootstrap,learner}_rows=2`. Train a
  `decision_memory="none"` control, then start a `"gru"` branch from its
  checkpoint with `SelfPlay.pretrained`; the branch's gates read 0.0 and its
  shared tensors equal the control's. A control without the log capacities is
  refused as a trunk. Test reliance by seating the same checkpoint with
  `config={"blank_decision_memory": True}` through `manabot.eval.pairing`.
- Files a returning user wrote before a change: `git worktree add --detach
  /private/tmp/<x> main`, build its release extension with its own
  `CARGO_TARGET_DIR` and copy the `.so` into that worktree's `managym/`. Then,
  from inside that worktree with `PYTHONPATH=/private/tmp/<x>`, call this
  checkout's `.venv/bin/etude train manabot/recipes/local.py --experiment demo
  --config '{"record_games": 1}' --until updates=1` and `.venv/bin/python -m
  manabot.eval.local ... --deal-seed` four times with `--seconds 45` (print
  `managym.__file__` first). That leaves a run to `--resume`, recorded games
  for `managym academy replay` and an incomplete panel for `--status` /
  `--continue`, all written by main. Copy untracked `experiments/data`
  fixtures into the worktree before comparing a test there. Earlier passes'
  `.runs/qa*/` hold older checkpoints, panels and `traces/play.sqlite` too.
- A checkout without `academy/ui/node_modules` needs `npm --prefix academy/ui
  ci` before the static build. In the board UI, start a policy game from the
  header's "New game" (selects, then "Start game") and click
  `[data-testid=action-option]:not([disabled])`; stop a `Popen` server with
  `os.killpg(pid, SIGINT)`.
- Once a CI run has finished, `gh run view <run> --job <job-id> --log-failed`
  gives the pytest summary. CI names its test files in
  `.github/workflows/manabot-integration.yml`; `tests/sim/test_puct_compatibility.py`,
  `tests/sim/test_conditional_search.py` and `tests/verify/test_learning_regression.py`
  are not among them, and the first two fail on main (2026-11-18).
- Input readings (`manabot/policy/readings.py`, `input_readings` in a
  checkpoint): the returning user is a checkpoint written before a change.
  Train `manabot/recipes/local.py` in a base worktree (its own release
  extension, this checkout's `.venv/bin/etude`, `PYTHONPATH` at the
  worktree), then play the same seeded games with `scripts/pinned_pilot.py`'s
  `play` on both trees (the v11 pilot's decks) and diff probabilities (381
  decisions matched within 3e-8 in MTG-404). Warm-start a `.runs/` recipe
  file with `pretrained=<that checkpoint>` and read the child's
  `input_readings`: they must equal the parent's legacy readings. A copy with
  one reading renamed must show in `/api/opponents` as unavailable, naming
  the column.
