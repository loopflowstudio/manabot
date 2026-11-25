Read when the change touches Academy (`managym/academy`, `academy/ui`): the table, replays, launch, recovery or browser scripts.

- For the default Playwright config, choose unused `ACADEMY_API_PORT` and
  `ACADEMY_FRONTEND_PORT`. The config reuses servers on the selected ports,
  even from another checkout. Set `VIRTUAL_ENV` to the checkout's `.venv`
  for the backend's `uv run --active`. Tab enters an unused board region on
  its first control with an offer. Use arrow keys to move within the region.
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
- Snapshot replays (`GameSnapshot.to_replay_json()`): run the `docs/engine-python.md#save-and-restore-game-positions` example from an empty directory, then build a longer snapshot replay on authored decks with `player_policies=("agent", "agent")`, a `shuffle_libraries` and a `determinize` restore mid-game; Academy refuses heuristic/random seats with `replay.rejected`. Open it with `managym academy replay <snapshot.json> --no-open --port <unused> --traces-dir <dir>` (needs `ACADEMY_FRONTEND_BUILD`), press End and compare "Decisions · N" with the replay's action count; collect WebSocket and JSON bodies and check no `"seed"` or intervention names reach the browser. Tamper `state_digest`, `seed`, `actions`, `card_digest` and `schema` in copies: each must refuse. `recording-heading` ("Recorded game") is visually hidden: wait for it with `state: 'attached'`. The final frame of an interrupted snapshot replay has no decision and no perspective control; step back with the "Previous decision" button before checking "Show both hands", and compare both seats' plate Hand counts. To simulate older rules, edit a mid-game action and re-sign `replay_digest` with BLAKE3 over the sorted compact JSON (see `checksummed_replay` in `tests/env/test_game_snapshot.py`): `from_replay_json` must refuse, `walk` must stop `diverged` at that action.
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
- Trained-policy Play/Watch/Search: train a checkpoint at this checkout
  (`uv run etude train manabot/recipes/local.py`, about 30 s) and register
  `{"provider": "manabot", "artifact": "<abs path>/out/policies/<update>-raw.safetensors", "config": {}}`
  with `managym academy --policy <file>.json play|watch`. A checkpoint trained
  before syncing main can pass the picker yet fail Start with "Failed to load
  the selected opponent." (catalog vocabulary grew); retrain rather than debug
  it. Watch resume tokens live in that tab's storage, so reopen a live Watch
  trace in the same browser context. A completed Play game's evidence opens
  from `/replay?trace=<id>` (ids from `/api/traces`) via "Inspect bot search".
  An Elemental recipe policy (`etude train manabot/recipes/elemental.py
  --config '{"updates": 2, "streams": 4}' --until updates=1`, about 10 s)
  binds the roster: launch it with `--deck elemental_ur --opponent-deck
  elemental_gw`, otherwise `setup.invalid` names the roster decks. Offer
  texts carry a number prefix (`1 Play Plains`), so match offers without `^`.
  X offers read `X = n`: `24 Plains / 16 United Front` against `40 Forest`
  with `--opponent passive` reaches one after four lands. Checkpoints with
  world or exact-deck bindings written before an appended input column load
  on their saved prefix.
- Record `uptime` when the Academy table stalls or Vite fails startup. MTG-320
  QA observed both at load averages around 300, but did not establish a cause.
  Repeat on an idle host before attributing a stall to the change or to load.
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
- A checkout without `academy/ui/node_modules` needs `npm --prefix academy/ui
  ci` before the static build. In the board UI, start a policy game from the
  header's "New game" (selects, then "Start game") and click
  `[data-testid=action-option]:not([disabled])`; stop a `Popen` server with
  `os.killpg(pid, SIGINT)`.
