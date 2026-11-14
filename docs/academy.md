# Academy

Academy is the browser UI for inspecting bots and replays. It ships inside the
`managym` distribution as `managym.academy`: the experience server, the built
Svelte client it serves, and the curated deck packs. The server owns
legality, hidden information and game state; the client renders viewer-safe
frames and returns Commands bound to the frame that offered them. Module-level
detail and demo walkthroughs live in [server guide](../managym/academy/README.md);
the wire contract is
[experience protocol v1](../protocol/experience-protocol-v1.md), with state
digests defined by [the state hash](engine.md#forks-replay-and-determinization).

- [Running Academy](#running-academy): install, launch, play, replay, trained bots, troubleshooting
- [Shared decision surface](#shared-decision-surface)
- [Choice navigation](#choice-navigation)
- [Presentation runtime](#presentation-runtime)
- [Playing a trained challenger](#playing-a-trained-challenger)
- [Attempts and shared history](#attempts-and-shared-history)
- [Experience proof gate](#experience-proof-gate)

## Running Academy

The PyPI commands below apply once a release that includes Academy is
published. To try an unpublished candidate, build it with
[Build and verify](development.md#build-and-verify) and install its engine
wheel with the extra, in a fresh environment:
`uv pip install '<path>/managym-<version>-<python>-<platform>.whl[academy]'`.

```bash
uv venv --python 3.13
uv pip install --python .venv/bin/python --only-binary :all: 'managym[academy]'
uv run --no-project managym academy
```

```text
ACADEMY_PLAY_READY {"traces_dir": "/home/you/academy/traces", "url": "http://127.0.0.1:8000"}
  Academy: http://127.0.0.1:8000
  Ctrl-C stops the server
```

Your browser opens on the table at `http://127.0.0.1:8000`, with a game ready
against the Search 64 opponent. That is the whole installation: no manabot,
Torch, Node or Rust, and no checkout of this repository. The wheel carries the
compiled engine, the server and the built client, and the server makes no
network requests while you play.

### Install

Academy needs ordinary CPython 3.12 or 3.13 on a platform with a `managym`
wheel ([supported platforms](engine-python.md)). Install into a virtual
environment with uv:

```bash
uv venv --python 3.13
uv pip install --python .venv/bin/python --only-binary :all: 'managym[academy]'
```

The `academy` extra adds the server's two dependencies, FastAPI and Uvicorn.
Keep the quotes: zsh treats unquoted square brackets as a pattern. Without the
extra, `import managym` works as before and `uv run --no-project managym academy` tells you what
to install.

### Launch

```bash
uv run --no-project managym academy                  # or: uv run --no-project python -m managym.academy
uv run --no-project managym academy --port 8001 --no-open --traces-dir ~/academy-games
```

| Option | Default | Meaning |
|---|---|---|
| `--port` | `8000` | Port to listen on |
| `--host` | `127.0.0.1` | Address to listen on |
| `--traces-dir` | `./academy/traces` | Where games are saved (also `ACADEMY_TRACES_DIR`) |
| `--no-open` | off | Do not open a browser window |

When the server is accepting connections it prints one `ACADEMY_PLAY_READY`
line, a JSON record with the `url` to open and the `traces_dir` in use, and
then opens that URL in your default browser. On a machine without a desktop
(SSH, a container) pass `--no-open` and reach the port yourself.

The default address is reachable only from your own machine. Academy has no
accounts: a browser's stored credential is its player identity. Listen on
another address (`--host 0.0.0.0`) only on a network where you would share
your games with everyone on it.

### Stop

Press Ctrl-C in the terminal that runs it. The server finishes its writes,
exits with status 0 and frees the port. A game in progress is kept: it is
listed as interrupted, with no winner, the next time you start Academy from
the same traces directory.

### Play a game

The table opens with UR Lessons (you) against GW Allies. Before starting:

- **Opponent** chooses who you face: **Search 16**, **Search 64** or
  **Search 256** (the engine's built-in search, with that many simulated
  playouts per decision), **Random**, or **Passive**, which always passes.
- **Choose decks** assigns a deck to each seat from the bundled packs.
- **New Game** starts the selected matchup. During a game it ends the current
  attempt and starts another.

To use your own decks, open **Choose decks**, expand **Import decklist files**
and check **Use imported decks for the next game**. Upload a UTF-8 `.txt` or
`.dek` file for both **Your deck** and **Opponent deck**, or paste each list
into its source box. Write one `count card name` per line, with optional
`Main` and `Sideboard` headings. Select **Check decklists**, correct any
reported lines, choose a built-in opponent such as **Search 64**, then select
**New Game**. Imported decks use the development format and supported cards;
this is not tournament legality validation. Checkpoint opponents are unavailable
for imported decks. Uncheck the import option to return to bundled decks.

The table reads **Who decides / What is happening / Your choices**. Clicking a
card inspects it and sends nothing; choosing one of the listed legal actions
commits it. Decisions with a single legal action are taken for you and never
shown. Reloading the page restores the pending decision.

Cards are drawn with Academy's own procedural treatments. The package contains
no card images and fetches none.

### Inspect a replay

Every game you start is recorded as you play.

- **Games** (`/games`) lists them, newest first, with filters for player, deck,
  status, result and date. **My games** narrows the list to this browser.
- Opening a game shows it in **Replay** (`/replay`): step forward and back
  through each recorded decision, with the board, the legal actions offered
  and the action taken. While the server that played the game is still
  running, **Try another line** plays a different legal action on an isolated
  copy and **Return to recorded decision** goes back; the recording is never
  changed.
- **Watch** (`/replay?watch=1`) plays two bots against each other, one decision
  per step, and saves the result to the same list. A managym-only install
  offers the Random and Search players.

Games are stored in `play.sqlite` under the traces directory, which defaults
to `academy/traces` in the directory you launched from. Launch from the same
directory, or pass the same `--traces-dir`, to see earlier games. The file
holds private deal seeds; keep it private.

### Watch a game from an evaluation

Evaluation trace import currently needs the optional manabot integration.
Install it into the same environment as Academy, then restart the server:

```bash
uv pip install --python .venv/bin/python --only-binary :all: --torch-backend cpu manabot
uv run --no-project managym academy
```

Use matching `managym` and `manabot` release versions. For an unpublished
candidate, install both wheels from the same release build instead.

1. Find the evaluation's retained command trace, typically
   `traces/<cell-id>.commands.jsonl.gz` under its output directory. Copy or
   download that artifact locally if the evaluation ran remotely. Use the
   command trace rather than the evaluation's metrics or summary file; see
   [automatic capture and replay](training.md#automatic-capture-and-replay)
   for the evaluation workflow.
2. Open **Watch** at `http://127.0.0.1:8000/replay?watch=1` and use
   **Open arena / Tournament game** to select the trace. Individual game JSON
   and uncompressed JSONL are also accepted.
3. If it contains several games, select a **Recorded game** and click
   **Open selected game**. A single game opens automatically.
4. Use **Step decision**, **Previous decision**, or **Play** to inspect the
   recorded game. The importer verifies the retained Commands and state
   digests without running the original players again. Historical policy
   outputs that were not recorded are marked unavailable.

If the file picker is absent, check that manabot is installed in the server's
environment and restart Academy. A trace bound to an incompatible world must
be opened with the revision that produced it; import does not upgrade frozen
evaluation evidence. Replaying does not require the original checkpoint.

### Trained opponents and other bots

Passive, Random and Search need only the engine. Everything that runs a
trained policy comes from manabot. Install it into the same environment and
restart Academy:

```bash
uv pip install manabot
```

manabot brings Torch with it; see
[training installation](development.md#install) to choose a CPU or CUDA build.
`managym` and `manabot` release together and must be the same version. Academy
finds manabot through its installed entry point, with no setting to change.
It adds:

- **Trained opponents** in the Opponent menu: every policy checkpoint under
  `MANABOT_RUNS_DIR` (default `./.runs`) and each `.pt` file, run directory or
  Tournament file named in `ACADEMY_PLAY_CHECKPOINTS` (`:`-separated). A
  checkpoint trained in another world or on other decks stays listed,
  disabled, with the reason.
- **Greedy** and **Heuristic** players in Watch.
- **Arena import**: verified arena and Tournament game files open as replays.

Without manabot the Opponent menu and Watch show a line saying trained bots
need it, and a saved game that was played against a checkpoint still replays;
only continuing it needs the checkpoint. The decision-advice panel reads
fixtures from a checkout of this repository and is absent from an installed
package.

### Troubleshooting

A launch that cannot start prints one `ACADEMY_PLAY_ERROR` line with a `code`
and a message, and exits with status 2.

| What you see | Cause and fix |
|---|---|
| `managym: command not found`, or `No module named managym.academy` | The environment holds a `managym` from before Academy was included or the command is running outside the virtual environment. Upgrade with `uv pip install --upgrade 'managym[academy]'`. |
| `ACADEMY_PLAY_ERROR` with code `extra.missing` | `managym` is installed without the server dependencies. Run `uv pip install 'managym[academy]'`. |
| `ACADEMY_PLAY_ERROR` with code `port.in_use` | Another program, often an earlier Academy, holds the port. Stop it, or run `uv run --no-project managym academy --port 8001`. |
| `ACADEMY_PLAY_ERROR` with code `site.missing` | This `managym` was built from source without the browser client, as in a checkout. Use `./scripts/play` from the checkout, or install a released wheel. |
| No browser window opens | Open the printed `url` yourself. On a remote machine, forward the port (`ssh -L 8000:127.0.0.1:8000 host`) and use `--no-open`. |
| Trained opponents are not listed | manabot is not installed in the environment that runs Academy; the Opponent menu says so. Install it there and restart. If the menu reports a catalog error instead, manabot is present but Torch failed to import. |
| A checkpoint is listed but disabled | It was trained in another world or for other decks; the row states the mismatch (for example `trained in w5, engine is w6`). Run it from the revision that wrote it. |
| Earlier games are missing from **Games** | Academy was launched from another directory. Launch from the original one or pass its `--traces-dir`. |
| `No matching distribution found for managym` | No wheel exists for this Python or platform. See [supported platforms](engine-python.md). |

### From a checkout

Working on Academy itself uses the development launcher, which needs uv, Rust
and Node and serves the client with hot reload:

```bash
./scripts/play                     # install locked dependencies, start backend + frontend; Ctrl-C stops both
./scripts/play --demo learn        # checked prefix into Learn, then ordinary play
uv run uvicorn managym.academy.server:app --port 8000   # development server without the launcher
uv run --extra dev pytest tests/academy/test_server.py -q
```

In a checkout the trained-bot features above are on, because manabot is
installed in the development environment. `uv run --no-project managym academy` there reports
`site.missing` unless `ACADEMY_FRONTEND_BUILD` names a static build of
`academy/ui` (`ACADEMY_STATIC_BUILD=1 npm --prefix academy/ui run build`); the
released wheel is the one place that build is packaged.

`npm --prefix academy/ui run fetch:card-art` populates the optional local
Scryfall art cache for checkout and hosted builds; play never contacts Scryfall
and missing art keeps the procedural card treatment. Clean-machine and release
checks are in [development.md](development.md).

## Shared decision surface

Academy makes the acting agent's
viewer-safe observations and complete legal action space understandable to a
human. Navigation, grouping and labels may clarify that shared contract; they
must not hide strategic choices, silently select actions, reveal unavailable
information or introduce human-only game semantics. Awkward decisions are
evaluated at the shared observation/action authority, not disguised in the
browser. The invariants that follow:

- **Command binding and stale rejection.** Every Command names the exact offer
  and revision that produced it; a stale or duplicate Command leaves state
  unchanged. Inspecting, filtering or highlighting sends nothing.
- **Pending decisions on reconnect.** A reload or reconnect restores the actual
  pending decision (including mid-payment and trigger-ordering choices) from a
  recovery envelope. A lost response does not undo a submitted choice; **Back**
  cannot refund committed costs or hide revealed information.
- **Forced decisions are never shown.** Academy uses native `skip_trivial`
  like every game runner: a sole legal action, for
  either seat, never reaches a person or policy.
- **Automation stays identifiable.** Priority-stop and F6 automation retain
  their Commands and appear as **Auto** in live and replay logs. Disabling
  **Use priority stops** in the Stops panel surfaces every priority window with a
  real choice. Traces from before native skipping keep their recorded forced
  Auto steps.
- **Replay and fork identity.** Live, Watch, Replay and Study use one decision
  renderer. Study executes one exact semantic offer on a retained isolated root
  (a full clone of the native game; the engine has no undo) and **Return to
  recorded decision** restores the original address, frame and Command. Expired
  roots stay unavailable rather than being reconstructed from the screen.
  Imported arena traces get their own Academy identity while retaining the
  source digest; frozen `erd1` evidence keeps its original bindings.

`/api/advice` serves fixture advice for the decision-advice surface; Academy has
no live belief runtime.

## Choice navigation

`ActionPanel` offers **Find an action** when the current list has at least
eight choices. It matches card/action label text case-insensitively; all
offers remain visible until a filter is entered, and the complete offer count
stays visible. **Clear filter** or Escape restores the list. Recovery, a new
update, board selection and Learn-mode navigation clear the filter. Selecting a
result sends its original offer through the revision-bound Command path;
duplicate labels remain separate choices, and focusing a result highlights its
exact board objects. Learn keeps its Take a Lesson / Discard and draw / Decline
interaction with previews and Back; costs and other small choices stay direct.

The eight-choice threshold and text filtering are reversible product
hypotheses. Human usability acceptance (ETU-77) remains open.

[managym/academy/fixtures/choice-navigation-w17-token-cards.json](../managym/academy/fixtures/choice-navigation-w17-token-cards.json)
pins actual GameSession positions (UR Lessons versus GW Allies) for five
families: `CHOOSE_TARGET`, `DISCARD`, `LEARN`, `DECLARE_BLOCKER` and
`PAY_OR_NOT`. `tests/academy/test_choice_navigation.py` reconstructs each
prefix and checks offers and stale rejection;
`academy/ui/e2e/choice-navigation.spec.ts` exercises them in the browser. To
play a retained position by hand, stop any `./scripts/play` instance and run
with unused ports:

```bash
ACADEMY_REVIEW_CHOICE=CHOOSE_TARGET ACADEMY_API_PORT=8147 ACADEMY_FRONTEND_PORT=5347 \
  ACADEMY_TRACES_DIR=/tmp/academy-choice-review ACADEMY_PLAY_RECORD_ORIGIN=automated_validation \
  npm --prefix academy/ui run test:e2e -- choice-navigation.spec.ts --headed -g 'live retained'
```

The test pauses with the decision ready; resume the inspector to finish. Use
`ACADEMY_REVIEW_CHOICE=DISCARD` for the cleanup discard. These attempts are
scripted prefixes and do not count as human completions.

## Presentation runtime

Presentation is theater over committed authority, never a source of game
facts. `managym/academy/presentation.py`'s match-local `PresentationProjector` stages
exact identities from server-authored offers (casting and targeting can resolve
in one engine step) and emits `PresentationEvent`s only when the engine's
committed `recent_events` confirm them: `cast`, `targeted`, `resolved`,
`damage`, `died` for spells, and `attack_group`, `blocked`, `damage`, `died`,
`turn_started` for combat and turns. It never diffs observations; a zone change
without the corresponding domain events emits nothing, and lethal damage yields
`died`, never the rules-distinct `destroyed`. The Rust and Python learning
encoders filter these presentation-only event types, so policy inputs are
unchanged.

Ordering on the live socket: sequence-gate the `FrameUpdate`, commit its
complete `ExperienceFrame`, validate every event against the update's base and
resulting revisions, then enqueue events into the `PresentationPlayer`. The
same event list is returned live, persisted on the transition's final trace
step and replayed through a separate instance of the same player; traces
without the field replay with no events rather than inferred ones.

Recovery commits the complete frame, cancels current theater and loads
`presentation_tail` from `presentation_cursor`. The server keeps a bounded
ledger of the latest 256 events (`MAX_PRESENTATION_EVENTS`), not a durable
checkpoint. A non-contiguous tail is rejected and a fresh envelope requested; a
cursor outside the retained window restarts theater at the oldest retained
event. Malformed presentation clears theater and reports an error without
touching the committed frame. Skip, Fast-forward, Finish, Pause, reduced motion
and unmount cannot alter or delay canonical state.

Labels come only from viewer-safe frames. The current projection certifies
incarnation zero and reuses the spell card's ID as the stack render ID;
references it cannot name stay unnamed.

**Recent consequences** (ETU-76, 2026-10-04) keeps the latest 12 semantic
events in an expandable, keyboard-scrollable list that cursor movement does not
erase; reduced motion removes movement without shortening reading time.

## Playing a trained challenger

### Ten-deck Elemental play

```sh
doppler run --project manabot --config prd -- ./scripts/play-elemental
./scripts/play-elemental --offline     # later runs, no credentials or network
```

`scripts/play_elemental.py` reads MTG-183's published checkpoints (default
segment `f3748bcabac8`; `--run` selects another), picks the highest recorded
training update, verifies SHA-256 and size, checks world admission and CPU
placement, and pins those bytes for the server's lifetime. It never trains or
swaps the opponent mid-match. `--prepare-only`
downloads and verifies; `--port`/`--frontend-port` override 18479/18480.
Checkpoint records live in `.runs/elemental-play/checkpoints/`, attempts in
`.runs/elemental-play/human-play/play.sqlite` marked `human_exploratory`.

The printed **Elemental deck browser** link assigns either seat from the ten
`elemental-world-v2` setups (mirrors allowed) with **Use for me** / **Use for
opponent**. **Choose decks** prepares the next matchup; only **New Game**
replaces the current attempt. On a scripted seed-0 game (2026-10-26), stop and
forced-step automation cut deliberate inputs from 289 to 73.

### Other checkpoints

Any trained checkpoint is selectable from the Opponent menu (`GET
/api/opponents`): everything under `MANABOT_RUNS_DIR` (default `./.runs`) plus
`.pt` files, run directories and Tournament JSON in `ACADEMY_PLAY_CHECKPOINTS`
(`:`-separated). manabot supplies the catalog
([trained opponents and other bots](#trained-opponents-and-other-bots)) and
reads them with Torch's weights-only
loader. A checkpoint bound to another world, content or deck setup stays listed,
disabled, with the mismatch, and the server refuses it with the same sentence.
The browser sends only a digest; the server re-hashes the bytes before play,
fails clearly if they were replaced, and records the identity as the trace's
`opponent_checkpoint`. An install without manabot lists no trained opponents
and says why.

To serve the same SPA/ASGI entrypoint as the hosted app:

```sh
npm --prefix academy/ui ci
ACADEMY_STATIC_BUILD=1 npm --prefix academy/ui run build
ACADEMY_FRONTEND_BUILD="$PWD/academy/ui/build" \
ACADEMY_TRACES_DIR="$PWD/.runs/human-play" \
ACADEMY_PLAY_RECORD_ORIGIN=human_exploratory \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
uv run uvicorn deploy.play.asgi:app --host 127.0.0.1 --port 8000
```

For a parallel preview, give it a fresh absolute `ACADEMY_FRONTEND_BUILD`, its
own `ACADEMY_TRACES_DIR` and port. Scored candidate evaluation on the selected
matchup (`uv run python -m manabot.eval.evaluate`) is described in
[evaluation.md](evaluation.md).

## Attempts and shared history

`managym/academy/attempts.py` stores attempts in SQLite at
`$ACADEMY_TRACES_DIR/play.sqlite` (default `academy/traces/`); new traces and
attempt metadata commit in one transaction, and legacy JSON traces are
read-only. Attempts are saved at start and at each surfaced revision.
Replacement, expiry, runtime error and authoritative completion have distinct
ending reasons; a disconnect leaves the attempt active, and after a server
restart it becomes `interrupted` with no invented winner.

Record version 2 tables: `attempts`, `attempt_players` (both seats: public
`player_id`, `kind` human/bot, name, deck, `version` = exact checkpoint digest
for bots), `players` (private browser credential hash to public human ID) and
`feedback` (note, author, optional validated decision address).
`attempts.participant_id` is private table authorization. `deal_seed` is
hexadecimal text to keep all 64 bits. The file holds private seeds and
canonical Commands: keep it private and query it with
`sqlite3 -readonly`, never a public endpoint. Do not run old and new server
processes against one database during a schema migration.

`ACADEMY_PLAY_RECORD_ORIGIN` declares provenance: `human_exploratory`,
`automated_validation`, `bot_evaluation` or `training` (default `unknown`). It
is operator-supplied, not detection; never run automated proofs against a
human-labelled server.

`/games` lists games across players or **My games**, filterable by name,
fingerprint, pairing, deck, status, result and date, with filters and cursors
in the URL. `GET /api/traces` takes `scope=all|mine`, `q`, `player`, `other`,
`pairing`, `version`, `deck`, `status`, `result`, `after`, `before`, `cursor`
and `limit` (1–100, default 50), newest first, with `X-Etude-Next-Cursor` for
the next page and `x-etude-player-token` for the browser player. Permissions
apply in SQL before filtering. Only authoritatively completed games are
replayable by non-participants and enable the post-game reveal; unfinished
replays and feedback remain participant-only. Identity is a browser credential
with no account recovery.

## Experience proof gate

A repeatable regression gate for UR Lessons (human) versus GW Allies (Search
64). It is not a parity or release-readiness claim. The single source of truth
for reference profile, sample counts and budgets is
[academy/ui/e2e/experience-proof-baseline.ts](../academy/ui/e2e/experience-proof-baseline.ts);
each run attaches its distribution as `experience-proof.json`.

```bash
npm --prefix academy/ui run test:e2e -- experience-proof.spec.ts
npm --prefix academy/ui run test:e2e:release   # focused decision, Learn, combat, navigation, Elemental checks
```

The proof measures warm launch, input latency, frame pacing and renderer heap
in headless Chromium; force-closes the WebSocket at a decision and requires
recovery with identical board, legal actions and log; and requires accessible
names on enabled buttons. It does not cover cold startup, process memory,
touch, real assistive technology, stale-Command handling or replay
equivalence. CI selects the release suite for browser code
and the backend/rules paths it consumes.

History: [choice navigation](https://github.com/loopflowstudio/manabot/blob/a74f31828/docs/choice-navigation.md),
[experience proof](https://github.com/loopflowstudio/manabot/blob/a74f31828/docs/experience-proof.md),
[local trained challenger](https://github.com/loopflowstudio/manabot/blob/a74f31828/docs/local-trained-challenger.md),
[presentation runtime](https://github.com/loopflowstudio/manabot/blob/a74f31828/docs/architecture/presentation-runtime.md).
