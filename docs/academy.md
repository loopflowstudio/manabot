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

- [Running Academy](#running-academy): install, launch, chosen games and replays, Python, training games, card art, trained bots, troubleshooting
- [Shared decision surface](#shared-decision-surface)
- [Using the laptop table](#using-the-laptop-table)
- [Play and Inspect modes](#play-and-inspect-modes)
- [Pending input, waits and recovery](#pending-input-waits-and-recovery)
- [Planning and confirming combat](#planning-and-confirming-combat)
- [Choosing priority stops](#choosing-priority-stops)
- [Playing a series against a model](#playing-a-series-against-a-model)
- [Reading consequences and controlling narration](#reading-consequences-and-controlling-narration)
- [Inspect a replay](#inspect-a-replay): timeline, live Watch, perspectives and recovery
- [Choice navigation](#choice-navigation)
- [Presentation runtime](#presentation-runtime)
- [Playing a trained challenger](#playing-a-trained-challenger)
- [Attempts and shared history](#attempts-and-shared-history)
- [Experience proof gate](#experience-proof-gate)

## One game, from setup to its recorded decisions

**New game** and **Games** are the main destinations. **Current game** returns
this tab to its last table or watched game. Existing launch, replay and Watch
links continue to open their intended games.

**New game** opens setup for a separate session, starting from retained choices.
Edits here do not save into your current session; **Cancel** returns without
committing them. Setup shows two native seats. Choose each controller, deck and optional
**Search**. One controller can be you; choosing bots in both seats starts a
watched game through the same setup. Seat 1 plays first; seat 2 draws first.
Policies are chosen directly by name and execution identity. Their provider,
artifacts and runner remain available in details. **Browse decks and cards**
opens the full deck browser; imported sources and disclosure use the same
admission checks for played and watched games.

Inspect a card through hover, focus or its **Inspect card** action without
changing how you play. The Play/Inspect click preference controls gestures;
it never gates reading. **Open a recorded game file** in Games imports a training game or evaluation
trace without changing its original evidence. Games links back to recorded positions, their legal
choices and available recorded policy/search evidence. Trying another line
creates a separate Study branch with an explicit return to the source decision.
There is no live example Advice panel or example-scenario read selector.

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
ACADEMY_PLAY_READY {"open": "http://127.0.0.1:8000", "traces_dir": "/home/you/academy/traces", "url": "http://127.0.0.1:8000"}
  Academy: http://127.0.0.1:8000
  Ctrl-C stops the server
```

Your browser opens on the table at `http://127.0.0.1:8000`, with a game ready
against the Search 64 opponent. That is the whole installation: no manabot,
Torch, Node or Rust, and no checkout of this repository. The wheel carries the
compiled engine, the server and the built client. Cards show their Scryfall
art, which the server fetches the first time a card is shown and keeps on
disk ([Card art](#card-art)); it makes no other network requests while you
play.

To skip the setup screens, name the game or the replay on the command line:

```bash
uv run --no-project managym academy play --deck ur_lessons --opponent-deck ./my-deck.txt --seed 7
uv run --no-project managym academy replay out/traces/<cell-id>.commands.jsonl.gz --game 3
```

The first opens the table already in that game; the second opens game 3 of an
evaluation's command trace at its first decision. See
[Start in a chosen game](#start-in-a-chosen-game),
[Open a replay directly](#open-a-replay-directly) and
[Launch from Python](#launch-from-python). To play against a model you
trained, or watch it play, see
[Play and watch a trained checkpoint](#play-and-watch-a-trained-checkpoint).
To look at a game your own training loop played, see
[Watch a game from your training loop](#watch-a-game-from-your-training-loop).

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
| `--policy` | none | Saved policy specification JSON; repeat for several. Its file stem is its launch alias. |

When the server is accepting connections it prints one `ACADEMY_PLAY_READY`
line, a JSON record with the server's `url`, the `traces_dir` in use and
`open`, the address the browser is sent to, and then opens that address in
your default browser. `open` equals `url` for a plain launch and points into
the game or replay for `play` and `replay`. On a machine without a desktop
(SSH, a container) pass `--no-open` and reach the port yourself.

These options go after `academy`, `academy play`, `academy watch` or
`academy replay` alike. Policy specifications name installed providers
([registering policies](#play-and-watch-a-trained-checkpoint)).

The default address is reachable only from your own machine. Academy has no
accounts: a browser's stored credential is its player identity. Listen on
another address (`--host 0.0.0.0`) only on a network where you would share
your games with everyone on it.

### Stop

Press Ctrl-C in the terminal that runs it. The server finishes its writes,
exits with status 0 and frees the port. A game in progress is kept: it is
listed as interrupted, with no winner, the next time you start Academy from
the same traces directory.

### Start in a chosen game

```bash
uv run --no-project managym academy play --deck ur_lessons --opponent-deck ./my-deck.txt \
    --opponent search --sims 64 --seed 7
```

```text
ACADEMY_PLAY_READY {"open": "http://127.0.0.1:8000/?launch=3f9c2a7be1d04c58", "traces_dir": "/home/you/academy/traces", "url": "http://127.0.0.1:8000"}
  Academy: http://127.0.0.1:8000/?launch=3f9c2a7be1d04c58
  Ctrl-C stops the server
```

The browser opens with that game dealt and your first decision pending. No
setup screen is shown, and nothing is chosen for you beyond what you named.

| Option | Default | Meaning |
|---|---|---|
| `--deck` | `ur_lessons` | Your deck: a bundled deck name or the path of a decklist file |
| `--opponent-deck` | `gw_allies` | The opponent's deck, named the same way |
| `--opponent` | `search` | `search`, `random`, `passive`, a registered policy alias, or `policy:<id>` |
| `--sims` | `64` | Simulated playouts per decision for `search` |
| `--seed` | random | Deal seed, an unsigned 64-bit integer; the same seed and decks give the same deal |
| `--decklists` | `open` | `open` or `closed`: whether each player knows the other's list |

A decklist file is the UTF-8 `.txt` or `.dek` format the table imports: one
`count card name` per line, with optional `Main` and `Sideboard` headings.
A bundled name always selects that deck, even if a file or directory has the
same name. Use an explicit path such as `./ur_lessons` to select that file.
Decks go through the same native admission as an imported deck in the browser
(the development format and supported cards, not tournament legality). When
either seat is a file, the game is recorded as an imported matchup: a bundled
deck on the other seat is admitted from its own list, and no curated asset
pack is attached.

The setup is checked before the browser opens. An unknown deck name, a
rejected decklist or an unusable combination prints one `ACADEMY_PLAY_ERROR`
line with the reason and exits with status 2.
Search needs open decklists. Every registered policy admits the selected decks
before loading; a baseline bound to specific decks refuses another matchup.
Missing or changed policy artifacts are refused with the provider's reason.

Reloading the page returns to the same game and its pending decision; it does
not deal again. To continue the series, use
[Rematch or Swap sides](#playing-a-series-against-a-model). **New game** opens a separate session setup with the launched decks, opponent
and decklist mode selected.
For imported matchups, setup shows
both selected source files and their editable lists. Uncheck **Use imported
decks for the next game** to choose bundled decks instead.

The launch address is good for as long as that server runs. Opened in another
browser it starts that browser's own game with the same decks and opponent.
With `--seed`, each opening deals the same game; without it, each opening
deals afresh.
After the server stops, the address cannot restore its live game. A tab that
already started it shows **Interrupted**; choose **New game** explicitly to
play again. Opening an unknown launch without a retained game shows ordinary
setup; run the command again to create a new launch.

### Open a replay directly

```bash
uv run --no-project managym academy replay out/traces/<cell-id>.commands.jsonl.gz --game 3
uv run --no-project managym academy replay jace.game.jsonl.gz
uv run --no-project managym academy replay <recording-id>
```

The first command opens one game of an evaluation's command trace; the second
opens one exported training game
([Watch a game from your training loop](#watch-a-game-from-your-training-loop));
the third reopens a game already in **Games**, by the id shown there. All land
in **Replay**, ready to step from the first recorded choice or resume a saved
position ([replay controls](#inspect-a-replay)). The heading names both players and
says which kind of game it is: **Training game**, one the training loop itself
played, or **Evaluation game**, one played afterwards to measure policies.

A recording id keeps the recording's existing access rules: open it in the
same browser and at the same address (including port) where **Games** lists
it, or open a shared, finished recording. Another browser or address cannot
open a private or unfinished game without its participant credential. The
launcher checks that the id exists; access is checked when the browser opens
it. Imported game files (evaluation traces and exported training games) are
shared, finished recordings and can open in any browser that can reach the server.

An evaluation trace or an exported training game needs manabot installed
beside Academy, at the same release version
([Watch a game from an evaluation](#watch-a-game-from-an-evaluation) has the
install command and where traces are written). Opening one verifies its
recorded Commands and state digests and saves the game to this traces
directory as a finished recording. It runs no policy and loads no checkpoint,
and it never changes the trace file. Opening the same game again returns to
the recording made the first time. `.json`, `.jsonl` and gzipped files are
accepted, `.game.jsonl.gz` among them. Recordings in **Games** open without
manabot.

A trace file usually holds many games. `--game` takes either the game's
zero-based position in the file (the order **Recorded game** lists in the
browser) or the first characters of its `game_trace_sha256`. A position in
the file takes precedence when both fit. A file with one game needs no
`--game`. Leave it off a larger file to see what the file holds:

```text
ACADEMY_PLAY_ERROR {"code": "replay.ambiguous", "message": "out/traces/c41.commands.jsonl.gz holds 40 games; select one by index or trace digest."}
  Select a game with --game <index-or-digest>.
  game  deal seed   leg  players                 trace
     0  9120443     0    w17-final vs heuristic  5be01c9a2f31
     1  9120443     1    heuristic vs w17-final  e07a44d1c8b2
     …
  20 of 40 shown
```

A trace recorded in another world is refused with `replay.rejected`, naming
the mismatch; open it with the revision that produced it. An exported training
game states its world, so the message names both worlds.

### Launch from Python

`managym.academy.launch` starts the same server from a script, a notebook or
a test, and returns the addresses instead of opening them unasked.

```python
from pathlib import Path

from managym.academy.launch import serve

with serve(port=0) as academy:
    game = academy.new_game(
        deck="ur_lessons",
        opponent_deck=Path("my-deck.txt"),
        opponent="search",
        sims=64,
        seed=7,
    )
    replay = academy.replay(Path("out/traces/c41.commands.jsonl.gz"), game=3)
    print(game.url)      # http://127.0.0.1:52114/?launch=3f9c2a7be1d04c58
    print(replay.url)    # http://127.0.0.1:52114/replay?watch=1&trace=8b1e…
    academy.open(game.url)
    academy.wait()       # block until Ctrl-C
```

- `serve(host="127.0.0.1", port=8000, traces_dir=None)`
  starts the server on a background thread and returns once it accepts
  connections. `port=0` takes a free port; `academy.url` and
  `academy.traces_dir` report what is in use. It opens no browser.
- `academy.new_game(...)` takes the options of `managym academy play`. A deck
  given as `str` is a bundled deck name; a `pathlib.Path` is a decklist file.
  With `seed`, each opening of its address deals the same game; without it,
  each opening deals afresh. Reloading in the same tab resumes its game either way.
  `opponent` also takes a policy specification JSON as a `pathlib.Path`.
- `academy.watch(players=("challenger", "random"))` opens Watch setup with
  exact seats selected and leaves **Start** to you. A string is a registered
  alias, `policy:<id>` or a built-in player; a `Path` names a saved specification.
- `serve(..., policies={"challenger": factory})` registers an already constructed
  policy factory directly. It needs no manifest or installed entry point.
  Each factory must create a fresh mutable session per game and seat.
- `serve(..., policies={"challenger": PolicySpec(...)})` loads an installed provider.
  `sources=[PolicySource("manabot", {"runs_dir": "runs/elemental-a"})]` lets
  that provider discover a run root. Sources belong to this server; no environment
  variables are rewritten. Each installed provider also discovers its ambient
  configuration once at server start, even when explicit sources are supplied.
- `academy.replay(source, game=None)` takes a trace `Path` with the same
  `game` selection as the command line, or a recording id as `str`. An integer
  `game` selects only a position; strings can also select a digest prefix.
- `launch.read_games(path)` lists a trace file's games as `RecordedGame`
  values (`index`, `deal_seed`, `leg`, `players`, `decks`,
  `game_trace_sha256`), for choosing one in code.
- `new_game`, `watch` and `replay` return a `Launched` whose `url` opens that
  game, setup or replay, subject to the recording access rules above. `academy.open(url)`
  opens it in the default browser; `academy.open()` opens the bare table.

Leaving the `with` block, or calling `academy.close()`, stops the server and
frees the port; unfinished games are kept as interrupted, as with Ctrl-C.
`academy.wait()` blocks until Ctrl-C and then returns, for scripts that
should stay up while someone plays. One Academy runs per Python process:
a second `serve` before the first is closed raises `server.running`.

Every failure is an `AcademyLaunchError` with a `code` and a `message`, the
codes listed under [Troubleshooting](#troubleshooting); nothing is opened
when one is raised.

### Watch a game from your training loop

Wrap the loop in `record_games`, pick one of the games it played, export it and
open it:

```python
from pathlib import Path

from manabot.sim.recording import record_games
from manabot.training.loops.minimal import train
from managym.academy.launch import serve

with record_games(Path("runs/jace/games"), players=("jace", "jace")) as games:
    train(8)                                  # the loop's own self-play games
game = games.sample(seed=0)                   # one complete game it played
path = game.export(Path("jace.game.jsonl.gz"))

with serve(port=0) as academy:
    academy.open(academy.replay(path).url)
    academy.wait()
```

The browser opens in **Replay**, headed **Training game** with `jace vs jace`.
A first opening starts at the first recorded choice. Every decision the two
seats made is there to step through, with the board and the legal actions each
was offered. The original policy is never rerun and no checkpoint is loaded,
so the file opens on a machine that has never seen the model.

The same two steps from the command line, on a directory recorded earlier:

```bash
uv run manabot export-game runs/jace/games --game 0 --out jace.game.jsonl.gz
uv run --no-project managym academy replay jace.game.jsonl.gz
```

The file also opens in the browser through **Open a game file**
under **Watch**. All three routes need manabot beside Academy, as an
evaluation trace does.

**Recording.** `record_games(directory, players=None)` records every game
played by collectors built inside the `with` block: the deal seed, both decks
and each decision's chosen action, one receipt file per game in `directory`.
It adds no decisions and changes none; the loop trains as it would without
it. The receipts can be read again later with `TrainingGames(source)`, where
`source` is a directory or a single receipt file. A `TrainingJob` run with `training_exposure` writes the same receipts under its
`training/exposure`; download that directory and open it the same way.

**Choosing a game.** `games.games()` lists the complete games as
`TrainingGame` values (`receipt_sha256`, `seed`, `stream`, `decks`,
`decisions`); `games.sample(seed=None)` picks one uniformly, the same one for
the same seed. A game the loop stopped in the middle of is counted and never
offered: a replay needs a finished game. `sample` raises when the directory
holds no complete game, saying how many were cut off.

**Exporting.** `game.export(path, players=None)` replays the recorded actions
through the engine, checks at every decision that the engine offers the number
of options the loop saw, and writes one `.game.jsonl.gz` holding the game, its
world and where it came from. The receipt is not changed. Export needs
manabot but loads no Torch and no policy. `manabot export-game <directory>
--out <path>` does the same; on a directory with several games, leave off
`--game` to list them:

```text
runs/jace/games holds 6 complete games (2 cut off); select one with --game <index-or-digest>.
  game  seed  stream  decisions  receipt
     0  3     0       41         9c1f07ab52e0
     1  3     1       57         03be77d1a94c
     …
```

`--game` takes the position in that list or the first characters of the
receipt digest, and `--players A B` names the seats.

**Who played.** The two names are labels you supply, first seat first,
because only your code knows which policy sat where. In one-model self-play
both seats are the learner, so both carry its name. When a loop seats a frozen
opponent or a league member, pass that game's names to `export(players=...)`.
Without names the heading shows the deck seats' own names.

**A training game is not an evaluation game.** An exported training game is a
game the collector played, with its exploration and whatever the policy was
at that update. To see how the current policy plays, run an evaluation and
open one of its games
([Watch a game from an evaluation](#watch-a-game-from-an-evaluation)); those
are labelled **Evaluation game**. Neither is ever relabelled as the other.

| What you see | Cause and fix |
|---|---|
| Export stops with "action not offered" or a different option count at a decision | The engine no longer plays this game as recorded: the receipt was edited, or it was recorded in another world or revision. Export from the revision that recorded it. |
| Export or `sample` reports only cut-off games | The loop stopped before any game ended. Let it collect at least one complete game. |
| `replay.rejected` when opening | The file's world is not the running one, or the file was edited after export; the message says which. Open it with the revision that exported it. |
| `replay.needs_manabot` | Install manabot beside Academy at the same version. |

The exported file holds the deal seed and so both players' hidden cards. Share
it as you would share both hands.

### Play a game

The table opens with UR Lessons (you) against GW Allies. Before starting:

- **Opponent** chooses who you face: **Search 16**, **Search 64** or
  **Search 256** (the engine's built-in search, with that many simulated
  playouts per decision), **Random**, or **Passive**, which always passes.
  It also offers **Registered policy**, which adds a **Policy** menu for
  policies registered with this server
  ([Play and watch a trained checkpoint](#play-and-watch-a-trained-checkpoint)).
- **New game** assigns a deck to each seat from the bundled packs.
- **New Game** starts a new session with the selected matchup. Replacing an
  active game requires confirmation that it will be stopped. To keep games
  together, use [Rematch, Swap sides or Change opponent](#playing-a-series-against-a-model).

To use your own decks, open **New game**, expand **Import decklist files**
and check **Use imported decks for the next game**. Upload a UTF-8 `.txt` or
`.dek` file for both **Your deck** and **Opponent deck**, or paste each list
into its source box. Write one `count card name` per line, with optional
`Main` and `Sideboard` headings. Select **Check decklists**, correct any
reported lines, choose a built-in opponent such as **Search 64**, then select
**New Game**. Imported decks use the development format and supported cards;
this is not tournament legality validation. Registered policies check the complete
setup, including imported lists, against their own admission requirements. Uncheck
the import option to return to bundled decks.
`managym academy play` starts the same game from the command line
([Start in a chosen game](#start-in-a-chosen-game)).

The table reads **Who decides / What is happening / Your choices**. It starts in
**Inspect** mode: clicking a card inspects it and sends nothing; choosing one of
the listed legal actions commits it. Combat controls add an editable
plan and explicit **Confirm** step ([planning combat](#planning-and-confirming-combat)).
**Play** adds card selection and a hand-drag
shortcut ([controls below](#play-and-inspect-modes)). Decisions with a single
legal action are taken for you and never shown. Reloading the page restores the
pending decision.

### Playing a series against a model

Choose **Registered policy**, select the intended policy, and import your decks
through **New game**. Check the matchup, decklist disclosure, priority stops
and deal mode, then select **New Game**. After the game finishes, select
**Rematch** to play again with that setup. After the next game, select
**Swap sides**: you keep your deck and move from playing first to drawing first,
or the reverse. Both games and the next one belong to the same session.

A trained or independent policy must first be registered with this Academy
server; see [Play and watch a trained checkpoint](#play-and-watch-a-trained-checkpoint).
The provider checks that it can play the selected decks and configuration.
Built-in opponents need no registration.

The session strip at the table shows the game number, matchup, your play/draw
seat and deal mode. Open its configuration details with a pointer or Tab and
Enter to check the full Policy ID, model artifacts, runner and available search
configuration. A familiar model name alone does not identify the execution.
Each game's recording keeps the exact configuration used to start it, including
its admitted decks, disclosure, actual seed and native seats.

| Next action | What carries forward |
|---|---|
| **Rematch** | Both decks and imported source lists, exact opponent configuration, disclosure, stops and your seat |
| **Swap sides** | The rematch setup, with your native play/draw seat reversed; your deck stays yours |
| **Change opponent** | Your decks, disclosure, stops and seat/deal plan; only the explicitly edited opponent settings change |
| **New Game** | Starts a separate session from setup |

Your preferred **Play** or **Inspect** mode stays selected across games. To
exchange decks within the series, choose **Edit next game** in the session strip.
**Change opponent** opens only the opponent settings. Both editors show that
edits belong to this session's next game. **Save next setup** retains the draft
without starting or admitting a game; **Start next game** admits it and starts
one continuation in the same session. **Cancel** discards unsaved edits.
Earlier recordings keep their original setup. Stop edits during play also carry
into the next game.

If both controllers become bots in **Edit next game**, choose **Start new
watched game**. This creates a separate Watch game and preserves the Play
session and current attempt. **Current game** then returns to that watched game;
reading another session's history does not change that destination.
Admission refusal keeps the editor, edits, current game and saved draft usable.
Reload discards unsaved edits and restores committed setup and the actual
pending game; it never silently starts another game. Replacing an unfinished
game warns that it will be stopped; inspecting setup or opening a recording does not stop it.

#### Choose the deal deliberately

**Fresh deal** is the default when no seed was supplied. Each successfully
started game gets a new random seed, recorded with that game. **Repeat seed**
keeps an explicitly supplied seed, including one supplied with CLI `--seed` or
Python `seed=`. Enter a seed as a decimal unsigned 64-bit integer, from `0` to
`18446744073709551615`. The selected mode is visible before starting.

Rematching or swapping sides preserves this mode. Repeating a seed does not
repeat policy decisions, and swapping native seats changes which player gets
each seat's deal. These controls do not establish a paired evaluation.

#### Read results and return from a recording

Open session history from the table to see its games in **Games**. Each entry
links to its recording. Use Replay's return link to return to the filtered
history, then return to Play in the same tab to continue the current game.
Opening an earlier recording does not replace the live game.

Results count wins, draws and losses from your seat only among completed games,
with the completed-game denominator. Active, stopped and interrupted games are
counted separately. For example, two wins, one loss and one stopped game means
**2 wins / 0 draws / 1 loss out of 3 completed**, plus **1 stopped**. Counts cover
the full authorized session selection, including games on other history pages.

Breakdowns keep exact policy/search configuration, matchup, disclosure, native
seat and evidence origin distinct. Changing opponent stays in the session and
creates a separate breakdown. Human exploratory play, automated validation,
training and evaluation recordings retain their declared origins; unknown
origin stays unknown. An evaluation label alone does not establish a controlled
study. These counts describe recorded games, not a strength estimate or rating.

Each collapsed breakdown names its recorded opponent and policy identity, where
available. Expand it for decks and the full execution configuration.

Wall elapsed time includes pauses and appears only when start/end times were
recorded. Response measurements appear only when recorded by the provider,
with their measured sample count; otherwise they are unavailable.

#### Reloads, failures and privacy

Reload in the same tab to recover the session and its actual pending decision.
Independent tabs and launches start independently. Opening session history in
another tab is read access and does not resume that session's live game. Keep
using the same browser address, including its port. If browser session storage
is unavailable, Academy explains that reload continuity is unavailable; the
current in-memory game remains usable.

If the next policy or decks cannot be admitted or loaded, Academy shows the
reason and preserves the existing game and setup for correction and retry. A
missing or changed policy is never silently replaced. Restore its pinned
artifacts, or deliberately select the replacement offered by the picker.
Retrying a continuation after a lost reply recovers the game already started
by that request. After a backend restart, the unfinished game is interrupted;
start the next game explicitly rather than expecting its live state to resume.

Session setup, imported source lists and seeds remain owner-private. Sharing a
completed recording follows the existing replay rules; it grants neither session
configuration access nor control of the live table. Earlier ungrouped games stay
in **Games** without being assigned to a session retroactively.

### Using the laptop table

Start a game, inspect a land in your hand, then choose its play action in
**Your choices**. The board stays in view as the next prompt appears. Inspecting
the card sends nothing; selecting the legal action commits the play. If a spell
needs a target or payment choice, follow the next prompt before taking another
action.

At laptop widths and heights, the board and hand sit beside the current
decision. Scroll the legal choices while the current instruction, legal count,
**Show all** and filter controls remain available. Long instructions scroll
independently. Open **History and settings** for decision details, recorded
consequences and stop configuration without losing the pending decision. The current
priority-stop automation setting and **Edit stops** remain visible beside the choices. At narrow or short windows, board and
choices stack; use the board/choices jump links to move between them. Changing decisions
moves focus within the choices region without pulling the page away from the
board.

- **A crowded hand:** scroll the labeled hand region horizontally to reach
  the remaining cards.
- **A crowded battlefield:** use the previous/next controls beside each lane heading. Counts show
  the size of each group; focusing an action reveals the page holding its
  subject. Occupied lanes come first; empty groups retain compact counts. Scroll the labeled battlefield region to reach further groups and
  secondary zones. Identical card copies keep their own choices.
- **Other zones:** open their inspection controls to see viewer-permitted
  contents. Empty zones keep their labels and counts; hidden cards remain
  hidden.
- **Card details:** hover or keyboard-focus a visible card to preview it, or
  use **Inspect card** in Play; in Inspect, activate the card to open its details
  and related choices. Inspection leaves the pending decision unchanged.
  Use **Show all** to return from card-specific
  choices to the complete list.
- **Keyboard navigation:** Tab reaches the board, choices and inspection
  controls, including labeled scrolling regions. Focus a region to scroll it
  with the keyboard; activate a listed action only when ready to commit it.

The total legal choice count remains visible when narrowing a list. If a text
filter finds nothing, use **Clear filter** or Escape; neither chooses an action.
See [Choice navigation](#choice-navigation) for searching larger lists.

While disconnected or waiting for a submitted choice, the board remains
readable and action submission is disabled. Reconnect or reload in the same
tab to recover the actual pending decision. If a choice is rejected because
the game has advanced, inspect the refreshed prompt and choose again. A
submitted action cannot be cancelled by clearing inspection or going back.
Reloading a configured launch in its tab resumes that game while its server
session remains available; a separate tab or another launch has its own game.
See [Pending input, waits and recovery](#pending-input-waits-and-recovery) if
recovery fails.

For bot inspection, open **Watch** or a recording in **Replay**. The shared
decision surface shows the acting seat's permitted view and the complete legal
choices for that decision. Recorded policy outputs appear only when available;
missing outputs are not reconstructed. See [Inspect a replay](#inspect-a-replay).

### Play and Inspect modes

In a live game, choose **Play**, click a land in your hand, then activate its
labeled play action. The land enters play and the next decision appears on the
same board. For a spell, read the cast label before committing, then follow any
target or payment prompts. Selecting the card alone sends no action. Keyboard
users can focus the card, press Enter to select it, then activate the labeled
action control.

For a hand card with exactly one current play or cast action, you can also drag
it into the labeled play area. Move at least 8 CSS pixels to start the drag;
read the action label and release inside the area to commit. Escape, pointer
cancellation or releasing outside the area discards the drag. If the card has
several actions, choose an explicit alternative from the list. A card with no
current action remains available for inspection.

Hover or focus previews the card; **Inspect card** opens its details without
committing. Player targets have explicit named controls. Combat
controls let you edit an attack or block plan, then confirm the declaration
([planning combat](#planning-and-confirming-combat)). Mana-source choices follow the offered decisions;
dragging a spell does not configure manual tapping.

Choose **Inspect** to examine the acting viewer's observation and every legal
offer. Card selection narrows navigation; **Show all** restores the full list.
Both modes retain the legal count and complete choices. Switching modes preserves
a combat draft for the same pending decision; other uncommitted card selections
are cleared. It sends no game action and leaves priority automation at
its configured setting. Inspect is the initial default; an explicit mode
preference is remembered in this browser.

A new decision, a stop-setting change or connection recovery clears uncommitted
selection and drags. Wait for the current prompt, then select again. Actions are
unavailable while disconnected, awaiting a submitted action or changing settings.
These live-play shortcuts do not submit moves in Watch, recorded Replay or Study;
use their own decision controls. Cancellation only discards local selection:
after commitment, costs and revealed information remain committed.

### Pending input, waits and recovery

Cast a spell, then follow its target and payment prompts. If the bot's next
decision fails, your submitted choice stays committed and the board stays
visible. Read the failure, then choose **Retry bot decision**, or **Retry search**
for a search failure, when offered. Retrying asks the same bot to finish its
pending decision; it does not cast your spell again.

**Who chooses.** Read the chooser and requested input beside the current prompt.
The player choosing can differ from the player whose turn it is. Target and
cost prompts identify their source and required input. Focus or inspect a
visible card to discover its current cast or attack restriction; payment
guidance explains when the complete cost cannot be paid. A reason describes
the current decision and may change after you act. Finish a pending target or
cost choice before trying another spell. Inspection and **Show all** keep the
complete legal choices accessible. Older recordings without explanations say
they were not recorded.

**What is taking time.** Live status distinguishes loading a bot, choosing its
action, searching and advancing the live game. When a wait lasts more than a
second, elapsed time shows how long that operation has been running; it is
not a completion estimate. Recorded playback shows saved choices and
consequences without running a bot. Pausing narration affects presentation
only. **Disconnected** or **Reconnecting** takes precedence when the connection
cannot confirm the live state. The last board remains readable, with submission
disabled until recovery confirms the current decision.

**Retrying a bot.** An unsuccessful bot proposal leaves its native decision
unchanged. Retry is explicit; reloading, reconnecting or changing priority stops
does not retry it or select a replacement action. Costs already committed by
earlier choices remain spent. If the bot cannot safely continue, Academy
explains why retry is unavailable; start another game explicitly. A bot-load
failure leaves setup available to correct the selection and try again.

**Recovering a submitted choice.** If the connection drops just after you act,
reconnect or reload in the same tab and wait for recovery before choosing again.
Academy keeps the unresolved submission with that tab's game identity and
checks whether it was accepted. If it must resend, it resends that exact
submission so an accepted choice is not applied twice. If the decision or your
permission changed, Academy reports the stale or unknown outcome and refreshes
the table. Going back or clearing a selection cannot cancel a submission.
In live Watch, recovery checks the recorded position before another step;
do not repeat a step just because its reply was lost.

**When recovery is impossible.** A server restart or an expired live session
shows **Interrupted**, retaining the last known table and a recording link
when available. Saved recordings preserve available evidence; they cannot
restore the live game. Choose **New game** explicitly to return to setup.
Academy never silently deals a replacement or replays your submissions into
a new game. Keep the original tab open for temporary connection failures:
resume credentials and unresolved input belong to that tab's session storage.

Back, Escape and clear controls discard only unsubmitted local selections.
Once an action commits, its costs and revealed information remain committed.
Cancellation is possible only when the current legal choices explicitly offer
it; there is no general browser undo.
### Planning and confirming combat

At your attack declaration, choose **Play**, select an eligible creature on
the board, then its named player destination. Repeat for another creature and
read their assignments in **Your choices**. Each assignment names
the exact creature and the opposing player it will attack. Remove one creature,
inspect the other, then choose **Confirm**. Only the confirmed declaration makes
the creature attack and taps it if required. Editing or inspecting the plan
spends nothing and leaves the game at the same decision.

The board stays in view while you plan. Drafted creatures carry a **Planned attack**
or **Planned block** marker, also included in their accessible names. Clearing an
assignment removes its marker; changing focus leaves the other markers in place.
The summary lists selected creatures, their destinations, counts and current power
totals. Power is the creatures'
current power, not a prediction of damage or the opponent's remaining life.
Copies with the same name remain separate selectable objects.

- **Attack:** select a creature and its named attack destination using the
  board controls or the labeled choices. The current game offers the opposing
  player as the destination. Remove a selection to leave that creature out.
  **Select all eligible** adds the eligible attackers to the draft; review the
  result before confirming.
- **Block:** select a blocker, then the attacker it should block. Use the same
  controls to reassign it to another offered attacker or clear its assignment.
  The summary keeps all blocker-to-attacker links visible together. Academy
  does not choose blocker assignments for you.
- **Clear:** removes the editable assignments so you can start again. It does
  not submit an empty declaration. To attack or block with no creatures, clear
  the plan and explicitly confirm it, if the rules permit that declaration.
- **Inspect:** hover or focus a card for its preview, or open **Inspect card**
  for details. Switching to **Inspect** lets you examine the acting viewer's
  observation and complete legal choices without changing the draft. Return
  to **Play** to continue editing the same pending declaration.
- **Keyboard:** Tab to the labeled creature and destination or assignment
  controls, then activate them with Enter or Space. The same controls support
  pointer input; dragging is not required. Tab to **Confirm** when ready.

The rules engine checks the whole declaration, including restrictions involving
several creatures. For example, assigning only one blocker to an attacker with
menace leaves an invalid draft visible with an explanation. Add another eligible
blocker or change the plan before confirming. Invalid plans open the native combat
requirements for review; these describe the board’s restrictions rather than
claiming to identify which restriction the draft violated. Open **Rules response**
to read the engine's rejection. A creature's presence among the available choices
does not mean every combination is legal. Academy never drops
an invalid assignment or substitutes another plan to make confirmation succeed.

**Confirm** waits for validation of the current draft and checks it again when
submitted. While validation or submission is pending, wait for its result.
A correctable rules rejection keeps your draft for editing and commits none of
it. If validation reports a capacity limit, it cannot establish legality;
confirmation remains unavailable. Invalid and capacity results keep your draft
editable; retrying an unchanged plan does not resolve those results. **Retry
validation** is offered when a validation request could not be sent.
A connection or submission failure is shown separately from a rules rejection. Recover the connection and read the current
declaration before trying again; a lost confirmation reply does not mean the
declaration was cancelled.

If the game has advanced, Academy discards the outdated draft and shows the
current decision. Reloading or reconnecting also discards unconfirmed local
plans, with an explanation, and restores the game's actual pending and committed
choices. If part of a declaration was already committed through individual
decisions, those assignments remain visible and fixed; only the remaining part
can be edited. **Clear** cannot remove committed assignments, untap creatures
or undo costs and revealed information.

A confirmed declaration is recorded as its exact individual game decisions.
Replay can therefore show several decisions from one confirmation. Those are
the choices that were committed, not additional selections made by the browser.
Confirmation also lets forced consequences run normally. If no further choice
is needed, the returned board may already show combat damage or a later phase;
confirmation does not create another pause at the end of the declaration.
Watch, recorded Replay and Study keep their own decision controls; the live
combat plan does not submit actions there.

### Choosing priority stops

To hold up a response during the opponent's upkeep, choose **Edit stops**, keep
**Use priority stops** checked and check **Upkeep** under **Opponent turn**.
Academy pauses at a priority window there when you have a real choice. The
automation summary beside the choices keeps the current setting visible.

**Your turn** and **Opponent turn** say whose turn contains the window. A checked
step stops automatic priority passing there; an unchecked step allows it unless
the stack-response setting requires a stop. The available steps are upkeep,
draw, first main, beginning of combat, declare attackers, declare blockers,
combat damage, second main and end step.

| Setting | Default and effect |
|---|---|
| **Use priority stops** | On. Automatically passes priority outside configured stops. Turn it off to see every priority window with a real choice. |
| **Your turn** | Stops in both main phases. |
| **Opponent turn** | Stops in the end step. |
| **Stop when the stack offers a response choice** | On. Stops for a response when the stack is nonempty, even at an unchecked step. |
| **Reset to defaults** | Restores these settings. |

Changing a stop can immediately advance the game. Target, payment, Learn,
attack and block decisions still appear whenever they offer a choice; stops
only govern priority passing. Sole legal actions are skipped in either mode.
Settings are remembered in this browser, and reconnect restores the active
game's server settings.

When **Pass Turn (F6)** is available, activating it yields through priority
windows, including checked stops and stack responses, until the turn ends or
a non-priority decision needs you. With **Use priority stops** off, priority
windows continue to appear. Automatic passes retain **Auto** labels in the
action log and recording. Switching Play/Inspect or pausing narration does not
change these settings.

### Reading consequences and controlling narration

After casting a targeted damage spell, finish its target and payment prompts,
then open **History and settings**. Read the recorded cast, target, resolution
and damage consequences as they occur; if a creature dies, its recorded death
appears too. The board shows the current position, and **Your choices** remains
available while narration describes what already happened.

History groups consequences under recorded turns. Use the earlier/later page
controls to browse up to 100 events at a time; the counts describe the available
history. Scroll back to read an earlier event: incoming events leave your reading
position and keyboard focus in place and increase the unread count. **Latest**
returns to the newest page and resumes following events while you stay near the
bottom. Browsing an older page also stops following. Recovered older events also count
as unread and leave the event you are reading at the same position. Unavailable
intervals appear where they occur; the turn after a gap stays unknown until a
recorded turn change. Seeking before the event you are reading moves to the
nearest earlier available event (or the first available event). A removed
inspection closes and returns focus to history. Switching match, viewer or Study
scope starts at Latest with a fresh unread count.

When a consequence offers inspection of a card still available in your permitted
view, select that subject to inspect the exact object without making a move.
Departed or unavailable subjects keep their recorded text; another copy with the
same name is not a substitute. Tab reaches history controls, turn headings and
inspectable subjects; focus the scrolling region to read with the keyboard.

Narration occupies the history area, leaving cards and choices accessible.
Its controls affect presentation only:

| Control | Result |
|---|---|
| **Pause narration** / **Resume narration** | Pause or resume the current narration; live play can continue. |
| **Fast-forward** | Play narration at 4× speed; toggle again for normal speed. |
| **Skip beat** | Move past the current narration beat. |
| **Finish** | Finish the queued narration immediately. |
| System reduced-motion preference | Remove movement while keeping consequence text and reading controls. |

During live play, narration catches up automatically when its backlog grows;
skipped motion keeps its history entries. Pausing narration does not pause the
match or its configured priority-stop automation. Submitted actions, costs and
revealed information are unaffected by narration controls.

Consequences describe recorded game events. The action log separately identifies
submitted choices and **Auto** passes; observed-state summaries describe changes
visible in a frame. An action or a summary alone is not evidence that a spell
resolved or a creature died.

Reconnect preserves available history already held in this page and avoids
repeating seen motion. A fresh reload can recover only the latest 256 events of
an active match; earlier unavailable history and gaps are labeled explicitly.
An unknown earlier turn stays unknown. To review the saved consequences of a
completed game, open it from **Games** in **Replay**. History there follows the
selected timeline position: stepping back and forward does not duplicate rows.
**Try another line** has separate history, and **Return to recorded decision**
restores the recording's history. Replay pages the selected perspective's
recorded history; a choice shows only events preceding that choice.
Spectator reveal does not add names to acting-view history.
Older recordings without event data say **Consequences not recorded**; missing
events do not mean that nothing happened. A presentation error leaves the
current board usable and marks retained history incomplete.

### Card art

Cards on the table, in replays and in the deck browser show their Scryfall
art. There is nothing to run first: the first time a card is shown, the
Academy server fetches its image and artist credit from Scryfall, saves them
and serves them to the browser itself. Later views, and later launches, read
the saved copy. The browser never contacts Scryfall.

| Setting | Default | Meaning |
|---|---|---|
| `ACADEMY_CARD_ART` | on | `off` serves no art and makes no network request |
| `ACADEMY_CARD_ART_DIR` | `$XDG_CACHE_HOME/managym/card-art`, else `~/.cache/managym/card-art` | Where images and credits are kept |

```bash
ACADEMY_CARD_ART=off uv run --no-project managym academy
```

A card is drawn with Academy's own procedural treatment until its image
arrives, and keeps it when there is none: with art off, without a network
connection, and for the few objects that have no printing (back faces and one
token). A failed fetch is tried again after ten minutes. The first view of a
new deck is therefore slower than later ones; requests to Scryfall are spaced
100 ms apart.

Art is requested only for cards your view already shows. An opponent's closed
decklist causes no request, so the cache and its timing say nothing about
cards you have not seen.

**Card art credits** in the page footer lists the artist, source and copyright
of every image shown so far. The package itself contains no card images, and
deleting the cache directory is safe.

### Inspect a replay

Open **Games**, choose a finished game, then use **Next** to move from a recorded
choice to its consequences. The board shows the recorded state after the action
while narration follows its events. Keep stepping to reach the final consequences
and result board, including the last attack or spell. Human games, watched bot
games and imported training or evaluation games use these same controls.

Every game you start is recorded as you play. **Games** (`/games`) lists them,
newest first, with filters for player, deck, status, result and date. **My games**
narrows the list to this browser. A trained bot appears under its
[short name and digest](#how-a-trained-opponent-is-named). Open an evaluation
trace or exported training game through **Watch → Open a game file**, or
[launch a replay directly](#open-a-replay-directly).

#### Follow the timeline

| Position | What appears |
|---|---|
| Choice | The permitted recorded board, legal offers and action taken, with the chooser identified and policy/search evidence if recorded. History stops before the choice. |
| Consequences | The recorded post-action board and ordered event groups. Narration emphasizes events on that board; it does not reconstruct an intermediate board for each beat. An action with no recorded events still has a post-action position. |
| Result or latest state | The saved final board and outcome, or the latest retained state of an unfinished recording. An interrupted recording has no invented winner. |

**Previous** and **Next** move through positions. Select a score line, use the
labeled scrubber, or jump to a recorded turn and phase to seek; seeking pauses
playback. **Play** follows the recorded consequences before moving on, and
**Pause** holds your place. Choose 1×, 2×, 4× or 16× playback speed. The system
reduced-motion preference removes movement while keeping the ordered facts.
Use Tab to reach these controls, Enter or Space to activate buttons, and arrow
keys on the focused scrubber to seek. Shortcuts do not intercept text editing.

The replay address retains your selected position, information perspective and
seat orientation. Reloading restores them, with a saved position in this tab as
a fallback. A position that is invalid or no longer available is explained so
you can select an available one. Opening a recording still requires its normal
access permissions; copying an address does not copy a participant credential.

#### Keep your place and perspective

Seat 0 stays near you by default, even when the chooser changes. **Swap seats**
changes the table's orientation. The chooser and information perspective are
labeled separately: swapping seats does not reveal a hand or change the recorded
policy inputs.

At bot choices, the acting view shows what that bot was allowed to see.
**Show both hands** is available only where spectator reveal is authorized;
it does not enrich the bot's recorded evidence or acting-view history.
Human/shared replays keep their recorded seat-0 information boundary.
Consequences and result boards use the recorded seat-0 view when a private
post-action view for the actor was not captured. Read the perspective label;
unrecorded views are disabled with an explanation.

Scroll back in history to read without incoming events pulling you to the
bottom; **Latest** resumes following. Earlier/later history pages keep long
recordings manageable. Changing information perspective starts a separate
history-reading scope; swapping physical seats keeps your place. Card inspection
stays open only while that exact object remains visible and permitted. If it
becomes unavailable, inspection closes with an explanation and focus returns
to the control that opened it. A later copy with the same name cannot supply
missing historical details. Older snapshots that cannot certify the same object
appearance also close inspection.

Opening card details or reading earlier history pauses replay. Close the details
when ready, then press **Play** to resume. Returning to **Latest** does not start
playback. A live decision already being produced may finish saving while you read;
your selected board stays put. A pending mirror-search timeout is
retained across reload and shown with its message. At the live edge, **Retry
search** explicitly retries that decision; browsing saved positions never retries
it. The exact selected position remains selected after reload.

#### Continue a live Watch

Open **Watch** (`/replay?watch=1`), select both seats, decks and a seed, then
press **Start watching**. Each seat uses the same player menu as Play's **Opponent**.
A managym-only install offers Random and Search; manabot adds Greedy, Heuristic
and trained checkpoints. The game opens in the replay viewer and is saved in
**Games**. **Watch another matchup** opens setup for another bot game.

Browsing earlier positions reads the recording. At the latest position of a live
Watch, **Continue** requests the next bot decision; **Play** can keep producing
and presenting decisions from there. The status distinguishes reading saved
data, producing a decision, waiting, finished and interrupted. Imported and
completed recordings only play saved data and never run a policy.

If production fails, the recorded choice remains available. After a lost reply,
refresh the saved latest state before explicitly retrying; the action may already
have been recorded. Resume credentials stay in this tab's session storage.
Reload in the same tab to recover a live Watch while its server session exists.
After a server restart or session expiry, the saved recording remains readable
subject to its access rules, but the original live producer cannot be resumed.

#### Study and older recordings

At an eligible choice, **Try another line** pauses playback and live production
and opens an isolated branch. **Return to recorded decision** restores that
exact choice and its history; the original recording and result stay unchanged.
A fresh branch requires the server's retained Study state. If that state has
expired, recorded playback remains available.

Older recordings use the same viewer with the frames they retained. Missing
legal offers, Study support, consequences or policy outputs are explicitly
unavailable; they are never inferred from a later frame or a rerun. Reading
an existing recording needs no original checkpoint or policy provider. Import
verification and live continuation have their own prerequisites below.

Games are stored in `play.sqlite` under the traces directory, which defaults
to `academy/traces` in the directory you launched from. Launch from the same
directory, or pass the same `--traces-dir`, to see earlier games. The file
holds private deal seeds; keep it private.

### Watch a game from an evaluation

Evaluation trace import currently needs the optional manabot integration.
Stop any running Academy server and install it into the same environment:

```bash
uv pip install --python .venv/bin/python --only-binary :all: --torch-backend cpu manabot
```

Use matching `managym` and `manabot` release versions. For an unpublished
candidate, install both wheels from the same release build instead.

1. Find the evaluation's retained command trace, typically
   `traces/<cell-id>.commands.jsonl.gz` under its output directory. Copy or
   download that artifact locally if the evaluation ran remotely. Use the
   command trace rather than the evaluation's metrics or summary file; see
   [automatic capture and replay](training.md#automatic-capture-and-replay)
   for the evaluation workflow.
2. Open the game from the command line with
   `uv run --no-project managym academy replay <trace> --game <n>`
   ([Open a replay directly](#open-a-replay-directly)) and skip to step 4, or
   launch `uv run --no-project managym academy`, open **Watch** at
   `http://127.0.0.1:8000/replay?watch=1` and use
   **Open a game file** to select the trace. Individual game JSON
   and uncompressed JSONL are also accepted.
3. If it contains several games, select a **Recorded game** and click
   **Open selected game**. A single game opens automatically.
4. Use **Next**, **Previous**, or **Play** to follow choices, consequences and
   the result ([replay controls](#inspect-a-replay)). The importer verifies the
   retained Commands and state digests without running the original players again. Historical policy
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

- **Registered policy** in Play's Opponent menu and in both Watch seats
  ([Play and watch a trained checkpoint](#play-and-watch-a-trained-checkpoint)).
- **Greedy** and **Heuristic** registered policies.
- **Open a game file**: exported training games and evaluation traces open as
  verified replays.

Independent Python policies work without manabot. A saved game that was played
against a checkpoint still replays without its provider; continuing it needs
the original admitted policy. Recorded policy and search evidence comes from
the saved game and works in installed Academy; missing outputs stay unavailable.
The live table does not load example analysis from a repository checkout.

#### How a trained opponent is named

Academy shows the first and last parts of a long checkpoint name, followed by
its identity's first ten characters. For example, `run/checkpoints/step_8`
appears as `run/…/step_8 · 7ac5d442da`. Human names are never shortened this way.

The same compact identity appears at the table, in Games, Replay and Watch.
Open **status and mana** on the player's bar with a mouse or Tab and Enter to
read the full recorded name, identity, producer and available provenance.
Registered policies also retain the runner, artifact digests and configuration.
Their **Policy ID** binds that whole execution; historical checkpoint identities
remain the checkpoint file's SHA-256. A runner change creates a different recorded
player even with the same checkpoint. Artifact digests remain available separately
for grouping recordings by weights. Missing provenance stays absent.

Shortening is display only. Recordings, exported games and filter links keep
the full name and identity. Search Games by either; the Bot version filter uses
the full recorded identity.

### Play and watch a trained checkpoint

Register a baseline checkpoint using a saved specification, `challenger.json`:

```json
{"provider": "manabot", "artifact": "/models/self-play-raw.pt", "config": {}}
```

```bash
uv run --no-project managym academy --policy challenger.json watch \
    --player challenger --player random
```

Watch opens at setup with both seats selected. Choose decks and press **Start watching**.
Either seat can use a registered policy. **Registered policy** and **Policy**
use the same direct selection in Play and Watch; provider and runner details
remain available as provenance. They preserve an
unavailable selection rather than substituting a newer model. Keyboard navigation
uses the browser's select controls.

With manabot installed, its provider discovers `.pt` and portable `.safetensors`
checkpoints under `MANABOT_RUNS_DIR` (default `./.runs`) and explicit
`ACADEMY_PLAY_CHECKPOINTS` sources. Python can supply provider-owned discovery
without changing the environment:

```python
from managym.academy.launch import serve
from managym.policy import PolicySource, PolicySpec

with serve(
    port=0,
    policies={"challenger": PolicySpec("manabot", "/models/self-play-raw.pt")},
    sources=[PolicySource("manabot", {"runs_dir": "runs/elemental-a"})],
) as academy:
    print(academy.watch(players=("challenger", "random")).url)
    academy.wait()
```

Run, stage, raw/EMA, updates and transitions come from the run's `run.json`,
joined to the exact artifact digest. Missing counts say **not recorded**;
filenames never supply training counts. A world mismatch stays listed but
disabled with the precise reason. A selected policy is checked against the setup before Start in Play and Watch;
a mismatch shows the provider’s reason beside the picker. Start checks again,
so changed artifacts and setups are still refused precisely. Bound checkpoint
decks are also listed as **Bound decks**. Registered launch aliases appear
alongside provider-authored names in setup and recorded player identities (unless
they match the name ignoring case), so
configurations of the same artifact can have recognizable labels. A refused
Play launch keeps its selected policy and decks for an explicit retry.
Imported decks work when the provider admits them.

For a local fine-tuned Qwen saved with its tokenizer and safetensors weights:

```json
{"provider": "qwen", "artifact": "/models/qwen-finetuned", "config": {"device": "cpu", "precision": "float32", "max_new_tokens": 32, "seconds": 30}}
```

The Qwen provider needs manabot's optional LLM dependencies (`uv pip install
'manabot[llm]'`). It reads local files only, pins model and tokenizer bytes,
and uses the complete native text decision. It returns a single legal action;
probabilities and a critic stay absent. Invalid or late output leaves the
pending decision intact. Qwen does not support Search. A provider's
`supports_search` defaults false; a true flag alone is no search admission.
Search-capable policies can use the shared composition below.

<a id="mirror-search"></a>

#### Search

Choose a registered policy, then change its **Search** control from **None
(raw policy)** to **Search**. This works for the opponent in Play and
for either Watch seat. Its configuration includes the exact budget; details retain the magnetic
mirror update algorithm and recorded identities. While a decision is pending, the table shows
its search budget and elapsed waiting time. Reload preserves the configuration.

| Setting | Default | Range |
|---|---|---|
| Worlds | 1 | 1–64 |
| Depth, including the root choice | 1 | 1–40 |
| Decision ceiling | 30 seconds | Greater than zero, at most 30 seconds |

<a id="mirror-search"></a>

**Uniform belief (physical deal)** samples unseen cards with copy counts and
public knowledge. Three unseen copies of one card and one of another give a
hidden card a three-in-four chance of being the first. It ignores behavioral
history; it does not mean uniform actions or uniform distinct hands. Native
sampling preserves the bot's information and full legal choices. Retained belief
identity `managym.determinize/physical-deal/v1` names native determinization,
including foretold slots and rejection against retained selection commitments;
it is distinct from the historical `physical-compatible-hand/v1` sampler.

Every sampled world is tried under each root action with positive base-policy
probability. Both rollout seats use that base evaluator from their own views.
Finished branches use the actual result; cutoffs use the declared signed
expected-outcome value. The existing regularized mirror update shifts the
decision distribution without training weights. Exact base zeros stay zero.
Action priors come from the selected policy, not the uniform hidden-world belief.
The update also uses a uniform action reference for regularization; this is
separate from the policy prior. The regularization defaults remain alpha 0.02
and beta 1.

Search requires open decklists, complete legal-action probabilities, a compatible
signed-outcome or W/D/L value evaluator, and isolated forks. V1 evaluation must
depend on the current view alone. Recurrent and action-only policies still play
and Watch, with search unavailable and a reason. Qwen v1 does not support search.
No missing capability is replaced by a synthetic distribution, terminal rollout,
raw-policy fallback.

On a timeout, no bot action commits. The pending decision remains on the table
with **Retry search**. Only an explicit retry tries it again; reconnecting or
changing priority stops does not. The retry checks the match, revision and
pilot authority. Other search failures report their reason without substituting
another policy.

Watch shows **Raw policy**, **Search values** and **Updated policy**, alongside
belief, budget, elapsed/ceiling seconds, rollouts, regularization, the exact
hexadecimal seed and receipt digest. The full receipt is not retained and no
verifier ships; recomputing its digest requires the same provider, position and seed.
A completed Play replay offers **Inspect bot search** for the same retained evidence. The human opponent's live view and
viewer export do not contain these outputs. Sampled hands and private branch
audits are not stored in the summary. Replay needs no provider or new inference.
Raw, searched and differently budgeted configurations have distinct recorded
player identities, binding the base provider, artifacts, runner and configuration.

Python and the CLI accept the same explicit settings:

```python
from managym.academy.launch import serve
from managym.search import SearchConfig
from managym.policy import PolicySpec

with serve(policies={"challenger": PolicySpec("manabot", "/models/self-play-raw.pt")}) as academy:
    search = SearchConfig(worlds=4, depth=2)
    print(academy.new_game(opponent="challenger", search=search).url)
    print(academy.watch(players=("challenger", "random"), search=(search, None)).url)
    academy.wait()
```

```bash
uv run --no-project managym academy --policy challenger.json play \
    --opponent challenger --search '{"worlds":4,"depth":2}'
uv run --no-project managym academy --policy challenger.json watch \
    --player challenger --player random --search '[{"worlds":4,"depth":2},null]'
```

`managym.mirror_search.MirrorSearch`, `configure_mirror`, Python `mirror=`
and CLI `--mirror` remain deprecated aliases until 0.4.0 and warn when used.
Existing serialized `mirror` fields, error codes and receipt schema names retain
their spelling so saved games and frozen evidence remain readable. The magnetic
mirror optimizer remains in `managym.mirror`.

The initial budget tries each supported root action in one sampled world and
uses the value evaluator immediately after that action. Increase worlds or depth
explicitly for more search work. Even the initial budget can time out with a slow
provider or many legal choices. More search work does not by itself establish
stronger play.

#### Independent Python policies

Implement the framework-free contracts in `managym.policy`. A factory exposes
`describe() -> PolicyDescriptor`, `admit(setup)` and
`load(*, seat, seed) -> PolicySession`. Pass it directly as
`serve(policies={"my-bot": factory})`. Installed providers may also register
in the `managym.bots` entry-point group and implement `policy(spec)` plus
optional `DiscoveringPolicyProvider.discover(config | None)` returning factories.
The entry point names a provider object, not its class:

```toml
[project.entry-points."managym.bots"]
my-bot = "my_bot:provider"   # provider = Provider() in my_bot/__init__.py
```

A saved specification's `"provider"` is that entry-point name.
`None` means the provider's ambient configuration, called once at server start;
explicit `sources=` are additional discovery calls. Multiple providers coexist.
Academy does not import a training framework or know their artifact layouts.

A session implements `observe(view)`, `choose(view)` and `close()`. The shared
runner sends each seat its viewer-safe native observations after setup and
transitions, preserving live decision focus. `PolicyView` carries the observation,
seat, complete ordered `choices`, one-use `binding` and optional native text
capture. It contains no live environment. Return
`PolicyDecision(choice=view.choices[index], binding=view.binding)`.
Optional probabilities must cover that entire ordered set and sum to one;
optional `ValueEstimate` declares units and viewer seat. Action-only choices
never become synthetic model probabilities. The shared runner validates them
before creating the existing bound Command; stale or illegal outputs commit nothing.
Sessions close on replacement, completion, failed setup and server shutdown.

The descriptor identity pins provider, adapter contract, runner/source, artifacts
and semantic/runtime configuration. Providers author optional `group` (defaults
to the provider name), ordered `PolicyDetail(label, value)` lines and `alias` on
`PolicyDescriptor`. Academy renders these labels and values verbatim; presentation
never changes the Policy ID. Use a stable alias in launch code and saved-spec
examples. When the browser remembers an older selection and exactly one policy
matches its provider, artifacts and config with a different runner, it explains
that change and offers **Reselect**. Nothing is selected automatically. Unknown
or ambiguous old IDs stay unavailable. The browser
selects that opaque identity and never submits an import path or model path.
Recordings retain the complete descriptor and genuine outputs once. Finished
replays need no policy reload, original checkpoint or installed provider.
Live objects survive browser reload with their server; a backend restart needs
the factory or saved specification again. Arbitrary Python objects are never
pickled into recordings. Historical checkpoint recordings remain readable.

### Troubleshooting

A launch that cannot start prints one `ACADEMY_PLAY_ERROR` line with a `code`
and a message, and exits with status 2. From Python the same `code` and
`message` are on the raised `AcademyLaunchError`.

| What you see | Cause and fix |
|---|---|
| `managym: command not found`, or `No module named managym.academy` | The environment holds a `managym` from before Academy was included or the command is running outside the virtual environment. Upgrade with `uv pip install --upgrade 'managym[academy]'`. |
| `ACADEMY_PLAY_ERROR` with code `extra.missing` | `managym` is installed without the server dependencies. Run `uv pip install 'managym[academy]'`. |
| `ACADEMY_PLAY_ERROR` with code `port.in_use` | Another program, often an earlier Academy, holds the port. Stop it, or run `uv run --no-project managym academy --port 8001`. |
| `ACADEMY_PLAY_ERROR` with code `site.missing` | This `managym` was built from source without the browser client, as in a checkout. Use `./scripts/play` from the checkout, or install a released wheel. |
| `deck.unknown` | The value is not a bundled deck. The message lists bundled names; in Python, pass a `pathlib.Path` for a decklist file. |
| `deck.unreadable` | The decklist file cannot be read as UTF-8 text. |
| `deck.rejected` | Native admission refused a decklist. The message gives each seat's issues by line; fix them as you would after **Check decklists**. |
| `setup.invalid` | The combination is not allowed, for example `--opponent search` with `--decklists closed`, or a policy bound to different deck setups. The message says which. |
| `policy.unavailable`, `opponent.unavailable`, `player.unavailable` | The policy or provider is missing, its specification is invalid, or its artifacts cannot run in this engine. The message says which. See [Play and watch a trained checkpoint](#play-and-watch-a-trained-checkpoint). |
| `replay.missing` | No such trace file, and no recording with that id in this traces directory. Check the path (use `pathlib.Path` for files in Python), or pass the `--traces-dir` the game was recorded in. |
| `participant_credential_required` in the browser after READY | The recording is private or unfinished and this browser at this address has no participant credential. Use the same browser and address (including port) where **Games** lists it, or open a shared, finished recording. |
| `replay.unreadable` | The file is not a game file: not JSON, JSONL or gzip of either, or it holds no game records. Use an evaluation's `traces/<cell-id>.commands.jsonl.gz` (not its metrics or summary file) or an exported training game. |
| `replay.ambiguous` | The trace holds several games and none was selected. Select one by index or trace digest: `--game` in the CLI, or `game=` in Python. |
| `replay.game_unknown` | The selection is past the end of the file, or matches no digest or more than one. A recording id already names one game; omit `--game` (Python `game=`) when opening it. Selection applies only within a trace file. |
| `replay.needs_manabot` | Game files are verified by manabot. Install it at the same version: `uv pip install --python .venv/bin/python --only-binary :all: --torch-backend cpu manabot`. |
| `replay.rejected` | Verification failed: the trace was recorded in another world, or its Commands or digests do not replay. Open it with the revision that produced it. |
| `server.running` | (Python) `serve` was called while an Academy from this process is still open. Close it first. |
| `server.closed` | (Python) This Academy has already closed. Call `serve` again. |
| `server.startup` / `server.shutdown` | The server could not start or finish stopping. The message names the failure. |
| Cards show drawn plates instead of art | The server could not reach Scryfall, or `ACADEMY_CARD_ART=off` is set. Check the connection; a failed card is tried again after ten minutes. See [Card art](#card-art). |
| No browser window opens | Open the printed `open` address yourself. On a remote machine, forward the port (`ssh -L 8000:127.0.0.1:8000 host`) and use `--no-open`. |
| No policies are listed | Register a Python factory or a saved policy specification; check any provider discovery error shown below the picker. |
| `No trained checkpoints found.` | Register a saved specification with `--policy <spec.json>` or supply provider-owned discovery sources through Python. |
| A checkpoint is listed but disabled | It was trained in another world or on other card content; the row states the mismatch (for example `trained in w5, engine is w6`). Run it from the revision that wrote it. |
| A checkpoint shows **not recorded** for its update | No `run.json` in the listed directories records those bytes. List the run directory rather than the file alone. A run that recorded no counts for its checkpoints stays **not recorded**. |
| `The selected opponent checkpoint is unavailable.` | The file was moved or deleted after it was listed. Restore it or choose another; Academy does not pick a replacement. |
| `The selected policy changed; restore the pinned artifact and runner, or restart Academy and explicitly reselect the policy.` | The artifact or runner differs from the pinned Policy ID. Restore the original execution, or restart Academy and deliberately select the new entry. |
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

A checkout shows [card art](#card-art) the same way an installed package
does, from the same cache directory. Tests and CI run with
`ACADEMY_CARD_ART=off` or a seeded `ACADEMY_CARD_ART_DIR` and never contact
Scryfall. Clean-machine and release checks are in
[development.md](development.md).

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
  Imported game files get their own Academy identity while retaining the
  source digest; frozen `erd1` evidence keeps its original bindings.

The historical `/api/advice` fixture API remains available to conformance tools.
The live table does not fetch or display that example as current-game analysis.

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

Retained semantic history is separate from the narration cursor. Validated live
events append once, scoped by match, canonical or Study branch, and sequence.
The live narration backlog, including the current beat's remaining duration, is
bounded to 12 beats and 2,000ms of scheduled duration, even while paused. Older
optional motion finishes as needed to admit new events; history remains readable.
Replay plays its selected recorded sequence in order without this live cap.

Recovery commits the complete frame, cancels stale theater and merges validated
`presentation_tail` events with same-match in-memory history. Only unseen events
can animate. The server retains the latest 256 events
(`MAX_PRESENTATION_EVENTS`), so a fresh page labels any unavailable prefix.
Sequence gaps are explicit. Malformed presentation clears theater and reports an
error without touching the committed frame; valid retained history is marked
incomplete. Skip, Fast-forward, Finish, Pause, reduced motion and unmount cannot
alter or delay canonical state.

Labels come only from viewer-safe frames. The current projection certifies
incarnation zero and reuses the spell card's ID as the stack render ID;
references it cannot name stay unnamed.

History retains the active match's available events and renders a 100-row page.
Recorded `turn_started` events establish turn headings; `group` is not a turn or
spell-chain identifier. Canonical Replay history uses the prefix at its selected
frame; Study has an isolated scope. See
[Reading consequences and controlling narration](#reading-consequences-and-controlling-narration)
for reading, inspection and recovery behavior.

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
`elemental-world-v2` setups (mirrors allowed) with a deck assignment for each
native seat. **Edit next game** prepares the next matchup in the current session;
starting it replaces the current attempt. **New game** starts a separate session.
On a scripted seed-0 game (2026-10-26), stop and forced-step automation cut deliberate inputs from 289 to 73.

### Other checkpoints

Any trained checkpoint is selectable in Play and in either Watch seat
([Play and watch a trained checkpoint](#play-and-watch-a-trained-checkpoint)).
manabot supplies the catalog (`GET /api/opponents`) and reads checkpoints with
its provider-owned loader. The browser sends only an opaque runtime identity: `policy:<id>`
as Play's opponent or in `POST /api/watch` `players`. The server re-hashes the
bytes before play, refuses an incompatible or replaced file with the sentence
the menu shows, and records the identity as the trace's `opponent_checkpoint`
in Play and in each seat's recorded player in Watch.

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

The release proof drives ordinary UR Lessons versus GW Allies games at 1440×900
and 1280×800, with Random and Search 64, normal and reduced motion. It records
actual browser metadata, per-action samples and distributions in `timing.json`,
plus a Chromium `timeline.json` with layout, style and paint events.

```bash
ACADEMY_STATIC_BUILD=1 npm --prefix academy/ui run build
npm --prefix academy/ui run test:e2e:release -- experience-proof.spec.ts
```

Input acknowledgement is input-to-socket-send. Authority response is input to a
matching newer frame's arrival before parsing, including policy and network time.
The client measurements cover arrival-to-synchronous-listener completion and
arrival-to-second-animation-frame after the matching DOM revision. The latter is
an upper bound on a rendering opportunity, not physical display latency. The
receipt reports misses against 100ms client targets, sample counts and probe
attribution overhead; it does not pretend to measure server-only computation.
Same-origin art requests and cancellations are reported separately. Normal cache
state and a fresh isolated art directory must be identified when comparing runs.

The older [baseline](../academy/ui/e2e/experience-proof-baseline.ts) remains frozen
with its own metric definitions and historical browser/dev-server profile. Its
DOM update-sequence metric was never paint latency. The current proof does not
compare different metrics against those historical numbers. Recovery, stale
Commands, complete choices and Replay equality are covered by the companion
release workflow specs. This is machine evidence, not an expert endurance or
human-enjoyment claim. CI selects the release suite for browser code and the
backend/rules paths it consumes.

History: [choice navigation](https://github.com/loopflowstudio/manabot/blob/a74f31828/docs/choice-navigation.md),
[experience proof](https://github.com/loopflowstudio/manabot/blob/a74f31828/docs/experience-proof.md),
[local trained challenger](https://github.com/loopflowstudio/manabot/blob/a74f31828/docs/local-trained-challenger.md),
[presentation runtime](https://github.com/loopflowstudio/manabot/blob/a74f31828/docs/architecture/presentation-runtime.md).
