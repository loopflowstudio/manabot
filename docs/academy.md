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
[Launch from Python](#launch-from-python). To look at a game your own training
loop played, see
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

When the server is accepting connections it prints one `ACADEMY_PLAY_READY`
line, a JSON record with the server's `url`, the `traces_dir` in use and
`open`, the address the browser is sent to, and then opens that address in
your default browser. `open` equals `url` for a plain launch and points into
the game or replay for `play` and `replay`. On a machine without a desktop
(SSH, a container) pass `--no-open` and reach the port yourself.

These four options go after `academy`, `academy play` or `academy replay`
alike.

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
| `--opponent` | `search` | `search`, `random`, `passive`, or the 64-character digest of a catalog checkpoint |
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
The same restrictions apply as at the table: a checkpoint opponent needs
manabot installed and does not accept decklist files, and `search` needs open
decklists.

Reloading the page returns to the same game and its pending decision; it does
not deal again. **New Game** returns to ordinary setup with the launched
decks, opponent and decklist mode selected. For imported matchups, setup shows
both selected source files and their editable lists. Uncheck **Use imported
decks for the next game** to choose bundled decks instead.

The launch address is good for as long as that server runs. Opened in another
browser it starts that browser's own game with the same decks and opponent.
With `--seed`, each opening deals the same game; without it, each opening
deals afresh.
After the server stops the address means nothing: the page says the
launch is unknown and shows ordinary setup, so run the command again.

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
in **Replay** on decision 1, ready to step. The heading names both players and
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
- `academy.replay(source, game=None)` takes a trace `Path` with the same
  `game` selection as the command line, or a recording id as `str`. An integer
  `game` selects only a position; strings can also select a digest prefix.
- `launch.read_games(path)` lists a trace file's games as `RecordedGame`
  values (`index`, `deal_seed`, `leg`, `players`, `decks`,
  `game_trace_sha256`), for choosing one in code.
- `new_game` and `replay` return a `Launched` whose `url` opens that game or
  replay, subject to the recording access rules above. `academy.open(url)`
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

The browser opens in **Replay** on decision 1 of that game, headed **Training
game** with `jace vs jace`. Every decision the two seats made is there to step
through, with the board and the legal actions each was offered. The original policy is never rerun and no checkpoint is loaded,
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
`managym academy play` starts the same game from the command line
([Start in a chosen game](#start-in-a-chosen-game)).

The table reads **Who decides / What is happening / Your choices**. Clicking a
card inspects it and sends nothing; choosing one of the listed legal actions
commits it. Decisions with a single legal action are taken for you and never
shown. Reloading the page restores the pending decision.

### Using the laptop table

Start a game, inspect a land in your hand, then choose its play action in
**Your choices**. The board stays in view as the next prompt appears. Inspecting
the card sends nothing; selecting the legal action commits the play. If a spell
needs a target or payment choice, follow the next prompt before taking another
action.

At laptop widths, the board and hand sit beside the current decision. Long
choice lists scroll within their own region, so reaching an action does not
require scrolling away from the table. At narrow widths, board and choices
stack; use the board/choices jump links to move between them. Changing decisions
moves focus within the choices region without pulling the page away from the
board.

- **A crowded hand:** scroll the labeled hand region horizontally to reach
  the remaining cards.
- **A crowded battlefield:** use its previous/next page controls. Counts show
  the size of each group; focusing an action reveals the page holding its
  subject. Scroll the labeled battlefield region to reach further groups and
  secondary zones. Identical card copies keep their own choices.
- **Other zones:** open their inspection controls to see viewer-permitted
  contents. Empty zones keep their labels and counts; hidden cards remain
  hidden.
- **Card details:** hover or keyboard-focus a visible card to preview it, or
  select it to inspect its details and related choices. Inspection leaves the
  pending decision unchanged. Use **Show all** to return from card-specific
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
Reloading a configured launch in its tab resumes that game; a separate tab or
another launch has its own game.

For bot inspection, open **Watch** or a recording in **Replay**. The shared
decision surface shows the acting seat's permitted view and the complete legal
choices for that decision. Recorded policy outputs appear only when available;
missing outputs are not reconstructed. See [Inspect a replay](#inspect-a-replay).

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

A finished bot recording shows **Replay**, its actual players and decision
controls first. A game still being played keeps the **Watch two bots** heading,
including when reopened. Expand **Watch another matchup** to set up a new bot game.

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
- **Open a game file**: exported training games and evaluation traces open as
  verified replays.

Without manabot the Opponent menu and Watch show a line saying trained bots
need it, and a saved game that was played against a checkpoint still replays;
only continuing it needs the checkpoint. The decision-advice panel reads
fixtures from a checkout of this repository and is absent from an installed
package.

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
| `setup.invalid` | The combination is not allowed, for example `--opponent search` with `--decklists closed`, or a checkpoint opponent with a decklist file. The message says which. |
| `opponent.unavailable` | The checkpoint digest is not in the catalog, or manabot is not installed. See [Trained opponents and other bots](#trained-opponents-and-other-bots). |
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
