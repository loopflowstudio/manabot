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
- [Playing on the board](#playing-on-the-board)
- [Previewing payments](#previewing-payments)
- [Pending input, waits and recovery](#pending-input-waits-and-recovery)
- [Choosing priority stops](#choosing-priority-stops)
- [Rematch, swap sides and resuming](#rematch-swap-sides-and-resuming)
- [Reading the history](#reading-the-history)
- [Inspect a replay](#inspect-a-replay): timeline, live Watch, perspectives and recovery
- [Choice navigation](#choice-navigation)
- [Presentation runtime](#presentation-runtime)
- [Playing a trained challenger](#playing-a-trained-challenger)
- [Attempts and history](#attempts-and-history)
- [Experience proof gate](#experience-proof-gate)

## One game, from setup to its recorded decisions

Academy has a single top bar (**Academy**, **Play**, **History**, theme) and
three views:

- **`/` (Play).** If you have an unfinished played game in progress, `/` opens
  straight onto its board at the pending decision. Otherwise it shows one setup
  screen: **Play vs Bot** or **Watch Bot vs Bot**, decks, opponent, **Start**,
  and **Play again** for the last matchup.
- **`/games` (History).** Every game in `traces_dir`, newest first, with search,
  filters, and **Open a recorded game file**. Every row has **Replay**
  (`/replay?trace=<id>`), and every unfinished played game has **Resume**
  (`/?game=<id>`).
- **`/replay?trace=<id>` (Replay).** Step through any saved game — finished or
  unfinished — with analysis, **See its line**, **Try another line**, and
  **Play on from here**. Bare `/replay` redirects to `/games`.

On setup, choose **Play vs Bot** or **Watch Bot vs Bot** and configure two
native seats. For each seat, choose a **Deck** and, for a bot seat, pick a
built-in bot or a **Registered policy** with optional **Search** for a
compatible policy. Built-in **Search 16/64/256** runs that many random playouts
per legal action without a policy; **Random** chooses uniformly among legal
actions; **Passive** passes when possible and otherwise takes the first legal
action. Seat 1 plays first; seat 2 draws first. Policies are chosen directly by
name and execution identity; their provider, artifacts and runner remain
available in details. **Browse decks and cards** opens the full deck browser;
imported sources and disclosure use the same admission checks for played and
watched games.

Closing the tab, quitting the browser, or restarting the server loses nothing:
`/` and History's **Resume** return to the same game, with the same game id and
recorded history, at the exact decision that was pending. Clicking **New game**
from a board opens setup in place with no confirmation dialog, leaving the
unfinished game resumable in History.

Hover or focus any visible card to read it; reading never sends an action.
**Open a recorded game file** in History opens any recorded game, from your own
loop, training or evaluation, without changing its original evidence. Replay
links back to recorded positions, their legal choices and available recorded
policy/search evidence. Trying another line creates a separate Study branch
with an explicit return to the source decision. There is no live example Advice
panel or example-scenario read selector.

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

Your browser opens at `http://127.0.0.1:8000` — straight onto the board of your
unfinished game if one is in progress, or onto setup with a game ready to start
against the default opponent. That is the whole installation: no manabot,
Torch, Node or Rust, and no checkout of this repository. The wheel carries the
compiled engine, the server and the built client. Cards show their Scryfall
art, which the server fetches the first time a card is shown and keeps on
disk ([Card art](#card-art)); it makes no other network requests while you
play.

To skip the setup screens, name the game or the replay on the command line:

```bash
uv run --no-project managym academy play --deck ur_lessons --opponent-deck ./my-deck.txt --seed 7
uv run --no-project managym academy replay .runs/demo/games --game 3
```

The first opens the board already in that game; the second opens game 3 of the
games a training run recorded, at its first decision. See
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

The default address is reachable only from your own machine. Academy is a local
tool with no accounts, no credentials and no per-browser privacy walls: every
game in `traces_dir` is readable and resumable from any browser that reaches
the server. Listen on another address (`--host 0.0.0.0`) only on a network
where you intend to share your local game store with everyone on it.

### Stop

Press Ctrl-C in the terminal that runs it. The server finishes its writes,
exits with status 0 and frees the port. An unfinished game stays unfinished,
with no winner invented: the next time you start Academy on the same
`--traces-dir`, opening `/` resumes it at its pending decision, and **History**
lists it with both **Resume** and **Replay**.

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
| `--sims` | `64` | Random playouts per legal action for `search` |
| `--seed` | random | Deal seed, an unsigned 64-bit integer; the same seed and decks give the same deal |
| `--decklists` | `open` | `open` or `closed`: whether each player knows the other's list |

A decklist file is the UTF-8 `.txt` or `.dek` format Academy imports: one
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

Once the game starts, the browser address updates to `/?game=<id>`, so
reloading the page returns to the same game and its pending decision — even
after an idle timeout or a server restart — and never deals again. To play
again with the same setup or reversed seats, use
[Rematch, Swap sides or Deal new decks](#rematch-swap-sides-and-resuming).
**New game** opens setup in place with the launched decks, opponent and
decklist mode selected, leaving the current game unfinished in **History**.
For imported matchups, setup shows both selected source files and their
editable lists; uncheck **Use imported decks for the next game** to choose
bundled decks instead.

The `?launch=<id>` address is good for as long as that server runs and starts
its game once (reloading before attach reconciles the same game). Opening
`/?launch=<id>` while another game is unfinished starts the launch and leaves
the earlier game resumable in **History**. Opening an unknown launch (for
example, an unstarted `?launch=` link after a server restart) clears it and
shows ordinary setup; run the command again to create a new launch.

### Open a replay directly

```bash
uv run --no-project managym academy replay games.jsonl
uv run --no-project managym academy replay .runs/demo/games --game 3
uv run --no-project managym academy replay <recording-id>
```

The first command opens a file of recorded games; the second opens one game
from a directory of them, here a training run's; the third reopens a game
already in **History** (`/games`), by the id shown there. All land in **Replay**, ready to
step from the first recorded choice or resume a saved position
([replay controls](#inspect-a-replay)). The heading says which kind of game
it is: **Training game**, one the training loop itself
played, **Evaluation game**, one played afterwards to measure policies, or
**Recorded game** for a record that says neither.

Any game managym recorded opens this way with only `managym` installed: your
own loop's ([Record games](engine-python.md#record-games)), a training run's,
an evaluation's, or a private snapshot replay exported by
`GameSnapshot.to_replay_json()` ([Save and restore game positions](engine-python.md#save-and-restore-game-positions)).
A record holds the seed, both setups, the actions, and any recorded boundary
interventions (such as library reshuffles or determinizations), so
Academy plays the game again through the engine, checks the recorded state
digests wherever the source retained them, and saves what it verified to this traces
directory as a finished recording. A snapshot replay carries its final digest.
It opens as a **Recorded game**. The acting view hides the opponent’s hand.
**Show both hands** includes both hands in the spectator response. The board
draws the bottom seat’s cards; the top seat’s **Hand** count opens a list of
its cards, by pointer or keyboard, as the graveyard does. Where the record
re-dealt the position before a decision, the move list shows a neutral note
in its place, such as **Branch · libraries reshuffled** or **Branch · hidden
cards re-dealt**, which jumps to that decision. Branch seeds stay private in
either view.
Replay runs no policy, loads no checkpoint, and never changes the file. Opening the same game again returns to the
recording made the first time. At each choice the panel shows what the chooser
recorded about it (the probability of each legal action, the value and the
update number) and **not retained** for whatever it did not record.

A game does not have to be whole to open:

| Heading | What you can step through |
|---|---|
| **Interrupted · saved recording only** | The run stopped during the game. Every decision up to the last recorded one, then the reason it stopped. No winner is shown. |
| **Stopped at decision N · saved recording only** | This engine did not reproduce the game from decision N: the digest differed, the recorded action was not offered, or the engine raised. Decisions before N are shown with the message; nothing after N is. |

Neither can be continued, and no later board is invented. An opened recording
from a file is read-only: it only plays saved data, even when its game never
finished.

Any recording id in this `traces_dir` opens in any browser that can reach the
server, finished or unfinished, with no credential. Opened game files are saved
in `traces_dir` and open at `/replay?trace=<id>`. A file holds the deal seed
and so both players' hidden cards; share it as you would share both hands. The
viewer still shows each seat only what it could see, and the file's run labels
and build identities stay on the server.

`.json`, `.jsonl` and gzipped files are accepted, and a directory stands for
the games in all its files. `--game` takes either the game's zero-based
position (the order **Recorded game** lists in the browser) or the first
characters of its digest. A position takes precedence when both fit. A file
with one game needs no `--game`. Leave it off to see what is there:

```text
ACADEMY_PLAY_ERROR {"code": "replay.ambiguous", "message": ".runs/demo/games holds 40 games; select one by index or digest."}
  Select a game with --game <index-or-digest>.
  game  seed     stream  players       decisions  ended        digest
     0  9120443  0       jace vs jace  41         complete     5be01c9a2f31
     1  9120443  1       jace vs jace  57         interrupted  e07a44d1c8b2
     …
  20 of 40 shown
```

A game recorded in another world, or with other card content, is refused with
`replay.rejected`. The message names the managym version, world and engine
digest that recorded it and the ones running, so you know which build opens
it. The same world on another engine build opens and says the build differs.
A record whose two players had different decklist disclosure, or that was
played without skipping forced decisions, is refused with the reason.

Files written before this record still open: an evaluation's command trace
(`traces/<cell-id>.commands.jsonl.gz`), a game exported by
`manabot export-game`, and a training exposure receipt. Command traces and
exported games are verified by manabot and need it installed beside Academy at
the same release version; everything else does not.

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
    replay = academy.replay(Path(".runs/demo/games"), game=3)
    print(game.url)      # http://127.0.0.1:52114/?launch=3f9c2a7be1d04c58
    print(replay.url)    # http://127.0.0.1:52114/replay?trace=8b1e…
    academy.open(game.url)
    academy.wait()       # block until Ctrl-C
```

- `serve(host="127.0.0.1", port=8000, traces_dir=None)`
  starts the server on a background thread and returns once it accepts
  connections. `port=0` takes a free port; `academy.url` and
  `academy.traces_dir` report what is in use. It opens no browser.
- `academy.new_game(...)` takes the options of `managym academy play`. A deck
  given as `str` is a bundled deck name; a `pathlib.Path` is a decklist file.
  Opening `game.url` starts the game once and updates the page to `/?game=<id>`,
  so reloading resumes that game even across a server restart.
  `opponent` also takes a policy specification JSON as a `pathlib.Path`.
- `academy.watch(players=("challenger", "random"))` opens Play setup in
  **Watch Bot vs Bot** mode with exact seats selected and leaves **Start** to
  you. A string is a registered alias, `policy:<id>` or a built-in player; a
  `Path` names a saved specification.
- `serve(..., policies={"challenger": factory})` registers an already constructed
  policy factory directly. It needs no manifest or installed entry point.
  Each factory must create a fresh mutable session per game and seat.
- `serve(..., policies={"challenger": PolicySpec(...)})` loads an installed provider.
  `sources=[PolicySource("manabot", {"runs_dir": "runs/elemental-a"})]` lets
  that provider discover a run root. Sources belong to this server; no environment
  variables are rewritten. Each installed provider also discovers its ambient
  configuration once at server start, even when explicit sources are supplied.
- `academy.replay(source, game=None)` takes a `Path` to a game file or a
  directory of them with the same `game` selection as the command line, or a
  recording id as `str`. An integer `game` selects only a position; strings
  can also select a digest prefix.
- `launch.read_games(path)` lists the games of a file or directory as
  `RecordedGame` values (`index`, `seed`, `stream`, `players`,
  `decisions`, `ended`, `digest`), for choosing one in code. The former
  `deal_seed`, `leg`, `decks` and `game_trace_sha256` properties warn and are
  removed in 0.3.0.
- `new_game`, `watch` and `replay` return a `Launched` whose `url` opens that
  game, setup or replay. `academy.open(url)`
  opens it in the default browser; `academy.open()` opens Academy at `/`.

Leaving the `with` block, or calling `academy.close()`, stops the server and
frees the port; unfinished played games remain unfinished and resumable the
next time you open Academy on the same `traces_dir`.
`academy.wait()` blocks until Ctrl-C and then returns, for scripts that
should stay up while someone plays. One Academy runs per Python process:
a second `serve` before the first is closed raises `server.running`.

Every failure is an `AcademyLaunchError` with a `code` and a `message`, the
codes listed under [Troubleshooting](#troubleshooting); nothing is opened
when one is raised.

### Watch a game from your training loop

Ask the run to record games, then open its output:

```bash
uv run etude train manabot/recipes/local.py --experiment demo --config '{"record_games": 0.25}'
uv run --no-project managym academy replay ~/.etude/track/demo/out/games --game 0
```

The browser opens in **Replay**, headed **Training game**. Every decision the
two seats made is there to step through, with the board, the legal actions each
was offered, the probability the policy gave each one, its value estimate and
the update it was playing at. The policy is never rerun and no checkpoint is
loaded, so the directory opens on a machine that has never seen the model and
has no manabot. There is no export step.

`record_games` takes 0 (the default), 1 for every game, or a fraction between
them. A fraction selects games by their seed and stream, so it does not disturb
the run and the same run records the same games. All shipped self-play recipe
configs accept it alongside `statistics`, `diagnostics` and `stall_seconds`,
locally and in a launched job. The learner digest excludes these options.
Etude compares them with the rest of the recipe config on resume. The rows are written under the run's
`out/games/`, one file per producer; download that directory to open a remote
run's games. A game the run stopped
in the middle of is kept and opens as **Interrupted**.

A loop of your own records the same way by passing the capture options to its
collector:

```python
collector = open_collector(..., games_path=Path("runs/jace/games/0.jsonl"))
```

or, with no manabot at all, by stepping through a
[`ScalarCapture` or `RecordedVectorEnv`](engine-python.md#record-games).

**Who played.** The heading takes the players' names from the record: the
recipe's learner and opponents, first seat first. In one-model self-play both
seats are the learner, so both carry its name.

**A training game is not an evaluation game.** A training game is one the
collector played, with its exploration and whatever the policy was at that
update. To see how the current policy plays, run an evaluation and open one of
its games ([Watch a game from an evaluation](#watch-a-game-from-an-evaluation));
those are labelled **Evaluation game**. Neither is ever relabelled as the
other. Statistics over a recorded fraction describe that fraction
([Ask questions of recorded games](engine-python.md#ask-questions-of-recorded-games)).

| What you see | Cause and fix |
|---|---|
| `replay.missing` for `<out>/games` | The run recorded nothing: `record_games` was 0, or no selected game had started. Run with 1 or a larger fraction. |
| **Stopped at decision N** | This build does not play the game as the run did. The message names both builds; open it with the one that recorded it. |
| `replay.rejected` | The games were recorded in another world or with other card content; the message names the build to use. |

`record_games(...)`, `TrainingGames` and `manabot export-game` still work,
warn, and are removed in 0.3.0. Receipts and
exported games they already wrote keep opening.

### Play a game

Open Academy (`/`): if this browser remembers an unfinished played game
(`localStorage['academy.game']`), `/` opens straight onto its board at the
pending decision. Otherwise `/` shows the setup screen:

- **Play again** starts the same opponent, decks, play/draw seat, deal mode and
  stops as your last matchup in one click (or **Play** with the default matchup
  the first time). The last matchup is a browser preference
  (`academy.last-matchup` in local storage).
- Choose **Play vs Bot** or **Watch Bot vs Bot**, pick decks, an opponent,
  decklist disclosure, deal mode and priority stops, then click **Start**.

Before a first matchup, setup proposes the default opponent and, where a deck
pool is installed, a dealt held-out deck for each seat. A matchup named by the
address, or the one just played, is never replaced by those defaults.

With closed decklists, the opponent's deck reads **Closed decklist** on the
board, in **History** (where the Deck filter does not find it) and in Replay
until the rules end the game; a finished game names both decks everywhere. A
side dealt from a deck pool is remembered as its pool, so **Play again** and
**New game** deal fresh decks, while **Rematch** keeps the decks just played;
"Your last matchup" names that side by its pool, never by the deck it dealt.

A deck saved from an import starts **Watch Bot vs Bot** as it starts a played
game, on either seat, against a named deck, a pool or another saved deck.

From a live board, clicking **New game** opens setup in place, with the matchup
just played selected and no confirmation dialog; the game you left stays
unfinished in **History** with a **Resume** link. Until it is resumed, `/`
opens setup rather than that game.

To use your own decks, open setup, expand **Import decklist files**
and check **Use imported decks for the next game**. Upload a UTF-8 `.txt` or
`.dek` file for both **Your deck** and **Opponent deck**, or paste each list
into its source box. Write one `count card name` per line, with optional
`Main` and `Sideboard` headings. Select **Check decklists**, correct any
reported lines, choose a built-in opponent such as **Search 64**, then select
**Start**. Imported decks use the development format and supported cards;
this is not tournament legality validation. Registered policies check the complete
setup, including imported lists, against their own admission requirements. Uncheck
the import option to return to bundled decks.
`managym academy play` starts the same game from the command line
([Start in a chosen game](#start-in-a-chosen-game)).

The game fills the window: the board in the centre, your hand along the
bottom and a rail with the stack, the current decision and every legal choice.
Cards a legal action applies to are raised off the board; clicking one with a
single action commits it ([controls below](#playing-on-the-board)). Decisions with a single
legal action are taken for you and never shown. Reloading the page restores the
pending decision.

### Rematch, swap sides and resuming

Start a game against a registered policy or built-in opponent. When the game
ends, a result card sits over the board, which stays visible around it
(**Look at the board** folds the card to a corner). It says who won and how
("Heuristic · scripted went to −2 life on turn 9"), the final life totals, and
one line from the history: turns, and each player's spells cast and attacks.

| Action on the result card | What carries forward |
|---|---|
| **Rematch** | Both decks (including imported lists), exact opponent configuration, disclosure, priority stops and your seat from the game just played |
| **Swap sides** | The rematch setup with your native play/draw seat reversed; your deck stays yours |
| **Deal new decks** | Same deck pools and opponent configuration, drawing fresh decks from the pools |
| **Review this game** | Opens `/replay?trace=<id>` from the first recorded position |

Priority-stop changes made during play are saved into the game's setup and
carry forward into **Rematch** and **Swap sides**.

Academy names every opponent with how it decides: **Search 64 · search**,
**Heuristic · scripted**, **Passive · scripted**, a checkpoint by its name and
the world it was trained in (**<name> · checkpoint w17**). The setup picker,
the top bar and the result card use that full name. The opponent's plate is
titled with the short name ("Heuristic", "Search 64") and puts how it decides
in the caption beside the deck ("Scripted · GW Allies"), which wraps to a
second line rather than cutting a long deck name.

A trained or independent policy must first be registered with this Academy
server; see [Play and watch a trained checkpoint](#play-and-watch-a-trained-checkpoint).
The provider checks that it can play the selected decks and configuration.
Built-in opponents need no registration.

#### Choose the deal deliberately

**Fresh deal** is the default when no seed was supplied. Each started game gets
a new random seed, recorded with that game. **Repeat seed** keeps an explicitly
supplied seed, including one supplied with CLI `--seed` or Python `seed=`.
Enter a seed as a decimal unsigned 64-bit integer, from `0` to
`18446744073709551615`. The selected mode is visible before starting.

Rematching or swapping sides preserves this mode. Repeating a seed does not
repeat policy decisions, and swapping native seats changes which player gets
each seat's deal. These controls do not establish a paired evaluation.

#### Automatic resume across closes, idle timeouts and restarts

Suppose you start a game against a registered policy, make three decisions,
close the browser tab, stop the server with Ctrl-C, and start it again on the
same `--traces-dir`:

1. Open `http://127.0.0.1:8000` (`/`), or open **History** (`/games`) and
   click **Resume** (`/?game=<id>`).
2. Academy steps the engine through the stored record (rebuilding opponent
   memory and Study roots without re-recording), installs the stored history
   unchanged, and lands on the board at your fourth decision — same game id,
   same recorded history. If the server stopped while the bot was replying to
   your last move, the bot finishes its reply on resume.
3. Make the fourth decision, then open `/replay?trace=<id>` to step through
   every decision before and after the restart in one unbroken record.

Every unfinished played game in **History** is resumable, not only the latest
one. Starting another game, closing the tab, 15 minutes of idle eviction from
server memory, and stopping/restarting the server never end a game; only rules
completion (`game_over`) or a replay error (`record_diverged`) does.

- **Explicit addresses take precedence.** Opening `/` with `?launch=`, `?demo=`,
  `?played_from=`, a Watch launch address (`managym academy watch`), or
  `?hero`/`?villain`/`?opponent` opens what the address asked for; any
  unfinished game remembered in `academy.game` stays resumable in **History**.
- **A game with nothing to resume.** The server refuses a resume with a code:
  `game_finished`, `game_is_watch`, `game_stopped` or `game_not_found`.
  Opening `/?game=<id>` for a finished or watched game redirects to its Replay
  (`/replay?trace=<id>`), always: reloading a result card opens the game's
  Replay, and the next game starts from setup's **Play again** or **History**.
  For a stopped one it shows setup with the reason and a link to its Replay. An unknown game id, or a game `/` remembered that is
  finished or stopped, clears `academy.game` and opens setup without comment.
- **Two tabs on the same game.** Opening `/` while another tab holds the game
  in progress opens setup with a **Play it here instead** link and leaves the
  other tab playing. Opening `/?game=<id>` (that link, or **Resume** in
  History) takes the game over and closes the older tab's socket.
  `localStorage['academy.game']` stores only the current game id, while each
  tab keeps its own in-flight Command envelope and last frame in
  `sessionStorage['academy.tab.<game>']`, so two tabs never overwrite each
  other's pending input.
- **Missing policy vs engine divergence.** If the stored opponent policy cannot
  be loaded (for example, the server restarted without `--policy` or a
  checkpoint was moved), Academy refuses resume with the load error, links to
  the game's Replay, clears `academy.game`, and leaves the game unfinished so
  restarting with the policy resumes it. If the engine no longer reproduces the
  stored steps (such as a rules change), Academy refuses with
  `"no longer plays as recorded at step N"`, marks the game `record_diverged`
  (no winner invented), clears `academy.game`, and links to Replay.
- **Unfinished Watch games.** Eviction or server restart stops live Watch
  production; the saved Watch game stays openable in Replay.

### Playing on the board

Start a game and look at your hand: the cards you can play now are raised off
the board with a soft shadow, and the ones you cannot are dimmed.
Click a land to play it; click a spell to cast it, then follow its target and
payment prompts. A card with exactly one legal action commits that action when
clicked, and its accessible name says which action that is. A card
with several actions opens them beside the card; choose one, or press Escape to
close the list without acting. A card with no action sends nothing when
clicked.

You can also play or cast from hand by dragging (mouse, pen or touch). Press a
playable land or castable spell and drag it onto the battlefield: the full card
follows the pointer — even when lifted from a covered fan — while its slot stays
stepped back in your hand. Any lands the automatic payer would tap stay outlined
on the board, and a dashed frame over the battlefield names what releasing does
(**Play Island**, **Cast Lightning Bolt**, or **Choose how to play …** when the
card has several legal offers). Over your hand, the rail or the bars, the frame
softens and reads **Release to cancel**. Release over the battlefield to commit
a single offer — the card travels from where you dropped it onto the stack or
battlefield — or to open the card's choices at the drop point when it has
several; choosing an offer commits it, while **Cancel** or Escape returns the
card to your hand and restores focus to it. Release outside the battlefield, or
press Escape while dragging, to return the card with nothing sent. If the server
rejects a dropped play, the card returns to your hand and the notice explains
why. Discard and "choose a card" prompts in hand, and choices in an out-of-board
tray, stay click-only, and casting a spell stays separate from choosing its
target.

When a spell or ability asks for a target, click a legal target or pull a line
from the source — the casting spell or ability on the stack, or the asked
permanent on the battlefield — onto any legal permanent, player plate or stack
entry (including non-spell abilities). A dashed outline marks each legal
permanent while you drag. The outline turns solid when the line snaps to it.
When the line snaps to a target, the board previews any lands it would tap.
Release to choose that target, or release elsewhere or press Escape to cancel.

The board is split at the centre line: the opponent's creatures face yours,
with each side's lands and other permanents behind them. A plate at the left of
each half shows the player, life, zone counts and floating mana as mana
symbols, with any restriction or expiry written under them and wrapped within
the plate ("Floating {R}{R} normally expires at step end" shows its pips, not
braces, and is never cut off); open a graveyard or
exile count to list its cards. A player who can be targeted has an
outlined plate. Identical permanents with the same effective abilities, status
and choices share a pile, its count after the top copy's name ("Forest ×4"; the
copies beneath show no name of their own), and each copy remains its own
control. Crowded rows shrink their cards and wrap onto a further line before
anything scrolls, and each half gives the crowded row more of its height.

Every card has a printed card's proportions (63:88). A permanent's name is on
a bar above its art, wrapping to a second line rather than being cut; its mana
cost is left to the hand and the inspection. The name bar carries the card's
colours: a pale tint in light mode and a solid bar in dark mode, white a
neutral ivory-grey and colourless a cool grey, a land the colour of its basic
land types, and a card of several colours blended left to right as printed
hybrid frames are. Power and toughness sit on a plate at the lower right, and
above it sit the permanent's active state chips, in priority order: damage
marked (`2 damage` in red), a small spiral mark when a creature is summoning
sick, active restrictions and statuses (`Can't be blocked`, `Can't block`,
`Can't attack or block`, `Doesn't untap`, `Must attack`, `Struck first`, or
`No abilities` / `Lost base abilities`), counters (`+1/+1`, `Charge ×2`), and
the effective battlefield keywords and keyword abilities it has now (evergreen
keywords plus `Prowess`, `Ward`, and active `Firebending N`; anything granted
by an effect or another permanent is edged in green). Mana symbols carry their
letter, and ◇ for colourless. A tapped permanent turns a quarter onto its side,
in room its row keeps for it so it never covers a neighbour; its name bar stays
across the top.

Battlefield chips show what applies right now rather than static printed text.
For example, **Platypus-Bear** displays plain `Defender` when you have no
Lesson in your graveyard and strikes `Defender` through in muted ink once a
Lesson is in your graveyard to show it can attack; **Fire Nation Cadets**
displays `Firebending 2` only while a Lesson is in your graveyard; **Mai,
Jaded Edge** shows `Prowess`, and an Ally beside **Sokka, Tenacious Tactician**
shows green-edged `Prowess` alongside granted `Menace`; printed and granted
Ward show their cost (`Ward {2}`, `Ward—Pay 7 life`, `Ward—Discard a card`);
and **Ty Lee, Chi Blocker** shows `Prowess` on the board while leaving `Flash`
to the inspection. Permanents with different effective abilities or status
never share a pile, so a vanilla 1/1 Soldier token stays separate from a
Firebending 1 Soldier token. Attacking and blocking do not take chip space: an
attacker steps toward the player it attacks and tilts a few degrees (one that
tapped lies just past the quarter, a vigilant one stays upright), and a blocker
is joined to its attacker by a line. Chips are placed whole and never clipped
mid-word. A narrow card can show `Firebend 2` or `Ward`. Hover the chip or
inspect the card for the full ability and Ward cost. Chips use the space beside
power/toughness when available. Hover a trailing `+N` for the hidden badges.
The inspection and the card’s accessible name include every badge. On a short
tapped card the count can sit beside the name. Something a spell on the stack targets has a dashed red edge.
Cards in hand have the type line under the name and the cost on the art; a hand
card's name stays on one line, shortened with an ellipsis when long, and its
full name is in the inspection.

While you may act, what you can act on is raised and what you cannot steps
back: a card in hand you cannot play now is dimmed most, a permanent of yours
nothing acts on is slightly muted, and lands and the opponent's side are never
dimmed. Summoning sickness is told by its spiral mark, not by more dimming. The
selected card, such as the spell whose targets you are choosing, rises higher
with a thin ink edge. In light mode the grounds are white and grey and what you
act with is charcoal; in dark mode they are near-black and it is near-white.
Academy follows the system setting. Card and player names, life, and the
rail's moves and legal choices are set in Spectral; everything else is in Work
Sans. The top bar shows the connection only when it has a problem.

Rest the pointer on a card for a moment, or focus it with the keyboard, to
inspect it: its rules text, type line and cost, effective keywords and keyword
abilities (including `Flash`, and `Defender · can attack anyway` when Defender
is overridden) with why it has each (printed, the effect that granted it with
source and duration, the counter for a keyword counter such as a double strike
counter, or the permanent granting the ability, such as Sokka), where its power and toughness come from (printed or set,
counters, each effect by its source and duration, and whatever those do not
explain as "other effects"), and its full battlefield state (summoning sickness
in words, attacking or blocking, active restrictions from static abilities or
effects with their provenance, and ability loss). For example, a creature
enchanted by Watery Grasp reads “Doesn’t untap · Watery Grasp”. Older recordings without
extended permanent details still render plain `Defender`, effective keywords,
recorded effect statuses and granted Firebending. The
inspection floats beside the card and closes when the pointer leaves, when you
click or when the game moves on; a card you just clicked does not reopen it
until the pointer leaves and returns; **Pin the card preview** in Settings keeps it in a pane in the left sidebar
instead, showing the last card you inspected. Along
the bottom of the screen runs the step bar: the turn number and whose turn it is,
then one cell per step, a line between
phases, the current step shaded in that side's colour, and in each cell the stop for the opponent's turn above
the step's name and the stop for your turn below it. Press a stop to
change it (see [priority stops](#choosing-priority-stops)).
Cards in hand are large, 136 to 150 px wide, and sit side by side while the
strip has room; with 24 permanents or more on the battlefield the board may
take the hand down to about 100 px wide. A hand too long for the strip first
shrinks its cards toward that floor; only then do they overlap, left over
right, and each covered card's visible slice gives its name the whole width,
with its cost on the line beneath in place of the type line, so a nine-card
hand reads every card's name and cost. Pointing at or focusing a covered card lifts it whole. An empty hand is a
label.

The rail holds:

- **Stack**, top first, each entry with its card and who controls it; a stack
  entry an action applies to is outlined. A line joins every spell or ability
  on the stack to each of its targets. A spell you are casting, or an ability
  you are activating, is on top of the stack with a dashed border (**casting**,
  **activating**) from the moment you choose it, through its targets and
  payment, with lines to the targets already chosen; the card leaves your hand.
- **Since your last move**: the unread tail of the history from your last
  choice to now (hidden while walking or looking back so it never contradicts a
  historical stack). Hovering a card name inspects it and clicking a line seeks
  the board to it. The first three show; the rest open with **more**. A spell
  that already resolved keeps its card here, and its line to a target that is
  still on the board, until your next move or the turn ends. Cards that
  attacked, blocked, were targeted or were dealt damage this turn are ringed on
  the board at now, and damage dealt this turn shows in the card's corner (while
  still marked on it) or on the player's plate ("−2"); transient board marks
  clear when the turn ends.
- **The decision** shows who decides, what they are deciding ("Pay {1}{U} for
  Accumulate Wisdom"), and the buttons. For mana payments the hint below the
  prompt says how to pay. It stays steady as different previews appear.
  Mana in the prompt, the buttons and the choices is drawn as pips.
  **Rules detail** holds the engine's full instruction.
  Passing priority and paying a cost are the primary button (Space), labelled
  with what it does next: **Resolve** and the spell on top of the stack, the
  step it moves to ("To combat", "To blockers"), or the cost ("Pay {1}{U}",
  with the lands it taps dashed on the board and named in the button's
  accessible name; see [Previewing payments](#previewing-payments)). The label
  stays on one line; a long one is shortened and its full text is the
  button's accessible name and tooltip. Declining,
  and a combat answer that does not name another creature, are separate
  buttons, so a click on a card never declines.
- **All legal choices**: every offer of the decision, numbered, with the total
  count. It is the complete list; press a number key to choose one of the first
  nine. During a live combat draft the per-creature answers fold under a closed
  disclosure (see combat below).
- **History**: everything that happened, in the order it happened, grouped by
  turn. A move names the player who made it, and what followed from it is
  indented beneath it with the rule that did it. Select a line to see the
  board as it was after it; see [Reading the history](#reading-the-history).
  Priority-stop automation is marked **auto** and never counts as your move.
  A run of the opponent's and automation's priority passes folds into one
  quiet **passed** line that counts each one's passes; your own passes stay
  moves. Hover or focus a card name to see the card.

While the opponent decides, the rail and the opponent's plate say so, with
the seconds it has taken ("Search 64 is searching · 3 s"), and the rail's
choice list reads "Waiting for Search 64" until there is a decision of yours.
An attacking creature steps toward the player it attacks, and a blocker is
joined to the attacker it blocks by a line. Combat
usually ends before your next decision, so the line stays after it, until your
next move, while both creatures are on the board. Cards
move between zones (hand to stack, stack to battlefield, an opponent's new
card from their plate) instead of appearing, and a triggered ability's stack
entry leaves from the card it came from; with reduced motion they appear
in place and every fact above stays.

Choices with no card on the board, such as Lessons outside the game, appear in
a tray over the board. Trays, the inspection, menus and the result card are
opaque and sit above the lines between targets, so a line never crosses them.

Combat is drafted whole. At an attack, every creature that can attack is
outlined; click one (or focus it and press Enter) to add it to the attack or
leave it out, or press inside the battlefield rows and drag a box across the
creatures you want. Every declarable creature touching the box highlights as you
sweep and joins the drafted attack on release; dragging a box only adds (click a
creature or **Clear the draft** to remove one), and box selection starts only
inside the battlefield rows during an attack draft — never on player plates,
your hand, the rail or the bars, and never for lands or cards that commit one
action at a time. Drafted attackers are ringed with a dashed line and lean in,
and a pile of identical creatures takes a count ("2 of 3 Tiger-Seal") in the
rail. At a block, drag a declarable blocker onto an attacker it can block, or
click the blocker and then the attacker. Dragging draws a live blocking line,
lights up legal attackers and hides an already-assigned blocker's old line while
you move it; release on a legal attacker to assign or move the block, or release
elsewhere or press Escape to put it back without disturbing a blocker you
clicked. On click, a blocker that can block only one attacker still takes it at
once, and clicking still takes a block back. When combat preview is unavailable
and blocks are asked one creature at a time, dragging from the blocker onto an
attacker still draws a blocking line and submits that block. The prompt reads
the whole plan ("Attack with 3 · not with
Tiger-Seal", "Badgermole Cub and Ally on Invasion Reinforcements") and
nothing is declared while drafting. The engine still asks about one creature
at a time: Academy's read-only combat preview replays the draft on a copy of
the game and returns the exact answers that declare it, or the first one it
refuses and why (a lone blocker of a creature with menace, a creature that
cannot attack). Confirm (Space) sends those answers one by one; if the game
asks anything the preview did not predict, sending stops and the board shows
where the game stands. While a priority stop updates, your draft and selected
blocker stay in place. If the first declaration is refused at the same
decision, you can edit the draft and confirm again. If a later declaration
is refused, the board keeps the accepted declarations and lets you draft
from that decision. Escape stops a confirmed plan between answers. A
reload keeps the draft for the same decision in that tab. While a draft is
open in live play it is the only way to declare: each creature's own answers
stay in the list of legal choices under a closed **Each creature's choice**
disclosure and are not numbered keys. Replay and Study keep them listed, since
recorded policy probabilities attach to them. **Pass turn** (F6) passes
priority until the turn ends.

Keys: `1`–`9` choose a numbered choice, Space presses the primary button (or
returns the board to now while looking back, unless a control with its own
action has focus), F6 passes the turn, ← and → step through the history, `?`
opens the shortcuts sheet (which also lists drag gestures), and Escape cancels
an active drag, closes an open list or, with none open, returns the board to
now — cancelling a drag does not also close a menu or clear a click-selected
blocker. Dragging adds no actions a click or key cannot take: short clicks
still act as clicks, floating card inspection stays closed mid-drag, a new
frame, disconnect or looking back cancels a drag with nothing sent, and
read-only recordings and Watch never start a drag. With reduced motion, dragged
cards, lines and selection boxes still follow the pointer directly, without the
return-to-hand or post-drop travel transitions.
**Settings** holds game controls, priority stops, live analysis, view options
(including **Card art credits**), and **New game**.

While disconnected or waiting for a submitted choice, the board remains
readable and actions are unavailable. Reconnect or reload to recover the actual
pending decision. If a choice is rejected because the game has advanced, read
the refreshed prompt and choose again. A submitted action cannot be cancelled;
after commitment, costs and revealed information remain committed. Reloading a
configured launch resumes that game even after a server restart (`/?game=<id>`).
See [Pending input, waits and recovery](#pending-input-waits-and-recovery)
if recovery fails.

For bot inspection, switch setup to **Watch Bot vs Bot** or open any recording
in **Replay**. Their decision surface shows the acting seat's permitted view
and the complete legal choices for that decision. Recorded policy outputs
appear only when available; missing outputs are not reconstructed. See
[Inspect a replay](#inspect-a-replay).

### Previewing payments

You control two Mountains and a Forest, all untapped, and hold Badgermole Cub
(`{1}{G}`). Point at the Cub in your hand, focus it, or drag it toward the
battlefield. The lands the
automatic payer would tap get a dashed outline and a mark with their order and
what each adds: the first Mountain `1 {R}`, the Forest `2 {G}`. A planned land
in a pile of identical lands leaves the pile while the plan is shown, so each
planned land has its own outline and its mark sits at the same corner of its
art. Nothing has
happened yet; pointing, focusing and dragging before release send no action.
Pointing at a row of **All legal choices**, or snapping a targeting line onto a
target that previews a payment, previews that choice the same way.

Cast the Cub. At the payment prompt the same lands stay dashed and the primary
button reads **Pay {1}{G}**; its accessible name reads the plan, "Pay {1}{G} ·
Mountain, Forest". Press Space or the button to tap exactly those lands.

The preview is the automatic payer's own plan for the position in front of
you. Academy asks the game for it; the browser never works one out. What it
shows is what **Pay the cost** does, including the lands it keeps back for other
cards in your hand and any mana it would produce beyond the cost.

**Choosing sources yourself.** When a cost leaves a real choice of sources,
each land you may tap is raised.

1. To accept the plan, press Space or the primary button.
2. Point at another land to preview "tap this one, then pay the rest
   automatically": that land becomes `1` and the marks move. The prompt, the
   **Pay** button and the line under it keep describing Pay's own plan, so the
   rail holds still under the pointer.
3. Click a land to tap it. It shows as tapped, its mana appears on your plate,
   and the plan for the rest is asked again and redrawn. Tap another land or
   press Space to finish. When your tap leaves no choice for the rest, the game
   finishes the payment at once, tapping what the plan showed.

A cost with no strategic source choice pays as soon as you cast, with no
payment prompt, so the marks you see while pointing at the spell are where you
see those taps beforehand. That applies to dual lands and other simple
choice-of-color sources just like basics — casting a `{1}{W}` spell with
Kyoshi Village and one Plains, `{2}{R}` with four identical Mountains, or
`{1}{G}` with Kyoshi Village and two Plains all finish without stopping
(tapping only Kyoshi Village and one Plains in the last case).

When a real choice stays open — such as Kyoshi Village and three Plains for
`{1}{W}` — **All legal choices** labels simple choice-of-color taps to match
basics: `Tap Kyoshi Village for {G}/{W}` (or `Tap Birds of Paradise for any
color`) beside `Tap Plains for {W}` and `Pay the cost`. Selecting a
choice-of-color source asks you to pick a color only when its colors lead to
different remaining options, spent colors or floating mana; if every color
finishes the same way, the color prompt is skipped. Non-simple mana abilities (sacrificing a
Treasure, paying life, or paying a filter fee) keep `Activate ability: <Name>`
and always stop.

**Mana payment** defaults to **Manual**. Choose **Automatic** to have the engine select sources, colors and spending after you accept a cost. You still choose whether to pay optional costs, which spells to cast, and which abilities to activate.

Under **Mana floats**, **Off** is the default. Choose **Only when the mana could be used** to float when a legal action can spend mana or tapping the source
can trigger an ability. Usable also offers an activation that pays life,
sacrifices a permanent or has another effect. **All** offers every legal
mana activation at priority. With **Manual** payment you can select these abilities and choose a land’s output while paying. A deliberate float at priority also lets you choose its output. The setting follows you when you swap sides.
Your priority stops still determine which windows pause for a decision: with
**Auto-pass outside stops** on, priority pauses only when you also have a
non-mana play; turn it off for **Full control** to pause solely to float mana.

**When there is no plan to draw.** The primary button then shows only the
cost, and a line under it says why when the game does:

| The game says | Meaning |
| --- | --- |
| A choice is needed first | X, an optional or alternative cost, an additional cost or a target can still change the price. Nothing is marked until that choice is made; the plan returns at the completing target or at the payment prompt when there is one. When the announced cost leaves no real source choice there is no prompt, and the cost is paid as soon as the last choice is made. |
| No sources needed | The cost is zero or your floating mana already covers it; the button reads "· floating mana". |
| No automatic payment | The automatic payer cannot fund this cost with its available taps. Use an offered mana ability and make its choices first. |
| Nothing | There is no automatic payment to preview. When choosing which floating mana to spend, choose the unit directly. Choice-output mana abilities and mana abilities with their own fee also have no preview. |

Mana with a spending restriction, an effect when spent or an end-of-combat
expiry is written under the pips on your plate. Restricted amounts are part of
the pool, not extra mana.

**Proposal and commitment.** A dashed outline means "would tap". A tapped
land means "has tapped". The pips on your plate are the only statement of mana
you actually have. Moving the pointer away changes nothing in the game. A
tapped source stays tapped and its mana stays in your pool until it is spent or
empties; there is no undo and no refund for a tap you regret.

**When a plan goes away.** A plan belongs to one decision. Submitting an
action, a new decision or a lost connection clears it, and an answer that
arrives late for an earlier decision is discarded. If the request fails, the
line under the button says so; **Pay** still pays automatically and every legal
choice stays usable. After a reload or reconnect in the same tab the pending
payment is restored and its plan is asked for again. Nothing about a plan is
saved in the recording or the browser.

Replay, Watch history and Study show the taps and spending that were recorded.
They show no plan: a preview was never part of the recording, and a live game
is never consulted to decorate a recorded position.

### Pending input, waits and recovery

Cast a spell, then follow its target and payment prompts. If the bot's next
decision fails, your submitted choice stays committed and the board stays
visible. Read the failure, then choose **Retry bot decision**, or **Retry search**
for a search failure, when offered. Retrying asks the same bot to finish its
pending decision; it does not cast your spell again. Academy acknowledges your
committed choice before starting the following bot work. Its receipt remains
fixed while later bot actions advance the board.

**Who chooses.** Read the chooser and requested input beside the current prompt.
The player choosing can differ from the player whose turn it is. Target and
cost prompts identify their source and required input. Finish a pending target
or cost choice before trying another spell. The rail's list keeps the complete
legal choices accessible. A pending cost can require a deliberate
mana activation before **Pay the cost** becomes available;
[Previewing payments](#previewing-payments) covers what **Pay the cost** would tap. Mana you have produced
and not yet spent appears as a floating-mana readout beside the board, by
color, until it is spent or empties. The list is the complete
set of legal actions at this decision. Academy does not say why an action is
absent from it: a card with no listed action has none right now, and its
details show its printed text and current state, not a reason.

**What is taking time.** Live status distinguishes loading a bot, choosing its
action, searching and advancing the live game. When a wait lasts more than a
second, elapsed time shows how long that operation has been running; it is
not a completion estimate. Recorded playback shows saved choices and
consequences without running a bot. Looking at an earlier line of the history
changes only what the board draws. **Disconnected** or **Reconnecting** takes
precedence when the connection cannot confirm the live state. The last board remains readable, with submission
disabled until recovery confirms the current decision.

**Retrying a bot.** An unsuccessful bot proposal leaves its native decision
unchanged. Retry is explicit; reloading, reconnecting or changing priority stops
does not retry it or select a replacement action. Costs already committed by
earlier choices remain spent. If the bot cannot safely continue, Academy
explains why retry is unavailable; start another game explicitly. A bot-load
failure leaves setup available to correct the selection and try again.

**Starting Watch.** Loading identifies each seat and shows elapsed time before
the recording opens. If a start reply is lost, the next **Start watching** in
that tab checks the retained setup attempt; it cannot create a second pair of
bots or apply changed form choices to that attempt. A missing setup after a
server restart is reported explicitly; choose Start again to submit a new
attempt. While a live decision is being produced, you can read earlier saved
positions. **Live frontier** status stays separate from the selected recorded
position and names the revision it belongs to.

**Recovering a submitted choice.** If the connection drops just after you act,
reconnect or reload in the same tab and wait for recovery before choosing again.
Academy keeps the unresolved submission in that tab's
`sessionStorage['academy.tab.<game>']` and checks whether it was accepted. If it
must resend, it resends that exact submission so an accepted choice is not
applied twice. If the decision has moved on (including across a server restart,
where prompt ids advance past every stored prompt id), Academy reports the
stale outcome and refreshes the board. Going back or clearing a selection
cannot cancel a submission.
In live Watch, recovery checks the recorded position before another step;
do not repeat a step just because its reply was lost. If that read fails,
**Refresh saved head** must establish the frontier before production is available.
After a bot failure, recorded navigation remains available, but playback stops
at the live edge. Only **Retry bot decision** or **Retry search** retries it.

**Across idle eviction and server restarts.** A game with an open board never
idles out, however long you think. A game nobody has open leaves server memory
after 15 minutes, and neither that nor a server restart ends it. Reloading the
page, opening `/`, or pressing **Resume** in **History** rebuilds the live game
by stepping the engine through the stored record and resumes at the exact
pending decision with the same game id and history. Rebuilding a game advances
its revision, so a choice submitted before the game left memory is refused as
stale and the board shows the recovered decision to choose again. Academy never
silently deals a replacement or replays a stale submission into a newer state. If the opponent's policy
cannot be loaded on resume, Academy shows the error with a link to **Replay**
and leaves the game unfinished so you can restart with the policy and resume.
If the engine contradicts the stored steps, Academy refuses with
`"no longer plays as recorded at step N"`, marks `record_diverged` (no winner
invented), and links to **Replay**. Unfinished Watch games are not rebuilt
after eviction or restart; they stay readable in Replay.

Back, Escape and clear controls discard only unsubmitted local selections.
Once an action commits, its costs and revealed information remain committed.
Cancellation is possible only when the current legal choices explicitly offer
it; there is no general browser undo.

### Choosing priority stops

Attack with Savannah Lions while holding Razor Rings and the mana to cast it:
with default stops, Academy pauses in **Declare attackers** (and again in
**Declare blockers** after blocks, while creatures are still attacking) and
offers **Cast Razor Rings**. The step bar along the bottom shows filled squares
on **Attackers** and **Blockers** for both turns. On a turn where nobody
attacks, or when you have no legal non-mana play, combat runs straight through
and each automated pass is marked **auto** in the history.

To hold up a response at another step — such as the opponent's upkeep — press
the upper stop in the **Upkeep** cell of the step bar, or open **Menu**, keep
**Auto-pass outside stops** checked, and check **Upkeep** under **Opponent
turn**; the two are the same setting. Press an active stop such as **Attackers**
or **Blockers** on the step bar to change it to **Once** or **Off**.

Each stop on the step bar has three states, and a press moves to the next:
**Off** (a hollow square), **Every time** (filled) and **Once** (a dot inside).
A once stop pauses the next time the game arrives at an eligible window for that
step on that turn and then turns itself off; the history says so ("Stopped
once at Upkeep on the opponent's turn; that stop is now off"), marked as
automation like the passes Academy makes for you. A once stop on **Attackers**
or **Blockers** waits for a combat with attacking creatures before spending
itself. The Menu shows a once stop as checked. The stops are one Tab stop: arrow
keys move between them, up and down between the two turns, and Space or Enter
presses one. Stops are a setting of live play: a recording's bar shows only the
steps of the turn, with the current one marked. A stops change on a finished
game's result card is this browser's preference for the next game; the finished
game and its record do not change.

**Your turn** and **Opponent turn** say whose turn contains the window. The
available steps are upkeep, draw, first main, beginning of combat, declare
attackers, declare blockers, combat damage, second main and end step. With
**Auto-pass outside stops** on, a priority window pauses only when you have at
least one legal play other than passing priority and activating mana abilities
(untapped lands or mana-producing abilities alone never hold priority, even with
**Mana floats** enabled), and either the stack is nonempty (with stack stops on)
or the current step is checked. **Declare attackers** and **Declare blockers**
stops hold only while creatures are attacking; if no creatures attack, or if the
only attacker leaves combat before blockers, those steps auto-pass without
phantom prompts. To pause before combat on an empty board, set a stop on
**Beginning of combat** or turn on **Full control**.

| Setting | Default and effect |
|---|---|
| **Auto-pass outside stops** | On. Automatically passes priority outside active stops, and whenever your only legal actions are passing and mana abilities. Turn it off for **Full control** (the Menu badge reads **full control** and the step bar reads **Full control: every priority choice pauses**): every priority window with more than one legal action pauses, including empty combat and mana floats. |
| **Your turn** | Stops in **First main**, **Declare attackers** (while attacking), **Declare blockers** (while attacking) and **Second main**. |
| **Opponent turn** | Stops in **Declare attackers** (while attacking), **Declare blockers** (while attacking) and **End step**. |
| **Stop when the stack offers a response choice** | On. Pauses whenever the stack is nonempty and you have a non-mana play, even at an unchecked step. |
| **Reset to defaults** | Restores these settings. |

Changing a stop can immediately advance the game. Target, payment, Learn,
attack and block decisions still appear whenever they offer a choice; stops
only govern priority passing. Sole legal actions are skipped in either mode.
Settings are remembered in this browser, and reconnect restores the active
game's server settings.

When **Pass Turn (F6)** is available, activating it yields through priority
windows, including checked stops and stack responses, until the turn ends or
a non-priority decision needs you. In **Full control** (with **Auto-pass
outside stops** off), priority windows continue to appear. Automatic passes are
marked **auto** in the history and recording. The move pace does not change
these settings.

### Reading the history

The opponent casts Lightning Strike at your Earth Kingdom Jailer, and its
Vengeful Bloodwitch drains you when another of its creatures dies later in
the turn. When the decision comes back to you, the rail's **History** reads:

```
Turn 6 · Heuristic
Cast Lightning Strike, targeting Earth Kingdom Jailer
  3 damage to Earth Kingdom Jailer
  Earth Kingdom Jailer died · lethal damage
Attack with Gray Ogre
  South Pole Voyager blocked Gray Ogre
  Gray Ogre died · lethal damage
  Trigger · Vengeful Bloodwitch, targeting you
Vengeful Bloodwitch's trigger resolved
  You lost 1 life (now 17)
  Heuristic gained 1 life (now 21)
```

For anything that changed on the board, the history says what happened, in
what order and why.

**What happened.** A move is something a player chose to do (cast, activate,
play a land, attack, block, pass) and sits at the left edge. Turn-based
actions, state-based losses, and triggers with no recent cause also appear at
the left edge, naming their rule or source instead of a player move. The lines
indented under a line are what followed from it: damage, deaths, destruction,
exile, token creation, power/toughness changes, control changes (and their
cleanup return), counters, triggers reaching the stack, abilities resolving or
being countered, cards drawn, discarded, revealed or sacrificed, a library
searched, a scry or surveil, life gained or lost other than by damage, a player
losing. An
ability names its targets as a spell does. A fact about a card you may not
see names it by role ("a card", "a permanent"). Tapping for mana is never a line.

**In what order.** The list is strictly the order things happened and is never
rearranged. A spell that resolves with nothing in between (ignoring priority
passes) reads as one folded line with its consequences beneath, as Lightning
Strike does above. When something happened in between, such as your response,
the resolution is its own line in its own place:

```
Cast Lightning Strike, targeting Earth Kingdom Jailer
You cast Accumulate Wisdom
  You drew a card
Lightning Strike resolved
  3 damage to Earth Kingdom Jailer
  Earth Kingdom Jailer died · lethal damage
```

Selecting "Lightning Strike resolved" marks its cast above. A triggered
ability whose event is long past, such as one at the beginning of upkeep, is
a line at the left edge that names its source and no player.

**Why.** A line is indented under the line that caused it. When no player
chose it, it ends with the rule that did:

| The line says | The rule |
|---|---|
| lethal damage | A creature with damage at least its toughness dies. |
| toughness 0 | A creature with toughness 0 or less dies. |
| loyalty 0 | A planeswalker with no loyalty dies. |
| legend rule | Of two legendary permanents with one name under one player's control, one stays. |
| Aura had nothing to enchant | An Aura attached to nothing legal goes to the graveyard. |
| 0 life | A player at 0 life or less loses. |
| drew from an empty library | A player who drew from an empty library loses. |
| 10 poison counters | A player with 10 or more poison counters loses. |
| illegal targets | Every target of a spell or ability was illegal when it tried to resolve, so it left the stack without resolving. |
| draw step, cleanup | The turn itself did it: the draw for the turn, the discard to hand size. |

The history says why something happened in the game. Why a creature is a 4/4
is answered by [inspecting the card](#playing-on-the-board), which lists where its power and
toughness come from.

On a turn with more than four consequences, each move shows a count ("3
consequences") in place of its indented lines; press it to open them.

#### Looking back

Select "Earth Kingdom Jailer died". Any active walk stops and the board
becomes the board as it was after that line, with what the line is about
raised. The Jailer is gone, and the top bar reads

```
Turn 6 · Heuristic's Lightning Strike · Earth Kingdom Jailer died · lethal damage · Esc for now
```

When a folded spell has consequences under it (like Lightning Strike above),
selecting the cast line shows the spell on the stack with its target lines
before its effects land on the next line; when a folded spell has no
consequences (such as a creature resolving cleanly), selecting it shows the
resolved permanent on the board. Lines that happened at once in the rules (two
creatures dying in one state-based check, a spell's cast and target, a
sweeper's deaths, or multiple effects of a single resolution) share one board.

← and → move the cursor one line at a time when no control has focus. Esc
returns to now, and `?` opens keyboard help. The step bar follows the board
you are looking at. Damage window totals and the result card wait until the
cursor reaches now. The decision in the rail stays live throughout: pressing
one of its buttons or a number key (`1`–`9`) returns to now and commits that
choice. A click on an earlier board, or Space, only returns to now and never
acts. History lines, scrubber ticks, ← →, caption hover and **Replay turn**
always respond directly. Reconnecting or reloading returns the cursor
to now and shows the actual pending decision; turns already told are not
walked again.

**Every move lands on the table.** After any move—yours or the opponent's—the
board starts at the first unread board stop and walks forward one kept board at
a time: first Lightning Strike on the stack targeting Earth Kingdom Jailer, then
Jailer on the battlefield with 3 damage, then Jailer gone with a dashed slot
reading **lethal damage** where it died, then a death trigger on the stack.
Passes are never a stop, bare turn-start draw steps alone do not walk, and lines
that happened at once share one stop. When a combat involves three or more
creatures, the combat damage stop also opens a compact breakdown table
(Attacker, Blocked by, Dealt, Took, Result) whose rows raise their combatants on
hover and pause the walk while you read it; smaller combats keep the single-line
caption. Under the board, in a steady-height bar that never shifts the board or
your hand once a move has occurred, sits one caption telling that stop's line or
lines with any rule, a pace control (**1.4s** Study default, **Quick** 0.7 s, or
**Off**, also in **Game → Move pace**), **Replay turn** to replay from that
turn's first stop, and a scrubber with a tick for each stop of the turn (heavier
at the first stop of each move). Click a tick to put the cursor on that stop and
hold; rest the pointer on the caption or combat table to hold the walk while you
read it. The rail highlights the matching history line throughout. With **Off**
or reduced motion the board stays on now and nothing walks automatically, while
the scrubber, captions, **Replay turn**, the combat table and **Since your last
move** all still work by hand.

**Motion** is the drawn difference between one board stop and the next,
180 ms at most, and never holds a choice:

| Fact | On the board | With reduced motion |
|---|---|---|
| A trigger or activation | Its stack entry leaves from its source card | The entry appears; the line names its source |
| A death | The card goes to its owner's graveyard count; its rule reads on the card as it leaves | The rule reads as a chip where the card was |
| Damage, counters | The number on the card changes ("−3", "+1/+1 · 2") | The same number, without the count |
| A block | The blocker stands opposite its attacker, joined by a line | The same line |

Every fact is a line of the history whatever the motion setting.

Sentences name a player shortly ("Heuristic drew a card"); the plate and top
bar keep how it decides ("Heuristic · scripted"). A move alone is not
evidence that a spell resolved or a creature died; the lines beneath it are.

#### What is kept

| Situation | What you see |
|---|---|
| Earlier in this page's session | Every line and its board, for the latest 512 boards. Lines that happened at once share one board; lines past one Command's position cap share the nearest later board and say **shared with N later lines**. |
| Before that, or before a reconnect or reload | Every line with its exact card names intact; selecting it raises its subjects on the current board where they still are, and says **Board not kept for this step**. |
| A fresh reload | The latest 256 events of the active match, with their card names intact; an earlier part is labelled unavailable. |
| After the game | Every line and its per-line board, in [Review](#inspect-a-replay). |

In Replay the same history is read from the recording. For a recorded game you
played, the rail holds your full history across the game and selecting any
line jumps to its transition and its per-line board inside that transition (a
recording without kept per-line boards falls back to the transition's board
and says which lines share it; a board patch that cannot be reconstructed says
**Board unavailable for this step**). For a watched bot game, history stays strictly
the selected acting viewer's track before that decision's cursor, and
spectator reveal never adds names to it. **Try another line** has its own
history, and **Return to recorded decision** restores the recording's.

A recording made before moves were recorded as events has no causes: its
history is a flat list of moves and facts in their recorded wording, with no
indentation and no rules. Older recordings without event data say
**Consequences not recorded**; missing events do not mean that nothing
happened.

### Card art

Cards on the board, in replays and in the deck browser show their Scryfall
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

**Card art credits** in the Board's **Settings → View** and on the setup screen
lists the artist, source and copyright of every image shown so far. The package
itself contains no card images, and deleting the cache directory is safe.

### Inspect a replay

A recording opens on the same board as live play, read-only. Open **History**
(`/games`), choose a game, and press **Replay** (`/replay?trace=<id>`). Human
games (finished or unfinished), watched bot games, and imported training or
evaluation games all use these controls. At the end of a game you played,
**Review this game** on the result card opens it from the first position. Bare
`/replay` redirects to `/games`.

Every game you start is recorded as you play. **History** (`/games`) lists
every game in `traces_dir`, newest first, with search, **Open a recorded game
file**, and filters for pairing (**Human vs Bot**, **Bot vs Bot**), status
(**In progress**, **Completed**, **Stopped**), bot, deck and date. Player 1 is
always on the play; the human seat is recorded as **You**. Every row has
**Replay**, and every unfinished played game also has **Resume** (`/?game=<id>`).
Played games show their turns, duration, recorded steps, and how many of your
choices were made with [analysis shown](#analysis-while-you-play). A trained
bot appears under its [short name and digest](#how-a-trained-opponent-is-named).
You can also [launch a replay directly](#open-a-replay-directly) from the
command line.

#### Follow the timeline

| Position | What the board shows |
|---|---|
| Choice | The chooser's recorded board and every legal offer, on its card and in the rail's numbered list, read-only. The played offer is marked ✓; where the policy was recorded, each offer's probability is on its card and row, with the value, update and any search detail beside the prompt ("not retained" where nothing was recorded). |
| Consequences | The board after the action, with what it did, and why, in the rail's history; selecting a line inside the transition shows its per-line board. In a game you played this is your own view, your hand included, whoever acts next. |
| Result or latest state | The final board under a result card, or the latest retained state of an unfinished recording. An unfinished or stopped recording has no invented winner, and one opened from a file says why and where it stops. |

The transport bar sits where live play has your hand's input:

| Control | Key | Result |
|---|---|---|
| **Previous** / **Next** | ← / → | One position back or on |
| ⇤ / ⇥ | Shift+← / Shift+→ | The previous or next recorded choice |
| **Play** / **Pause** | Space | Advance a position per beat, at 1×, 2×, 4× or 16× |
| Scrubber | arrows when focused | Any position; turns are ticked and choices marked under it |
| | Home / End | The first or last position |

Under it, the **Decisions** strip draws the game as one line: each choice is a
tick, coloured by whose it was, or a point on that chooser's value line where
a value was recorded; turns are ruled and numbered. Click it to jump. It is
open in Replay and folded in live play, where it counts decisions; positions
from before a reload open only in Review after the game. A line of the rail's
history jumps to its transition position and per-line board, with what it is
about raised; in Review each move is numbered by turn in its gutter, with the
whole move in its tooltip, and the history takes the rail's remaining height.
The rail draws the stack only while something is on it in Review, and the
answers only in the choices list. Keys never act while you type in a field.

The replay address keeps the position, the perspective and the information
view. Reloading restores them, with a saved position in this tab as a
fallback. A position that is no longer available is explained so you can
select another.

#### Perspective

The **Perspective** control in the top bar chooses who sits at the bottom:
**Follow the chooser** (the default for bot games: each choice is drawn from
its chooser's seat, hand included, and the board stays turned through that
choice's consequences), or a fixed seat (the default for a game you played).
Turning the board never reveals a hand or changes the recorded policy inputs.

At bot choices, the acting view shows what that bot was allowed to see.
**Show both hands** appears at Watch choices that captured a spectator view and
at every position of a played game once it has finished (`game_over`). It does
not enrich the bot's recorded evidence or acting-view history. In Watch and
imported games, consequences and result boards use the recorded view of the
seat named in the menu's perspective line when a private post-action view was
not captured.

**Settings** holds the recording's kind and status, each seat's identity
(policy ID, runner, provenance), **Refresh saved head**, analysis controls, and
game numbers.

A live decision already being produced may finish saving while you read; your
selected board stays put. A pending Search timeout is retained across reload
and shown with its message. At the live edge, **Retry search** explicitly
retries that decision; browsing saved positions never retries it.

#### Analysis

An analysis reads a played game against a policy: at each of your decisions
with more than one legal action, what that policy would have done in your
seat. Academy replays the game with the policy in your seat; it sees only what
your seat saw, and the game follows the moves you made. Opening Replay on a
finished played game starts an analysis automatically when the game has none
yet: by your opponent's policy when it is a registered trained policy,
otherwise by Academy's recommended policy, the one setup offers first. On an
unfinished played game, analysis runs only when you click **Analyse** (never
automatically). Replay is usable at once; **Analysing 34 of 81** counts up
while the marks fill in.

At one of your choices the heading reads **Analysis by** the policy, and every
legal option carries its probability and the policy's estimate after it, 0 to
100 from your seat. Your choice is labelled **you** and the policy's first
choice **its choice**. An estimate after an option reads the policy at the
next decision: your own when you choose next, otherwise the opponent's in a
world re-dealt from what you knew, never the real hidden cards. With closed
decklists nothing can be re-dealt, so options after which the opponent
chooses have no estimate.

Each of your moves in the rail reads **same** when you chose as the policy
would and **differs** otherwise, with a small track of the two estimates on
one 0–100 scale: the filled dot after your choice, the ring after its own.
These are the policy's readings, not grades; a policy weaker than you will
often differ for good reason.

With an analysis, the **Decisions** strip draws the policy's estimate at each
of your decisions on a fixed 0–100 axis with a rule at 50. Its caption names
the measure: **Expected score** for a policy whose values are signed outcomes
(a draw counts half), **Win chance** for one whose values are win logits.
Dots mark where you and the policy chose differently.

**Next key moment**, beside the replay's step buttons, jumps to the next of up
to eight decisions worth a look,
in game order: the five where the policy's first choice had the most
probability over yours, and the three where its estimate moved most by your
next decision. At one of them a line names why, such as **Key moment ·
Differs · 62% on its first choice, 8% on yours** or **Its expected score moved
48 → 71 by your next decision**. A reading that barely moves, as an untrained
policy's does, has none.

**See its line**, at any analysed decision of a played game (finished or
unfinished), plays the policy's first choice there and carries on: the policy's
first choice at each of your decisions and your opponent's own choice at theirs,
to the end of the next turn or the end of the game. The board shows each move
from your seat; **Previous** and **Next**, or the arrow keys, step through it,
and **Return to the game**, or Escape, goes back to the decision. The line is
one play-out of the real game's cards, not a forecast, and nothing about it is
saved. Your opponent's moves are named once the game is over; in an unfinished
or stopped game they read **Opponent's move**. Priority passes fold into the
move before them.

**Play on from here**, at any of your deliberate decisions of any saved played
game (finished or unfinished), opens Play with that game's decks, cards and
opponent and replays its moves up to the decision; **Play on** starts it, and
from there you choose and the opponent plays as in any game. It is a new game
in History: Replay names the game it came from and its row in History reads
**played on from another game**. The moves before the decision are not listed
in the new game's live move list; its Replay lists them all. If this build no
longer plays the old game the same way, Play says which step differs and
nothing starts.

**Analyse with…** in **Settings** runs any other registered trained policy; each
run is saved beside the game, and **Shown** switches between them. Analyses
are never part of the game's recording or its exports.

| You see | Why | What to do |
|---|---|---|
| **No analysis yet · Analyse with… in Settings** | Your opponent was Random, Passive, Heuristic or Search, and the recommended policy cannot read these decks, or the game is unfinished. | Choose a policy under **Analyse with…** (or click **Analyse**). |
| **No analysis: no registered policy can read this game** | This server has no trained policy for these decks. | [Register a policy](#trained-opponents-and-other-bots), then restart Academy. |
| **Analysis stopped after 12 of 81 decisions**, with a reason | The policy failed there, the game stopped replaying as recorded, or Academy restarted. Earlier decisions keep their readings. | **Analyse again**. |
| A line naming another build | The game was recorded by another world or card set and cannot be replayed here. | Open it with that build. |

A policy with Search is not offered: Search plays out from the real hidden
cards. One analysis runs per game at a time; an unfinished or stopped game is
analysed up to its latest recorded position.

#### Analysis while you play

**Show analysis**, in the top bar's Settings during your game (the bar then
shows an **Analysis on** chip that turns it off), reads your
decisions as you play. It is off at the start of every game. Turned on, it
reads with your opponent's policy when that is a registered trained policy,
otherwise with Academy's recommended policy; with more than one, a menu
beside the switch changes it. The reading is the same as Review's: the
heading reads **Analysis by** the policy, and each option on its card and row
carries the policy's probability and its estimate after it. Nothing is
chosen, raised or reordered for you, and no key plays the policy's choice.

Turning it on also opens the strip under the board, which draws the policy's
estimate from your seat at each of your decisions this game, those made
before you turned it on included (read after the fact, so they mark nothing), on Review's
fixed 0–100 axis; hovering it names the nearest point by the move before it
and its score. The line keeps its points when you turn analysis off and on
again, and starts over if you switch to another policy.

Academy first replays the game so far with the policy in your seat, which takes
a few seconds, then reads each new decision once its board is on the table.
The board never waits for a reading: it arrives in its own message after the
decision, and a choice you make before it arrives reads as made without it.

Each choice you commit after its reading reached you is saved as made with
analysis shown. Replay marks that move **analysis shown**, **This game**
counts them, and **History** shows them on the game's row.
Turning **Show analysis** off stops the marking from your next decision; it
never unmarks a choice. A game you play without it saves exactly as before.

| You see | Why | What to do |
|---|---|---|
| **No registered policy can read this game** under the switch | This server has no trained policy for these decks. | [Register a policy](#trained-opponents-and-other-bots), then restart Academy. |
| A reason under the switch, and no probabilities | The policy failed, or the game could not be replayed here. Your game goes on. | Turn the switch off and on to start again. |

#### The game in numbers

The result card of a game you played ends with one line of its numbers, such as
**9 turns · you cast 6 spells and attacked 4 times; Reader cast 3 and attacked
2 times**. In Review, **This game** in **Settings** gives turns, whether you were on
the play or the draw and how long the game took, and a table of both seats:

| Count | What it counts |
|---|---|
| Lands played | Your decisions that played a land. |
| Spells cast | Cards you cast. Copies of a spell are never cast and do not count. |
| Attacks | Your decisions that declared an attacker. |
| Cards drawn | Cards drawn from your library after the opening hand. |
| Damage dealt | Damage your sources dealt to your opponent. |
| Damage taken | Damage dealt to you by any source, your own included. |

With an analysis shown, **This game** also reads **Decisions with a choice: 20 ·
chose as Reader would: 8 of 20**, where the policy's first choice was yours,
and the policy's estimate from your seat at the start, at its highest and at
the end. A game with choices made while [analysis was
shown](#analysis-while-you-play) reads **Chosen with analysis shown: 3
decisions**. An unfinished or stopped game is counted up to its latest recorded
step, and the cached counts clear automatically when a resumed game appends new
moves so the next read recounts fresh numbers. An older game is counted the
first time you open it. A game recorded by another build has no counts.

#### Continue a live Watch

Open Play setup (`/`), switch to **Watch Bot vs Bot** (or launch with
`managym academy watch --player X --player Y`), select both seats, decks and a
seed, then press **Start**. Each seat uses the same player menu as Play's
**Opponent**. A managym-only install offers Random and Search; manabot adds
Greedy, Heuristic and trained checkpoints. The game opens in **Replay**
(`/replay?trace=<id>`) and is saved in **History**.

Browsing earlier positions reads the recording. At the latest position of a live
Watch, **Continue** requests the next bot decision; **Play** can keep producing
and presenting decisions from there. The status distinguishes reading saved
data, producing a decision, waiting, finished and stopped. Imported and
completed recordings only play saved data and never run a policy.

If production fails, the recorded choice remains available. After a lost reply,
refresh the saved latest state before explicitly retrying; the action may already
have been recorded. Reloading the page recovers a live Watch while its server
state is still in memory. After a server restart or 15-minute idle eviction,
the saved Watch recording remains readable in Replay, though live Watch
production cannot be resumed.

#### Study and older recordings

At an eligible choice, **Try another line** pauses playback and live production
and opens an isolated branch on the same board: the historical offers become
live, and choosing one plays it against the branch. **Return to recorded decision** restores that
exact choice and its history; the original recording and result stay unchanged.
Study roots live in server memory only, so **Try another line** appears only
while that game's roots are in memory. Starting a rematch or a new game leaves
the game just played in memory until it idles out, so its lines stay open to
Study. After idle eviction or a server restart,
**Resume** (or **Play on from here**) on a played game rebuilds Study roots for
every recorded decision; on a game nobody has resumed yet, recorded playback
and **Play on from here** remain available while **Try another line** waits
until the game is resumed.

Older recordings use the same viewer with the frames they retained. Missing
legal offers, Study support, consequences or policy outputs are explicitly
unavailable; they are never inferred from a later frame. Reading an existing
recording needs no original checkpoint or policy provider. Import verification
and live continuation have their own prerequisites below.

A game you played is also played again from its seed, decks and actions as you
step through it, which gives your own view at every position and both hands
after the game. Each step is checked against the stored track: the acting
board and the action must match, and the final board and result. Where they
stop matching, or for a game recorded in another world or card set, the board
is the recorded one, with a line saying how far your own view was replayed and
which world recorded the game.

Games are stored in `play.sqlite` under the traces directory, which defaults
to `academy/traces` in the directory you launched from. Launch from the same
directory, or pass the same `--traces-dir`, to see earlier games. The file
holds private deal seeds; keep it private.

### Watch a game from an evaluation

1. Find the evaluation's recorded games, `games/<cell-id>.jsonl` under its
   output directory. Copy or download that artifact locally if the evaluation
   ran remotely. Use the games file rather than the evaluation's metrics or
   summary file.
2. Open one from the command line with
   `uv run --no-project managym academy replay <file> --game <n>`
   ([Open a replay directly](#open-a-replay-directly)) and skip to step 4, or
   launch `uv run --no-project managym academy`, open **History**
   (`http://127.0.0.1:8000/games`) and use **Open a game file**.
3. If it contains several games, select a **Recorded game** and click
   **Open selected game**. A single game opens automatically.
4. Use **Next**, **Previous**, or **Play** to follow choices, consequences and
   the result ([replay controls](#inspect-a-replay)). Academy re-executes the
   game and checks its state digests without running the original players
   again. Search outputs the evaluation recorded appear at each choice; what
   it did not record is marked **not retained**.

This needs only `managym`. Replaying does not require the original checkpoint.
A game bound to an incompatible world must be opened with the build that
recorded it, which the refusal names; opening never upgrades frozen evaluation
evidence.

Two kinds of evaluation game are command traces
(`traces/<cell-id>.commands.jsonl.gz`) instead: every game of an evaluation
run before this record, and games that start from a constructed scenario
position, which no seed and decklist can reproduce. They open through the same
command and file picker, verified by manabot. Stop any running Academy server
and install it into the same environment, at the matching release version:

```bash
uv pip install --python .venv/bin/python --only-binary :all: --torch-backend cpu manabot
```

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
- Verification of earlier evaluation command traces and exported training
  games in **Open a game file**. Games recorded by managym open without it.

Independent Python policies work without manabot. A saved game that was played
against a checkpoint still replays without its provider; continuing it needs
the original admitted policy. Recorded policy and search evidence comes from
the saved game and works in installed Academy; missing outputs stay unavailable.
The live board does not load example analysis from a repository checkout.

#### How a trained opponent is named

Academy shows the first and last parts of a long checkpoint name, followed by
its identity's first ten characters. For example, `run/checkpoints/step_8`
appears as `run/…/step_8 · 7ac5d442da`. The human seat is recorded as **You**.

The same compact identity appears on the board, in History, Replay and Watch.
Open **status** on the player's bar with a mouse or Tab and Enter to
read the full recorded name, identity, producer and available provenance.
Registered policies also retain the runner, artifact digests and configuration.
Their **Policy ID** binds that whole execution; historical checkpoint identities
remain the checkpoint file's SHA-256. A runner change creates a different recorded
player even with the same checkpoint. Artifact digests remain available separately
for grouping recordings by weights. Missing provenance stays absent.

Shortening is display only. Recordings, exported games and filter links keep
the full name and identity. Search **History** by either, or filter with the
**Bot** filter.

### Play and watch a trained checkpoint

Register a baseline checkpoint using a saved specification, `challenger.json`:

```json
{"provider": "manabot", "artifact": "/models/self-play-raw.pt", "config": {}}
```

```bash
uv run --no-project managym academy --policy challenger.json watch \
    --player challenger --player random
```

Play opens at setup in **Watch Bot vs Bot** mode with both seats selected.
Choose decks and press **Start**.
Either seat can use a registered policy. **Registered policy** and **Policy**
use the same direct selection in Play and Watch; provider and runner details
remain available as provenance. They preserve an
unavailable selection rather than substituting a newer model. Keyboard navigation
uses the browser's select controls.

With manabot installed, its provider discovers `.pt` and portable `.safetensors`
checkpoints under `MANABOT_RUNS_DIR` (default `./.runs`), explicit
`ACADEMY_PLAY_CHECKPOINTS` sources and the runs `etude train` keeps on this
computer (`~/.etude/track`, moved by `ETUDE_HOME`). Python can supply provider-owned discovery
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
filenames never supply training counts. An incompatible checkpoint stays listed with the loader’s refusal reason. A selected policy is checked against the setup before Start in Play and Watch;
a mismatch shows the provider’s reason beside the picker. Start checks again,
so changed artifacts and setups are still refused precisely. If the checkpoint recorded exact decks, Academy lists them as **Bound decks**. Registered launch aliases appear
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

#### Search

Choose a registered policy, then change its **Search** control from **None
(raw policy)** to **Search**. This works for the opponent in Play and
for either Watch seat. Its configuration includes the exact budget; details retain the magnetic
mirror update algorithm and recorded identities. While a decision is pending, the board shows
its search budget and elapsed waiting time. Reload preserves the configuration.

| Setting | Default | Range |
|---|---|---|
| Worlds | 1 | 1–64 |
| Depth, including the root choice | 1 | 1–40 |
| Decision ceiling | 30 seconds | Greater than zero, at most 30 seconds |

**Uniform belief (physical deal)** samples unseen cards with copy counts and
public knowledge. Three unseen copies of one card and one of another give a
hidden card a three-in-four chance of being the first. It ignores behavioral
history; it does not mean uniform actions or uniform distinct hands. Native
sampling preserves the bot's information and full legal choices. Retained belief
identity `managym.determinize/physical-deal/v2` names native determinization,
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

On a timeout, no bot action commits. The pending decision remains on the board
with **Retry search**. Only an explicit retry tries it again; reconnecting or
changing priority stops does not. The retry checks the match and revision.
Other search failures report their reason without substituting another policy.

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

Search is the only name in the browser, Python and the CLI. Recordings store
its summary under the serialized key `mirror`, and its error codes and receipt
schema names keep that spelling, so saved games stay readable. `managym.search`
holds the configuration, the summary and the magnetic mirror update.

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
runner calls `observe(view)` once after every committed state transition (after
setup, after each committed Command by either seat, and on terminal `game_over`),
with `view.choices` populated only when this seat is acting and
`view.observation.recent_events` carrying the last step's public events.
`choose(view)` proposes an action for the already-observed pending decision and
may be retried without another `observe()` call if a proposal fails or times out;
recurrent policies should update state in `observe()` only (native decision
memory is also populated on `view.observation.decision_memory_json()`).
`PolicyView` carries the observation, seat, complete ordered `choices`, one-use
`binding` and optional native text capture. It contains no live environment. Return
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
Recordings retain the complete descriptor and genuine outputs once. Replays
need no policy reload, original checkpoint or installed provider.
Unfinished played games survive browser reload, idle eviction and server
restart as long as the factory or saved specification is registered on the
restarted server. Arbitrary Python objects are never pickled into recordings.
Historical checkpoint recordings remain readable.

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
| Resume refuses because a policy cannot be loaded | The opponent policy is missing or not registered on this server (for example, restarted without `--policy`). The game stays unfinished in **History**; restart with the policy and click **Resume**, or open **Replay** to inspect what was played. |
| Resume refuses with `no longer plays as recorded at step N` | This engine build contradicts the stored moves. The game is marked `record_diverged` (no winner invented) and stays openable in **Replay**. |
| `replay.unreadable` | The file is not a game file: not JSON, JSONL or gzip of either, or it holds no game records. Use a run's `games/` directory or one of its files, not a metrics or summary file. |
| `replay.ambiguous` | The file or directory holds several games and none was selected. Select one by index or digest: `--game` in the CLI, or `game=` in Python. |
| `replay.game_unknown` | The selection is past the end of the file, or matches no digest or more than one. A recording id already names one game; omit `--game` (Python `game=`) when opening it. Selection applies only within a file or directory. |
| `replay.needs_manabot` | The file is an evaluation command trace or a game exported by `manabot export-game`, which manabot verifies. Install it at the same version: `uv pip install --python .venv/bin/python --only-binary :all: --torch-backend cpu manabot`. Games recorded by managym need no manabot. |
| `replay.rejected` | The game was recorded in another world or with other card content, mixes decklist disclosure, or was played without skipping forced decisions. The message says which and names the recording and running builds. |
| `server.running` | (Python) `serve` was called while an Academy from this process is still open. Close it first. |
| `server.closed` | (Python) This Academy has already closed. Call `serve` again. |
| `server.startup` / `server.shutdown` | The server could not start or finish stopping. The message names the failure. |
| Cards show drawn plates instead of art | The server could not reach Scryfall, or `ACADEMY_CARD_ART=off` is set. Check the connection; a failed card is tried again after ten minutes. See [Card art](#card-art). |
| No browser window opens | Open the printed `open` address yourself. On a remote machine, forward the port (`ssh -L 8000:127.0.0.1:8000 host`) and use `--no-open`. |
| No policies are listed | Register a Python factory or a saved policy specification; check any provider discovery error shown below the picker. |
| `No trained checkpoints found.` | Register a saved specification with `--policy <spec.json>` or supply provider-owned discovery sources through Python. |
| A checkpoint is listed but disabled | The row states what this revision cannot give it, for example `incompatible agent_player columns: checkpoint …, runtime …` or `unexplained record_token key …; classify it in managym/src/cards/token_renames.rs`. Added cards are compatible unless the checkpoint recorded exact decks and its content manifest changed. [What refuses a checkpoint](observation-and-model.md#what-a-checkpoint-is-bound-to) lists each case and its fix; otherwise run it from the revision that wrote it. |
| A checkpoint shows **not recorded** for its update | No `run.json` in the listed directories records those bytes. List the run directory rather than the file alone. A run that recorded no counts for its checkpoints stays **not recorded**. |
| `The selected opponent checkpoint is unavailable.` | The file was moved or deleted after it was listed. Restore it or choose another; Academy does not pick a replacement. |
| `The selected policy changed; restore the pinned artifact and runner, or restart Academy and explicitly reselect the policy.` | The artifact or runner differs from the pinned Policy ID. Restore the original execution, or restart Academy and deliberately select the new entry. |
| Earlier games are missing from **History** | Academy was launched from another directory. Launch from the original one or pass its `--traces-dir`. |
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
  their Commands and appear as **Auto** in live and replay logs. Turning off
  **Auto-pass outside stops** in the Stops panel enters **Full control** and
  surfaces every priority window with a real choice. Traces from before native
  skipping keep their recorded forced Auto steps.
- **Replay and fork identity.** Live, Watch, Replay and Study use one decision
  renderer. Study executes one exact semantic offer on a retained isolated root
  (a full clone of the native game; the engine has no undo) and **Return to
  recorded decision** restores the original address, frame and Command. Expired
  roots stay unavailable rather than being reconstructed from the screen.
  Imported game files get their own Academy identity while retaining the
  source digest; frozen `erd1` evidence keeps its original bindings.

The historical `/api/advice` fixture API remains available to conformance tools.
The live board does not fetch or display that example as current-game analysis.

## Choice navigation

Academy numbers every choice in the rail and places it on its card, in
live play, Replay, Watch and Study alike; there is no filter. Selecting a
choice sends its original offer through the revision-bound Command path, and
duplicate labels remain separate choices. Human usability acceptance (ETU-77)
remains open.

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

The presentation track is the one record of what happened in a game as one
viewer may read it. The rail's history, the recap, the move walk
(caption, scrubber and large-combat table), board marks, game-over summary
counts, the live decision strip count, and motion are all views of it; none of
them is a source of game facts, and none can alter or delay the committed frame.

```json
[
  {"seq": 41, "kind": "cast",      "group": 41, "labels": {"objects": {"12:0": "Lightning Strike"}}},
  {"seq": 42, "kind": "targeted",  "cause": 41, "group": 41, "labels": {"objects": {"19:0": "Earth Kingdom Jailer"}}},
  {"seq": 43, "kind": "resolved",  "cause": 41, "group": 41, "labels": {"objects": {"12:0": "Lightning Strike"}}},
  {"seq": 44, "kind": "damage",    "cause": 43, "group": 41, "labels": {"objects": {"19:0": "Earth Kingdom Jailer"}}},
  {"seq": 45, "kind": "died",      "cause": 43, "group": 41, "rule": "lethal_damage", "labels": {"objects": {"19:0": "Earth Kingdom Jailer"}}},
  {"seq": 46, "kind": "triggered", "cause": 45, "group": 41, "labels": {"objects": {"24:0": "Tiger-Seal"}}}
]
```

(Subjects, revisions, amounts, passes and `caused_by` are omitted here;
`cause`, `rule` and `labels` are optional fields omitted when absent.)

| Field | Meaning |
|---|---|
| `seq` | Position on this viewer's track. |
| `kind` | `cast`, `targeted`, `resolved`, `activated`, `triggered`, `ability_resolved`, `countered`, `passed`, `damage`, `destroyed`, `died`, `exiled`, `token_created`, `pt_changed`, `control_changed`, `counters`, `searched`, `looked` (scry, surveil), `sacrificed`, `drew`, `discarded`, `revealed`, `land_played`, `attack_group`, `blocked`, `turn_started`, `lost`, `life` (gained or lost other than by damage; damage to a player is told only as `damage`). State-based deaths are told as `died` (with `rule`); spell/ability destruction is told as `destroyed`. |
| `cause` | Optional; omitted when root. The `seq` of the earlier event this one follows from. A move is one of the player-choice kinds (`cast`, `activated`, `land_played`, `attack_group`, `blocked`, `passed`) with no `cause` and no `rule`. |
| `rule` | Optional; omitted when chosen by a player. The state-based, turn-based, or resolution rule (`illegal_targets`) that did it. |
| `labels` | Optional; omitted on older recordings or when no object or stack entry is cited. Viewer-safe `PresentationLabels` (`objects`, `stacks`) naming only the objects and stack entries this event mentions, at exact incarnation. |
| `group` | The `seq` of the event's root, its move. It is not a turn identifier; `turn_started` events establish turn headings. |
| `caused_by` | The Command that committed the step, or null. A move by the viewer's seat with a null `caused_by` is automation. |
| `from_revision`, `to_revision` | The frame revisions the event lies between. |

**Source.** The engine keeps an optional, boxed presentation sink on `Game`,
beside its statistics and card-capture sinks and under their rule: search
clones omit it, rollback snapshots preserve it, and it is never hashed or
exposed to a policy. Academy turns it on at setup; with it off the read
returns nothing. Every committed `GameEvent` is mirrored into it, and the
engine notes directly what has no event: an activated ability reaching the
stack, an ability resolving or leaving the stack, a priority pass, and each
state-based or turn-based action with its rule. Mana abilities are never
noted. During stack resolution the sink holds the resolution as its active
cause and carries it across stabilization until state-based actions finish, so
state-based actions following a resolution or combat damage inherit that cause
beside their `rule`; triggers carry the triggering event (or nothing for
upkeep/delayed triggers).

One non-destructive native read per viewer returns a step's facts with the
per-event `labels` that viewer may read; a hidden-zone move names no card to
another seat. This is the only place viewer privacy is applied to the track.
A separate `take_presentation_positions()` hands Python the step's cloned
positions and clears the sink after both seats have read it.
`managym/academy/server.py` keeps one `PresentationProjector`
(`managym/academy/presentation.py`) per viewer seat (`0` and `1`); each reads
its own filtered stream, assigns its own monotonic `seq`, resolves each `cause`
to a `seq` on that viewer's track (climbing to the nearest told ancestor, or
omitting it), sets `group` to the root ancestor's `seq`, and attaches `labels`
directly to the event. The projector never diffs observations or boards. The
sink is not observation state: the policy input, the LLM text observation,
learning tokens, state hashes and a recording's replay identity do not include
it.

**Boards.** With the sink enabled, the engine clones a read-only position
(with no sink and no pending decision context) at each whole-state boundary
inside a step where at least one non-pass fact was noted: before and after a
non-pass action, after a stack object resolves, when stabilization begins
(separating combat damage or a turn-based action from the state-based check
after it), after a state-based batch, and after a trigger round reaches the
stack. Priority passes never cut a position. Each fact records the index of the
position cut after it, so lines noted inside one boundary (including multiple
effects of a single resolution) share a board. Python projects each position
for the recorded person's seat through the same `decision_surface_view` used
for frames and chains `boards` patches (`first`, `last`, `patch`) back from the
final frame, deduplicating equal boards. Play traces retain each transition's
mid-step `boards` patches beside `presentation` (outside replay identity) so
Review draws per-line boards with the same patcher; Watch keeps its own
acting-viewer transport and stores no person-seat `boards`.

**Names.** Each event's `labels` carry the viewer-safe names for every object
and stack entry it mentions, at exact incarnation, across live updates,
reconnect `presentation_tail`, Review and Study. No reader recovers a name
from text or from a neighbouring frame (frame labels remain only as a fallback
for older recordings); an object the viewer may not name stays unnamed.

**Ordering on the live socket.** Sequence-gate the `FrameUpdate`, commit its
complete `ExperienceFrame`, validate every event against the update's base and
resulting revisions, then append the events to the track. The same event list
is returned live and persisted on the transition's final trace step; Replay
reads it back, and traces without the field replay with no events rather than
inferred ones.

**In the browser.** `board/tell.ts` words an event for every view;
`board/history.ts` turns the track (plus client-local automation notes, such
as a spent browser-only "once" stop) into turns, lines with depth, and board
stops (seq order, indented under a cause that is the nearest line above, an
immediate resolution folded into its cast with the cast board when
consequences follow and the resolved board otherwise, runs of passes folded,
and one walk stop per kept board). One cursor (`selection: { seq: number;
scope: 'stop' | 'line' } | null` plus timer state `walking: boolean`) selects
what the board draws in both live play and Review sub-step inspection: in live
play `null` is now, walks and scrubber ticks select `scope: 'stop'` (keeping
multi-line stops intact when paused) with the caption and scrubber in the bottom
bar, and selecting a history line or pressing ← → selects `scope: 'line'`;
`board/result.ts` and the live decision strip also read the track.
`board/boards.ts` answers `boardAt(seq)` from the frames and per-line boards
this page was sent, up to 512; the server serves no past board to a live
session. Review rebuilds each `consequences` transition's retained `boards`
patches through the same `mergeBoards` / `findBoard` path before orienting and
feeds `cursorSeq` into `Board.svelte`.

**Recovery** commits the complete frame, returns the cursor to now and merges
validated `presentation_tail` events (with each event's `labels` intact) with
same-match in-memory history. The tail brings events and no boards. The server
retains the latest 256 events (`MAX_PRESENTATION_EVENTS`), so a fresh page
labels any unavailable prefix. Sequence gaps are explicit. A malformed track
reports an error without touching the committed frame, and valid retained
history is marked incomplete. A fact the projector does not know fails tests;
in a live session it is dropped with a logged warning.

**Older recordings** have no move events, no `cause`/`rule`, and no per-event
`labels`. Because those fields are optional on `PresentationEvent`, older
recordings validate unchanged. Their moves are worded by
`managym/academy/recording.py`'s `narrate_move`, which remains frozen for that
use alone and is not extended. The `log` field of protocol v1 frames is no
longer filled.

Watch history is the acting viewer's track strictly before that decision's
`presentation_cursor`. Recorded Play review exposes the recorded person's full
track across the game; Study has an isolated scope. See
[Reading the history](#reading-the-history) for what a person reads.

## Playing a trained challenger

### Top-level CLI and hosted policies

```sh
managym play                                             # Play vs recommended policy (TLA trophies)
managym play --order draw --deck ur_lessons --opponent search-64
managym play --opponent ./my_bot.py:choose               # Play against any Python callable
managym watch --bot1 heuristic --bot2 manabot            # Watch bot vs bot
managym opponents                                        # List available built-ins & policies
```

When `manabot` is installed, its `managym.bots` provider discovers the hosted
default policy from `https://policies.manabot.gg/index.json`, verifies its
SHA-256 into `~/.cache/manabot/sha256/`, and loads it alongside any local
checkpoints. Set `ACADEMY_HOSTED_POLICIES=0` to disable hosted catalog fetches
offline, or pass `--no-discover` to hide discovered policies when testing a
single custom policy. In a repo checkout, `./scripts/play` runs the same
packaged launcher with the `play-runtime` dependency group.

From Python, pass any `Policy` or plain callable `(obs) -> int | Sequence[float] | PolicyDecision`:

```python
from managym.academy.launch import play, watch

play(opponent=lambda obs: 0, order="draw")
watch(bot1="./my_bot.py:choose", bot2="heuristic")
```

### Other checkpoints

Any trained checkpoint is selectable in Play and in either Watch seat
([Play and watch a trained checkpoint](#play-and-watch-a-trained-checkpoint)).
manabot supplies the catalog (`GET /api/opponents`) and reads checkpoints with
its provider-owned loader. The browser sends only an opaque runtime identity: `policy:<id>`
as Play's opponent or in `POST /api/watch` `players`. The server re-hashes the
bytes before play, refuses an incompatible or replaced file with the sentence
the menu shows, and records the identity as the trace's `opponent_checkpoint`
in Play and in each seat's recorded player in Watch.

Scored candidate evaluation on the selected matchup (`uv run python -m manabot.eval.evaluate`) is described in
[evaluation.md](evaluation.md).

## Attempts and history

`managym/academy/attempts.py` stores games in SQLite at
`$ACADEMY_TRACES_DIR/play.sqlite` (default `academy/traces/`); new traces and
attempt metadata commit in one transaction, and legacy JSON traces remain
read-only. Each game is keyed by its id (`attempts.id`) and saved at start and
at each surfaced revision. Disconnecting, starting another game, 15-minute idle
eviction and server restart do not end a game — only authoritative rules
completion (`game_over`) or a runtime/divergence error (`record_diverged`) does.
History and Replay name three statuses: `active` (in progress), `completed`
and `stopped`. Existing databases keep legacy `interrupted` rows and legacy
`stopped` rows (`server_restart`, `server_shutdown`, `session_expired`,
`new_game`) readable and resumable without rewriting them; they read as
`active`, and leave legacy tables/columns unread so
an older binary can still open the file.

Fresh databases create `attempts`, `attempt_players` (both seats: public
`player_id`, `kind` human/bot, name, deck, `version` = exact checkpoint digest
for bots; the human seat is recorded as `"You"`, `human:local`), `analyses`,
and `analysed_decisions`. `deal_seed` is hexadecimal text to keep all 64 bits.
The file holds private seeds and canonical Commands: keep it private and query
it with `sqlite3 -readonly`, never a public endpoint.

`ACADEMY_PLAY_RECORD_ORIGIN` declares provenance: `human_exploratory`,
`automated_validation`, `bot_evaluation` or `training` (default `unknown`). It
is operator-supplied, not detection; never run automated proofs against a
human-labelled server.

`/games` lists every game in `traces_dir`, newest first, with search and
filters for pairing (`human-bot`, `bot-bot`), status (In progress, Completed,
Stopped), bot, deck and date, with filters and cursors in the URL.
`GET /api/traces` returns those summaries newest first (`limit` 1–100, default
50) with `resumable` and `X-Etude-Next-Cursor` pagination, with no player
credential headers and no 403 wall. Every saved game, finished or unfinished,
is readable in Replay.

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
