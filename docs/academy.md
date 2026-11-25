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
this tab to its last live or watched game. Existing launch, replay and Watch
links continue to open their intended games.

**New game** opens setup for a separate session, starting from retained choices.
Edits here do not save into your current session; **Cancel** returns without
committing them. Setup shows two native seats. For each seat, choose a **Controller** (**You**
or **Bot**) and a **Deck**. A bot seat also has a player picker: choose a
built-in bot or a **Registered policy**, with optional **Search** for a compatible
policy. Built-in **Search 16/64/256** runs that many random playouts per
legal action without a policy; **Random** chooses uniformly among legal actions;
**Passive** passes when possible and otherwise takes the first legal action.
One controller can be you; choosing bots in both seats starts a
watched game through the same setup. Seat 1 plays first; seat 2 draws first.
Policies are chosen directly by name and execution identity. Their provider,
artifacts and runner remain available in details. **Browse decks and cards**
opens the full deck browser; imported sources and disclosure use the same
admission checks for played and watched games.

Hover or focus any visible card to read it; reading never sends an action.
**Open a recorded game file** in Games opens any recorded game, from your own
loop, training or evaluation, without changing its original evidence. Games links back to recorded positions, their legal
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

Your browser opens on the board at `http://127.0.0.1:8000`, with a game ready
against the Search 64 opponent. That is the whole installation: no manabot,
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
uv run --no-project managym academy replay games.jsonl
uv run --no-project managym academy replay .runs/demo/games --game 3
uv run --no-project managym academy replay <recording-id>
```

The first command opens a file of recorded games; the second opens one game
from a directory of them, here a training run's; the third reopens a game
already in **Games**, by the id shown there. All land in **Replay**, ready to
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
is closed: it only plays saved data, even when its game never finished.

A recording id keeps the recording's existing access rules: open it in the
same browser and at the same address (including port) where **Games** lists
it, or open a shared, finished recording. Another browser or address cannot
open a private or unfinished game without its participant credential. The
launcher checks that the id exists; access is checked when the browser opens
it. Opened game files are shared recordings and can open in any browser that
can reach the server. A file holds the deal seed and so both players' hidden
cards; share it as you would share both hands. The viewer still shows each
seat only what it could see, and the file's run labels and build identities
stay on the server.

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
  game, setup or replay, subject to the recording access rules above. `academy.open(url)`
  opens it in the default browser; `academy.open()` opens Academy with no game.

Leaving the `with` block, or calling `academy.close()`, stops the server and
frees the port; unfinished games are kept as interrupted, as with Ctrl-C.
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

Open Academy (`/`) in a tab with no game and it offers your last matchup:
**Play again** starts the same opponent, decks, play/draw seat, deal mode and
stops as the last game this browser started, in one click. The first time, it
offers **Play** with the default matchup, UR Lessons (you) against Search 64 ·
search with GW Allies. **Set up a different game** opens the setup. Play again
always starts a new session; it never resumes another tab's game. The last
matchup is a browser preference (`academy.last-matchup` in local storage) and
holds no credential.

Open **New game**
to use the [two-seat setup](#one-game-from-setup-to-its-recorded-decisions):
choose controllers, decks and bot players, then select **Start game** to start
a separate session. Replacing an active game requires confirmation that it
will be stopped. To keep games together, use
[Rematch, Swap sides or Change opponent](#playing-a-series-against-a-model).

To use your own decks, open **New game**, expand **Import decklist files**
and check **Use imported decks for the next game**. Upload a UTF-8 `.txt` or
`.dek` file for both **Your deck** and **Opponent deck**, or paste each list
into its source box. Write one `count card name` per line, with optional
`Main` and `Sideboard` headings. Select **Check decklists**, correct any
reported lines, choose a built-in opponent such as **Search 64**, then select
**Start game**. Imported decks use the development format and supported cards;
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

### Playing a series against a model

Choose **Registered policy**, select the intended policy, and import your decks
through **New game**. Check the matchup, decklist disclosure, priority stops
and deal mode, then select **Start game**. When the game ends, a result card
sits over the board, which stays visible around it (**Look at the board**
folds the card to a corner). It says who won and how ("Heuristic · scripted
went to −2 life on turn 9"), the final life totals, and one line from the move
list: turns, and each player's spells cast and attacks. Select **Rematch** to
play again with that setup. After the next game, select **Swap sides**: you
keep your deck and move from playing first to drawing first, or the reverse.
**Change opponent** opens only the opponent settings, and **Review this game**
opens the game in Replay from its first position. Both games and the next one belong to the same
session.

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

The session strip in the top bar's **Menu** shows the game number, matchup, your play/draw
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
| **Start game** | Starts a separate session from setup |

To exchange decks within the series, choose **Edit next game** in the session strip.
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

Open session history from the board to see its games in **Games**. Each entry
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
using the same browser address, including its port. Session history requires
the owner credential stored in that browser; copying its link does not grant
access. If access is denied, open the link in the original browser and address,
or browse Games outside that session. Cleared owner credentials cannot be
recovered from the link. Temporary loading failures still offer **Try again**.
If browser session storage
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
configuration access nor control of the live game. Earlier ungrouped games stay
in **Games** without being assigned to a session retroactively.

### Playing on the board

Start a game and look at your hand: the cards you can play now are raised off
the board with a soft shadow, and the ones you cannot are dimmed.
Click a land to play it; click a spell to cast it, then follow its target and
payment prompts. A card with exactly one legal action commits that action when
clicked, and its accessible name says which action that is. A card
with several actions opens them beside the card; choose one, or press Escape to
close the list without acting. A card with no action sends nothing when
clicked.

The board is split at the centre line: the opponent's creatures face yours,
with each side's lands and other permanents behind them. A plate at the left of
each half shows the player, life, zone counts and floating mana as mana
symbols, with any restriction or expiry written under them and wrapped within
the plate ("Floating {R}{R} normally expires at step end" shows its pips, not
braces, and is never cut off); open a graveyard or
exile count to list its cards. A player who can be targeted has an
outlined plate. Identical permanents with the same choices share a pile, its
count after the top copy's name ("Forest ×4"; the copies beneath show no name of
their own), and each copy remains its own control. Crowded rows shrink their cards
and wrap onto a further line before anything scrolls, and each half gives the
crowded row more of its height.

Every card has a printed card's proportions (63:88). A permanent's name is on
a bar above its art, wrapping to a second line rather than being cut; its mana
cost is left to the hand and the inspection. The name bar carries the card's
colours: a pale tint in light mode and a solid bar in dark mode, white a
neutral ivory-grey and colourless a cool grey, a land the colour of its basic
land types, and a card of several colours blended left to right as printed
hybrid frames are. Power and toughness sit on a plate at the lower right, and
on a row above it the permanent's state: attacking, blocking, damage marked on
it, summoning sickness, counters and the keywords it has now (one it was
granted is edged in green). Mana symbols carry their letter, and ◇ for
colourless. A tapped permanent turns a quarter onto its side, in room its row
keeps for it so it never covers a neighbour; its name bar stays across the
top. An attacker steps toward the player it attacks and tilts a few degrees:
one that tapped lies just past the quarter, a vigilant one stays upright.
Something a spell on the stack targets has a dashed red edge. Cards in hand
have the type line under the name and the cost on the art; a hand card's name
stays on one line, shortened with an ellipsis when long, and its full name is
in the inspection.

While you may act, what you can act on is raised and what you cannot steps
back: a card in hand you cannot play now is dimmed most, a permanent of yours
nothing acts on is slightly muted, and lands and the opponent's side are never
dimmed. Summoning sickness is told by its label, not by more dimming. The
selected card, such as the spell whose targets you are choosing, rises higher
with a thin ink edge. In light mode the grounds are white and grey and what you
act with is charcoal; in dark mode they are near-black and it is near-white.
Academy follows the system setting. Card and player names, life, and the
rail's moves and legal choices are set in Spectral; everything else is in Work
Sans. The top bar shows the connection only when it has a problem.

Rest the pointer on a card for a moment, or focus it with the keyboard, to inspect it: its rules text, type line and cost,
keywords with why it has each (printed, the effect that granted it, or
"another ability" for a grant from another permanent's static ability), where its power and toughness come
from (printed or set, counters, each effect by its source and duration, and
whatever those do not explain as "other effects"), and its state. The
inspection floats beside the card and closes when the pointer leaves, when you
click or when the game moves on; a card you just clicked does not reopen it
until the pointer leaves and returns; **Dock** in the top bar keeps it in a pane at the top of the rail
instead, showing the last card you inspected. The top bar shows the turn. Along
the bottom of the screen runs the step bar: one cell per step, a line between
phases, the current step shaded, and in each cell the stop for your turn above
the step's name and the stop for the opponent's turn below it. Press a stop to
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
  entry an action applies to is outlined. A line joins a spell on the stack to
  each of its targets. A spell you are casting, or an ability you are
  activating, is on top of the stack with a dashed border (**casting**,
  **activating**) from the moment you choose it, through its targets and
  payment, with lines to the targets already chosen; the card leaves your hand.
- **Since your last move**: what happened after your last choice, as the
  server reports it: spells cast and what they targeted, attacks, blocks,
  damage and deaths. The first three lines show; the rest open with **more**. A spell that already resolved keeps its card here, and its
  line to a target that is still on the board, until your next move. Cards that
  attacked, blocked, were targeted or were dealt damage are ringed on the board,
  and damage dealt since your last move shows on the card or the player's plate
  as it lands ("−2").
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
  nine.
- **Moves**: the game's moves grouped by turn, each turn headed by whose turn
  it was. Each move names the player who made it; priority-stop automation is
  marked **auto** and never counts as your move. A run of the opponent's and
  automation's priority passes folds into one quiet **passed** line that counts
  each one's passes; your own passes stay moves. Hover or focus a card name to
  see the card.

While the opponent decides, the rail and the opponent's plate say so, with
the seconds it has taken ("Search 64 is searching · 3 s"), and the rail's
choice list reads "Waiting for Search 64" until there is a decision of yours. An attacking creature steps toward the player it
attacks, and a blocker is joined to the attacker it blocks by a line. Combat
usually ends before your next decision, so the line stays after it, until your
next move, while both creatures are on the board. Cards
move between zones (hand to stack, stack to battlefield, an opponent's new
card from their plate) instead of appearing; with reduced motion they appear
in place and every fact above stays.

Choices with no card on the board, such as Lessons outside the game, appear in
a tray over the board. Trays, the inspection, menus and the result card are
opaque and sit above the lines between targets, so a line never crosses them.

Combat is drafted whole. At an attack, every creature that can attack is
outlined; click one (or focus it and press Enter) to add it to the attack or
leave it out. Drafted attackers are ringed with a dashed line and lean in, and
a pile of identical creatures takes a count ("2 of 3 Tiger-Seal") in the
rail. At a block, click a blocker, then the attacker it blocks; a blocker
that can block only one attacker takes it at once, and the block's line is
drawn. The prompt reads the whole plan ("Attack with 3 · not with
Tiger-Seal", "Badgermole Cub and Ally on Invasion Reinforcements") and
nothing is declared while drafting. The engine still asks about one creature
at a time: Academy's read-only combat preview replays the draft on a copy of
the game and returns the exact answers that declare it, or the first one it
refuses and why (a lone blocker of a creature with menace, a creature that
cannot attack). Confirm (Space) sends those answers one by one; if the game
asks anything the preview did not predict, sending stops and the board shows
where the game stands. Escape stops a confirmed plan between answers. A
reload keeps the draft for the same decision in that tab. Each creature's own
answers stay in the full list of legal choices. **Pass turn** (F6) passes
priority until the turn ends.

Keys: `1`–`9` choose a numbered choice, Space presses the primary button
(unless a control with its own action has focus), F6 passes the turn and Escape
closes an open list. **Menu** holds the session controls, priority stops, the
sharing and Study tools and feedback.

While disconnected or waiting for a submitted choice, the board remains
readable and actions are unavailable. Reconnect or reload in the same tab to
recover the actual pending decision. If a choice is rejected because the game
has advanced, read the refreshed prompt and choose again. A submitted action
cannot be cancelled; after commitment, costs and revealed information remain
committed. Reloading a configured launch in its tab resumes that game while its
server session remains available; a separate tab or another launch has its own
game. See [Pending input, waits and recovery](#pending-input-waits-and-recovery)
if recovery fails.

For bot inspection, open **Watch** or a recording in **Replay**. Their decision
surface shows the acting seat's permitted view and the complete legal choices
for that decision. Recorded policy outputs appear only when available; missing
outputs are not reconstructed. See [Inspect a replay](#inspect-a-replay).

### Previewing payments

You control two Mountains and a Forest, all untapped, and hold Badgermole Cub
(`{1}{G}`). Point at the Cub in your hand, or focus it. The lands the
automatic payer would tap get a dashed outline and a mark with their order and
what each adds: the first Mountain `1 {R}`, the Forest `2 {G}`. A planned land
in a pile of identical lands leaves the pile while the plan is shown, so each
planned land has its own outline and its mark sits at the same corner of its
art. Nothing has
happened yet; pointing and focusing send no action. Pointing at a row of **All
legal choices** previews that choice the same way.

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

A cost that every available source is needed for has no source choice. The
game pays it as soon as you cast, with no payment prompt, so the marks you see
while pointing at the spell are where you see those taps beforehand.

**Mana payment** defaults to **Manual**. Choose **Automatic** to have the engine select sources, colors and spending after you accept a cost. You still choose whether to pay optional costs, which spells to cast, and which abilities to activate.

Under **Mana floats**, **Off** is the default. Choose **Only when the mana could be used** to float when a legal action can spend mana or tapping the source
can trigger an ability. Usable also offers an activation that pays life,
sacrifices a permanent or has another effect. **All** offers every legal
mana activation at priority. With **Manual** payment you can select these abilities and choose a land’s output while paying. A deliberate float at priority also lets you choose its output. The setting follows you when you swap sides.
Your priority stops still determine which windows pause for a decision.

**When there is no plan to draw.** The primary button then shows only the
cost, and a line under it says why when the game does:

| The game says | Meaning |
| --- | --- |
| A choice is needed first | X, an optional or alternative cost, an additional cost or a target can still change the price. Nothing is marked until that choice is made; the plan returns at the completing target or at the payment prompt when there is one. When the announced cost needs every useful source there is no prompt, and the cost is paid as soon as the last choice is made. |
| No sources needed | The cost is zero or your floating mana already covers it; the button reads "· floating mana". |
| No automatic payment | The automatic payer cannot fund this cost with its available taps. Use an offered mana ability and make its choices first. |
| Nothing | There is no automatic payment to preview. When choosing which floating mana to spend, choose the unit directly. Mana abilities with their own fee also have no preview. |

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
Academy keeps the unresolved submission with that tab's game identity and
checks whether it was accepted. If it must resend, it resends that exact
submission so an accepted choice is not applied twice. If the decision or your
permission changed, Academy reports the stale or unknown outcome and refreshes
the board. Going back or clearing a selection cannot cancel a submission.
In live Watch, recovery checks the recorded position before another step;
do not repeat a step just because its reply was lost. If that read fails,
**Refresh saved head** must establish the frontier before production is available.
After a bot failure, recorded navigation remains available, but playback stops
at the live edge. Only **Retry bot decision** or **Retry search** retries it.

**When recovery is impossible.** A server restart or an expired live session
shows **Interrupted**, retaining the last known board and a recording link
when available. Saved recordings preserve available evidence; they cannot
restore the live game. A read-only snapshot of the last published board,
opponent identity and interruption reason survives further reloads in the same
tab. The header reads **Interrupted** and the choices are labeled **Last recorded
choices**; they do not invite live input, even when the server returns. The snapshot
contains no resume credential and cannot submit actions. Choose
**New game** explicitly to return to setup and retire that snapshot.
Academy never silently deals a replacement or replays your submissions into
a new game. Keep the original tab open for temporary connection failures:
resume credentials and unresolved input belong to that tab's session storage.

Back, Escape and clear controls discard only unsubmitted local selections.
Once an action commits, its costs and revealed information remain committed.
Cancellation is possible only when the current legal choices explicitly offer
it; there is no general browser undo.

### Choosing priority stops

To hold up a response during the opponent's upkeep, press the lower stop in the
**Upkeep** cell of the step bar, or open **Menu**, keep **Use priority stops**
checked and check **Upkeep** under **Opponent turn**; the two are the same
setting. Academy pauses at a priority window there when you have a real choice.

Each stop on the step bar has three states, and a press moves to the next:
**Off** (a hollow square), **Every time** (filled) and **Once** (a dot inside).
A once stop pauses the next time the game arrives at that step on that turn and
then turns itself off; the moves list says so ("Stopped once at Upkeep on the
opponent's turn; that stop is now off"), marked as automation like the passes
Academy makes for you. The Menu shows a once stop as checked. The stops are one
Tab stop: arrow keys move between them, up and down between the two turns, and
Space or Enter presses one. A recording's bar shows its stops read-only.

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
windows continue to appear. Automatic passes are marked **auto** in the
move list and recording. Pausing narration does not change these settings.

### Reading consequences

The rail tells what happened in prose. In live play it is **Since
your last move**: spells cast and their targets, lands played, discards,
reveals, attacks, blocks, damage and deaths, read from the authority's presentation
events after your last commit, with the move list beneath grouped by turn.
When the opponent moved, those facts are read out one at a time before your
decision, with the cards each is about outlined on the
board. The fact being read heads the decision block, above your dimmed prompt.
Each fact is shown on the board after the engine step it happened in: the
server sends your view of the board after each step with the frame, so an
attacker that died is usually seen attacking before it leaves, and cards travel
from one board to the next. One step can run from an attack through damage into
the next turn, when nobody had a choice between; such a fact is shown on the
board at that next turn. A past board outlines nothing. While the reading lasts,
the top bar names the turn the fact happened in, and the step bar marks that
turn's step when the board shows it and no step when the board is a later
turn's. Your decision's buttons and numbered choices stay in place
but wait. They are announced as unavailable and still take focus. Clicking
the board or rail, activating a control, or pressing a key such as Space,
Enter, Escape or a number key ends the reading. Focus keys still move and
`?` opens keyboard help. The current board and your live decision answer the
next press. So when
Heuristic attacks, you watch the attack land before you are asked to block, and
pressing `1` early shows the blocking board rather than declaring a block.
Damage marks ("−2") and a finished game's result card wait with the decision,
so a past board never shows damage it has not yet taken. A
new frame or a reconnect ends the reading and shows the actual pending
decision. **Opponent's moves** in **Menu** sets the pace: **Quick** (under a
second a fact), **Slow** or **Off**. With **Off** or reduced motion nothing is
read out and the current board shows at once; the recap lists every fact. Sentences name a player
shortly ("Heuristic cast…"); the top bar keeps how it decides
("Heuristic · scripted"). In Replay the same recap
covers **This move** at a consequences position and **Since the last
decision** at a choice (from the same chooser's previous decision); the move
list jumps to any move.

Consequences describe recorded game events. The move list separately
identifies submitted choices and **Auto** passes. An action alone is not
evidence that a spell resolved or a creature died.

Reconnect preserves the history already held in this page. A fresh reload can
recover only the latest 256 events of an active match. Replay reads the
selected perspective's recorded history; a choice shows only events preceding
that choice, and spectator reveal never adds names to it. **Try another line**
has its own recap, and **Return to recorded decision** restores the
recording's. Older recordings without event data say **Consequences not
recorded**; missing events do not mean that nothing happened.

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

**Card art credits** in the page footer lists the artist, source and copyright
of every image shown so far. The package itself contains no card images, and
deleting the cache directory is safe.

### Inspect a replay

A recording opens on the same board as live play, read-only. Open **Games**,
choose a game, and step through it; human games, watched bot games and
imported training or evaluation games all use these controls. At the end of a
game you played, **Review this game** on the result card opens it from the
first position.

Every game you start is recorded as you play. **Games** (`/games`) lists them,
newest first, with filters for player, deck, status, result and date. Player 1
is always on the play. A finished game you played shows its turns and how long
it took. **My games** narrows the list to this browser, under **My record**:
your completed games across sessions, grouped by opponent and by the two decks,
with wins, losses and draws and the win rate with its 95% Wilson interval.
Draws count as games, not wins. Rows that differ only in seat or disclosure are
combined. These are counts of recorded games, not a rating. A trained bot appears under its
[short name and digest](#how-a-trained-opponent-is-named). Open any recorded
game file through **Watch → Open a game file**, or
[launch a replay directly](#open-a-replay-directly). `/replay` alone lists
recent recordings.

#### Follow the timeline

| Position | What the board shows |
|---|---|
| Choice | The chooser's recorded board and every legal offer, on its card and in the rail's numbered list, read-only. The played offer is marked ✓; where the policy was recorded, each offer's probability is on its card and row, with the value, update and any search detail beside the prompt ("not retained" where nothing was recorded). |
| Consequences | The board after the action, with what it did in the rail's recap. In a game you played this is your own view, your hand included, whoever acts next. |
| Result or latest state | The final board under a result card, or the latest retained state of an unfinished recording. An interrupted recording has no invented winner, and one opened from a file says why and where it stops. |

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
open in Replay and folded in live play, where it counts decisions but past
positions open only in Review after the game. Moves in the rail jump to their
positions too. Keys never act while you type in a field.

The replay address keeps the position, the perspective and the information
view. Reloading restores them, with a saved position in this tab as a
fallback. A position that is no longer available is explained so you can
select another. Opening a recording still requires its normal access
permissions; copying an address does not copy a participant credential.

#### Perspective

The **Perspective** control in the top bar chooses who sits at the bottom:
**Follow the chooser** (the default for bot games: each choice is drawn from
its chooser's seat, hand included, and the board stays turned through that
choice's consequences), or a fixed seat (the default for a game you played).
Turning the board never reveals a hand or changes the recorded policy inputs.

At bot choices, the acting view shows what that bot was allowed to see.
**Show both hands** appears where spectator reveal is authorized: at Watch
choices that captured a spectator view, and at every position of a game you
played once it has finished. It does not enrich the bot's recorded evidence or
acting-view history. In Watch and imported games, consequences and result
boards use the recorded view of the seat named in the menu's perspective line
when a private post-action view was not captured.

**Menu** holds the recording's kind and status, each seat's identity (policy
ID, runner, provenance), **Refresh saved head**, a note for the position,
**Watch another matchup** for a Watch, and feedback for your own games.

A live decision already being produced may finish saving while you read; your
selected board stays put. A pending Search timeout is retained across reload
and shown with its message. At the live edge, **Retry search** explicitly
retries that decision; browsing saved positions never retries it.

#### Analysis

An analysis reads a game you played against a policy: at each of your
decisions with more than one legal action, what that policy would have done
in your seat. Academy replays the game with the policy in your seat; it sees
only what your seat saw, and the game follows the moves you made. Opening
Review of your own game starts an analysis by your opponent's policy when it
is a registered trained policy and the game has none yet. Review is usable at
once; **Analysing 34 of 81** counts up while the marks fill in.

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

**Next key moment** jumps to the next of up to eight decisions worth a look,
in game order: the five where the policy's first choice had the most
probability over yours, and the three where its estimate moved most by your
next decision. At one of them a line names why, such as **Key moment ·
Differs · 62% on its first choice, 8% on yours** or **Its expected score moved
48 → 71 by your next decision**. A reading that barely moves, as an untrained
policy's does, has none.

**See its line**, at any analysed decision of your own game, plays the
policy's first choice there and carries on: the policy's first choice at each
of your decisions and your opponent's own choice at theirs, to the end of the
next turn or the end of the game. The board shows each move from your seat;
**Previous** and **Next**, or the arrow keys, step through it, and **Return to
the game**, or Escape, goes back to the decision. The line is one play-out of
the real game's cards, not a forecast, and nothing about it is saved. Your
opponent's moves are named once the game is over; in a game that stopped
early they read **Opponent's move**. Priority passes fold into the move
before them.

**Analyse with…** in **Menu** runs any other registered trained policy; each
run is saved beside the game, and **Shown** switches between them. Analyses
are never part of the game's recording or its exports. Readers of a shared
finished game see its saved analyses; only you can start one.

| You see | Why | What to do |
|---|---|---|
| **No analysis yet · Analyse with… in the Menu** | Your opponent was Random, Passive, Heuristic or Search. | Choose a policy under **Analyse with…**. |
| **No analysis: no registered policy can read this game** | This server has no trained policy for these decks. | [Register a policy](#trained-opponents-and-other-bots), then restart Academy. |
| **Analysis stopped after 12 of 81 decisions**, with a reason | The policy failed there, the game stopped replaying as recorded, or Academy restarted. Earlier decisions keep their readings. | **Analyse again**. |
| A line naming another build | The game was recorded by another world or card set and cannot be replayed here. | Open it with that build. |

A policy with Search is not offered: Search plays out from the real hidden
cards. One analysis runs per game at a time; a stopped or interrupted game is
analysed up to its last recorded position.

#### The game in numbers

The result card of a game you played ends with one line of its numbers, such as
**9 turns · you cast 6 spells and attacked 4 times; Reader cast 3 and attacked
2 times**. In Review, **This game** in **Menu** gives turns, whether you were on
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
the end. A game that stopped early is counted to where it stopped. Academy
counts each game once, by replaying it when it ends. An older game is counted
the first time you open it. A game recorded by another build has no counts.

#### Continue a live Watch

Open **Watch** (`/replay?watch=1`), select both seats, decks and a seed, then
press **Start watching**. Each seat uses the same player menu as Play's **Opponent**.
A managym-only install offers Random and Search; manabot adds Greedy, Heuristic
and trained checkpoints. The game opens on the replay board and is saved in
**Games**. **Watch another matchup** in the menu opens setup for another bot game.

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
and opens an isolated branch on the same board: the historical offers become
live, and choosing one plays it against the branch. **Return to recorded decision** restores that
exact choice and its history; the original recording and result stay unchanged.
A fresh branch requires the server's retained Study state. If that state has
expired, recorded playback remains available.

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
   launch `uv run --no-project managym academy`, open **Watch** at
   `http://127.0.0.1:8000/replay?watch=1` and use **Open a game file**.
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
appears as `run/…/step_8 · 7ac5d442da`. Human names are never shortened this way.

The same compact identity appears on the board, in Games, Replay and Watch.
Open **status** on the player's bar with a mouse or Tab and Enter to
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
