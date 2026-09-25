# Train, play, and retain a local challenger

This workflow runs the existing search-distillation recipe on the checked-out
Allies/Lessons world. Exp-03 supports that method over early PPO in its
historical world; it does **not** rank this small new candidate as the strongest
current bot. The generic `manabot train --preset local` uses a different deck
and is a training smoke, not this matchup's challenger recipe.

No cloud account is needed. The fixed local workflow uses search-64 teacher
self-play for eight games, alternating deck assignment, then eight supervised
epochs (lr 0.001, batch 64, seed 79, whole-game 20% validation). Both seats
contribute labels. The selected artifact is the final epoch of this one run,
without tuning or a strength claim. Each execution has one CPU worker/thread,
a ten-minute wall cap and a 4 GiB process-tree RSS cap. Overflow of a fixed
training observation rejects the run rather than dropping legal choices.

From the repository root, prepare the locked environment and native runtime:

```sh
uv sync --locked --python 3.12 --extra play
uv run maturin build --release -i .venv/bin/python -m managym/Cargo.toml -o managym/target/wheels
uv run python - <<'PY'
from pathlib import Path
from zipfile import ZipFile
wheel = next(Path('managym/target/wheels').glob('*cp312*.whl'))
with ZipFile(wheel) as archive:
    library = next(name for name in archive.namelist() if name.endswith('.so'))
    Path('managym', Path(library).name).write_bytes(archive.read(library))
PY
uv run scripts/train_challenger.py --out .runs/my-challenger --seed 79
```

Use a new output directory for every execution. `recipe.json` and an exact
source/native archive `sources.zip` are written before work; per-game shards
and `games.json` survive failures. `phases.json` reports
seconds and teacher decisions/second; `receipt.json` reports worker elapsed
`wall_seconds`, total script `operator_wall_seconds` including source capture,
sampled worker process-tree RSS (excluding the supervisor), actual new cloud
spend and unknown electricity cost. Build, browser and human-play time are
separate; missing phase measurements remain unknown.
`candidate.json` binds the checkpoint digest, source/recipe, data, observation
configuration, content manifest and inference setting. Export reloads the
checkpoint and checks exact logits against the trained model. A failed receipt
cannot be configured as an opponent.

Reproducibility means retained inputs and procedure, complete legal games and
replay witnesses. It does not promise identical stochastic checkpoints across
devices or independent seeds. These receipts are the ETU-80 measurement input;
do not copy results into a separate measurement store. Early ETU-79 feasibility
uses the pre-ETU-75 world. Corrected rules/Lesson-pool acceptance remains open
until that world is integrated and the same workflow is repeated.

Build and serve the same SPA/ASGI entrypoint as the hosted app, locally:

```sh
npm --prefix frontend ci
ETUDE_STATIC_BUILD=1 npm --prefix frontend run build
ETUDE_PLAY_CANDIDATE="$PWD/.runs/my-challenger/candidate.json" \
ETUDE_FRONTEND_BUILD="$PWD/frontend/build" \
ETUDE_TRACES_DIR="$PWD/.runs/human-play" \
ETUDE_PLAY_RECORD_ORIGIN=human_exploratory \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
uv run uvicorn deploy.play.asgi:app --host 127.0.0.1 --port 8000
```

Open `http://localhost:8000`. The trained opponent is selected automatically;
its name and short digest remain visible through reconnect and Play Again.
Search, Random and Passive remain selectable controls. The browser never needs
a checkpoint path. The server verifies the artifact digest, rules binary,
content manifest and decks. Replacing checkpoint bytes fails clearly.

Play, optionally save a note, and choose Play Again. Open replay beside the note
to revisit that attempt. Attempts are saved at startup and each surfaced
revision, including incomplete prefixes. Replacement, expiry, runtime error
and authoritative completion have different ending reasons. Disconnect leaves
the attempt active; after server restart its status becomes `interrupted`, with
no invented winner or stopping time. Private records use the table participant
credential, retained in this browser's local storage. Clearing browser storage
removes that browser's access; this is not an account system. Legacy JSON traces
remain readable. Hidden reveal requires authoritative completion.

The database is `.runs/human-play/play.sqlite` with the launch configuration
above. New traces and attempt metadata share one transaction; no parallel JSON
trace is written. Use local read-only SQL (never a public SQL endpoint):

```sh
sqlite3 -readonly .runs/human-play/play.sqlite
```

```sql
-- All attempts against exact trained bytes, including incomplete games.
SELECT a.id, a.started_at, a.status, a.ending_reason, a.winner
FROM attempts a
WHERE EXISTS (SELECT 1 FROM attempt_players p
              WHERE p.attempt_id = a.id AND p.version = 'FULL_CHECKPOINT_SHA256')
ORDER BY a.started_at;

-- Both players, their decks/versions, results and private notes.
SELECT a.id, p.seat, p.player_id, p.kind, p.name, p.deck, p.version,
       a.status, a.winner, f.note, '/replay?trace=' || a.id AS replay
FROM attempts a JOIN attempt_players p ON p.attempt_id = a.id
LEFT JOIN feedback f ON f.attempt_id = a.id
ORDER BY a.started_at, p.seat;

-- Find reported confusion and reopen its replay.
SELECT a.id, f.note, f.decision_address, '/replay?trace=' || a.id AS replay
FROM attempts a JOIN feedback f ON f.attempt_id = a.id
WHERE f.note LIKE '%confus%' OR f.note LIKE '%awkward%';

-- A graph-ready count by bot version and status; unfinished games stay visible.
SELECT p.player_id, p.version, a.origin, a.status, count(DISTINCT a.id) AS games
FROM attempts a JOIN attempt_players p ON p.attempt_id = a.id
WHERE p.kind = 'bot'
GROUP BY p.player_id, p.version, a.origin, a.status;

```

`attempt_players.player_id` permits restricting these queries to one stable public player.
`attempts.participant_id` is private table authorization, not a public player ID.
The database includes private seeds and canonical Command/replay evidence;
browser APIs expose the authorized viewer projection. Keep this file private.
New `deal_seed` values use hexadecimal text to preserve all unsigned 64 bits;
the trace JSON retains the exact integer. Earlier feasibility rows used integers.
Before any hosted deployment, configure persistent storage and bundle the
training runtime and candidate. The checked Fly deployment has no volume mount;
this local workflow does not deploy anything.


## Shared game history

Open `/games` for games across players, or select **My games**. Search names,
bot fingerprint prefixes or game IDs; click either participant for their games,
then **Filter this pairing** for the other participant. Bot versions, Human/Bot
pairings, decks, status, player-relative results and UTC dates can be combined.
Filters and page cursors live in the URL. Replay's **Back to games** preserves
that selection. Incomplete games show their actual state, never an invented draw.

Set **Your player name** on Games before starting a game. A separate random
browser credential identifies that player across tables; public IDs are unrelated
to credential hashes. Names are snapshots on each game, so renaming does not
change historical records. Clearing browser storage loses that browser identity;
there is no account recovery or cross-device sign-in in this increment.

New attempts join shared listings. Completed shared games expose the existing
seat-0 replay projection; only authoritative completion enables the established
post-game reveal. Active/stopped/interrupted replay prefixes and feedback remain
participant-only. Existing SQLite attempts migrate privately, retaining their
exact trace JSON and unknown historical provenance. Frozen JSON traces remain
readable, with unknown players; they appear in Everyone's games, not My games.
Do not run old and new server processes against the same database during a
schema migration. No historical private games are automatically published.

SQLite record version 2 owns identity in `attempt_players` (two seats per game).
It replaces the former `attempts.bot_sha256`, `hero_deck`, and `villain_deck`
summary columns. Original trace configurations remain retained evidence, not
another mutable player directory. `players` maps private browser credentials to
public human IDs; bot identities come from the configured artifact's optional
`bot_id`, separate from producer, digest and loading path. Older artifacts lacking
lineage retain an identity for their exact digest; we do not guess lineage from
similar names. New local challenger exports declare `etude:local-challenger`.

Declare record origin when launching the server with `ETUDE_PLAY_RECORD_ORIGIN`:
`human_exploratory`, `automated_validation`, `bot_evaluation`, or `training`.
The default is `unknown`. This is operator-supplied provenance, not automatic
human detection; do not use a human-labeled server for automated proof runs.
Records retain world/source digests, configuration, seed and canonical decisions.
Feedback adds author identity and accepts an optional validated existing decision
address. Legacy unknown values remain unknown. Future graphs and training exports
should select from this store and pin attempt/revision IDs, world/schema versions,
selection filters and viewer projection; those export tools are not built here.

The existing `/api/traces` list now accepts `scope=all|mine` (default all), `q`,
`player`, `other`, `pairing`, `version`, `deck`, `status`, `result`, `after`,
`before`, `cursor` and `limit` (1–100, default 50). Results are newest first with
stable ID tie breaking. `X-Etude-Next-Cursor` supplies the next page. Results
are typed summary data; private traces, credentials and deal seeds are absent.
`x-etude-player-token` authenticates the browser player, and existing table
credentials still work. Listing is not mutation or private replay permission.

For a separate built preview without replacing another running server's assets,
set `ETUDE_FRONTEND_BUILD` to a fresh absolute output path for both the static
build and ASGI launch. Use a separate `ETUDE_TRACES_DIR` and port as well.
