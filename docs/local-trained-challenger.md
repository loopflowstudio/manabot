# Play, evaluate and retain a local challenger

Build and serve the same SPA/ASGI entrypoint as the hosted app, locally:

```sh
npm --prefix academy/ui ci
ACADEMY_STATIC_BUILD=1 npm --prefix academy/ui run build
ACADEMY_FRONTEND_BUILD="$PWD/academy/ui/build" \
ACADEMY_TRACES_DIR="$PWD/.runs/human-play" \
ACADEMY_PLAY_RECORD_ORIGIN=human_exploratory \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
uv run uvicorn deploy.play.asgi:app --host 127.0.0.1 --port 8000
```

Open `http://localhost:8000` and choose a trained opponent under **Trained
checkpoints**. Checkpoints from `uv run manabot train`, regime runs and
Tournament files are listed there. Everything under `.runs/` appears
automatically; add other files, run directories or Tournament JSON with
`ACADEMY_PLAY_CHECKPOINTS="/path/a.pt:/path/run-dir:/path/grid.json"`.
Tournament entrants resolve beside the Tournament file or in a listed
directory, and only with their pinned bytes. A checkpoint bound to another
world, content or deck setup is listed with the mismatch and cannot be started.
Choose the checkpoint's decks in either seat; its world, model, step and
producer show under the menu, and its exact identity is recorded on the trace.
The browser names only a digest, never a checkpoint path; the server verifies
the bytes before play, and replacing them fails clearly. Search, Random and
Passive remain selectable controls.

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
training runtime and checkpoints. The checked Fly deployment has no volume mount;
this local workflow does not deploy anything.


## Evaluate the selected matchup

Use the bounded evaluation instrument command on the authored Allies
versus Lessons setup, including both sideboards:

```bash
uv run python -m manabot.eval.evaluate \
  --out-dir .runs/allies-lessons-arena-1 --deal-seeds 83001
```

Without a candidate registration this runs Random against the shipped default
demo Search-64. It calls `manabot.sim.search.flat.FlatSearchPolicy` itself: 64 simulations
per legal action, four rollouts per world, 2,000-step playout cap, deterministic
first-maximum selection, CPU, one worker and one Torch thread. The baseline
registration pins the policy source bytes; the protocol additionally pins the
native extension, rules sources, compiled content, complete setup, observation
and action ABIs and evaluator sources. It does not substitute a historical
trained opponent whose runtime no longer matches.

Each deal seed produces **four games**. Legs 0/1 keep UR in starting seat 0 and
GW in seat 1 while swapping players; legs 2/3 put GW in starting seat 0 and UR
in seat 1 and swap players again. Thus each player gets both decks and both
starting seats. The same seed fixes each per-seat deal within its pair. Reversing
the decks does not promise identical opening hands across those two pairs.
Bootstrap resampling keeps the entire four-game deal block together.

To evaluate exported bytes, pass `--candidate registration.json` and
`--candidate-checkpoint candidate.pt`. Use the existing `PlayerRegistration`
schema: world `w4`, suite `w4-allies-lessons-v1`, exact checkpoint SHA-256/size,
parameter count, training seed, immutable artifact ID and explicit inference
spec (`checkpoint`, `deterministic`, `cpu`, batch size 1). Compute the setup and
ABI fingerprints with `runtime_fingerprints(match_hypers=selected_match(),
observation_space=loaded_space, world="w4")`; both are existing Python APIs.
The ordinary checkpoint loader supplies `loaded_space`; the simulator retains its
bounds in every trace. No checkpoint port or loader bypass is performed.
Stochastic policies are reseeded per player decision, with the exact seed in
the trace. Treat an untrained fixture as a fixture, never as training evidence.

The default limits are 120 wall seconds per game (including worker startup)
and 10,000 Commands; change them explicitly with `--game-seconds` and
`--max-commands` before starting a cohort. Player exceptions and native worker
crashes attributable to a policy are forfeits. Total-game timeouts, Command
caps and environment failures are retained as unscored failures and reject the
cohort's statistical admission. They never earn draw credit or disappear from
the denominator. Only authoritative terminal draws score 0.5. Any failed game,
replay mismatch or evaluator source change causes a nonzero exit after writing
the attempt artifacts. A passing replay of a failed prefix is not a completed
game. Admission errors before gameplay, such as wrong bytes or ABI, fail before
creating the output directory.

`protocol.json` is written before play; `players.json`, `matches.jsonl`, the
existing compressed Command traces, `replay.json`, `rating.json`, the resource
ledger and digest manifest retain the result. Existing output directories are
rejected. Replay rebuilds the exact retained setup and observation bounds and
checks frames, offers, Commands, actors, state digests and terminal results
without invoking policies. Frozen historical arena contracts keep their
interactive-mirror interpretation and are not regenerated.

This command builds the instrument. Its four-game default is not a strength
comparison or useful uncertainty estimate; source-pinned current Learn rules
are not a declaration that ETU-75's remaining w3 certification passed. A scored
comparison still needs that gate, independent training seeds, held-out deals,
a preregistered cohort and matched inference budgets.

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
public human IDs; a trained bot's identity is its exact checkpoint digest,
separate from producer and loading path. We do not guess lineage from similar
names.

Declare record origin when launching the server with `ACADEMY_PLAY_RECORD_ORIGIN`:
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
set `ACADEMY_FRONTEND_BUILD` to a fresh absolute output path for both the static
build and ASGI launch. Use a separate `ACADEMY_TRACES_DIR` and port as well.
