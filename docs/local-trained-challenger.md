# Train, play, and retain a local challenger

This workflow runs the existing search-distillation trainer on the corrected
Allies/Lessons world: the authored decks with their sideboards, so every Learn
decision offers Take a Lesson, Discard and draw, and Decline. Exp-03 supports
distillation over early PPO in its historical world; it does **not** rank this
small new candidate as the strongest current bot. The generic
`manabot train --preset local` uses a different deck and is a training smoke,
not this matchup's challenger recipe.

No cloud account is needed. The fixed local workflow follows the
[approved starting configuration](plans/strongest-manabot-training.md):
uniform-prior determinized PUCT with 64 simulations across four compatible
worlds plays eight teacher games, alternating which deck moves first. A
width-64 policy then learns the complete visit distributions for eight epochs
(policy weight 1, value weight 0, lr 0.001, batch 128, seed 79, whole-game 10%
validation). Both seats contribute labels. The selected artifact is the final
epoch of this one run, without tuning or a strength claim. Each execution has
one CPU worker/thread, a ten-minute wall cap and a 4 GiB process-tree RSS cap.

Teacher data must pass admission before training. `admission.json` compares
the rules engine's own legal-offer count with the encoded training row at
every decision, overall and by decision kind. The run is rejected if any
choice is missing, if no Learn decision offered a Lesson, if a game used
another deck or sideboard, or if more than 1% of search continuations hit
their step cap. Overflow of a fixed training observation also rejects the run
rather than dropping legal choices. Training rows hold only the acting
player's observation.

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
and `games.json` survive failures. `recipe.json` also records the world the
run binds to: rules runtime digest, content manifest, deck and sideboard
setup, Lesson pool, and the observation/action tensor shapes and enumerations,
each with its own digest. `phases.json` reports
seconds and teacher decisions/second; `receipt.json` reports worker elapsed
`wall_seconds`, total script `operator_wall_seconds` including source capture,
sampled worker process-tree RSS (excluding the supervisor), actual new cloud
spend and unknown electricity cost. Build, browser and human-play time are
separate; missing phase measurements remain unknown.
`candidate.json` binds the checkpoint digest, source/recipe, data, admission
report, world identities, observation configuration, content manifest and
inference setting. Export reloads the checkpoint and checks exact logits
against the trained model. The run then loads the candidate the way the play
server does and plays one complete game in each deck assignment against a
random player, recording both in `demo_check.json` and
`demo-check/play.sqlite` with `automated_validation` origin. This proves the
checkpoint loads and finishes legal games; it measures no strength. A run that
fails this check has a failed receipt, and a failed receipt cannot be
configured as an opponent.

Reproducibility means retained inputs and procedure, complete legal games and
replay witnesses. It does not promise identical stochastic checkpoints across
devices or independent seeds. These receipts are the ETU-80 measurement input;
do not copy results into a separate measurement store. Early ETU-79 feasibility
used the pre-ETU-75 world and a flat search-64 teacher; those checkpoints do
not bind to this world. The first corrected-world executions are in the
[2026-09-29 record](evidence/corrected-world-training-2026-09-29.md).

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


## Evaluate the selected matchup

Use the existing arena's bounded instrument command on the authored Allies
versus Lessons setup, including both sideboards:

```bash
uv run python -m experiments.runners.run_skill_arena evaluate-matchup \
  --out-dir .runs/allies-lessons-arena-1 --deal-seeds 83001
```

Without a candidate registration this runs Random against the shipped default
demo Search-64. It calls `etude.villain.SearchVillain` itself: 64 simulations
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
schema: world `w3`, suite `w3-allies-lessons-v1`, exact checkpoint SHA-256/size,
parameter count, training seed, immutable artifact ID and explicit inference
spec (`checkpoint`, `deterministic`, `cpu`, batch size 1). Compute the setup and
ABI fingerprints with `runtime_fingerprints(match_hypers=selected_match(),
observation_space=loaded_space, world="w3")`; both are existing Python APIs.
The ordinary checkpoint loader supplies `loaded_space`; the arena retains its
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
