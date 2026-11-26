"""SQLite authority for game attempts, their two players, traces and analyses.

A game is named by its attempt id. Every game in the store is readable; a
played game nothing has ended is unfinished (`UNFINISHED_SQL`) and can be
taken up again from its stored steps.

Databases written by earlier builds also hold `feedback`, `players`,
`play_sessions`, `session_games` and `continuation_receipts` tables and the
`attempts.visibility` and `attempts.participant_id` columns. Nothing reads
them and nothing drops them, so an earlier build can still open the file.
"""

import base64
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import secrets
import sqlite3
import threading
from typing import Any

PROCESS_ID = secrets.token_hex(16)
RECORD_VERSION = 2
ORIGINS = {
    "unknown",
    "human_exploratory",
    "automated_validation",
    "bot_evaluation",
    "training",
}
# Endings of a record import that stopped before game over (`watch.import_record`).
CLOSED_RECORD_ENDINGS = frozenset({"record_interrupted", "record_diverged"})
# A played game is unfinished until the rules end it or its record breaks.
# Closing the tab, idling out and stopping the server end nothing, so rows an
# earlier build closed for those reasons are unfinished too; they are read
# through this predicate and never rewritten. A watched game has no seat to
# take up again, so it is never unfinished. Over `attempts AS a`.
UNFINISHED_SQL = """(
    EXISTS(SELECT 1 FROM attempt_players h WHERE h.attempt_id=a.id AND h.kind='human')
    AND (a.status IN ('active', 'interrupted') OR (a.status='stopped' AND a.ending_reason IN
        ('server_restart', 'server_shutdown', 'session_expired', 'new_game'))))"""


def shown_status(status: str, resumable: bool) -> str:
    """A game's status as History and Replay name it: `active` (in progress),
    `completed` (the rules ended it) or `stopped`. The stored value keeps an
    earlier build's spelling (`interrupted`, a resumable `stopped`)."""
    if resumable or status == "active":
        return "active"
    return "completed" if status == "completed" else "stopped"


# The one person who plays here: every human seat is recorded as this player.
HUMAN_PLAYER_ID = "human:local"
HUMAN_PLAYER_NAME = "You"


@dataclass(frozen=True)
class RecordedPlayer:
    seat: int
    player_id: str
    kind: str
    name: str
    deck: str
    version: str | None = None
    inference: dict | None = None


@dataclass(frozen=True)
class RetainedAttempt:
    trace: dict[str, Any]
    world_json: str


def bot_player(config: dict[str, Any]) -> RecordedPlayer:
    identity = config.get("opponent") or {}
    policy = config["villain_type"]
    digest = identity.get("sha256")
    inference = {"policy": policy}
    if policy == "checkpoint":
        inference["deterministic"] = config.get("villain_deterministic", False)
    elif policy == "search":
        inference["simulations"] = config.get("villain_sims")
    provenance = config.get("opponent_checkpoint") or {}
    if provenance.get("schema") == "managym-policy-1":
        inference.update(runtime=provenance["policy"], sha256=digest)
    version = (
        digest
        or hashlib.sha256(json.dumps(inference, sort_keys=True).encode()).hexdigest()
    )
    # Older artifacts identify exact bytes only; never infer lineage from a name.
    bot_id = identity.get("bot_id") or (
        f"checkpoint:{digest}" if digest else f"control:{policy}"
    )
    return RecordedPlayer(
        1 - config.get("human_seat", 0),
        bot_id,
        "bot",
        identity.get("name") or policy.title(),
        config.get("villain_deck_name", "custom"),
        version,
        inference,
    )


def page_cursor(row: dict) -> str:
    return base64.urlsafe_b64encode(
        json.dumps([row["timestamp"], row["id"]]).encode()
    ).decode()


def decode_cursor(cursor: str | None) -> tuple[str, str] | None:
    if cursor is None:
        return None
    try:
        value = json.loads(base64.b64decode(cursor, altchars=b"-_", validate=True))
        if (
            not isinstance(value, list)
            or len(value) != 2
            or not all(isinstance(v, str) for v in value)
        ):
            raise ValueError
        return tuple(value)
    except (ValueError, TypeError) as exc:
        raise ValueError("Invalid history cursor.") from exc


class AttemptStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA foreign_keys=ON")
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS attempts (
                    id TEXT PRIMARY KEY,
                    process_id TEXT NOT NULL, started_at TEXT NOT NULL,
                    ended_at TEXT, status TEXT NOT NULL, ending_reason TEXT,
                    deal_seed TEXT, winner INTEGER, revision INTEGER NOT NULL,
                    world_json TEXT NOT NULL, trace_json TEXT NOT NULL,
                    record_version INTEGER NOT NULL DEFAULT 2,
                    origin TEXT NOT NULL DEFAULT 'unknown',
                    num_events INTEGER NOT NULL DEFAULT 0,
                    -- The game's numbers so far (`numbers.GameNumbers`); NULL
                    -- until first computed and again whenever a step is added.
                    numbers_json TEXT
                );
                CREATE TABLE IF NOT EXISTS attempt_players (
                    attempt_id TEXT NOT NULL REFERENCES attempts(id),
                    seat INTEGER NOT NULL CHECK(seat IN (0, 1)),
                    player_id TEXT NOT NULL, kind TEXT NOT NULL CHECK(kind IN ('human', 'bot')),
                    name TEXT NOT NULL, deck TEXT NOT NULL,
                    version TEXT, inference_json TEXT,
                    PRIMARY KEY(attempt_id, seat)
                );
                CREATE INDEX IF NOT EXISTS history_order ON attempts(started_at DESC, id DESC);
                CREATE INDEX IF NOT EXISTS history_player ON attempt_players(player_id, attempt_id);
                CREATE INDEX IF NOT EXISTS history_version ON attempt_players(version, attempt_id);
                CREATE INDEX IF NOT EXISTS arena_source ON attempt_players(
                    json_extract(inference_json, '$.source_sha256'), attempt_id
                );
                -- A policy's reading of one seat of a game (`analysis.py`): never
                -- part of the trace, any frame or any export.
                CREATE TABLE IF NOT EXISTS analyses (
                    id TEXT PRIMARY KEY,
                    attempt_id TEXT NOT NULL REFERENCES attempts(id),
                    seat INTEGER NOT NULL CHECK(seat IN (0, 1)),
                    policy_json TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('running', 'complete', 'stopped')),
                    reason TEXT, total INTEGER NOT NULL,
                    process_id TEXT NOT NULL, created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS analyses_attempt ON analyses(attempt_id, created_at);
                CREATE TABLE IF NOT EXISTS analysed_decisions (
                    analysis_id TEXT NOT NULL REFERENCES analyses(id),
                    event INTEGER NOT NULL, evidence_json TEXT NOT NULL,
                    PRIMARY KEY(analysis_id, event)
                );
            """)
            self._migrate(db)
            # A database an earlier build created requires an owner on every row.
            self._owner_column = "participant_id" in {
                row[1] for row in db.execute("PRAGMA table_info(attempts)")
            }
            # A watched game cannot be rebuilt from its record, so one another
            # process left active is over. A played game stays unfinished.
            db.execute(
                """UPDATE attempts SET status='interrupted',
                ending_reason='server_restart' WHERE status='active' AND process_id<>?
                AND NOT EXISTS(SELECT 1 FROM attempt_players h
                    WHERE h.attempt_id=attempts.id AND h.kind='human')""",
                (PROCESS_ID,),
            )

    def _migrate(self, db):
        columns = {row[1] for row in db.execute("PRAGMA table_info(attempts)")}
        if "record_version" not in columns:
            for column in (
                "record_version INTEGER NOT NULL DEFAULT 1",
                "origin TEXT NOT NULL DEFAULT 'unknown'",
                "num_events INTEGER NOT NULL DEFAULT 0",
            ):
                db.execute(f"ALTER TABLE attempts ADD COLUMN {column}")
            for row in db.execute("SELECT * FROM attempts").fetchall():
                trace = json.loads(row["trace_json"])
                seats = [
                    RecordedPlayer(
                        0,
                        HUMAN_PLAYER_ID,
                        "human",
                        HUMAN_PLAYER_NAME,
                        row["hero_deck"],
                    ),
                    bot_player(trace["config"]),
                ]
                self._save_players(db, row["id"], seats)
                db.execute(
                    "UPDATE attempts SET num_events=? WHERE id=?",
                    (len(trace.get("events", [])), row["id"]),
                )
            for column in ("bot_sha256", "hero_deck", "villain_deck"):
                db.execute(f"ALTER TABLE attempts DROP COLUMN {column}")
        if "numbers_json" not in {
            row[1] for row in db.execute("PRAGMA table_info(attempts)")
        }:
            db.execute("ALTER TABLE attempts ADD COLUMN numbers_json TEXT")

    @contextmanager
    def connect(self):
        with self._lock, self._db:
            yield self._db

    @staticmethod
    def players_for_game(config: dict[str, Any]) -> list[RecordedPlayer]:
        """The person and the configured bot, each at its native seat."""
        return [
            RecordedPlayer(
                config.get("human_seat", 0),
                HUMAN_PLAYER_ID,
                "human",
                HUMAN_PLAYER_NAME,
                config.get("hero_deck_name", "custom"),
            ),
            bot_player(config),
        ]

    @staticmethod
    def _save_players(db, attempt, players):
        if len(players) != 2 or {p.seat for p in players} != {0, 1}:
            raise ValueError("A game must identify exactly two seats.")
        for player in players:
            db.execute(
                "INSERT INTO attempt_players VALUES(?,?,?,?,?,?,?,?)",
                (
                    attempt,
                    player.seat,
                    player.player_id,
                    player.kind,
                    player.name,
                    player.deck,
                    player.version,
                    json.dumps(player.inference)
                    if player.inference is not None
                    else None,
                ),
            )

    def save(
        self,
        attempt: str,
        revision: int,
        world: dict[str, Any],
        trace: dict[str, Any],
        *,
        trace_json: str,
        finished: bool,
        players: list[RecordedPlayer],
        origin: str | None = None,
    ) -> None:
        config = trace["config"]
        ending = trace["end_reason"] if finished else None
        status = (
            "interrupted"
            if ending == "server_shutdown"
            else ("completed" if ending == "game_over" else "stopped")
            if finished
            else "active"
        )
        origin = origin or os.getenv("ACADEMY_PLAY_RECORD_ORIGIN", "unknown")
        if origin not in ORIGINS:
            raise ValueError("Unknown game record origin.")
        # `trace_json` encodes `trace`; a live game encodes it incrementally
        # (`Trace.to_json`), before taking the writer lock.
        world_json = json.dumps(world)
        with self.connect() as db:
            exists = db.execute(
                "SELECT 1 FROM attempts WHERE id=?", (attempt,)
            ).fetchone()
            owner = (",participant_id", ",''") if self._owner_column else ("", "")
            db.execute(
                f"""INSERT INTO attempts
                (id,process_id,started_at,ended_at,status,ending_reason,deal_seed,winner,
                 revision,world_json,trace_json,record_version,origin,num_events{owner[0]})
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?{owner[1]})
                ON CONFLICT(id) DO UPDATE SET
                ended_at=excluded.ended_at, status=excluded.status, ending_reason=excluded.ending_reason,
                winner=excluded.winner, revision=excluded.revision, trace_json=excluded.trace_json,
                num_events=excluded.num_events,
                numbers_json=CASE WHEN excluded.num_events=attempts.num_events
                    THEN attempts.numbers_json END
                WHERE attempts.status='active'""",
                (
                    attempt,
                    PROCESS_ID,
                    trace["timestamp"],
                    datetime.now(timezone.utc).isoformat() if finished else None,
                    status,
                    ending,
                    hex(config["seed"]),
                    trace["winner"] if ending == "game_over" else None,
                    revision,
                    world_json,
                    trace_json,
                    RECORD_VERSION,
                    origin,
                    len(trace["events"]),
                ),
            )
            if not exists:
                self._save_players(db, attempt, players)

    def arena_recording(self, source_sha256: str) -> str | None:
        """Find one finished import by retained source identity, without reading traces."""
        with self.connect() as db:
            row = db.execute(
                """SELECT p.attempt_id FROM attempt_players AS p
                JOIN attempts AS a ON a.id=p.attempt_id
                WHERE json_extract(p.inference_json, '$.source_sha256')=?
                AND a.status IN ('completed', 'stopped') LIMIT 1""",
                (source_sha256,),
            ).fetchone()
        return None if row is None else str(row[0])

    def load(self, attempt: str) -> dict | None:
        retained = self.load_attempt(attempt)
        return retained.trace if retained is not None else None

    def has_attempt(self, attempt: str) -> bool:
        with self.connect() as db:
            return (
                db.execute("SELECT 1 FROM attempts WHERE id=?", (attempt,)).fetchone()
                is not None
            )

    def load_attempt(self, attempt: str) -> RetainedAttempt | None:
        """Read the private trace and its world from the same database snapshot."""
        with self.connect() as db:
            row = db.execute(
                "SELECT trace_json, world_json, ending_reason FROM attempts WHERE id=?",
                (attempt,),
            ).fetchone()
        if row is None:
            return None
        trace = json.loads(row["trace_json"])
        # A row closed without its trace being rewritten (a legacy restart, a
        # record that stopped replaying) reads with the row's ending.
        if trace["end_reason"] == "active" and row["ending_reason"]:
            trace["end_reason"] = row["ending_reason"]
        return RetainedAttempt(trace, row["world_json"])

    def unfinished(self, attempt: str) -> bool:
        """Whether this is a played game nothing has ended (`UNFINISHED_SQL`)."""
        with self.connect() as db:
            return (
                db.execute(
                    f"SELECT 1 FROM attempts a WHERE a.id=? AND {UNFINISHED_SQL}",
                    (attempt,),
                ).fetchone()
                is not None
            )

    def reopen(self, attempt: str) -> int:
        """Make an unfinished game this process's active one, so `save` appends
        to it, and return its revision, advanced by one: nothing addressed to
        the game before it was reopened addresses it now. The only write that
        changes a row an earlier build closed."""
        with self.connect() as db:
            db.execute(
                f"""UPDATE attempts AS a SET status='active', ending_reason=NULL,
                ended_at=NULL, process_id=?, revision=revision+1
                WHERE a.id=? AND {UNFINISHED_SQL}""",
                (PROCESS_ID, attempt),
            )
            row = db.execute(
                "SELECT revision FROM attempts WHERE id=?", (attempt,)
            ).fetchone()
        return int(row["revision"])

    def diverge(self, attempt: str) -> None:
        """End an unfinished game whose record no longer replays. Its stored
        steps stay as they are and stay replayable."""
        with self.connect() as db:
            db.execute(
                f"""UPDATE attempts AS a SET status='stopped',
                ending_reason='record_diverged', ended_at=? WHERE a.id=? AND {UNFINISHED_SQL}""",
                (datetime.now(timezone.utc).isoformat(), attempt),
            )

    def metadata(self, attempt: str) -> dict[str, Any]:
        with self.connect() as db:
            row = db.execute(
                f"""SELECT record_version,origin,revision,status,ending_reason,
                    started_at,ended_at,{UNFINISHED_SQL} AS resumable
                    FROM attempts a WHERE id=?""",
                (attempt,),
            ).fetchone()
            if row is None:
                return {}
            players = self._players(db, attempt)
        return {
            **dict(row),
            "resumable": bool(row["resumable"]),
            "players": players,
        }

    @staticmethod
    def _players(db: sqlite3.Connection, attempt: str) -> list[dict[str, Any]]:
        return AttemptStore._players_for(db, [attempt]).get(attempt, [])

    @staticmethod
    def _players_for(
        db: sqlite3.Connection, attempts: list[str]
    ) -> dict[str, list[dict[str, Any]]]:
        if not attempts:
            return {}
        placeholders = ",".join("?" for _ in attempts)
        result: dict[str, list[dict[str, Any]]] = {}
        for row in db.execute(
            f"SELECT attempt_id,seat,player_id,kind,name,deck,version,inference_json FROM attempt_players WHERE attempt_id IN ({placeholders}) ORDER BY attempt_id,seat",
            attempts,
        ):
            player = dict(row)
            attempt = player.pop("attempt_id")
            inference = player.pop("inference_json")
            player["inference"] = json.loads(inference) if inference else None
            result.setdefault(attempt, []).append(player)
        return result

    def summaries(
        self,
        *,
        q: str = "",
        bot: str | None = None,
        pairing: str | None = None,
        deck: str | None = None,
        status: str | None = None,
        after: str | None = None,
        before: str | None = None,
        cursor: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """One page of games, newest first. `status` is `active` (in progress:
        an unfinished played game or a watched game still running),
        `completed` or `stopped`; `bot` is a bot seat's player id."""
        if status not in {None, "active", "completed", "stopped"}:
            raise ValueError("Invalid history status.")
        if pairing not in {None, "human-bot", "bot-bot"}:
            raise ValueError("Invalid pairing.")
        if not 1 <= limit <= 101 or len(q) > 200:
            raise ValueError(
                "History requires a limit of 1–100 and search up to 200 characters."
            )
        for value in (after, before):
            if value:
                try:
                    datetime.strptime(value, "%Y-%m-%d")
                except ValueError as exc:
                    raise ValueError("Dates must use YYYY-MM-DD.") from exc
        in_progress = f"(a.status='active' OR {UNFINISHED_SQL})"
        clauses = ["1"]
        args: list[Any] = []
        if q:
            pattern = (
                "%"
                + q.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
                + "%"
            )
            clauses.append(
                "(a.id LIKE ? ESCAPE '\\' OR EXISTS(SELECT 1 FROM attempt_players s WHERE s.attempt_id=a.id AND (s.name LIKE ? ESCAPE '\\' OR s.player_id LIKE ? ESCAPE '\\' OR s.version LIKE ? ESCAPE '\\')))"
            )
            args.extend([pattern] * 4)
        if bot:
            clauses.append(
                "EXISTS(SELECT 1 FROM attempt_players p WHERE p.attempt_id=a.id AND p.kind='bot' AND p.player_id=?)"
            )
            args.append(bot)
        if deck:
            clauses.append(
                "EXISTS(SELECT 1 FROM attempt_players s WHERE s.attempt_id=a.id AND s.deck=?)"
            )
            args.append(deck)
        if pairing:
            clauses.append(
                "(SELECT count(*) FROM attempt_players s WHERE s.attempt_id=a.id AND s.kind='human')=?"
            )
            args.append(pairing.count("human"))
        if status == "active":
            clauses.append(in_progress)
        elif status == "completed":
            clauses.append("a.status='completed'")
        elif status == "stopped":
            clauses.append(
                f"a.status IN ('stopped', 'interrupted') AND NOT {UNFINISHED_SQL}"
            )
        if after:
            clauses.append("a.started_at>=?")
            args.append(after)
        if before:
            clauses.append("substr(a.started_at,1,10)<=?")
            args.append(before)
        position = decode_cursor(cursor)
        if position:
            clauses.append("(a.started_at,a.id)<(?,?)")
            args.extend(position)
        with self.connect() as db:
            rows = db.execute(
                f"""SELECT a.id,a.started_at,a.ended_at,a.winner,a.status,a.ending_reason,
                    a.num_events,a.revision,a.record_version,a.origin,a.numbers_json,
                    {UNFINISHED_SQL} AS resumable
                FROM attempts a WHERE {" AND ".join(clauses)}
                ORDER BY started_at DESC,id DESC LIMIT ?""",
                [*args, limit],
            ).fetchall()
            players_by_attempt = self._players_for(db, [row["id"] for row in rows])
        return [
            {
                "id": row["id"],
                "timestamp": row["started_at"],
                "ended_at": row["ended_at"],
                "winner": row["winner"],
                "status": shown_status(row["status"], bool(row["resumable"])),
                "end_reason": row["ending_reason"] or "active",
                "num_events": row["num_events"],
                "numbers": json.loads(row["numbers_json"])
                if row["numbers_json"]
                else None,
                "revision": row["revision"],
                "players": players_by_attempt.get(row["id"], []),
                "record_version": row["record_version"],
                "origin": row["origin"],
                "resumable": bool(row["resumable"]),
            }
            for row in rows
        ]

    # ———————————————— Numbers

    def numbers(self, attempt: str) -> tuple[str, str | None] | None:
        """The game's status and saved numbers JSON, or None for no such game."""
        with self.connect() as db:
            row = db.execute(
                "SELECT status, numbers_json FROM attempts WHERE id=?", (attempt,)
            ).fetchone()
        return None if row is None else (row["status"], row["numbers_json"])

    def save_numbers(self, attempt: str, numbers_json: str, num_events: int) -> None:
        """Save the numbers of a game's first `num_events` steps, unless it has
        moved on since they were read; `save` clears them when it appends."""
        with self.connect() as db:
            db.execute(
                "UPDATE attempts SET numbers_json=? WHERE id=? AND num_events=? AND numbers_json IS NULL",
                (numbers_json, attempt, num_events),
            )

    # ———————————————— Analyses

    def start_analysis(
        self, attempt: str, seat: int, policy: dict[str, Any], total: int
    ) -> str:
        """A new running analysis of `seat`, owned by this process."""
        identifier = secrets.token_hex(8)
        with self.connect() as db:
            db.execute(
                "INSERT INTO analyses VALUES(?,?,?,?,'running',NULL,?,?,?)",
                (
                    identifier,
                    attempt,
                    seat,
                    json.dumps(policy, sort_keys=True),
                    total,
                    PROCESS_ID,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
        return identifier

    def add_analysed_decision(self, analysis: str, event: int, evidence: str) -> None:
        with self.connect() as db:
            db.execute(
                "INSERT INTO analysed_decisions VALUES(?,?,?)",
                (analysis, event, evidence),
            )

    def end_analysis(self, analysis: str, reason: str | None) -> None:
        """Complete without a reason, else stopped with it."""
        with self.connect() as db:
            db.execute(
                "UPDATE analyses SET status=?, reason=? WHERE id=?",
                ("stopped" if reason else "complete", reason, analysis),
            )

    def analyses(self, attempt: str) -> list[dict[str, Any]]:
        """Every analysis of a game, oldest first, with its decisions so far.

        A running analysis another process started stopped with that process.
        """
        with self.connect() as db:
            rows = db.execute(
                "SELECT * FROM analyses WHERE attempt_id=? ORDER BY created_at, id",
                (attempt,),
            ).fetchall()
            decisions = {
                row["id"]: [
                    json.loads(evidence)
                    for (evidence,) in db.execute(
                        "SELECT evidence_json FROM analysed_decisions WHERE analysis_id=? ORDER BY event",
                        (row["id"],),
                    )
                ]
                for row in rows
            }
        result = []
        for row in rows:
            orphaned = row["status"] == "running" and row["process_id"] != PROCESS_ID
            result.append(
                {
                    "id": row["id"],
                    "seat": row["seat"],
                    "policy": json.loads(row["policy_json"]),
                    "status": "stopped" if orphaned else row["status"],
                    "reason": "Academy stopped before this analysis finished."
                    if orphaned
                    else row["reason"],
                    "total": row["total"],
                    "created_at": row["created_at"],
                    "decisions": decisions[row["id"]],
                }
            )
        return result
