"""SQLite authority for game attempts, their two players, traces, and feedback."""

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

from managym.academy.play_session import (
    PlaySessionRecord,
    SessionMembership,
    SessionStart,
)

PROCESS_ID = secrets.token_hex(16)
RECORD_VERSION = 2
ORIGINS = {
    "unknown",
    "human_exploratory",
    "automated_validation",
    "bot_evaluation",
    "training",
}


def participant_id(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


@dataclass(frozen=True)
class RecordedPlayer:
    seat: int
    player_id: str
    kind: str
    name: str
    deck: str
    version: str | None = None
    inference: dict | None = None


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
                    id TEXT PRIMARY KEY, participant_id TEXT NOT NULL,
                    process_id TEXT NOT NULL, started_at TEXT NOT NULL,
                    ended_at TEXT, status TEXT NOT NULL, ending_reason TEXT,
                    deal_seed TEXT, winner INTEGER, revision INTEGER NOT NULL,
                    world_json TEXT NOT NULL, trace_json TEXT NOT NULL,
                    record_version INTEGER NOT NULL DEFAULT 2,
                    visibility TEXT NOT NULL DEFAULT 'shared',
                    origin TEXT NOT NULL DEFAULT 'unknown',
                    num_events INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS players (
                    id TEXT PRIMARY KEY, credential_hash TEXT UNIQUE NOT NULL,
                    name TEXT NOT NULL
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
                CREATE TABLE IF NOT EXISTS feedback (
                    attempt_id TEXT NOT NULL REFERENCES attempts(id),
                    request_id TEXT NOT NULL, created_at TEXT NOT NULL,
                    note TEXT NOT NULL, author_id TEXT, decision_address TEXT,
                    PRIMARY KEY (attempt_id, request_id)
                );
            """)
            db.executescript("""
                CREATE INDEX IF NOT EXISTS feedback_history ON feedback(attempt_id,created_at,request_id);
                CREATE TABLE IF NOT EXISTS play_sessions (
                    id TEXT PRIMARY KEY, owner TEXT NOT NULL, created_at TEXT NOT NULL,
                    title TEXT NOT NULL, revision INTEGER NOT NULL, draft_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS session_owner ON play_sessions(owner,id);
                CREATE TABLE IF NOT EXISTS session_games (
                    attempt_id TEXT PRIMARY KEY REFERENCES attempts(id),
                    session_id TEXT NOT NULL REFERENCES play_sessions(id),
                    game_number INTEGER NOT NULL,
                    matchup_json TEXT NOT NULL,
                    UNIQUE(session_id,game_number)
                );
                CREATE TABLE IF NOT EXISTS continuation_receipts (
                    owner TEXT NOT NULL, request_id TEXT NOT NULL, fingerprint TEXT NOT NULL,
                    attempt_id TEXT NOT NULL REFERENCES attempts(id),
                    PRIMARY KEY(owner,request_id)
                );
            """)
            self._migrate(db)
            db.execute(
                """UPDATE attempts SET status='interrupted',
                ending_reason='server_restart' WHERE status='active' AND process_id<>?""",
                (PROCESS_ID,),
            )

    def _migrate(self, db):
        columns = {row[1] for row in db.execute("PRAGMA table_info(attempts)")}
        if "record_version" not in columns:
            for column in (
                "record_version INTEGER NOT NULL DEFAULT 1",
                "visibility TEXT NOT NULL DEFAULT 'private'",
                "origin TEXT NOT NULL DEFAULT 'unknown'",
                "num_events INTEGER NOT NULL DEFAULT 0",
            ):
                db.execute(f"ALTER TABLE attempts ADD COLUMN {column}")
            for row in db.execute("SELECT * FROM attempts").fetchall():
                trace = json.loads(row["trace_json"])
                human = self._profile(db, "legacy:" + row["participant_id"])
                seats = [
                    RecordedPlayer(
                        0, human["id"], "human", human["name"], row["hero_deck"]
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
        feedback_columns = {row[1] for row in db.execute("PRAGMA table_info(feedback)")}
        for column in ("author_id", "decision_address"):
            if column not in feedback_columns:
                db.execute(f"ALTER TABLE feedback ADD COLUMN {column} TEXT")

    @contextmanager
    def connect(self):
        with self._lock, self._db:
            yield self._db

    @staticmethod
    def _profile(db, credential_hash, name=None):
        row = db.execute(
            "SELECT id,name FROM players WHERE credential_hash=?", (credential_hash,)
        ).fetchone()
        if row is None:
            identifier = "human:" + secrets.token_hex(12)
            db.execute(
                "INSERT INTO players VALUES(?,?,?)",
                (identifier, credential_hash, name or f"Player {identifier[-6:]}"),
            )
        elif name is not None:
            db.execute(
                "UPDATE players SET name=? WHERE credential_hash=?",
                (name, credential_hash),
            )
        return dict(
            db.execute(
                "SELECT id,name FROM players WHERE credential_hash=?",
                (credential_hash,),
            ).fetchone()
        )

    def profile(self, token: str, name: str | None = None) -> dict:
        if not isinstance(token, str) or not 32 <= len(token) <= 128:
            raise ValueError("Player credential must contain 32–128 characters.")
        if name is not None and (
            not isinstance(name, str) or not 1 <= len(name.strip()) <= 60
        ):
            raise ValueError("Player name must contain 1–60 characters.")
        with self.connect() as db:
            return self._profile(
                db, participant_id(token), name.strip() if name is not None else None
            )

    def players_for_game(
        self,
        owner: str,
        config: dict[str, Any],
        token: str | None = None,
        name: str | None = None,
    ) -> list[RecordedPlayer]:
        if token is not None:
            human = self.profile(token, name)
        else:
            with self.connect() as db:
                human = self._profile(db, "legacy:" + owner)
        return [
            RecordedPlayer(
                config.get("human_seat", 0),
                human["id"],
                "human",
                human["name"],
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
        owner: str,
        revision: int,
        world: dict[str, Any],
        trace: dict[str, Any],
        *,
        finished: bool,
        players: list[RecordedPlayer],
        origin: str | None = None,
        session_start: SessionStart | None = None,
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
        # Serialize the growing trace before taking the writer lock.
        world_json = json.dumps(world)
        trace_json = json.dumps(trace)
        with self.connect() as db:
            exists = db.execute(
                "SELECT 1 FROM attempts WHERE id=?", (attempt,)
            ).fetchone()
            db.execute(
                """INSERT INTO attempts
                (id,participant_id,process_id,started_at,ended_at,status,ending_reason,deal_seed,winner,
                 revision,world_json,trace_json,record_version,visibility,origin,num_events)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,'shared',?,?)
                ON CONFLICT(id) DO UPDATE SET
                ended_at=excluded.ended_at, status=excluded.status, ending_reason=excluded.ending_reason,
                winner=excluded.winner, revision=excluded.revision, trace_json=excluded.trace_json,
                num_events=excluded.num_events WHERE attempts.status='active'""",
                (
                    attempt,
                    owner,
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
                if session_start is not None:
                    self._start_session_game(
                        db, attempt, trace["timestamp"], session_start, config
                    )

    @staticmethod
    def _start_session_game(
        db: sqlite3.Connection,
        attempt: str,
        timestamp: str,
        start: SessionStart,
        config: dict[str, Any],
    ) -> None:
        if start.revision == 0:
            db.execute(
                "INSERT INTO play_sessions VALUES(?,?,?,?,1,?)",
                (
                    start.session_id,
                    start.owner,
                    timestamp,
                    "Play session",
                    json.dumps(start.draft),
                ),
            )
        else:
            changed = db.execute(
                "UPDATE play_sessions SET revision=revision+1,draft_json=? WHERE id=? AND owner=? AND revision=?",
                (
                    json.dumps(start.draft),
                    start.session_id,
                    start.owner,
                    start.revision,
                ),
            ).rowcount
            if changed != 1:
                raise ValueError("Session setup changed; recover before continuing.")
        db.execute(
            "INSERT INTO session_games SELECT ?,?,coalesce(max(game_number),0)+1,? FROM session_games WHERE session_id=?",
            (
                attempt,
                start.session_id,
                json.dumps(
                    {
                        key: config.get(key)
                        for key in (
                            "hero_deck",
                            "villain_deck",
                            "hero_sideboard",
                            "villain_sideboard",
                            "decklists",
                        )
                    },
                    sort_keys=True,
                ),
                start.session_id,
            ),
        )
        db.execute(
            "INSERT INTO continuation_receipts VALUES(?,?,?,?)",
            (start.owner, start.request_id, start.fingerprint, attempt),
        )

    def continuation_receipt(
        self, owner: str, request_id: str, fingerprint: str
    ) -> str | None:
        with self.connect() as db:
            row = db.execute(
                "SELECT fingerprint,attempt_id FROM continuation_receipts WHERE owner=? AND request_id=?",
                (owner, request_id),
            ).fetchone()
        if row is None:
            return None
        if row["fingerprint"] != fingerprint:
            raise ValueError("Request id was already used for different setup.")
        return str(row["attempt_id"])

    def session(self, identifier: str, owners: set[str]) -> PlaySessionRecord:
        with self.connect() as db:
            row = db.execute(
                "SELECT * FROM play_sessions WHERE id=?", (identifier,)
            ).fetchone()
        if row is None or row["owner"] not in owners:
            raise ValueError("Session owner credential required.")
        return PlaySessionRecord(
            id=row["id"],
            created_at=row["created_at"],
            title=row["title"],
            revision=row["revision"],
            draft=json.loads(row["draft_json"]),
        )

    def session_for_attempt(self, attempt: str) -> SessionMembership | None:
        with self.connect() as db:
            row = db.execute(
                "SELECT g.session_id,g.game_number,s.revision FROM session_games g JOIN play_sessions s ON s.id=g.session_id WHERE g.attempt_id=?",
                (attempt,),
            ).fetchone()
        return (
            SessionMembership(
                session_id=row["session_id"],
                game_number=row["game_number"],
                revision=row["revision"],
            )
            if row
            else None
        )

    def latest_session_attempt(self, identifier: str) -> str:
        with self.connect() as db:
            row = db.execute(
                "SELECT attempt_id FROM session_games WHERE session_id=? ORDER BY game_number DESC LIMIT 1",
                (identifier,),
            ).fetchone()
        if row is None:
            raise ValueError("Session has no recorded games.")
        return str(row[0])

    def update_session(
        self, identifier: str, owner: str, revision: int, draft: dict[str, Any]
    ) -> None:
        with self.connect() as db:
            changed = db.execute(
                "UPDATE play_sessions SET draft_json=?,revision=revision+1 WHERE id=? AND owner=? AND revision=?",
                (json.dumps(draft), identifier, owner, revision),
            ).rowcount
            if changed != 1:
                raise ValueError(
                    "Session setup changed or owner credential is missing."
                )

    def session_results(
        self, identifier: str, owners: set[str], *, offset: int = 0, limit: int = 101
    ) -> list[dict[str, Any]]:
        self.session(identifier, owners)
        if offset < 0 or not 1 <= limit <= 101:
            raise ValueError("Invalid session results page.")
        # Each group is an exact execution/matchup/origin/seat, never a rating.
        with self.connect() as db:
            rows = db.execute(
                """
                SELECT a.origin,h.seat AS human_seat,h.deck AS human_deck,
                    b.name AS opponent_name,b.player_id AS opponent_id,
                    b.deck AS opponent_deck,b.version,b.inference_json,
                    json_extract(g.matchup_json,'$.decklists') AS decklists,g.matchup_json,
                    count(*) AS total,
                    sum(a.status='completed') AS completed,
                    coalesce(sum(a.status='completed' AND a.winner=h.seat),0) AS wins,
                    sum(a.status='completed' AND a.winner IS NULL) AS draws,
                    sum(a.status='completed' AND a.winner IS NOT NULL AND a.winner<>h.seat) AS losses,
                    sum(a.status='active') AS active,sum(a.status='stopped') AS stopped,
                    sum(a.status='interrupted') AS interrupted,
                    sum(CASE WHEN a.ended_at IS NOT NULL THEN (julianday(a.ended_at)-julianday(a.started_at))*86400 END) AS wall_elapsed_seconds,
                    sum(a.ended_at IS NOT NULL) AS elapsed_samples
                FROM session_games g JOIN attempts a ON a.id=g.attempt_id
                JOIN attempt_players h ON h.attempt_id=a.id AND h.kind='human'
                JOIN attempt_players b ON b.attempt_id=a.id AND b.kind='bot'
                WHERE g.session_id=?
                GROUP BY a.origin,h.seat,h.deck,b.deck,b.version,b.inference_json,g.matchup_json,b.player_id,b.name
                ORDER BY a.origin,h.seat,b.version,g.matchup_json,h.deck,b.deck,b.inference_json,b.player_id,b.name
                LIMIT ? OFFSET ?
            """,
                (identifier, limit, offset),
            ).fetchall()
        return [dict(row) | {"response_measurements": None} for row in rows]

    def session_totals(self, identifier: str, owners: set[str]) -> list[dict[str, Any]]:
        self.session(identifier, owners)
        with self.connect() as db:
            rows = db.execute(
                """SELECT a.origin, a.status, count(*) AS count
                FROM session_games g JOIN attempts a ON a.id=g.attempt_id
                WHERE g.session_id=? GROUP BY a.origin,a.status ORDER BY a.origin,a.status""",
                (identifier,),
            ).fetchall()
        return [dict(row) for row in rows]

    def arena_recording(self, source_sha256: str) -> str | None:
        """Find one finished import by retained source identity, without reading traces."""
        with self.connect() as db:
            row = db.execute(
                """SELECT p.attempt_id FROM attempt_players AS p
                JOIN attempts AS a ON a.id=p.attempt_id
                WHERE json_extract(p.inference_json, '$.source_sha256')=?
                AND a.status='completed' LIMIT 1""",
                (source_sha256,),
            ).fetchone()
        return None if row is None else str(row[0])

    def owner(self, attempt: str) -> str | None:
        with self.connect() as db:
            row = db.execute(
                "SELECT participant_id FROM attempts WHERE id=?", (attempt,)
            ).fetchone()
        return None if row is None else row[0]

    def load(self, attempt: str) -> dict | None:
        with self.connect() as db:
            row = db.execute("SELECT * FROM attempts WHERE id=?", (attempt,)).fetchone()
        if row is None:
            return None
        trace = json.loads(row["trace_json"])
        if row["status"] == "interrupted":
            trace["end_reason"] = "server_restart"
        return trace

    def metadata(self, attempt: str) -> dict[str, Any]:
        with self.connect() as db:
            row = db.execute(
                "SELECT record_version,origin,revision,status,ending_reason FROM attempts WHERE id=?",
                (attempt,),
            ).fetchone()
            if row is None:
                return {}
            players = self._players(db, attempt)
        return {**dict(row), "players": players}

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

    @staticmethod
    def _mine(owners, token):
        values = sorted(owners)
        clauses = (
            [f"a.participant_id IN ({','.join('?' for _ in values)})"] if values else []
        )
        if token:
            clauses.append(
                "EXISTS(SELECT 1 FROM attempt_players mp JOIN players p ON p.id=mp.player_id WHERE mp.attempt_id=a.id AND p.credential_hash=?)"
            )
            values.append(participant_id(token))
        return "(" + " OR ".join(clauses or ["0"]) + ")", values

    def access(self, attempt, owners, token=None):
        mine, args = self._mine(owners, token)
        with self.connect() as db:
            row = db.execute(
                f"SELECT {mine} AS mine, visibility,status FROM attempts a WHERE id=?",
                [*args, attempt],
            ).fetchone()
        return (
            None
            if row is None
            else {
                "mine": bool(row["mine"]),
                "replay": bool(row["mine"])
                or (row["visibility"] == "shared" and row["status"] == "completed"),
            }
        )

    def summaries(
        self,
        owners: set[str],
        *,
        token: str | None = None,
        session: str | None = None,
        scope: str = "mine",
        q: str = "",
        player: str | None = None,
        other: str | None = None,
        pairing: str | None = None,
        version: str | None = None,
        deck: str | None = None,
        status: str | None = None,
        result: str | None = None,
        after: str | None = None,
        before: str | None = None,
        cursor: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        if scope not in {"all", "mine"} or status not in {
            None,
            "active",
            "completed",
            "stopped",
            "interrupted",
        }:
            raise ValueError("Invalid history scope or status.")
        if pairing not in {
            None,
            "human-human",
            "human-bot",
            "bot-bot",
        } or result not in {None, "win", "loss", "draw"}:
            raise ValueError("Invalid pairing or result.")
        if result in {"win", "loss"} and not player:
            raise ValueError("Choose a player to filter wins or losses.")
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
        mine, mine_args = self._mine(owners, token)
        if session is not None:
            self.session(session, owners)
        clauses = [mine if scope == "mine" else f"(a.visibility='shared' OR {mine})"]
        args = list(mine_args)
        if session is not None:
            clauses.append(
                "a.id IN (SELECT attempt_id FROM session_games WHERE session_id=?)"
            )
            args.append(session)
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
        if player:
            clauses.append(
                "EXISTS(SELECT 1 FROM attempt_players p WHERE p.attempt_id=a.id AND p.player_id=?"
                + (
                    " AND EXISTS(SELECT 1 FROM attempt_players o WHERE o.attempt_id=a.id AND o.seat<>p.seat AND o.player_id=?)"
                    if other
                    else ""
                )
                + ")"
            )
            args.extend([player, other] if other else [player])
        elif other:
            clauses.append(
                "EXISTS(SELECT 1 FROM attempt_players p WHERE p.attempt_id=a.id AND p.player_id=?)"
            )
            args.append(other)
        for column, value in (("version", version), ("deck", deck)):
            if value:
                clauses.append(
                    f"EXISTS(SELECT 1 FROM attempt_players s WHERE s.attempt_id=a.id AND s.{column}=?)"
                )
                args.append(value)
        if pairing:
            clauses.append(
                "(SELECT count(*) FROM attempt_players s WHERE s.attempt_id=a.id AND s.kind='human')=?"
            )
            args.append(pairing.count("human"))
        if status:
            clauses.append("a.status=?")
            args.append(status)
        if result == "draw":
            clauses.append("a.status='completed' AND a.winner IS NULL")
        elif result:
            clauses.append(
                "a.status='completed' AND a.winner IS NOT NULL AND EXISTS(SELECT 1 FROM attempt_players p WHERE p.attempt_id=a.id AND p.player_id=? AND p.seat "
                + ("=" if result == "win" else "<>")
                + " a.winner)"
            )
            args.append(player)
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
                f"SELECT a.id,a.started_at,a.ended_at,a.winner,a.status,a.ending_reason,a.num_events,a.revision,a.record_version,a.origin,a.visibility,g.session_id,g.game_number, {mine} AS mine FROM attempts a LEFT JOIN session_games g ON g.attempt_id=a.id WHERE {' AND '.join(clauses)} ORDER BY started_at DESC,id DESC LIMIT ?",
                [*mine_args, *args, limit],
            ).fetchall()
            players_by_attempt = self._players_for(db, [row["id"] for row in rows])
            own_attempts = [row["id"] for row in rows if row["mine"]]
            notes_by_attempt: dict[str, list[dict[str, Any]]] = {}
            if own_attempts:
                placeholders = ",".join("?" for _ in own_attempts)
                # Each selected attempt reads only its latest 101 notes through
                # the date index. The extra row makes the visible cap explicit.
                note_rows = db.execute(
                    f"""SELECT a.id, (SELECT json_group_array(json_object(
                        'created_at',created_at,'note',note,'author_id',author_id,'decision_address',decision_address))
                        FROM (SELECT created_at,note,author_id,decision_address FROM feedback
                            WHERE attempt_id=a.id ORDER BY created_at DESC,request_id DESC LIMIT 101)) AS notes
                        FROM attempts a WHERE a.id IN ({placeholders})""",
                    own_attempts,
                )
                for note_row in note_rows:
                    notes_by_attempt[note_row["id"]] = list(
                        reversed(json.loads(note_row["notes"]))
                    )
            summaries = []
            for row in rows:
                seats = players_by_attempt.get(row["id"], [])
                notes = notes_by_attempt.get(row["id"], [])
                summaries.append(
                    {
                        "id": row["id"],
                        "timestamp": row["started_at"],
                        "ended_at": row["ended_at"],
                        "winner": row["winner"],
                        "status": row["status"],
                        "end_reason": row["ending_reason"] or "active",
                        "num_events": row["num_events"],
                        "revision": row["revision"],
                        "players": seats,
                        "play_session_id": row["session_id"] if row["mine"] else None,
                        "game_number": row["game_number"] if row["mine"] else None,
                        "record_version": row["record_version"],
                        "origin": row["origin"],
                        "mine": bool(row["mine"]),
                        "replay_available": bool(row["mine"])
                        or (
                            row["visibility"] == "shared"
                            and row["status"] == "completed"
                        ),
                        "feedback": notes[-100:],
                        "feedback_truncated": len(notes) > 100,
                    }
                )
        return summaries

    def feedback(
        self,
        attempt: str,
        request_id: str,
        note: str,
        *,
        author_id=None,
        decision_address=None,
    ) -> None:
        if (
            not request_id
            or len(request_id) > 128
            or not note.strip()
            or len(note) > 4000
        ):
            raise ValueError(
                "Feedback needs a request id and a note of 1–4000 characters."
            )
        with self.connect() as db:
            existing = db.execute(
                "SELECT note,author_id,decision_address FROM feedback WHERE attempt_id=? AND request_id=?",
                (attempt, request_id),
            ).fetchone()
            if existing is not None and tuple(existing) != (
                note,
                author_id,
                decision_address,
            ):
                raise ValueError(
                    "Feedback request id was already used for different feedback."
                )
            db.execute(
                "INSERT OR IGNORE INTO feedback VALUES(?,?,?,?,?,?)",
                (
                    attempt,
                    request_id,
                    datetime.now(timezone.utc).isoformat(),
                    note,
                    author_id,
                    decision_address,
                ),
            )
