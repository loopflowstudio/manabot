"""
metrics.py
DuckDB-backed metrics store for sim/eval data.

All sim runs (training eval, verify steps, standalone sim) write to the same
database. Each run gets a sim_id; games and actions reference it.

Usage:
    db = MetricsDB()  # opens .runs/metrics.duckdb
    sim_id = db.start_sim(model_name="step4", model_step=5000000, opponent="random")
    db.log_game(sim_id, game_index=0, outcome="hero_win", steps=42, ...)
    db.log_action(sim_id, game_index=0, step=3, player=0, action_type="attack", ...)
    db.finish_sim(sim_id, win_rate=0.85, attack_rate=0.4, mean_steps=35.0)

    # Query:
    db.query("SELECT * FROM sims WHERE opponent = 'random'")
"""

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb


def _default_db_path() -> Path:
    return Path(os.getenv("MANABOT_RUNS_DIR", str(Path.cwd() / ".runs"))) / "metrics.duckdb"


class MetricsDB:
    def __init__(self, path: Path | str | None = None, read_only: bool = False):
        self.path = Path(path) if path else _default_db_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.con = duckdb.connect(str(self.path), read_only=read_only)
        if not read_only:
            self._create_tables()

    def _create_tables(self):
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS sims (
                sim_id TEXT PRIMARY KEY,
                created_at TIMESTAMP,
                finished_at TIMESTAMP,
                model_name TEXT,
                model_step INTEGER,
                opponent TEXT,
                num_games INTEGER,
                win_rate DOUBLE,
                attack_rate DOUBLE,
                mean_steps DOUBLE,
                config TEXT
            )
        """)
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS games (
                sim_id TEXT,
                game_index INTEGER,
                outcome TEXT,
                steps INTEGER,
                duration DOUBLE,
                hero_attacks INTEGER,
                hero_actions INTEGER,
                attack_rate DOUBLE,
                could_attack INTEGER,
                attacked_when_able DOUBLE,
                PRIMARY KEY (sim_id, game_index)
            )
        """)
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS actions (
                sim_id TEXT,
                game_index INTEGER,
                step INTEGER,
                player INTEGER,
                action_type TEXT,
                choice_set TEXT,
                is_trivial BOOLEAN,
                is_attack BOOLEAN,
                attack_available BOOLEAN,
                pass_available BOOLEAN,
                land_available BOOLEAN,
                spell_available BOOLEAN,
                pass_prob DOUBLE,
                land_prob DOUBLE,
                spell_prob DOUBLE,
                attack_prob DOUBLE,
                num_valid_actions INTEGER
            )
        """)
        self._ensure_column("actions", "choice_set", "TEXT")
        self._ensure_column("actions", "is_trivial", "BOOLEAN")
        self._ensure_column("actions", "pass_available", "BOOLEAN")
        self._ensure_column("actions", "pass_prob", "DOUBLE")
        self._ensure_column("actions", "land_prob", "DOUBLE")
        self._ensure_column("actions", "spell_prob", "DOUBLE")
        self._ensure_column("actions", "attack_prob", "DOUBLE")

    def _ensure_column(self, table: str, column: str, definition: str) -> None:
        existing = {
            row[1]
            for row in self.con.execute(f"PRAGMA table_info('{table}')").fetchall()
        }
        if column not in existing:
            self.con.execute(
                f"ALTER TABLE {table} ADD COLUMN {column} {definition}"
            )

    def start_sim(
        self,
        model_name: str = "",
        model_step: int = 0,
        opponent: str = "",
        num_games: int = 0,
        config: dict[str, Any] | None = None,
    ) -> str:
        sim_id = uuid.uuid4().hex[:12]
        self.con.execute(
            """INSERT INTO sims (sim_id, created_at, model_name, model_step,
               opponent, num_games, config)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            [
                sim_id,
                datetime.now(timezone.utc),
                model_name,
                model_step,
                opponent,
                num_games,
                json.dumps(config) if config else None,
            ],
        )
        return sim_id

    def log_game(
        self,
        sim_id: str,
        game_index: int,
        outcome: str,
        steps: int,
        duration: float = 0.0,
        hero_attacks: int = 0,
        hero_actions: int = 0,
        could_attack: int = 0,
    ):
        attack_rate = hero_attacks / hero_actions if hero_actions > 0 else 0.0
        attacked_when_able = hero_attacks / could_attack if could_attack > 0 else 0.0
        self.con.execute(
            """INSERT INTO games (sim_id, game_index, outcome, steps, duration,
               hero_attacks, hero_actions, attack_rate, could_attack, attacked_when_able)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                sim_id, game_index, outcome, steps, duration,
                hero_attacks, hero_actions, attack_rate,
                could_attack, attacked_when_able,
            ],
        )

    def log_action(
        self,
        sim_id: str,
        game_index: int,
        step: int,
        player: int,
        action_type: str,
        choice_set: str = "",
        is_trivial: bool = False,
        is_attack: bool = False,
        attack_available: bool = False,
        pass_available: bool = False,
        land_available: bool = False,
        spell_available: bool = False,
        pass_prob: float | None = None,
        land_prob: float | None = None,
        spell_prob: float | None = None,
        attack_prob: float | None = None,
        num_valid_actions: int = 0,
    ):
        self.con.execute(
            """INSERT INTO actions (sim_id, game_index, step, player,
               action_type, choice_set, is_trivial, is_attack, attack_available,
               pass_available, land_available, spell_available, pass_prob,
               land_prob, spell_prob, attack_prob, num_valid_actions)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                sim_id,
                game_index,
                step,
                player,
                action_type,
                choice_set,
                is_trivial,
                is_attack,
                attack_available,
                pass_available,
                land_available,
                spell_available,
                pass_prob,
                land_prob,
                spell_prob,
                attack_prob,
                num_valid_actions,
            ],
        )

    def finish_sim(
        self,
        sim_id: str,
        win_rate: float = 0.0,
        attack_rate: float = 0.0,
        mean_steps: float = 0.0,
    ):
        self.con.execute(
            """UPDATE sims SET finished_at = ?, win_rate = ?, attack_rate = ?,
               mean_steps = ? WHERE sim_id = ?""",
            [datetime.now(timezone.utc), win_rate, attack_rate, mean_steps, sim_id],
        )

    def query(self, sql: str, params: list | None = None) -> list[dict]:
        result = self.con.execute(sql, params or [])
        columns = [desc[0] for desc in result.description]
        return [dict(zip(columns, row)) for row in result.fetchall()]

    def close(self):
        self.con.close()
