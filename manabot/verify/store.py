"""SQLite-backed storage for first-light verification runs."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sqlite3
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from manabot.experiment.execution import ExperimentRun
    from manabot.training.records import TrainingRun

RUN_CONFIG_FIELDS = (
    "experiment",
    "train",
    "reward",
    "agent",
    "match",
    "observation",
)

EVALUATION_METRIC_FIELDS = (
    "win_rate",
    "win_ci_lower",
    "mean_steps",
    "attack_rate",
    "attacked_when_able",
    "landed_when_able",
    "cast_when_able",
    "passed_when_able",
    "could_attack",
    "could_land",
    "could_spell",
    "could_pass",
    "land_plays",
    "spell_casts",
    "pass_count",
    "single_valid_decisions",
    "multi_valid_decisions",
    "pass_land_decisions",
    "pass_land_pass_rate",
    "pass_land_land_rate",
    "pass_land_spell_rate",
    "mean_pass_prob",
    "mean_land_prob",
    "mean_spell_prob",
    "mean_attack_prob",
    "mean_pass_prob_when_land_available",
    "mean_land_prob_when_land_available",
    "mean_pass_prob_when_pass_land",
    "mean_land_prob_when_pass_land",
    "action_space_truncations",
    "card_space_truncations",
    "permanent_space_truncations",
)

EVALUATION_ACTION_FIELDS = (
    "game_index",
    "step",
    "player",
    "action_type",
    "choice_set",
    "is_trivial",
    "num_valid_actions",
    "attack_available",
    "land_available",
    "spell_available",
    "pass_available",
    "pass_prob",
    "land_prob",
    "spell_prob",
    "attack_prob",
)

BOOLEAN_ACTION_FIELDS = {
    "is_trivial",
    "attack_available",
    "land_available",
    "spell_available",
    "pass_available",
}


def _default_db_path() -> Path:
    runs_dir_env = os.getenv("MANABOT_RUNS_DIR")
    runs_dir = Path(runs_dir_env) if runs_dir_env else Path.cwd() / ".runs"
    return runs_dir / "verify.sqlite"


def training_stage_payloads(con: sqlite3.Connection, run_id: str) -> list[dict]:
    """Return one run's stage records in execution order.

    Self-play update rows are Etude metrics, so a stage payload stays bounded
    and is rewritten whole on every save.
    """
    return [
        json.loads(payload)
        for (payload,) in con.execute(
            "SELECT payload FROM training_stages WHERE run_id=? ORDER BY rowid",
            (run_id,),
        )
    ]


def _clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _clean_json(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_clean_json(item) for item in value]
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return None
    if isinstance(value, Path):
        return str(value)
    return value


class VerifyStore:
    """Simple SQLite store for experiment runs, evals, and markdown reports."""

    def __init__(
        self, path: Path | str | None = None, *, read_only: bool = False
    ) -> None:
        """Read-only admission skips schema creation and journal changes."""
        self.path = Path(path) if path else _default_db_path()
        if read_only:
            self.con = sqlite3.connect(
                f"{self.path.absolute().as_uri()}?mode=ro", uri=True
            )
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.con = sqlite3.connect(self.path)
        self.con.row_factory = sqlite3.Row
        self.con.execute("PRAGMA foreign_keys=ON")
        if not read_only:
            self.con.execute("PRAGMA journal_mode=WAL")
            self._create_schema()

    def __enter__(self) -> VerifyStore:
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        self.close()

    def close(self) -> None:
        self.con.close()

    def save_experiment_run(self, run: ExperimentRun) -> None:
        """Persist actual execution separately from the declarative Experiment."""
        with self.con:
            self.con.execute(
                "INSERT INTO experiment_runs VALUES (?, ?) ON CONFLICT(id) DO UPDATE SET payload=excluded.payload",
                (run.id, run.model_dump_json()),
            )

    def experiment_run(self, run_id: str) -> ExperimentRun:
        from manabot.experiment.execution import ExperimentRun

        row = self.con.execute(
            "SELECT payload FROM experiment_runs WHERE id=?", (run_id,)
        ).fetchone()
        if row is None:
            raise KeyError(run_id)
        return ExperimentRun.model_validate_json(row[0])

    def save_training_run(self, run: TrainingRun) -> None:
        """Commit one canonical regime execution and its stage records atomically."""
        with self.con:
            self.con.execute(
                "INSERT INTO training_runs VALUES (?, ?) ON CONFLICT(id) DO UPDATE SET payload=excluded.payload",
                (run.id, run.model_dump_json(exclude={"stages"})),
            )
            for stage in run.stages:
                self.con.execute(
                    "INSERT INTO training_stages VALUES (?, ?, ?) ON CONFLICT(run_id, stage_id) DO UPDATE SET payload=excluded.payload",
                    (run.id, stage.id, stage.model_dump_json()),
                )

    def claim_training_recovery(self, parent_id: str, run: TrainingRun) -> None:
        """Atomically allow one continuation per stopped attempt, never a fork."""
        with self.con:
            self.con.execute("BEGIN IMMEDIATE")
            if self.con.execute(
                "SELECT 1 FROM training_runs WHERE json_extract(payload, '$.parent_run_id') = ?",
                (parent_id,),
            ).fetchone():
                raise ValueError(
                    "attempt already has a recovery child; continue that child"
                )
            parent = self.training_run(parent_id)
            if parent.status not in {"failed", "interrupted", "paused"}:
                raise ValueError("recovery parent is not stopped")
            self.con.execute(
                "INSERT INTO training_runs VALUES (?, ?)",
                (run.id, run.model_dump_json(exclude={"stages"})),
            )

    def training_run(self, run_id: str) -> TrainingRun:
        from manabot.training.records import TrainingRun

        row = self.con.execute(
            "SELECT payload FROM training_runs WHERE id=?", (run_id,)
        ).fetchone()
        if row is None:
            raise KeyError(run_id)
        payload = json.loads(row[0])
        payload["stages"] = training_stage_payloads(self.con, run_id)
        return TrainingRun.model_validate(payload)

    def _create_schema(self) -> None:
        self.con.executescript(
            """
            CREATE TABLE IF NOT EXISTS experiment_runs (
                id TEXT PRIMARY KEY,
                payload TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS training_runs (
                id TEXT PRIMARY KEY,
                payload TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS training_stages (
                run_id TEXT NOT NULL REFERENCES training_runs(id),
                stage_id TEXT NOT NULL,
                payload TEXT NOT NULL,
                PRIMARY KEY (run_id, stage_id)
            );
            CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY,
                created_at TEXT NOT NULL,
                label TEXT,
                status TEXT NOT NULL,
                mode TEXT NOT NULL,
                seed INTEGER NOT NULL,
                git_commit TEXT,
                git_branch TEXT,
                tree_dirty INTEGER NOT NULL,
                recipe_name TEXT NOT NULL,
                opponent_policy TEXT NOT NULL,
                num_envs INTEGER NOT NULL,
                num_steps INTEGER NOT NULL,
                total_timesteps INTEGER NOT NULL,
                eval_interval INTEGER,
                eval_num_games INTEGER NOT NULL,
                baseline_auto INTEGER NOT NULL,
                notes TEXT
            );

            CREATE TABLE IF NOT EXISTS run_configs (
                run_id INTEGER PRIMARY KEY REFERENCES runs(id) ON DELETE CASCADE,
                experiment_json TEXT NOT NULL,
                train_json TEXT NOT NULL,
                reward_json TEXT NOT NULL,
                agent_json TEXT NOT NULL,
                match_json TEXT NOT NULL,
                observation_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS evaluations (
                id INTEGER PRIMARY KEY,
                run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
                created_at TEXT NOT NULL,
                kind TEXT NOT NULL,
                step INTEGER NOT NULL,
                update_index INTEGER,
                opponent_policy TEXT NOT NULL,
                num_games INTEGER NOT NULL,
                deterministic INTEGER NOT NULL,
                win_rate REAL NOT NULL,
                win_ci_lower REAL NOT NULL,
                mean_steps REAL NOT NULL,
                attack_rate REAL NOT NULL,
                attacked_when_able REAL NOT NULL,
                landed_when_able REAL NOT NULL,
                cast_when_able REAL NOT NULL,
                passed_when_able REAL NOT NULL,
                could_attack REAL NOT NULL,
                could_land REAL NOT NULL,
                could_spell REAL NOT NULL,
                could_pass REAL NOT NULL,
                land_plays REAL NOT NULL,
                spell_casts REAL NOT NULL,
                pass_count REAL NOT NULL,
                single_valid_decisions REAL NOT NULL,
                multi_valid_decisions REAL NOT NULL,
                pass_land_decisions REAL NOT NULL,
                pass_land_pass_rate REAL NOT NULL,
                pass_land_land_rate REAL NOT NULL,
                pass_land_spell_rate REAL NOT NULL,
                mean_pass_prob REAL,
                mean_land_prob REAL,
                mean_spell_prob REAL,
                mean_attack_prob REAL,
                mean_pass_prob_when_land_available REAL,
                mean_land_prob_when_land_available REAL,
                mean_pass_prob_when_pass_land REAL,
                mean_land_prob_when_pass_land REAL,
                action_space_truncations REAL NOT NULL,
                card_space_truncations REAL NOT NULL,
                permanent_space_truncations REAL NOT NULL,
                explained_variance REAL
            );

            CREATE TABLE IF NOT EXISTS evaluation_choice_sets (
                evaluation_id INTEGER NOT NULL REFERENCES evaluations(id) ON DELETE CASCADE,
                scope TEXT NOT NULL,
                choice_set TEXT NOT NULL,
                count REAL NOT NULL,
                PRIMARY KEY (evaluation_id, scope, choice_set)
            );

            CREATE TABLE IF NOT EXISTS evaluation_actions (
                id INTEGER PRIMARY KEY,
                evaluation_id INTEGER NOT NULL REFERENCES evaluations(id) ON DELETE CASCADE,
                game_index INTEGER NOT NULL,
                step INTEGER NOT NULL,
                player INTEGER NOT NULL,
                action_type TEXT NOT NULL,
                choice_set TEXT NOT NULL,
                is_trivial INTEGER NOT NULL,
                num_valid_actions INTEGER NOT NULL,
                attack_available INTEGER NOT NULL,
                land_available INTEGER NOT NULL,
                spell_available INTEGER NOT NULL,
                pass_available INTEGER NOT NULL,
                pass_prob REAL,
                land_prob REAL,
                spell_prob REAL,
                attack_prob REAL
            );

            CREATE TABLE IF NOT EXISTS reports (
                id INTEGER PRIMARY KEY,
                run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
                created_at TEXT NOT NULL,
                report_kind TEXT NOT NULL,
                markdown_path TEXT,
                recommendation TEXT NOT NULL,
                summary_json TEXT
            );
            """
        )
        self.con.commit()
