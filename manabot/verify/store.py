"""SQLite-backed storage for first-light verification runs."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sqlite3
from typing import Any

from .util import EvaluationArtifacts

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
    runs_dir = Path(os.getenv("MANABOT_RUNS_DIR", str(Path.cwd() / ".runs")))
    return runs_dir / "verify.sqlite"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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

    def __init__(self, path: Path | str | None = None):
        self.path = Path(path) if path else _default_db_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(self.path)
        self.con.row_factory = sqlite3.Row
        self.con.execute("PRAGMA journal_mode=WAL")
        self.con.execute("PRAGMA foreign_keys=ON")
        self._create_schema()

    def __enter__(self) -> VerifyStore:
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        self.close()

    def close(self) -> None:
        self.con.close()

    def _create_schema(self) -> None:
        self.con.executescript(
            """
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

    @staticmethod
    def _json_blob(value: Any) -> str:
        return json.dumps(_clean_json(value), sort_keys=True)

    @staticmethod
    def _row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        return {key: row[key] for key in row.keys()}

    def create_run(
        self,
        *,
        label: str | None,
        status: str,
        mode: str,
        seed: int,
        git_commit: str | None,
        git_branch: str | None,
        tree_dirty: bool,
        recipe_name: str,
        opponent_policy: str,
        num_envs: int,
        num_steps: int,
        total_timesteps: int,
        eval_interval: int | None,
        eval_num_games: int,
        baseline_auto: bool,
        notes: str | None,
        configs: dict[str, Any],
    ) -> int:
        cursor = self.con.execute(
            """
            INSERT INTO runs (
                created_at, label, status, mode, seed, git_commit, git_branch,
                tree_dirty, recipe_name, opponent_policy, num_envs, num_steps,
                total_timesteps, eval_interval, eval_num_games, baseline_auto, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                _utc_now(),
                label,
                status,
                mode,
                seed,
                git_commit,
                git_branch,
                int(tree_dirty),
                recipe_name,
                opponent_policy,
                num_envs,
                num_steps,
                total_timesteps,
                eval_interval,
                eval_num_games,
                int(baseline_auto),
                notes,
            ),
        )
        run_id = int(cursor.lastrowid)
        self.con.execute(
            """
            INSERT INTO run_configs (
                run_id, experiment_json, train_json, reward_json,
                agent_json, match_json, observation_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (run_id, *(self._json_blob(configs[field]) for field in RUN_CONFIG_FIELDS)),
        )
        self.con.commit()
        return run_id

    def update_run_status(
        self, run_id: int, status: str, notes: str | None = None
    ) -> None:
        self.con.execute(
            "UPDATE runs SET status = ?, notes = COALESCE(?, notes) WHERE id = ?",
            (status, notes, run_id),
        )
        self.con.commit()

    def record_evaluation(
        self,
        *,
        run_id: int,
        kind: str,
        step: int,
        update_index: int | None,
        opponent_policy: str,
        num_games: int,
        deterministic: bool,
        artifacts: EvaluationArtifacts,
        explained_variance: float | None = None,
        retain_actions: bool = False,
    ) -> int:
        metrics = artifacts.metrics
        metric_values = tuple(metrics[field] for field in EVALUATION_METRIC_FIELDS)
        cursor = self.con.execute(
            """
            INSERT INTO evaluations (
                run_id, created_at, kind, step, update_index, opponent_policy,
                num_games, deterministic, win_rate, win_ci_lower, mean_steps,
                attack_rate, attacked_when_able, landed_when_able, cast_when_able,
                passed_when_able, could_attack, could_land, could_spell, could_pass,
                land_plays, spell_casts, pass_count, single_valid_decisions,
                multi_valid_decisions, pass_land_decisions, pass_land_pass_rate,
                pass_land_land_rate, pass_land_spell_rate, mean_pass_prob,
                mean_land_prob, mean_spell_prob, mean_attack_prob,
                mean_pass_prob_when_land_available, mean_land_prob_when_land_available,
                mean_pass_prob_when_pass_land, mean_land_prob_when_pass_land,
                action_space_truncations, card_space_truncations,
                permanent_space_truncations, explained_variance
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                _utc_now(),
                kind,
                step,
                update_index,
                opponent_policy,
                num_games,
                int(deterministic),
                *metric_values,
                explained_variance,
            ),
        )
        evaluation_id = int(cursor.lastrowid)

        choice_rows = []
        for scope, counts in artifacts.choice_sets.items():
            for choice_set, count in counts.items():
                choice_rows.append((evaluation_id, scope, choice_set, count))
        if choice_rows:
            self.con.executemany(
                """
                INSERT INTO evaluation_choice_sets (
                    evaluation_id, scope, choice_set, count
                ) VALUES (?, ?, ?, ?)
                """,
                choice_rows,
            )

        if retain_actions and artifacts.actions:
            self.con.executemany(
                """
                INSERT INTO evaluation_actions (
                    evaluation_id, game_index, step, player, action_type, choice_set,
                    is_trivial, num_valid_actions, attack_available, land_available,
                    spell_available, pass_available, pass_prob, land_prob, spell_prob,
                    attack_prob
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        evaluation_id,
                        *(
                            int(getattr(action, field))
                            if field in BOOLEAN_ACTION_FIELDS
                            else getattr(action, field)
                            for field in EVALUATION_ACTION_FIELDS
                        ),
                    )
                    for action in artifacts.actions
                ],
            )

        self.con.commit()
        return evaluation_id

    def record_report(
        self,
        *,
        run_id: int,
        report_kind: str,
        markdown_path: str | None,
        recommendation: str,
        summary: dict[str, Any],
    ) -> int:
        cursor = self.con.execute(
            """
            INSERT INTO reports (
                run_id, created_at, report_kind, markdown_path, recommendation, summary_json
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                _utc_now(),
                report_kind,
                markdown_path,
                recommendation,
                self._json_blob(summary),
            ),
        )
        self.con.commit()
        return int(cursor.lastrowid)

    def get_run(self, run_id: int) -> dict[str, Any]:
        row = self.con.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        result = self._row_to_dict(row)
        if result is None:
            raise KeyError(f"Unknown run_id={run_id}")
        return result

    def get_run_configs(self, run_id: int) -> dict[str, Any]:
        row = self.con.execute(
            "SELECT * FROM run_configs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        result = self._row_to_dict(row)
        if result is None:
            raise KeyError(f"Unknown run_id={run_id}")
        return {
            field: json.loads(result[f"{field}_json"]) for field in RUN_CONFIG_FIELDS
        }

    def get_evaluations(self, run_id: int) -> list[dict[str, Any]]:
        rows = self.con.execute(
            "SELECT * FROM evaluations WHERE run_id = ? ORDER BY step, id",
            (run_id,),
        ).fetchall()
        return [self._row_to_dict(row) for row in rows if row is not None]

    def get_evaluation_choice_sets(
        self, evaluation_id: int
    ) -> dict[str, dict[str, float]]:
        rows = self.con.execute(
            """
            SELECT scope, choice_set, count
            FROM evaluation_choice_sets
            WHERE evaluation_id = ?
            ORDER BY scope, choice_set
            """,
            (evaluation_id,),
        ).fetchall()
        grouped: dict[str, dict[str, float]] = {}
        for row in rows:
            scope = row["scope"]
            grouped.setdefault(scope, {})[row["choice_set"]] = float(row["count"])
        return grouped

    def get_evaluation_actions(self, evaluation_id: int) -> list[dict[str, Any]]:
        rows = self.con.execute(
            """
            SELECT *
            FROM evaluation_actions
            WHERE evaluation_id = ?
            ORDER BY game_index, step, id
            """,
            (evaluation_id,),
        ).fetchall()
        return [self._row_to_dict(row) for row in rows if row is not None]

    def get_reports(self, run_id: int) -> list[dict[str, Any]]:
        rows = self.con.execute(
            "SELECT * FROM reports WHERE run_id = ? ORDER BY id",
            (run_id,),
        ).fetchall()
        return [self._row_to_dict(row) for row in rows if row is not None]
