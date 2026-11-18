"""Etude tables describing the games and decision settings behind a learner."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

from etude.train.tracking import TrainingRun

from manabot.model.writer_catalog import writer_catalog
from manabot.training.decks import deck_cards
from manabot.training.update_metrics import DECK_TABLE, OUTCOME_TABLE, SETTINGS_TABLE

if TYPE_CHECKING:
    from manabot.training.rollout.collection import Collector


class ParentCheckpoint(TypedDict):
    run: str | None
    update: int | None
    sha256: str


def log_corpus_setup(
    run: TrainingRun,
    collector: Collector,
    *,
    behavior: str,
    curriculum: str,
    disclosure: str,
    parent: ParentCheckpoint | None = None,
    **clocks: float,
) -> None:
    players = collector.uniform_roster or collector.match.to_rust()
    run.log_table(
        DECK_TABLE, deck_cards((p.deck for p in players), writer_catalog()), **clocks
    )
    observation = collector.observation_space.encoder.config
    run.log_table(
        SETTINGS_TABLE,
        [
            {
                "decklists": disclosure,
                "mana_taps": observation.model_dump().get("mana_taps", False),
                "max_actions": observation.max_actions,
                "action_overflow": observation.action_overflow,
                "policy_input_version": observation.policy_input_version,
                "behavior": behavior,
                "opponent_mode": collector.opponent_mode,
                "learning_seats": collector.learning_seats,
                "matchup_curriculum": curriculum,
                "parent_run": parent["run"] if parent else None,
                "parent_update": parent["update"] if parent else None,
                "parent_sha256": parent["sha256"] if parent else None,
            }
        ],
        **clocks,
    )


def log_corpus_outcomes(
    run: TrainingRun, collector: Collector, **clocks: float
) -> None:
    if collector.opponent_mode in ("self", "learning"):
        run.log_table(OUTCOME_TABLE, collector.stats.self_play.drain(), **clocks)
