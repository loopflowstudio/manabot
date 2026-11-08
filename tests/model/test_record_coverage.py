"""The catalog record vocabulary covers what real games emit (bounded sweep).

`uv run python -m manabot.verify.record_coverage --games 200` reruns the full
sweep before long catalog training.
"""

import pytest
import torch

from manabot.verify import record_coverage
import managym

# Enough games to deal every eligible card once (two decks of 12 spells each).
_GAMES = 14


def test_every_id_random_games_emit_has_a_catalog_row() -> None:
    torch.set_num_threads(1)
    report = record_coverage.coverage_sweep(_GAMES, seed=0)
    assert report.cards_played == report.eligible_cards
    assert report.fact_rows > 0 and report.effect_rows > 0


@pytest.mark.parametrize(
    "missing_prefix", ["Modification::", "execution/6/effect execution"]
)
def test_a_name_missing_from_the_vocabulary_fails_the_sweep(
    monkeypatch: pytest.MonkeyPatch,
    missing_prefix: str,
) -> None:
    torch.set_num_threads(1)
    names = managym.learning_record_names()
    monkeypatch.setattr(
        managym,
        "learning_record_names",
        lambda: [n for n in names if not n[1].startswith(missing_prefix)],
    )
    with pytest.raises(ValueError, match="outside the catalog record vocabulary"):
        record_coverage.coverage_sweep(_GAMES, seed=0)


def test_elemental_execution_ids_have_catalog_rows() -> None:
    torch.set_num_threads(1)
    report = record_coverage.coverage_sweep(4, seed=1000, decks="elemental")
    assert report.decisions > 0 and report.fact_rows > 0
