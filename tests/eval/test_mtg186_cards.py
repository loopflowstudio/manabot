"""MTG-186's card job: one deal's four UR/GW seatings deliver games and card data.

Untrained fixture checkpoints stand in for the two models; no trained model
plays here.
"""

from pathlib import Path

from etude.eval.match import evaluate
from etude.eval.query import statsdb
from etude.run.declaration import declare
import pytest
import torch

from experiments import mtg186_cards as cards
from experiments.mtg186_eval import GW, UR, Checkpoint, Pairing, Player
from manabot.env.match import Match as EnvMatch
from manabot.env.observation import ObservationSpace
from manabot.eval.card_query import CardSelection, card_data
from manabot.infra.config import ModelConfig
from manabot.infra.identity import file_sha256
from manabot.model.actor_critic import ActorCritic
from manabot.model.checkpoint import save_checkpoint


def _checkpoint(path: Path, seed: int, trial: str) -> Player:
    torch.manual_seed(seed)
    space = ObservationSpace()
    model = ActorCritic(
        space,
        ModelConfig(
            hidden_dim=8, num_attention_heads=2, semantic_pack="elemental-world-v2"
        ),
    )
    save_checkpoint(
        model,
        space,
        path,
        player_configs=EnvMatch(cards.matchup().match()).to_rust(),
    )
    checkpoint = Checkpoint(
        trial=trial,
        seed=seed,
        job="fixture",
        updates=0,
        sha256=file_sha256(path),
        size_bytes=path.stat().st_size,
        key=f"fixtures/{trial}.pt",
    )
    return Player(checkpoint=checkpoint, path=path)


def test_schedule_is_pinned_and_fresh() -> None:
    assert cards.schedule_sha256() == cards.SCHEDULE_SHA256
    deals = cards.deals()
    assert len(set(deals)) == cards.DEALS == 500
    # Disjoint from the pairings' UR/GW and rest-block deals.
    assert min(deals) > 1_869_100_000 + 500


def test_one_match_delivers_games_and_both_card_tables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    a = _checkpoint(tmp_path / "a.pt", 1, "mtg186")
    b = _checkpoint(tmp_path / "b.pt", 2, "mtg181")
    pairing = Pairing(name="endpoint", mtg186=a.checkpoint, mtg181=b.checkpoint)
    [match] = [
        m.model_copy(update={"conditions": {"deals": [cards.DEAL_BASE]}, "games": 4})
        for m in cards.matches(pairing)[:1]
    ]
    directory = tmp_path / "eval"
    root = declare(
        experiment=cards.EXPERIMENT,
        clocks=(),
        run=cards.RUN,
        kind="evaluation",
        directory=directory,
    ).root
    machine = cards.Machine(a=a, b=b, root=root.resolve(), work=tmp_path / "work")
    monkeypatch.setenv("MTG186_CARDS", machine.model_dump_json())
    evaluate(
        experiment=cards.EXPERIMENT,
        environment=cards.environment(),
        matches=[match],
        play=cards.play,
        run=cards.RUN,
        directory=directory,
    )
    with statsdb(cards.EXPERIMENT, directory) as db:
        games = db.execute(
            "select deck, opponent_deck, seat, valid from games order by deck, seat"
        ).fetchall()
        assert [(deck, seat) for deck, _, seat, _ in games] == [
            (GW, 0),
            (GW, 1),
            (UR, 0),
            (UR, 1),
        ]
        assert all(valid for *_, valid in games)
        for subject in ("mtg186", "mtg181"):
            data = card_data(
                db, CardSelection(experiment=cards.EXPERIMENT, subjects=(subject,))
            )
            assert data.cards
            assert data.games + sum(data.excluded.values()) == 4
