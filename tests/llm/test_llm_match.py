"""An LLM game's schedule and Etude rows, without a model or a game."""

from collections import Counter

from manabot.verify.llm_game import GameSummary
from manabot.verify.llm_match import Game, schedule, tables

DECKS = ("a", "b", "c", "d")


def test_schedule_balances_seats_and_extends_without_moving_games() -> None:
    one = schedule("pack", DECKS, deals=1, deal_base=100, opponent_seed_base=900)
    two = schedule("pack", DECKS, deals=2, deal_base=100, opponent_seed_base=900)
    assert two[: len(one)] == one
    assert [game.deal for game in two] == list(range(100, 132))
    for side in ("llm_deck", "opponent_deck"):
        on_play = Counter(getattr(game, side) for game in one if game.llm_seat == 0)
        assert set(on_play.values()) == {2}
    seats = Counter((game.llm_deck, game.opponent_deck, game.llm_seat) for game in two)
    assert len(seats) == 2 * len(DECKS) ** 2 and set(seats.values()) == {1}


def test_only_a_completed_replayed_game_scores() -> None:
    game = Game(
        pack="pack",
        llm_deck="a",
        opponent_deck="b",
        llm_seat=1,
        deal=7,
        opponent_seed=8,
    )
    won = GameSummary(mode="fixture", completed=True, winner=1, reported_cost_usd=0.25)
    found = tables(game, won, llm=0, record=b"record")
    llm, opponent = found["games"]
    assert (llm["seat"], llm["deck"], llm["game_seat"], llm["score"]) == (
        0,
        "a",
        1,
        1.0,
    )
    assert (opponent["seat"], opponent["game_seat"], opponent["score"]) == (1, 0, 0.0)
    (usage,) = found["llm_usage"]
    assert usage["seat"] == 0 and usage["reported_cost_usd"] == 0.25
    failed = GameSummary(
        mode="fixture", failure="TimeoutError: game wall limit reached"
    )
    assert {
        row["score"] for row in tables(game, failed, llm=0, record=b"")["games"]
    } == {None}
