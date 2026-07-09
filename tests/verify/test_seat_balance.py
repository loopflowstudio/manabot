"""Tests for seat-balanced evaluation (C0.5 protocol amendment A1).

The managym engine always gives player 0 the first turn, so seat balancing
works by swapping which PlayerConfig is handed to the engine as player 0.
These tests cover the Match swap, the per-seat summary accounting, and — the
part that matters — that the swap actually reaches the engine (an asymmetric
matchup whose winner is deck-determined must flip player index with the seat).
"""

from manabot.env import Match
from manabot.verify.decision_profile import (
    GameProfile,
    play_profile_games,
    summarize_profiles,
)
from manabot.verify.util import STANDARD_DECK, build_hypers

MOUNTAIN_ONLY = {"Mountain": 40}


def _make_profile(game_index: int, hero_seat: int, winner: int | None) -> GameProfile:
    return GameProfile(
        game_index=game_index,
        winner=winner,
        hero_seat=hero_seat,
        turns=10,
        surfaced_total=0,
        surfaced_hero=0,
        surfaced_villain=0,
        skipped=0,
        collapse_ratio=0.0,
    )


def test_match_swapped_exchanges_seats():
    hypers = build_hypers(
        match={"hero_deck": MOUNTAIN_ONLY, "villain_deck": STANDARD_DECK}
    )
    match = Match(hypers.match)
    swapped = match.swapped()

    assert swapped.hero == match.villain
    assert swapped.villain == match.hero
    assert swapped.hero_deck == match.villain_deck
    assert swapped.villain_deck == match.hero_deck

    # Player 0 in the engine is the first config; swapping must reorder.
    configs = match.to_rust()
    swapped_configs = swapped.to_rust()
    assert configs[0].name == swapped_configs[1].name
    assert configs[1].name == swapped_configs[0].name
    # Original must be untouched.
    assert match.hero_deck == MOUNTAIN_ONLY


def test_summarize_profiles_reports_per_seat_win_rates():
    # Hero wins both games on the play (winner == hero_seat == 0) and loses
    # both on the draw (winner == 0 != hero_seat == 1).
    profiles = [
        _make_profile(0, hero_seat=0, winner=0),
        _make_profile(1, hero_seat=1, winner=0),
        _make_profile(2, hero_seat=0, winner=0),
        _make_profile(3, hero_seat=1, winner=0),
    ]
    summary = summarize_profiles(profiles)

    assert summary["hero_wins"] == 2
    assert summary["hero_win_rate"] == 0.5
    assert summary["per_seat"]["0"]["games"] == 2
    assert summary["per_seat"]["0"]["hero_wins"] == 2
    assert summary["per_seat"]["0"]["hero_win_rate"] == 1.0
    assert summary["per_seat"]["1"]["games"] == 2
    assert summary["per_seat"]["1"]["hero_wins"] == 0
    assert summary["per_seat"]["1"]["hero_win_rate"] == 0.0
    # Player 0 (on the play) won all four games regardless of role.
    assert summary["on_the_play_wins"] == 4
    assert summary["on_the_play_win_rate"] == 1.0


def test_seat_balancing_alternates_and_swap_reaches_engine():
    # Hero plays a deck with no creatures (Mountain-only) against the
    # standard creature deck: the hero must lose every game, and the winning
    # *player index* must track the seat swap — villain's deck sits in seat 0
    # exactly when the hero is on the draw.
    profiles = play_profile_games(
        hero_policy_name="random",
        villain_policy_name="random",
        num_games=4,
        seed=0,
        hero_deck=MOUNTAIN_ONLY,
        villain_deck=STANDARD_DECK,
        seat_balanced=True,
    )

    assert [p.hero_seat for p in profiles] == [0, 1, 0, 1]
    for profile in profiles:
        assert not profile.aborted
        assert profile.winner is not None
        # The creature deck (villain) wins from whichever seat it occupies.
        assert profile.winner == 1 - profile.hero_seat

    summary = summarize_profiles(profiles)
    assert summary["per_seat"]["0"]["games"] == 2
    assert summary["per_seat"]["1"]["games"] == 2
    assert summary["hero_wins"] == 0
