import pytest

from manabot.study.allies_lessons import balance
from manabot.study.allies_lessons.matchup import authored_lists, schedule
from manabot.study.record import record_game


def test_edits_parse_and_apply_without_touching_the_source():
    edits = balance.parse_edits("+2 Tiger-Seal, -2 Pop Quiz")
    assert edits == [(2, "Tiger-Seal"), (-2, "Pop Quiz")]
    lists = authored_lists()
    changed = balance.edited(lists, "ur_lessons", edits)
    assert changed["ur_lessons"]["Tiger-Seal"] == 4
    assert "Pop Quiz" not in changed["ur_lessons"]
    assert lists["ur_lessons"]["Pop Quiz"] == 2
    assert sum(changed["ur_lessons"].values()) == sum(lists["ur_lessons"].values())
    assert balance.describe(changed) == {
        "ur_lessons": "-2 Pop Quiz, +2 Tiger-Seal",
        "gw_allies": "authored",
    }


def test_bad_edits_are_rejected():
    with pytest.raises(ValueError):
        balance.parse_edits("two Tiger-Seal")
    with pytest.raises(ValueError):
        balance.edited(authored_lists(), "ur_lessons", [(-3, "Pop Quiz")])


def test_removal_sweep_keeps_deck_size():
    lists = authored_lists()
    candidates = balance.removals(lists)
    assert candidates[0] == ("authored", lists)
    nonland = sum(card not in balance.LANDS for deck in lists.values() for card in deck)
    assert len(candidates) == nonland + 1
    for _, candidate in candidates:
        for deck in lists:
            assert sum(candidate[deck].values()) == sum(lists[deck].values())


def test_custom_lists_play_and_outcome_only_records_stay_small():
    lists = balance.edited(
        authored_lists(), "gw_allies", [(-2, "Kyoshi Warriors"), (2, "Glider Kids")]
    )
    specs = schedule(
        "random", "random", deals=1, seed=30, lists=lists, keep_decisions=False
    )
    games = [record_game(spec) for spec in specs]
    assert all(game.completed and not game.decisions for game in games)
    assert all(game.turns > 0 and game.first_player in (0, 1) for game in games)
    rate = balance.lessons_win_rate(games, "random")
    assert rate.total == 2


def test_results_page_orders_candidates_by_distance_from_even():
    def row(name, wins):
        return {
            "name": name,
            "changes": {"ur_lessons": "authored", "gw_allies": "-2 Badgermole Cub"},
            "player": "search",
            "lessons_wins": wins,
            "games": 100,
        }

    html = balance.results_page([row("far", 30), row("near", 48)])
    assert html.index("near") < html.index("far")
    assert "UR Lessons won 48 of 100" in html
