from manabot.study import measures as m
from manabot.study.allies_lessons import build_report, schedule
from manabot.study.record import ATTACK, read_games, record_games, write_games


def test_schedule_covers_every_seating():
    mirror = schedule("random", "random", deals=2, seed=10)
    assert len(mirror) == 4
    assert {spec.decks for spec in mirror} == {
        ("ur_lessons", "gw_allies"),
        ("gw_allies", "ur_lessons"),
    }
    versus = schedule("search", "random", deals=1, seed=10)
    assert sorted((spec.labels, spec.decks[0]) for spec in versus) == [
        (("random", "search"), "gw_allies"),
        (("random", "search"), "ur_lessons"),
        (("search", "random"), "gw_allies"),
        (("search", "random"), "ur_lessons"),
    ]
    assert len({spec.game_id for spec in versus}) == 4


def test_recorded_games_round_trip_and_render(tmp_path):
    games = list(record_games(schedule("random", "random", deals=2, seed=20)))
    assert all(game.completed for game in games), [game.error for game in games]

    for name in ("games.jsonl", "games.jsonl.gz"):
        path = tmp_path / name
        assert write_games(path, games) == 4
        assert read_games(path) == games
    assert (tmp_path / "games.jsonl.gz").stat().st_size < path.with_suffix(
        ""
    ).stat().st_size / 5

    decisions = [decision for game in games for decision in game.decisions]
    assert all(game.first_player in (0, 1) for game in games)
    attacks = [decision for decision in decisions if decision.kind == ATTACK]
    assert attacks
    for decision in attacks:
        assert decision.subject
        assert decision.declared == (not decision.label.startswith("Do not"))
    assert any(p.is_land and p.name for d in decisions for p in d.mine)
    assert m.land_drops(games, "random").total > 0

    html = build_report(games, subject="random", baseline="random")
    assert html.startswith("<!doctype html>")
    assert "Does it play a land every turn it can?" in html
    assert "4 games" in html
