import pytest

from manabot.study.allies_lessons.__main__ import parse_set
from manabot.study.allies_lessons.config import StudyConfig


def test_config_round_trips_and_overrides(tmp_path):
    config = StudyConfig(mirror_deals=3, ur_lessons="+1 Tiger-Seal, -1 Pop Quiz")
    config.save(tmp_path / "study.json")
    loaded = StudyConfig.load(tmp_path / "study.json")
    assert loaded == config
    changed = loaded.changed(
        **parse_set(["mirror_deals=5", "gw_allies=-1 Yip Yip!, +1 Plains"])
    )
    assert changed.mirror_deals == 5
    assert changed.lists()["gw_allies"]["Plains"] == 10
    assert changed.lists()["ur_lessons"]["Tiger-Seal"] == 3
    assert loaded.mirror_deals == 3


def test_unknown_fields_are_rejected():
    with pytest.raises(ValueError):
        StudyConfig().changed(mirror_deal=5)


def test_schedule_follows_the_config():
    config = StudyConfig(mirror_deals=2, versus_deals=1, baseline_deals=3)
    assert len(config.games()) == 2 * 2 + 1 * 4 + 3 * 2
    mirror_only = config.changed(versus_deals=0, baseline_deals=0, keep_decisions=False)
    specs = mirror_only.games()
    assert len(specs) == 4
    assert not any(spec.keep_decisions for spec in specs)
    same = StudyConfig(subject="random", baseline="random", mirror_deals=2)
    assert len(same.games()) == 4
