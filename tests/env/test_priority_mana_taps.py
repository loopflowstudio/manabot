"""Priority mana offers survive setup, encoding and replay."""

from pathlib import Path

import numpy as np
import pytest

from manabot.env.observation import ObservationSpace
from manabot.infra.config import ObservationConfig
import managym
from managym.recording import GameRecord, ScalarCapture


def forests() -> list[managym.PlayerConfig]:
    return [managym.PlayerConfig(seat, {"Forest": 12}) for seat in ("a", "b")]


def test_priority_tap_encodes_and_replays(tmp_path: Path) -> None:
    setup = managym.GameSetup(forests(), priority_mana_taps=(True, False))
    assert managym.GameSetup(setup).priority_mana_taps == (True, False)
    assert setup[::-1].priority_mana_taps == (False, True)
    vector = managym.VectorEnv(num_envs=1, seed=3)
    vector.reset_all(setup)
    assert vector.current_player_configs()[0].priority_mana_taps == (True, False)

    env = managym.Env(seed=3)
    capture = ScalarCapture(3, games_path=tmp_path / "games.jsonl")
    obs, _ = capture.reset(env, setup)
    play_land = next(
        i
        for i, a in enumerate(obs.action_space.actions)
        if a.action_type == managym.ActionEnum.PRIORITY_PLAY_LAND
    )
    obs, *_ = capture.step(env, play_land)
    taps = [
        i
        for i, a in enumerate(obs.action_space.actions)
        if a.action_type == managym.ActionEnum.TAP_FOR_COST
    ]
    assert len(taps) == 1 and obs.action_space.actions[taps[0]].focus
    space = ObservationSpace(ObservationConfig())
    reference = space.encode(obs)
    native = space.encoder.allocate()
    env.encode_observation_into(obs, native)
    for key in reference:
        np.testing.assert_array_equal(reference[key], native[key], err_msg=key)
    after, *_ = capture.step(env, taps[0])
    capture.close()
    (record,) = GameRecord.model_validate_json(
        (tmp_path / "games.jsonl").read_bytes()
    ).games
    replay = managym.Env(seed=record.seed)
    replay.reset(record.setup())
    for action in record.actions:
        assert isinstance(action, int)
        replayed, *_ = replay.step(action)
    assert replayed.toJSON() == after.toJSON()
    assert replay.state_digest() == env.state_digest()


def test_default_and_explicit_off_play_and_record_identically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("managym.recording.time.perf_counter", lambda: 3.0)
    seats = forests()
    games = []
    for setup in (
        managym.GameSetup(seats),
        managym.GameSetup(seats, priority_mana_taps=(False, False)),
    ):
        env = managym.Env(seed=3)
        path = tmp_path / f"game-{len(games)}.jsonl"
        capture = ScalarCapture(3, games_path=path)
        obs, _ = capture.reset(env, setup)
        frames = [obs.toJSON()]
        for _ in range(200):
            obs, _, done, truncated, _ = capture.step(env, 0)
            frames.append(obs.toJSON())
            if done or truncated:
                break
        else:
            pytest.fail("fixture game did not finish")
        capture.close()
        games.append((frames, path.read_bytes(), env.state_digest()))
    assert games[0] == games[1]
    assert b"priority_mana_taps" not in games[0][1]


@pytest.mark.parametrize("invalid", [(True,), (True, False, True), (1, False), "no"])
def test_priority_mana_taps_requires_two_booleans(invalid: object) -> None:
    with pytest.raises(ValueError, match="two booleans"):
        managym.GameSetup(forests(), priority_mana_taps=invalid)  # type: ignore[arg-type]


def test_skip_count_includes_priority_passes_after_playing_a_land() -> None:
    off = managym.Env(seed=3)
    on = managym.Env(seed=3)
    seats = forests()
    for env, taps in ((off, False), (on, True)):
        obs, _ = env.reset(managym.GameSetup(seats, priority_mana_taps=(taps, False)))
        before = env.skip_trivial_count()
        land = next(
            i
            for i, action in enumerate(obs.action_space.actions)
            if action.action_type == managym.ActionEnum.PRIORITY_PLAY_LAND
        )
        env.step(land)
        if taps:
            assert env.skip_trivial_count() == before
        else:
            # With only lands, the next surfaced decision is the opponent's land.
            # At least both seats' passes at the current main phase were skipped.
            assert env.skip_trivial_count() >= before + 2
