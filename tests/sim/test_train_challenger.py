"""Operator budget validation must reject unbounded runs before creating output."""

import pytest

from scripts import train_challenger


@pytest.mark.parametrize("seconds", ["nan", "inf", "-inf", "0"])
def test_rejects_invalid_wall_budget_before_starting_worker(
    seconds, tmp_path, monkeypatch, capsys
):
    out = tmp_path / "invalid-budget"
    monkeypatch.setattr(
        "sys.argv", ["train_challenger", "--out", str(out), f"--seconds={seconds}"]
    )

    def unexpected_worker(*args, **kwargs):
        pytest.fail("Invalid wall budget must not launch a training worker")

    monkeypatch.setattr(train_challenger.subprocess, "Popen", unexpected_worker)
    monkeypatch.setattr(train_challenger.platform, "platform", lambda: "test host")
    with pytest.raises(SystemExit) as error:
        train_challenger.main()
    assert error.value.code == 2
    assert "resource limits" in capsys.readouterr().err
    assert not out.exists()


OBSERVATION = {
    "max_actions": 128,
    "max_cards_per_player": 96,
    "max_permanents_per_player": 64,
}


def test_world_identity_binds_lesson_pool_and_tensor_shape():
    world = train_challenger.world_identity(OBSERVATION)
    assert world["lesson_pool"] == {
        "ur_lessons": {
            "Accumulate Wisdom": 1,
            "Firebending Lesson": 1,
            "It'll Quench Ya!": 1,
        },
        "gw_allies": {"Fancy Footwork": 1, "Yip Yip!": 1},
    }
    assert world["observation_action_abi"]["shapes"]["actions"] == [128, 17]
    assert "LEARN_TAKE_LESSON" in world["observation_action_abi"]["action_types"]
    assert world["content_manifest"]["compiled_semantics"]["pack_key"] == (
        train_challenger.PACK_KEY
    )

    narrower = train_challenger.world_identity({**OBSERVATION, "max_actions": 64})
    assert narrower["lesson_pool_sha256"] == world["lesson_pool_sha256"]
    assert (
        narrower["observation_action_abi_sha256"]
        != world["observation_action_abi_sha256"]
    )


def test_admission_covers_learn_and_rejects_omitted_or_unbound_data(tmp_path):
    from manabot.env.observation import ActionSpaceEnum
    from manabot.sim.distill import load_shards

    recipe = {
        "seed": 81,
        "teacher": {"kind": "search", "sims": 4, "max_steps": 2000},
        "observation": OBSERVATION,
        "world": train_challenger.world_identity(OBSERVATION),
    }
    shards = [tmp_path / f"shard_{index}.npz" for index in range(2)]
    summaries = [
        train_challenger.teacher_game(recipe, index, shard)
        for index, shard in enumerate(shards)
    ]
    dataset = load_shards(shards)

    report = train_challenger.admit(dataset, summaries, recipe)
    assert report["admitted"], report["failures"]
    assert report["omitted_legal_choices"] == 0
    assert report["learn"]["decisions_offering_a_lesson"] > 0
    assert report["learn"]["omitted_legal_choices"] == 0

    learn_row = int((dataset["decision_kind"] == int(ActionSpaceEnum.LEARN)).argmax())
    omitted = {**dataset, "num_valid": dataset["num_valid"].copy()}
    omitted["num_valid"][learn_row] -= 1
    report = train_challenger.admit(omitted, summaries, recipe)
    assert not report["admitted"]
    assert report["learn"]["omitted_legal_choices"] == 1

    without_learn = {
        **dataset,
        "actions": dataset["actions"] * 0,
        "decision_kind": dataset["decision_kind"] * 0 + 1,
    }
    assert (
        "no Learn decision offered a Lesson"
        in (train_challenger.admit(without_learn, summaries, recipe)["failures"])
    )

    summaries[0]["provenance"]["match"]["hero_sideboard"] = {}
    report = train_challenger.admit(dataset, summaries, recipe)
    assert report["failures"] == [
        "a teacher game used a different deck or sideboard setup"
    ]
