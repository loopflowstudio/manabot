"""The projection says what each stack entry targets."""

import random

import managym
from managym.experience import LegacyHeroObservation
from managym.experience_projection import decision_surface_view


def targeted_bolt(want_permanent: bool) -> tuple[managym.Env, managym.Observation]:
    """Play random games of Bolts and Ogres until a Bolt on the stack has the wanted target."""
    setup = managym.GameSetup(
        [
            managym.PlayerConfig(
                name, {"Mountain": 12, "Lightning Bolt": 12, "Gray Ogre": 6}
            )
            for name in ("a", "b")
        ]
    )
    for seed in range(200):
        env = managym.Env(seed=seed, skip_trivial=True)
        observation, _ = env.reset(setup)
        rng = random.Random(seed)
        for _ in range(400):
            if env.is_game_over():
                break
            viewer = env.observation_for_player(0)
            targets = [
                target for entry in viewer.stack_objects for target in entry.targets
            ]
            if targets and all(
                (target.permanent_id is not None) == want_permanent
                for target in targets
            ):
                return env, viewer
            observation, *_ = env.step(
                rng.randrange(len(observation.action_space.actions))
            )
    raise AssertionError("no game reached a targeted Lightning Bolt")


def test_stack_entry_names_a_targeted_player() -> None:
    env, viewer = targeted_bolt(want_permanent=False)
    [entry] = decision_surface_view(env, 0)["stack_objects"]
    [target] = viewer.stack_objects[0].targets
    assert entry["targets"] == [{"kind": "player", "id": int(target.player_id)}]


def test_stack_entry_names_a_targeted_permanent_as_the_object_on_the_board() -> None:
    env, viewer = targeted_bolt(want_permanent=True)
    view = decision_surface_view(env, 0)
    [entry] = view["stack_objects"]
    [target] = entry["targets"]
    on_board = {
        permanent["id"]
        for side in ("agent", "opponent")
        for permanent in view[side]["battlefield"]
    }
    assert target["kind"] == "object"
    assert target["id"]["entity"] == int(
        viewer.stack_objects[0].targets[0].permanent_id
    )
    assert target["id"]["entity"] in on_board
    LegacyHeroObservation.model_validate(view)


def test_untargeted_entries_omit_targets() -> None:
    env, _ = targeted_bolt(want_permanent=False)
    view = decision_surface_view(env, 0)
    view["stack_objects"][0]["targets"] = []
    dumped = LegacyHeroObservation.model_validate(view).model_dump(mode="json")
    assert "targets" not in dumped["stack_objects"][0]
