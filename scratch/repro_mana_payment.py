"""Expose the 0.2.6 empty decision without leaving a spinning worker."""

import managym


def probe(second_land: str, life: int) -> None:
    spell = "Sokka, Bold Boomeranger"
    citadel = "Tarnished Citadel"
    board = (citadel, second_land, "Heartbeat of Spring")
    deck = {name: 1 for name in (spell, *board)}
    deck["Island"] = 30
    env = managym.Env(seed=111, skip_trivial=False)
    setup = managym.GameSetup(
        [
            managym.PlayerConfig("payer", deck),
            managym.PlayerConfig("opponent", {"Forest": 40}),
        ],
        mana_payment=("manual", "manual"),
    )
    obs, _ = env.reset(setup)
    while obs.turn.turn_number == 0:
        obs, *_ = env.step(0)
    env.scenario_clear_hand(0)
    env.scenario_set_life(0, life)
    for name in board:
        env.scenario_force_battlefield(0, name)
    env.scenario_force_card_in_hand(0, spell)
    obs = env.scenario_refresh()
    while obs.turn.turn_number == 1:
        actions = obs.action_space.actions
        cast = next(
            (
                i
                for i, a in enumerate(actions)
                if a.action_type == managym.ActionEnum.PRIORITY_CAST_SPELL
            ),
            None,
        )
        if cast is not None:
            obs, *_ = env.step(cast)
            break
        obs, *_ = env.step(
            next(
                i
                for i, a in enumerate(actions)
                if a.action_type == managym.ActionEnum.PRIORITY_PASS_PRIORITY
            )
        )
    else:
        raise AssertionError("Sokka was never offered")

    land = next(p for p in obs.agent_permanents if p.characteristics.name == citadel)
    activation = next(
        (
            i
            for i, a in enumerate(obs.action_space.actions)
            if a.action_type == managym.ActionEnum.PRIORITY_ACTIVATE_ABILITY
            and land.id in a.focus
            and a.ability_index == 0
        ),
        None,
    )
    if activation is None:
        print(f"{second_land}, {life} life: Citadel colored activation absent")
        return
    obs, *_ = env.step(activation)
    modes = [a.mode_index for a in obs.action_space.actions]
    print(
        f"{second_land}, {life} life: {obs.action_space.action_space_type}, modes {modes}"
    )


if __name__ == "__main__":
    probe("Airship Engine Room", 2)
    probe("Airship Engine Room", 3)
    probe("Sunbaked Canyon", 3)
    probe("Sunbaked Canyon", 5)
