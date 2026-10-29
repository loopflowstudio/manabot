"""Explicit existing-card root for trigger-order proofs across consumers.

Setup precedes the recorded root. Every subsequent choice uses ordinary native
actions or Commands; no waiting trigger or card definition is injected.
"""

import managym

DECK: dict[str, int] = {
    "Swamp": 32,
    "Bastion of Remembrance": 2,
    "Callous Inspector": 2,
    "Deadly Dispute": 4,
}
SEED = 7


def configure_root(env: managym.Env) -> managym.Observation:
    """Apply the retained root recipe to a newly reset, hero-priority environment."""
    for seat in range(2):
        env.scenario_clear_hand(seat)
    env.scenario_set_life(0, 1)
    env.scenario_set_life(1, 20)
    env.scenario_force_card_in_hand(0, "Deadly Dispute")
    for name in ("Bastion of Remembrance", "Callous Inspector", "Swamp", "Swamp"):
        env.scenario_force_battlefield(0, name, ready=True)
    return env.scenario_refresh()


def root() -> tuple[managym.Env, managym.Observation]:
    env = managym.Env(seed=SEED)
    env.reset(
        [
            managym.PlayerConfig("Hero", DECK),
            managym.PlayerConfig("Villain", {"Swamp": 40}),
        ]
    )
    return env, configure_root(env)


def prefix_index(obs: managym.Observation) -> int:
    """Choose the explicit cast/sacrifice/payment prefix, without ordering triggers."""
    actions = obs.action_space.actions
    for kind in (
        managym.ActionEnum.PRIORITY_CAST_SPELL,
        managym.ActionEnum.SELECT_CARD,
        managym.ActionEnum.PAY_COST,
    ):
        for index, action in enumerate(actions):
            if action.action_type == kind:
                return index
    raise AssertionError(f"unexpected trigger-demo prefix: {obs.toJSON()}")


def ordering_root() -> tuple[managym.Env, managym.Observation]:
    env, obs = root()
    for _ in range(8):
        if obs.action_space.action_space_type == managym.ActionSpaceEnum.ORDER_TRIGGERS:
            return env, obs
        obs, _, done, _, _ = env.step(prefix_index(obs))
        assert not done
    raise AssertionError("trigger order was not reached through the bounded prefix")
