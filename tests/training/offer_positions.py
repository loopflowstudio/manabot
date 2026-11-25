"""Real decisions with a known small legal-offer support."""

from manabot.env.match import Match
from manabot.env.observation import ObservationSpace, encode_policy_batch
from manabot.infra.config import ObservationConfig
from manabot.policy.spans import ObservationSelection
import managym

# Four offer slots: three legal offers and one padding slot.
FOUR_OFFERS = ObservationSpace(ObservationConfig(max_actions=4))


def three_offer_root() -> managym.Observation:
    """A real decision with three legal offers; the first two share a type."""
    env = managym.Env(seed=0, skip_trivial=True)
    env.reset(Match().to_rust())
    for _ in range(3):
        env.step(0)
    obs = env.observation_for_player(env.current_agent_index())
    types = [int(action.action_type) for action in obs.action_space.actions]
    assert len(types) == 3 and types[0] == types[1] != types[2]
    return obs


def three_offers() -> ObservationSelection:
    batch = encode_policy_batch(FOUR_OFFERS, [three_offer_root()])
    return ObservationSelection(FOUR_OFFERS, ((batch, 0),))
