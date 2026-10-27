# Local imports
<<<<<<< HEAD
from manabot.infra import AgentSpec, TrainHypers

# Local directory imports
from .agent import Agent
from .train import Trainer

__all__ = ["Agent", "Trainer", "TrainHypers", "AgentSpec"]
=======
from manabot.infra import ModelConfig

# Local directory imports
from .actor_critic import ActorCritic

__all__ = ["ActorCritic", "ModelConfig"]
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f
