"""Deprecated: the torch policy moved to `manabot.backends.torch.policy`."""

from manabot.infra.deprecation import moved

__getattr__ = moved(
    __name__,
    dict.fromkeys(
        (
            "ActorCritic",
            "EmbeddedObservation",
            "MeanPoolingLayer",
            "AttentionReadout",
            "ProjectionLayer",
            "GameObjectAttention",
            "layer_init",
        ),
        "manabot.backends.torch.policy",
    ),
)
