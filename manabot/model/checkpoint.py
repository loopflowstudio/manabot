"""Deprecated: policy checkpoints moved to `manabot.backends.torch.checkpoint`."""

from manabot.infra.deprecation import moved

__getattr__ = moved(
    __name__,
    dict.fromkeys(
        (
            "load_checkpoint",
            "save_checkpoint",
            "load_etude_checkpoint",
            "local_checkpoint_file",
            "fetch_etude_checkpoint",
        ),
        "manabot.backends.torch.checkpoint",
    ),
)
