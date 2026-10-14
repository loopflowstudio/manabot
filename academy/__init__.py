"""Academy experience server package: authoritative play, presentation, study."""

from typing import Any

__all__ = ["app"]


def __getattr__(name: str) -> Any:
    """Keep ``academy.app`` importable without loading the native engine eagerly."""
    if name == "app":
        from .server import app

        return app
    raise AttributeError(name)
