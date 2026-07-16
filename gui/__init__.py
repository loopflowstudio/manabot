"""GUI backend package."""

from typing import Any

__all__ = ["app"]


def __getattr__(name: str) -> Any:
    """Keep ``gui.app`` compatible without importing the native engine eagerly."""
    if name == "app":
        from .server import app

        return app
    raise AttributeError(name)
