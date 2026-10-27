<<<<<<< HEAD
"""Executable training regimes over the existing manabot trainers."""

from .models import TrainingRegime, TrainingRun
=======
"""Training regimes and run records; execute_regime orchestrates their execution."""

from .records import TrainingRun
from .regime import TrainingRegime
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f

__all__ = ["TrainingRegime", "TrainingRun"]
